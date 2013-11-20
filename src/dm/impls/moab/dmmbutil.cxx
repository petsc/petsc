#include <petsc-private/dmmbimpl.h> /*I  "petscdm.h"   I*/
#include <petsc-private/vecimpl.h> /*I  "petscdm.h"   I*/

#include <petscdmmoab.h>
#include <MBTagConventions.hpp>
#include <moab/ReadUtilIface.hpp>
#include <moab/ScdInterface.hpp>
#include <moab/CN.hpp>


#undef __FUNCT__
#define __FUNCT__ "DMMoabComputeDomainBounds_Private"
PetscErrorCode DMMoabComputeDomainBounds_Private(moab::ParallelComm* pcomm, PetscInt dim, PetscInt neleglob, PetscInt *ise)
{
  PetscInt size,rank;
  PetscInt fraction,remainder;
  PetscInt neleadim;
  PetscInt starts[3],sizes[3];

  PetscFunctionBegin;
  if(dim<1 && dim>3) SETERRQ1(PETSC_COMM_WORLD,PETSC_ERR_ARG_OUTOFRANGE,"The problem dimension is invalid: %D",dim);
  
  size=pcomm->size();
  rank=pcomm->rank();
  neleadim=(dim==3?neleglob*neleglob:(dim==2?neleglob:1));
  fraction=neleglob/size;    /* partition only by the largest dimension */
  remainder=neleglob%size;   /* remainder after partition which gets evenly distributed by round-robin */

  if(fraction==0) SETERRQ2(PETSC_COMM_WORLD,PETSC_ERR_ARG_OUTOFRANGE,"The leading dimension size should be greater than number of processors: %D < %D",neleglob,size);

  PetscPrintf(PETSC_COMM_SELF, "[%D] Input Dim=%D IFR=[%D]; IREM=[%D]; NCOUNT=[%D]\n", rank, dim, fraction, remainder, neleadim);

  starts[0]=starts[1]=starts[2]=0;       /* default dimensional element = 1 */
  sizes[0]=sizes[1]=sizes[2]=neleglob;   /* default dimensional element = 1 */

  if(rank < remainder) {
    /* This process gets "fraction+1" elements */
    sizes[dim-1] = (fraction + 1);
    starts[dim-1] = rank * (fraction+1);
  } else {
    /* This process gets "fraction" elements */
    sizes[dim-1] = fraction;
    starts[dim-1] = (remainder*(fraction+1) + fraction*(rank-remainder));
  }

  for(int i=dim-1; i>=0; --i) {
    ise[2*i]=starts[i];ise[2*i+1]=starts[i]+sizes[i];
    PetscPrintf(PETSC_COMM_SELF, "[%D] Dim=%D ISTART=[%D]; IEND=[%D]; NCOUNT=[%D]\n", rank, i, ise[2*i], ise[2*i+1], sizes[i]);
  }
  PetscPrintf(PETSC_COMM_SELF, "[%D] X=[%D, %D]; Y=[%D,%D]; Z=[%D,%D]\n", rank, ise[0], ise[1], ise[2], ise[3], ise[4], ise[5]);
  
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "DMMoabCreateBoxMesh"
PetscErrorCode DMMoabCreateBoxMesh(MPI_Comm comm, PetscInt dim, PetscInt nele, PetscInt nghost, DM *dm)
{
  PetscErrorCode ierr;
  moab::ErrorCode merr;
  PetscInt  rank,nprocs;
  DM_Moab        *dmmoab;
  moab::Interface *mbiface;
  moab::ParallelComm *pcomm;
  moab::Tag  id_tag=PETSC_NULL;
  moab::Range range;
  moab::EntityType etype;
  moab::ScdInterface *scdiface;
  PetscInt    ise[6];
  PetscReal   xse[6];

  // Determine which elements (cells) the current process owns:
  const PetscInt npts=nele+1;
  PetscInt my_nele,my_npts;      // Number of elements owned by this process
  PetscInt my_estart;    // The starting element for this process
  PetscInt vpere;

  PetscFunctionBegin;
  ierr = DMMoabCreateMoab(comm, PETSC_NULL, PETSC_NULL, PETSC_NULL, PETSC_NULL, dm);CHKERRQ(ierr);

  dmmoab = (DM_Moab*)(*dm)->data;
  mbiface = dmmoab->mbiface;
  pcomm = dmmoab->pcomm;
  id_tag = dmmoab->ltog_tag;

  nprocs = pcomm->size();
  rank = pcomm->rank();

  // Begin with some error checking:
  if(npts < 2) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_ARG_OUTOFRANGE,"Number of points must be >= 2");
  if(nprocs >= npts) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_ARG_OUTOFRANGE,"Number of processors must be less than number of points");

  // No errors,proceed with building the mesh:
  merr = mbiface->query_interface(scdiface);MBERRNM(merr); // get a ScdInterface object through moab instance
  
  moab::ReadUtilIface* readMeshIface;
  mbiface->query_interface(readMeshIface);

  ierr = PetscMemzero(ise,sizeof(PetscInt)*6);CHKERRQ(ierr);  
  ierr = DMMoabComputeDomainBounds_Private(pcomm, dim, nele, ise);CHKERRQ(ierr);

  my_estart = ise[2*(dim-1)];
  switch(dim) {
   case 1:
    vpere = 2;
    my_nele = (ise[1]-ise[0]);
    my_npts = (ise[1]-ise[0]+1);
    etype = moab::MBEDGE;
    break;
   case 2:
    vpere = 4;
    my_nele = (ise[1]-ise[0])*(ise[3]-ise[2]);
    my_npts = (ise[1]-ise[0]+1)*(ise[3]-ise[2]+1);
    etype = moab::MBQUAD;
    break;
   case 3:
    vpere = 8;
    my_nele = (ise[1]-ise[0])*(ise[3]-ise[2])*(ise[5]-ise[4]);
    my_npts = (ise[1]-ise[0]+1)*(ise[3]-ise[2]+1)*(ise[5]-ise[4]+1);
    etype = moab::MBHEX;
    break;
  }

  /* we have a domain of size [1,1,1] - now compute local co-ordinate box */
  ierr = PetscMemzero(xse,sizeof(PetscReal)*6);CHKERRQ(ierr);  
  for(int i=0; i<6; ++i) {
    xse[i]=(PetscReal)ise[i]/nele;
    PetscPrintf(PETSC_COMM_SELF, "[%D] Coords %d ; nele = %D; [%D, %G]\n", rank, i, nele, ise[i], xse[i]);
  }
  PetscPrintf(PETSC_COMM_SELF, "[%D] Coords X=[%G, %G]; Y=[%G,%G]; Z=[%G,%G]\n", rank, xse[0], xse[1], xse[2], xse[3], xse[4], xse[5]);

  PetscPrintf(PETSC_COMM_SELF, "\n[%D] My start_ele = %D and tot_nele = %D\n", rank,my_estart, my_nele);

  /*
  // Compute the co-ordinates of vertices
  std::vector<double> coords(my_npts*vpere*3); // vertices_per_edge = 2, 3 doubles/point
  std::vector<int>    vgid(my_npts);
  int vcount=0;
  double hxyz=1.0/nele;
  for (int k = ise[4]; k <= ise[5]; k++) {
    for (int j = ise[2]; j <= ise[3]; j++) {
      for (int i = ise[0]; i <= ise[1]; i++, vcount++) {
        coords[vcount*3]   = i*hxyz;
        coords[vcount*3+1] = j*hxyz;
        coords[vcount*3+2] = k*hxyz;
        vgid[vcount] = (k*nele+j)*nele+i;
      }
    }
  }


  // 1. Creates a IxJxK structured mesh, which includes I*J*K vertices and (I-1)*(J-1)*(K-1) hexes.
  moab::ScdBox *box;
  moab::HomCoord low(ise[0], ise[2], ise[4]);
  moab::HomCoord high(ise[1], ise[3], ise[5]);
//  low.normalize(); high.normalize();
  merr = scdiface->construct_box(low, high, // low, high box corners in parametric space
                                 coords.data(), vcount,   // NULL coords vector and 0 coords (don't specify coords for now)
                                 box);      // box is the structured box object providing the parametric
                                            // structured mesh interface for this tensor grid of elements
  MBERRNM(merr);
  */

    // Create vertexes and set the coodinate of each vertex:
  moab::EntityHandle vfirst;
  std::vector<double*> vcoords;
  const int sequence_size = (my_nele + 2) + 1;
  merr = readMeshIface->get_node_coords(3,my_npts,0,vfirst,vcoords,sequence_size);MBERRNM(merr);

    // Compute the co-ordinates of vertices and global IDs
  std::vector<int>    vgid(my_npts);
  int vcount=0;
  double hxyz=1.0/nele;
  for (int k = ise[4]; k <= ise[5]; k++) {
    for (int j = ise[2]; j <= ise[3]; j++) {
      for (int i = ise[0]; i <= ise[1]; i++, vcount++) {
        vcoords[0][vcount] = i*hxyz;
        vcoords[1][vcount] = j*hxyz;
        vcoords[2][vcount] = k*hxyz;
        vgid[vcount] = (k*nele+j)*nele+i;
      }
    }
  }

  moab::Range ownedvtx,ownedelms;  
  merr = mbiface->get_entities_by_type(0,moab::MBVERTEX,ownedvtx,true);MBERRNM(merr);

  // Get the global ID tag. The global ID tag is applied to each
  // vertex. It acts as an global identifier which MOAB uses to
  // assemble the individual pieces of the mesh:
  merr = mbiface->tag_get_handle(GLOBAL_ID_TAG_NAME,id_tag);MBERRNM(merr);

  // set the global id for all the owned vertices
  merr = mbiface->tag_set_data(id_tag,ownedvtx,vgid.data());MBERRNM(merr);
  
  // Create elements between mesh points. This is done so that VisIt
  // will interpret the output as a mesh that can be plotted...
  moab::EntityHandle efirst;
  moab::EntityHandle *connectivity = 0;
  std::vector<int> subent_conn(vpere);

  merr = readMeshIface->get_element_connect (my_nele,vpere,etype,1,efirst,connectivity);MBERRNM(merr);

  int ecount=0;
  for (int k = ise[4]; k < std::max(ise[5],1); k++) {
    for (int j = ise[2]; j < std::max(ise[3],1); j++) {
      for (int i = ise[0]; i < std::max(ise[1],1); i++,ecount++) {
        const int offset = ecount*vpere;
        moab::CN::SubEntityVertexIndices(etype, dim, 0, subent_conn.data());

        switch(dim) {
          case 1:
            connectivity[offset+subent_conn[0]] = vfirst+i;
            connectivity[offset+subent_conn[1]] = vfirst+(i+1);
            break;
          case 2:
            connectivity[offset+subent_conn[0]] = vfirst+i+j*(nele+1);
            connectivity[offset+subent_conn[1]] = vfirst+(i+1)+j*(nele+1);
            connectivity[offset+subent_conn[2]] = vfirst+(i+1)+(j+1)*(nele+1);
            connectivity[offset+subent_conn[3]] = vfirst+i+(j+1)*(nele+1);
            break;
          case 3:
            connectivity[offset+subent_conn[0]] = vfirst+i+(nele+1)*(j+(nele+1)*k);
            connectivity[offset+subent_conn[1]] = vfirst+(i+1)+(nele+1)*(j+(nele+1)*k);
            connectivity[offset+subent_conn[2]] = vfirst+(i+1)+(nele+1)*((j+1)+(nele+1)*k);
            connectivity[offset+subent_conn[3]] = vfirst+i+(nele+1)*((j+1)+(nele+1)*k);
            connectivity[offset+subent_conn[4]] = vfirst+i+(nele+1)*(j+(nele+1)*(k+1));
            connectivity[offset+subent_conn[5]] = vfirst+(i+1)+(nele+1)*(j+(nele+1)*(k+1));
            connectivity[offset+subent_conn[6]] = vfirst+(i+1)+(nele+1)*((j+1)+(nele+1)*(k+1));
            connectivity[offset+subent_conn[7]] = vfirst+i+(nele+1)*((j+1)+(nele+1)*(k+1));
            break;
        }
      }
    }
  }
  merr = readMeshIface->update_adjacencies(efirst,my_nele,vpere,connectivity);MBERRNM(merr);
  
    // 2. Get the vertices and hexes from moab and check their numbers against I*J*K and (I-1)*(J-1)*(K-1), resp.
   // first '0' specifies "root set", or entire MOAB instance, second the entity dimension being requested
  merr = mbiface->get_entities_by_dimension(0, dim, ownedelms);MBERRNM(merr);

  if (my_nele != (int) ownedelms.size())
    SETERRQ2(PETSC_COMM_WORLD,PETSC_ERR_ARG_OUTOFRANGE,"Created the wrong number of elements! (%D!=%D)",my_nele,ownedelms.size());
  else if(my_npts != (int) ownedvtx.size())
    SETERRQ2(PETSC_COMM_WORLD,PETSC_ERR_ARG_OUTOFRANGE,"Created the wrong number of vertices! (%D!=%D)",my_npts,ownedvtx.size());    
  else
    PetscPrintf(PETSC_COMM_WORLD, "Created %D elements and %D vertices.\n", ownedelms.size(), ownedvtx.size());

    // 3. Loop over elements in 3 nested loops over i, j, k; for each (i,j,k):
  /*
  std::vector<double> coords(vpere*3); // vertices_per_edge = 2, 3 doubles/point
  std::vector<moab::EntityHandle> connect;
  int count=0;
  for (int k = ise[4]; k < std::max(ise[5],1); k++) {
    for (int j = ise[2]; j < std::max(ise[3],1); j++) {
      for (int i = ise[0]; i < std::max(ise[1],1); i++,count++) {
          // 3a. Get the element corresponding to (i,j,k)
        moab::EntityHandle ehandle = box->get_element(i, j, k);
        if (0 == ehandle) MBERRNM(moab::MB_FAILURE);

        PetscPrintf(PETSC_COMM_SELF, "[%D] element[%D,%D,%D]=%D\n", rank, i,j,k, ehandle);

          // 3b. Get the connectivity of the element
        merr = mbiface->get_connectivity(&ehandle, 1, connect);MBERRNM(merr); // get the connectivity, in canonical order

          // 3c. Get the coordinates of the vertices comprising that element
        merr = mbiface->get_coords(connect.data(), connect.size(), coords.data());MBERRNM(merr); // get the coordinates of those vertices

        for (int iv=0; iv<vpere; ++iv)
          PetscPrintf(PETSC_COMM_SELF, "[%D] \t iv=%D [X,Y,Z]=[%G, %G, %G]\n", rank, iv, coords[iv*3], coords[iv*3+1], coords[iv*3+2]);
        PetscPrintf(PETSC_COMM_SELF, "\n");
      }
    }
  }

  merr = readMeshIface->update_adjacencies(box->get_element(ise[0],ise[2],ise[4]),my_nele,vpere,connect.data());MBERRNM(merr);
  

    // 4. Release the structured mesh interface 
  mbiface->release_interface(scdiface); // tell MOAB we're done with the ScdInterface
  */

  // The global ID tag is applied to each
  // vertex. It acts as an global identifier which MOAB uses to
  // assemble the individual pieces of the mesh:
  // Set the global ID indices
//  std::vector<int> global_ids(my_npts);
//  for (int i = 0; i < my_npts; i++) {
//    global_ids[i] = i+my_estart;
//  }

  // set the global id for all the owned vertices
//  merr = mbiface->tag_set_data(id_tag,ownedvtx,global_ids.data());MBERRNM(merr);
  
  merr = pcomm->check_all_shared_handles();MBERRNM(merr);

  if (rank)
    reinterpret_cast<moab::Core*>(mbiface)->print_database();
    
  // resolve the shared entities by exchanging information to adjacent processors
  merr = mbiface->get_entities_by_type(0,etype,ownedelms,true);MBERRNM(merr);
  merr = pcomm->resolve_shared_ents(0,ownedelms,dim,0);MBERRNM(merr);

  // Reassign global IDs on all entities.
  merr = pcomm->assign_global_ids(0,dim,0,false,true);MBERRNM(merr);
  merr = pcomm->exchange_ghost_cells(dim,0,nghost,0,true);MBERRNM(merr);

  // Everything is set up, now just do a tag exchange to update tags
  // on all of the ghost vertexes:
  merr = pcomm->exchange_tags(id_tag,ownedvtx);MBERRNM(merr);
  merr = pcomm->exchange_tags(id_tag,ownedelms);MBERRNM(merr);

  // set the dimension of the mesh
  merr = mbiface->set_dimension(dim);MBERRNM(merr);


  std::stringstream sstr;
  sstr << "test_" << rank << ".vtk";
  mbiface->write_mesh(sstr.str().c_str());
  PetscFunctionReturn(0);
}




PetscErrorCode resolve_and_exchange(moab::ParallelComm* mbpc, PetscInt dim)
{
  moab::EntityHandle entity_set;
  moab::ErrorCode merr;
  moab::Interface *mbint=mbpc->get_moab();
  moab::Range range;
  moab::Tag tag;
  PetscInt rank=mbpc->rank();

  // Create the entity set:
  merr = mbint->create_meshset(moab::MESHSET_SET, entity_set);MBERRNM(merr);

  // Get a list of elements in the current set:
  merr = mbint->get_entities_by_dimension(0, dim, range, true);MBERRNM(merr);

  // Add entities to the entity set:
  merr = mbint->add_entities(entity_set, range);MBERRNM(merr);

  // Add the MATERIAL_SET tag to the entity set:
  merr = mbint->tag_get_handle(MATERIAL_SET_TAG_NAME, 1, moab::MB_TYPE_INTEGER, tag);MBERRNM(merr);
  merr = mbint->tag_set_data(tag, &entity_set, 1, &rank);MBERRNM(merr);

  // Set up partition sets. This is where MOAB is actually told what
  // entities each process owns:
  merr = mbint->get_entities_by_type_and_tag(0, moab::MBENTITYSET,
					    &tag, NULL, 1,
					    mbpc->partition_sets());MBERRNM(merr);

  // Finally, determine which entites are shared and exchange the
  // ghosted entities:
  merr = mbpc->resolve_shared_ents(0, -1, -1);MBERRNM(merr);
  merr = mbpc->exchange_ghost_cells(-1, 0, 1, 0, true);MBERRNM(merr);
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "DMMoabLoadFromFile_Private"
PetscErrorCode DMMoabLoadFromFile_Private(moab::ParallelComm* pcomm,PetscInt dim,PetscInt npts,PetscInt nghost)
{
//  moab::ErrorCode merr;
  PetscInt rank,nprocs;
//  moab::ScdInterface *scdiface;

  /*
  // Determine which elements (cells) this process owns:
  const PetscInt nele = npts-1;
  PetscInt my_nele; // Number of elements owned by this process
  PetscInt my_estart;    // The starting element for this process
  const PetscInt vertices_per_edge=2;
*/
  PetscFunctionBegin;
  MPI_Comm_size( PETSC_COMM_WORLD,&nprocs );
  MPI_Comm_rank( PETSC_COMM_WORLD,&rank );


//  std::stringstream sstr;
//  sstr << "test_" << rank << ".vtk";
//  mbiface->write_mesh(sstr.str().c_str());
  PetscFunctionReturn(0);
}



