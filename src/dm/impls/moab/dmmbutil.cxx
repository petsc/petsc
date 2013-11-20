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
  PetscInt starts[3],sizes[3];

  PetscFunctionBegin;
  if(dim<1 && dim>3) SETERRQ1(PETSC_COMM_WORLD,PETSC_ERR_ARG_OUTOFRANGE,"The problem dimension is invalid: %D",dim);
  
  size=pcomm->size();
  rank=pcomm->rank();
  fraction=neleglob/size;    /* partition only by the largest dimension */
  remainder=neleglob%size;   /* remainder after partition which gets evenly distributed by round-robin */

  if(fraction==0) SETERRQ2(PETSC_COMM_WORLD,PETSC_ERR_ARG_OUTOFRANGE,"The leading dimension size should be greater than number of processors: %D < %D",neleglob,size);

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
  }
  
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "DMMoabCreateBoxMesh"
PetscErrorCode DMMoabCreateBoxMesh(MPI_Comm comm, PetscInt dim, PetscInt nele, PetscInt nghost, DM *dm)
{
  PetscErrorCode ierr;
  moab::ErrorCode merr;
  PetscInt            nprocs;
  DM_Moab        *dmmoab;
  moab::Interface *mbiface;
  moab::ParallelComm *pcomm;
  moab::ReadUtilIface* readMeshIface;

  moab::Tag  id_tag=PETSC_NULL;
  moab::Range         ownedvtx,ownedelms;
  moab::EntityHandle  vfirst,efirst;
  std::vector<double*> vcoords;
  moab::EntityHandle  *connectivity = 0;
  std::vector<int>    subent_conn;
  moab::EntityType etype;
  PetscInt    ise[6];
  PetscReal   xse[6];

  const PetscInt npts=nele+1;        /* Number of points in every dimension */
  PetscInt vpere,locnele,locnpts;    /* Number of verts/element, vertices, elements owned by this process */

  PetscFunctionBegin;
  /* Create the basic DMMoab object and keep the default parameters created by DM impls */
  ierr = DMMoabCreateMoab(comm, PETSC_NULL, PETSC_NULL, PETSC_NULL, PETSC_NULL, dm);CHKERRQ(ierr);

  /* get all the necessary handles from the private DM object */
  dmmoab = (DM_Moab*)(*dm)->data;
  mbiface = dmmoab->mbiface;
  pcomm = dmmoab->pcomm;
  id_tag = dmmoab->ltog_tag;
  nprocs = pcomm->size();

  /* do some error checking */
  if(pow(npts,dim) < 2) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_ARG_OUTOFRANGE,"Number of points must be >= 2");
  if(nprocs > pow(nele,dim)) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_ARG_OUTOFRANGE,"Number of processors must be less than or equal to number of elements");

  /* No errors yet; proceed with building the mesh */
  merr = mbiface->query_interface(readMeshIface);MBERRNM(merr);

  ierr = PetscMemzero(ise,sizeof(PetscInt)*6);CHKERRQ(ierr);  

  /* call the collective routine that computes the domain bounds for a structured mesh using MOAB */
  ierr = DMMoabComputeDomainBounds_Private(pcomm, dim, nele, ise);CHKERRQ(ierr);

  /* set some variables based on dimension */
  switch(dim) {
   case 1:
    vpere = 2;
    locnele = (ise[1]-ise[0]);
    locnpts = (ise[1]-ise[0]+1);
    etype = moab::MBEDGE;
    break;
   case 2:
    vpere = 4;
    locnele = (ise[1]-ise[0])*(ise[3]-ise[2]);
    locnpts = (ise[1]-ise[0]+1)*(ise[3]-ise[2]+1);
    etype = moab::MBQUAD;
    break;
   case 3:
    vpere = 8;
    locnele = (ise[1]-ise[0])*(ise[3]-ise[2])*(ise[5]-ise[4]);
    locnpts = (ise[1]-ise[0]+1)*(ise[3]-ise[2]+1)*(ise[5]-ise[4]+1);
    etype = moab::MBHEX;
    break;
  }

  /* we have a domain of size [1,1,1] - now compute local co-ordinate box */
  ierr = PetscMemzero(xse,sizeof(PetscReal)*6);CHKERRQ(ierr);  
  for(int i=0; i<6; ++i) {
    xse[i]=(PetscReal)ise[i]/nele;
  }

  /* Create vertexes and set the coodinate of each vertex */
  const int sequence_size = (locnele + vpere) + 1;
  merr = readMeshIface->get_node_coords(3,locnpts,0,vfirst,vcoords,sequence_size);MBERRNM(merr);

  /* Compute the co-ordinates of vertices and global IDs */
  std::vector<int>    vgid(locnpts);
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

  merr = mbiface->get_entities_by_type(0,moab::MBVERTEX,ownedvtx,true);MBERRNM(merr);
  merr = mbiface->add_entities (dmmoab->fileset, ownedvtx);MBERRNM(merr);

  /* The global ID tag is applied to each owned
     vertex. It acts as an global identifier which MOAB uses to
     assemble the individual pieces of the mesh:
     Set the global ID indices */
  merr = mbiface->tag_set_data(id_tag,ownedvtx,vgid.data());MBERRNM(merr);
  
  /* Create elements between mesh points using the ReadUtilInterface 
     get the reference to element connectivities for all local elements from the ReadUtilInterface */
  merr = readMeshIface->get_element_connect (locnele,vpere,etype,1,efirst,connectivity);MBERRNM(merr);

  /* offset appropriately so that only local ID and not global ID numbers are set for connectivity array */
  vfirst-=vgid[0];

   /* 3. Loop over elements in 3 nested loops over i, j, k; for each (i,j,k):
         and then set the connectivity for each element appropriately */
  int ecount=0;
  subent_conn.resize(vpere);
  for (int k = ise[4]; k < std::max(ise[5],1); k++) {
    for (int j = ise[2]; j < std::max(ise[3],1); j++) {
      for (int i = ise[0]; i < std::max(ise[1],1); i++,ecount++) {
        const int offset = ecount*vpere;
        moab::CN::SubEntityVertexIndices(etype, dim, 0, subent_conn.data());

        switch(dim) {
          case 1:
            connectivity[offset+subent_conn[0]] = vfirst+i;
            connectivity[offset+subent_conn[1]] = vfirst+(i+1);
            PetscPrintf(PETSC_COMM_WORLD, "ELEMENT[%D,%D,%D]: CONNECTIVITY = %D, %D\n", i,j,k,connectivity[offset+subent_conn[0]], connectivity[offset+subent_conn[1]]);
            break;
          case 2:
            connectivity[offset+subent_conn[0]] = vfirst+i+j*(nele+1);
            connectivity[offset+subent_conn[1]] = vfirst+(i+1)+j*(nele+1);
            connectivity[offset+subent_conn[2]] = vfirst+(i+1)+(j+1)*(nele+1);
            connectivity[offset+subent_conn[3]] = vfirst+i+(j+1)*(nele+1);
            PetscPrintf(PETSC_COMM_WORLD, "ELEMENT[%D,%D,%D]: CONNECTIVITY = %D, %D, %D, %D\n", i,j,k,connectivity[offset+subent_conn[0]], connectivity[offset+subent_conn[1]], connectivity[offset+subent_conn[2]], connectivity[offset+subent_conn[3]]);
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
  merr = readMeshIface->update_adjacencies(efirst,locnele,vpere,connectivity);MBERRNM(merr);
  
  /* 2. Get the vertices and hexes from moab and check their numbers against I*J*K and (I-1)*(J-1)*(K-1), resp.
        first '0' specifies "root set", or entire MOAB instance, second the entity dimension being requested */
  merr = mbiface->get_entities_by_dimension(0, dim, ownedelms);MBERRNM(merr);
  merr = mbiface->add_entities (dmmoab->fileset, ownedelms);MBERRNM(merr);

//  merr = mbiface->unite_meshset(dmmoab->fileset, 0);MBERRV(mbiface,merr);

  if (locnele != (int) ownedelms.size())
    SETERRQ2(PETSC_COMM_WORLD,PETSC_ERR_ARG_OUTOFRANGE,"Created the wrong number of elements! (%D!=%D)",locnele,ownedelms.size());
  else if(locnpts != (int) ownedvtx.size())
    SETERRQ2(PETSC_COMM_WORLD,PETSC_ERR_ARG_OUTOFRANGE,"Created the wrong number of vertices! (%D!=%D)",locnpts,ownedvtx.size());    
  else
    PetscInfo2(*dm, "Created %D elements and %D vertices.\n", ownedelms.size(), ownedvtx.size());

  /* check the handles */
  merr = pcomm->check_all_shared_handles();MBERRV(mbiface,merr);

  /* resolve the shared entities by exchanging information to adjacent processors */
  merr = mbiface->get_entities_by_type(dmmoab->fileset,etype,ownedelms,true);MBERRNM(merr);
  merr = pcomm->resolve_shared_ents(dmmoab->fileset,ownedelms,dim,dim-1,&id_tag);MBERRV(mbiface,merr);

  /* Reassign global IDs on all entities. */
  merr = pcomm->assign_global_ids(dmmoab->fileset,dim,1,false,true);MBERRNM(merr);
  merr = pcomm->exchange_ghost_cells(dim,0,nghost,0,true);MBERRV(mbiface,merr);

  /* Everything is set up, now just do a tag exchange to update tags
     on all of the ghost vertexes */
  merr = pcomm->exchange_tags(id_tag,ownedvtx);MBERRV(mbiface,merr);
  merr = pcomm->exchange_tags(id_tag,ownedelms);MBERRV(mbiface,merr);

//  ierr = DMMoabSetLocalVertices(*dm, &ownedvtx);CHKERRQ(ierr);
//  ierr = DMMoabSetLocalElements(*dm, &ownedelms);CHKERRQ(ierr);

  merr = mbiface->set_dimension(dim);MBERRNM(merr);

  std::stringstream sstr;
  sstr << "test_" << pcomm->rank() << ".vtk";
  mbiface->write_mesh(sstr.str().c_str());
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "DMMoab_GetReadOptions_Private"
PetscErrorCode DMMoab_GetReadOptions_Private(PetscBool by_rank, PetscInt numproc, PetscInt dim, MoabReadMode mode, PetscInt dbglevel, const char* extra_opts, const char** read_opts)
{
  std::ostringstream str;

  PetscFunctionBegin;
  // do parallel read unless only one processor
  if (numproc > 1) {
    str << "PARALLEL=" << mode << ";PARTITION=PARALLEL_PARTITION;PARTITION_DISTRIBUTE;";
    str << "PARALLEL_RESOLVE_SHARED_ENTS;PARALLEL_GHOSTS=" << dim << ".0.1;";
    if (by_rank)
      str << "PARTITION_BY_RANK;";
  }

  if (extra_opts)
    str << extra_opts << ";";

  if (dbglevel)
    str << "DEBUG_IO=" << dbglevel << ";DEBUG_PIO=" << dbglevel << ";CPUTIME;";

  *read_opts = str.str().c_str();
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "DMMoabLoadFromFile"
PetscErrorCode DMMoabLoadFromFile(MPI_Comm comm,PetscInt dim,const char* filename, const char* usrreadopts, DM *dm)
{
  moab::ErrorCode merr;
  PetscInt        nprocs,dbglevel=0;
  PetscBool       part_by_rank=PETSC_FALSE;
  DM_Moab        *dmmoab;
  moab::Interface *mbiface;
  moab::ParallelComm *pcomm;
  moab::Range verts,elems;
  const char *readopts;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidPointer(dm,5);

  /* Create the basic DMMoab object and keep the default parameters created by DM impls */
  ierr = DMMoabCreateMoab(comm, PETSC_NULL, PETSC_NULL, PETSC_NULL, PETSC_NULL, dm);CHKERRQ(ierr);

  /* get all the necessary handles from the private DM object */
  dmmoab = (DM_Moab*)(*dm)->data;
  mbiface = dmmoab->mbiface;
  pcomm = dmmoab->pcomm;
  nprocs = pcomm->size();

  /* TODO: Use command-line options to control by_rank, verbosity, MoabReadMode and extra options */
  ierr  = PetscOptionsBegin(PETSC_COMM_WORLD, "", "Options for reading/writing MOAB based meshes from file", "DMMoab");
  ierr  = PetscOptionsInt("-dmmb_rw_dbg", "The verbosity level for reading and writing MOAB meshes", "dmmbutil.cxx", dbglevel, &dbglevel, NULL);CHKERRQ(ierr);
  ierr  = PetscOptionsBool("-dmmb_part_by_rank", "Use partition by rank when reading MOAB meshes from file", "dmmbutil.cxx", part_by_rank, &part_by_rank, NULL);CHKERRQ(ierr);
  ierr  = PetscOptionsEnd();

  /* add mesh loading options specific to the DM */
  ierr = DMMoab_GetReadOptions_Private(part_by_rank, nprocs, dim, MOAB_PARROPTS_READ_PART, dbglevel, usrreadopts, &readopts);CHKERRQ(ierr);

  PetscInfo2(*dm, "Reading file %s with options: %s",filename,readopts);

  /* Load the mesh from a file. */
  merr = mbiface->load_file(filename, &dmmoab->fileset, readopts);MBERRVM(mbiface,"Reading MOAB file failed.", merr);

  /* Reassign global IDs on all entities. */
  merr = pcomm->assign_global_ids(dmmoab->fileset,dim,1,false,true);MBERRNM(merr);
  merr = pcomm->exchange_ghost_cells(dim,0,1,0,true);MBERRV(mbiface,merr);

  merr = pcomm->collective_sync_partition();MBERR("Collective sync failed", merr);

  merr = mbiface->set_dimension(dim);MBERRNM(merr);

  PetscInfo3(*dm, "MOAB file '%s' was successfully loaded. Found %D vertices and %D elements.\n", filename, verts.size(), elems.size());
  PetscFunctionReturn(0);
}

