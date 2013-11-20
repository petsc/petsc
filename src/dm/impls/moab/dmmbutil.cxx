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

static void set_structured_coordinates(PetscInt i, PetscInt j, PetscInt k, PetscReal hxyz, PetscInt vcount, std::vector<double*>& vcoords)
{
  vcoords[0][vcount] = i*hxyz;
  vcoords[1][vcount] = j*hxyz;
  vcoords[2][vcount] = k*hxyz;
}

static void set_element_connectivity(PetscInt dim, moab::EntityType etype, PetscInt offset, PetscInt nele, PetscInt i, PetscInt j, PetscInt k, PetscInt vfirst, moab::EntityHandle *connectivity)
{
  std::vector<int>    subent_conn(pow(2,dim));  /* only linear edge, quad, hex supported now */

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
    default:
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

#undef __FUNCT__
#define __FUNCT__ "DMMoabCreateBoxMesh"
PetscErrorCode DMMoabCreateBoxMesh(MPI_Comm comm, PetscInt dim, PetscInt nele, PetscInt nghost, DM *dm)
{
  PetscErrorCode  ierr;
  moab::ErrorCode merr;
  PetscInt        i,j,k,n,nprocs,rank;
  DM_Moab        *dmmoab;
  moab::Interface *mbiface;
  moab::ParallelComm *pcomm;
  moab::ReadUtilIface* readMeshIface;

  moab::Tag  id_tag=PETSC_NULL;
  moab::Range         ownedvtx,ownedelms;
  moab::EntityHandle  vfirst,efirst;
  std::vector<double*> vcoords;
  moab::EntityHandle  *connectivity = 0;
  moab::EntityType etype;
  PetscInt    ise[6];
  PetscReal   xse[6];

  const PetscInt npts=nele+1;        /* Number of points in every dimension */
  PetscInt vpere,locnele,locnpts,ghnele,ghnpts;    /* Number of verts/element, vertices, elements owned by this process */

  PetscFunctionBegin;
  if(dim < 1 || dim > 3) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_ARG_OUTOFRANGE,"Invalid dimension argument for mesh: dim=[1,3].\n");

  ierr = MPI_Comm_size(comm, &nprocs);CHKERRQ(ierr);
  /* total number of vertices in all dimensions */
  n=pow(npts,dim);

  /* do some error checking */
  if(n < 2) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_ARG_OUTOFRANGE,"Number of points must be >= 2.\n");
  if(nprocs > n) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_ARG_OUTOFRANGE,"Number of processors must be less than or equal to number of elements.\n");
  if(nghost < 0) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_ARG_OUTOFRANGE,"Number of ghost layers cannot be negative.\n");

  /* Create the basic DMMoab object and keep the default parameters created by DM impls */
  ierr = DMMoabCreateMoab(comm, PETSC_NULL, PETSC_NULL, PETSC_NULL, PETSC_NULL, dm);CHKERRQ(ierr);

  /* get all the necessary handles from the private DM object */
  dmmoab = (DM_Moab*)(*dm)->data;
  mbiface = dmmoab->mbiface;
  pcomm = dmmoab->pcomm;
  id_tag = dmmoab->ltog_tag;
  nprocs = pcomm->size();
  rank = pcomm->rank();
  dmmoab->dim = dim;

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
    ghnele = (nghost > 0 ? (ise[0] > nghost ? 1 : 0) + (ise[1] < nele - nghost ? 1 : 0) : 0 );
    ghnpts = (nghost > 0 ? (ise[0] > 0 ? 1 : 0) + (ise[1] < nele ? 1 : 0) : 0);
    etype = moab::MBEDGE;
    break;
   case 2:
    vpere = 4;
    locnele = (ise[1]-ise[0])*(ise[3]-ise[2]);
    locnpts = (ise[1]-ise[0]+1)*(ise[3]-ise[2]+1);
    ghnele = (nghost > 0 ? (ise[2] > 0 ? nele : 0) + (ise[3] < nele ? nele : 0) : 0);
    ghnpts = (nghost > 0 ? (ise[2] > 0 ? npts : 0) + (ise[3] < nele ? npts : 0) : 0);
    etype = moab::MBQUAD;
    break;
   case 3:
    vpere = 8;
    locnele = (ise[1]-ise[0])*(ise[3]-ise[2])*(ise[5]-ise[4]);
    locnpts = (ise[1]-ise[0]+1)*(ise[3]-ise[2]+1)*(ise[5]-ise[4]+1);
    ghnele = (nghost > 0 ? (ise[4] > 0 ? nele*nele : 0) + (ise[5] < nele ? nele*nele : 0) : 0);
    ghnpts = (nghost > 0 ? (ise[4] > 0 ? npts*npts : 0) + (ise[5] < nele ? npts*npts : 0) : 0);
    etype = moab::MBHEX;
    break;
  }

  /* we have a domain of size [1,1,1] - now compute local co-ordinate box */
  ierr = PetscMemzero(xse,sizeof(PetscReal)*6);CHKERRQ(ierr);  
  for(i=0; i<6; ++i) {
    xse[i]=(PetscReal)ise[i]/nele;
  }

  /* Create vertexes and set the coodinate of each vertex */
  merr = readMeshIface->get_node_coords(3,locnpts+ghnpts,0,vfirst,vcoords,n);MBERRNM(merr);

  /* Compute the co-ordinates of vertices and global IDs */
  std::vector<int>    vgid(locnpts+ghnpts);
  int vcount=0;
  double hxyz=1.0/nele;

  /* create all the owned vertices */
  for (k = ise[4]; k <= ise[5]; k++) {
    for (j = ise[2]; j <= ise[3]; j++) {
      for (i = ise[0]; i <= ise[1]; i++, vcount++) {
        set_structured_coordinates(i,j,k,hxyz,vcount,vcoords);
        vgid[vcount] = (k*npts+j)*npts+i+1;
      }
    }
  }

  /* create ghosted vertices requested by user - below the current plane */
  if (ise[2*dim-2] > 0) {
    for (k = (dim==3?ise[4]-nghost:ise[4]); k <= (dim==3?ise[4]-1:ise[5]); k++) {
      for (j = (dim==2?ise[2]-nghost:ise[2]); j <= (dim==2?ise[2]-1:ise[3]); j++) {
        for (i = (dim>1?ise[0]:ise[0]-nghost); i <= (dim>1?ise[1]:ise[0]-1); i++, vcount++) {
          set_structured_coordinates(i,j,k,hxyz,vcount,vcoords);
          vgid[vcount] = (k*npts+j)*npts+i+1;
        }
      }
    }
  }

  /* create ghosted vertices requested by user - above the current plane */
  if (ise[2*dim-1] < nele) {
    for (k = (dim==3?ise[5]+1:ise[4]); k <= (dim==3?ise[5]+nghost:ise[5]); k++) {
      for (j = (dim==2?ise[3]+1:ise[2]); j <= (dim==2?ise[3]+nghost:ise[3]); j++) {
        for (i = (dim>1?ise[0]:ise[1]+1); i <= (dim>1?ise[1]:ise[1]+nghost); i++, vcount++) {
          set_structured_coordinates(i,j,k,hxyz,vcount,vcoords);
          vgid[vcount] = (k*npts+j)*npts+i+1;
        }
      }
    }
  }

  if (locnpts+ghnpts != vcount) SETERRQ2(PETSC_COMM_WORLD,PETSC_ERR_ARG_OUTOFRANGE,"Created the wrong number of vertices! (%D!=%D)",locnpts+ghnpts,vcount);

  merr = mbiface->get_entities_by_type(0,moab::MBVERTEX,ownedvtx,true);MBERRNM(merr);
  merr = mbiface->add_entities (dmmoab->fileset, ownedvtx);MBERRNM(merr);

  /* The global ID tag is applied to each owned
     vertex. It acts as an global identifier which MOAB uses to
     assemble the individual pieces of the mesh:
     Set the global ID indices */
  merr = mbiface->tag_set_data(id_tag,ownedvtx,vgid.data());MBERRNM(merr);
  
  /* Create elements between mesh points using the ReadUtilInterface 
     get the reference to element connectivities for all local elements from the ReadUtilInterface */
  merr = readMeshIface->get_element_connect (locnele+ghnele,vpere,etype,1,efirst,connectivity);MBERRNM(merr);

  /* offset appropriately so that only local ID and not global ID numbers are set for connectivity array */
//  PetscPrintf(PETSC_COMM_SELF, "[%D] first local handle %D\n", rank, vfirst);
  vfirst-=vgid[0]-1;

   /* 3. Loop over elements in 3 nested loops over i, j, k; for each (i,j,k):
         and then set the connectivity for each element appropriately */
  int ecount=0;

  /* create ghosted elements requested by user - below the current plane */
  if (ise[2*dim-2] >= nghost) {
    for (k = (dim==3?ise[4]-nghost:ise[4]); k < (dim==3?ise[4]:std::max(ise[5],1)); k++) {
      for (j = (dim==2?ise[2]-nghost:ise[2]); j < (dim==2?ise[2]:std::max(ise[3],1)); j++) {
        for (i = (dim>1?ise[0]:ise[0]-nghost); i < (dim>1?std::max(ise[1],1):ise[0]); i++, ecount++) {
          set_element_connectivity(dim, etype, ecount*vpere, nele, i, j, k, vfirst, connectivity);
        }
      }
    }
  }

  /* create owned elements requested by user */
  for (k = ise[4]; k < std::max(ise[5],1); k++) {
    for (j = ise[2]; j < std::max(ise[3],1); j++) {
      for (i = ise[0]; i < std::max(ise[1],1); i++,ecount++) {
        set_element_connectivity(dim, etype, ecount*vpere, nele, i, j, k, vfirst, connectivity);
      }
    }
  }

  /* create ghosted elements requested by user - above the current plane */
  if (ise[2*dim-1] <= nele-nghost) {
    for (k = (dim==3?ise[5]:ise[4]); k < (dim==3?ise[5]+nghost:std::max(ise[5],1)); k++) {
      for (j = (dim==2?ise[3]:ise[2]); j < (dim==2?ise[3]+nghost:std::max(ise[3],1)); j++) {
        for (i = (dim>1?ise[0]:ise[1]); i < (dim>1?std::max(ise[1],1):ise[1]+nghost); i++, ecount++) {
          set_element_connectivity(dim, etype, ecount*vpere, nele, i, j, k, vfirst, connectivity);
        }
      }
    }
  }

  merr = readMeshIface->update_adjacencies(efirst,locnele+ghnele,vpere,connectivity);MBERRNM(merr);
  
  /* 2. Get the vertices and hexes from moab and check their numbers against I*J*K and (I-1)*(J-1)*(K-1), resp.
        first '0' specifies "root set", or entire MOAB instance, second the entity dimension being requested */
  merr = mbiface->get_entities_by_dimension(0, dim, ownedelms);MBERRNM(merr);
  merr = mbiface->add_entities (dmmoab->fileset, ownedelms);MBERRNM(merr);

  if (locnele+ghnele != (int) ownedelms.size())
    SETERRQ2(PETSC_COMM_WORLD,PETSC_ERR_ARG_OUTOFRANGE,"Created the wrong number of elements! (%D!=%D)",locnele+ghnele,ownedelms.size());
  else if(locnpts+ghnpts != (int) ownedvtx.size())
    SETERRQ2(PETSC_COMM_WORLD,PETSC_ERR_ARG_OUTOFRANGE,"Created the wrong number of vertices! (%D!=%D)",locnpts+ghnpts,ownedvtx.size());    
  else
    PetscInfo2(*dm, "Created %D elements and %D vertices.\n", ownedelms.size(), ownedvtx.size());

  /* check the handles */
  merr = pcomm->check_all_shared_handles();MBERRV(mbiface,merr);

#if 0
  std::stringstream sstr;
  sstr << "test_" << pcomm->rank() << ".vtk";
  mbiface->write_mesh(sstr.str().c_str());
#endif

  /* resolve the shared entities by exchanging information to adjacent processors */
  merr = mbiface->get_entities_by_type(dmmoab->fileset,etype,ownedelms,true);MBERRNM(merr);
  merr = pcomm->resolve_shared_ents(dmmoab->fileset,ownedelms,dim,dim-1,&id_tag);MBERRV(mbiface,merr);

  /* Reassign global IDs on all entities. */
  merr = pcomm->assign_global_ids(dmmoab->fileset,dim,1,true,true,true);MBERRNM(merr);

  merr = pcomm->exchange_ghost_cells(dim,0,1/*nghost*/,dim,true,false,&dmmoab->fileset);MBERRV(mbiface,merr);

  /* Everything is set up, now just do a tag exchange to update tags
     on all of the ghost vertexes */
  merr = mbiface->get_entities_by_type(dmmoab->fileset,moab::MBVERTEX,ownedvtx,true);MBERRNM(merr);
  merr = mbiface->get_entities_by_dimension(dmmoab->fileset, dim, ownedelms);MBERRNM(merr);
  
  merr = pcomm->exchange_tags(id_tag,ownedvtx);MBERRV(mbiface,merr);
  merr = pcomm->exchange_tags(id_tag,ownedelms);MBERRV(mbiface,merr);
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "DMMoab_GetReadOptions_Private"
PetscErrorCode DMMoab_GetReadOptions_Private(PetscBool by_rank, PetscInt numproc, PetscInt dim, MoabReadMode mode, PetscInt dbglevel, const char* extra_opts, const char** read_opts)
{
  std::ostringstream str;

  PetscFunctionBegin;
  /* do parallel read unless using only one processor */
  if (numproc > 1) {
    str << "PARALLEL=" << mode << ";PARTITION=PARALLEL_PARTITION;PARTITION_DISTRIBUTE;";
    str << "PARALLEL_RESOLVE_SHARED_ENTS;PARALLEL_GHOSTS=" << dim << ".0.1;";
    if (by_rank)
      str << "PARTITION_BY_RANK;";
  }

  if (dbglevel) {
    str << "CPUTIME;DEBUG_IO=" << dbglevel << ";";
    if (numproc>1)
      str << "DEBUG_PIO=" << dbglevel << ";";
  }

  if (extra_opts)
    str << extra_opts;

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
  dmmoab->dim = dim;

  /* TODO: Use command-line options to control by_rank, verbosity, MoabReadMode and extra options */
  ierr  = PetscOptionsBegin(PETSC_COMM_WORLD, "", "Options for reading/writing MOAB based meshes from file", "DMMoab");
  ierr  = PetscOptionsInt("-dmmb_rw_dbg", "The verbosity level for reading and writing MOAB meshes", "dmmbutil.cxx", dbglevel, &dbglevel, NULL);CHKERRQ(ierr);
  ierr  = PetscOptionsBool("-dmmb_part_by_rank", "Use partition by rank when reading MOAB meshes from file", "dmmbutil.cxx", part_by_rank, &part_by_rank, NULL);CHKERRQ(ierr);
  ierr  = PetscOptionsEnd();

  /* add mesh loading options specific to the DM */
  ierr = DMMoab_GetReadOptions_Private(part_by_rank, nprocs, dim, MOAB_PARROPTS_READ_PART, dbglevel, usrreadopts, &readopts);CHKERRQ(ierr);

  PetscInfo2(*dm, "Reading file %s with options: %s\n",filename,readopts);

  /* Load the mesh from a file. */
  merr = mbiface->load_file(filename, &dmmoab->fileset, readopts);MBERRVM(mbiface,"Reading MOAB file failed.", merr);

  /* Reassign global IDs on all entities. */
  merr = pcomm->assign_global_ids(dmmoab->fileset,dim,1,true,true,true);MBERRNM(merr);

  /* load the local vertices */
  merr = mbiface->get_entities_by_type(dmmoab->fileset, moab::MBVERTEX, verts, true);MBERRNM(merr);
  /* load the local elements */
  merr = mbiface->get_entities_by_dimension(dmmoab->fileset, dim, elems, true);MBERRNM(merr);

  /* Everything is set up, now just do a tag exchange to update tags
     on all of the ghost vertexes */
  merr = pcomm->exchange_tags(dmmoab->ltog_tag,verts);MBERRV(mbiface,merr);
  merr = pcomm->exchange_tags(dmmoab->ltog_tag,elems);MBERRV(mbiface,merr);

  merr = pcomm->exchange_ghost_cells(dim,0,1,0,true,true,&dmmoab->fileset);MBERRV(mbiface,merr);

  merr = pcomm->collective_sync_partition();MBERR("Collective sync failed", merr);

  PetscInfo3(*dm, "MOAB file '%s' was successfully loaded. Found %D vertices and %D elements.\n", filename, verts.size(), elems.size());

#if 1
  std::stringstream sstr;
  sstr << "test_" << pcomm->rank() << ".vtk";
  mbiface->write_mesh(sstr.str().c_str());
#endif
  PetscFunctionReturn(0);
}

