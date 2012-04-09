
/* 
   This file contains simple binary input routines for vectors.  The
   analogous output routines are within each vector implementation's 
   VecView (with viewer types PETSCVIEWERBINARY)
 */

#include <petscsys.h>
#include <petscvec.h>         /*I  "petscvec.h"  I*/
#include <petsc-private/vecimpl.h>
#include <petscmat.h> /* so that MAT_FILE_CLASSID is defined */

#undef __FUNCT__  
#define __FUNCT__ "PetscViewerBinaryReadVecHeader_Private"
static PetscErrorCode PetscViewerBinaryReadVecHeader_Private(PetscViewer viewer,PetscInt *rows)
{
  PetscErrorCode ierr;
  MPI_Comm       comm;
  PetscInt       tr[2],type;

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject)viewer,&comm);CHKERRQ(ierr);
  /* Read vector header */
  ierr = PetscViewerBinaryRead(viewer,tr,2,PETSC_INT);CHKERRQ(ierr);
  type = tr[0];
  if (type != VEC_FILE_CLASSID) {
    ierr = PetscLogEventEnd(VEC_Load,viewer,0,0,0);CHKERRQ(ierr);
    if (type == MAT_FILE_CLASSID) {
      SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Matrix is next in file, not a vector as you requested");
    } else {
      SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Not a vector next in file");
    }
  }
  *rows = tr[1];
  PetscFunctionReturn(0);
}

#if defined(PETSC_HAVE_MPIIO)
#undef __FUNCT__
#define __FUNCT__ "VecLoad_Binary_MPIIO"
static PetscErrorCode VecLoad_Binary_MPIIO(Vec vec, PetscViewer viewer)
{
  PetscErrorCode ierr;
  PetscMPIInt    gsizes[1],lsizes[1],lstarts[1];
  PetscScalar    *avec;
  MPI_Datatype   view;
  MPI_File       mfdes;
  MPI_Aint       ub,ul;
  MPI_Offset     off;

  PetscFunctionBegin;
  ierr = VecGetArray(vec,&avec);CHKERRQ(ierr);
  gsizes[0]  = PetscMPIIntCast(vec->map->N);
  lsizes[0]  = PetscMPIIntCast(vec->map->n);
  lstarts[0] = PetscMPIIntCast(vec->map->rstart);CHKERRQ(ierr);
  ierr = MPI_Type_create_subarray(1,gsizes,lsizes,lstarts,MPI_ORDER_FORTRAN,MPIU_SCALAR,&view);CHKERRQ(ierr);
  ierr = MPI_Type_commit(&view);CHKERRQ(ierr);

  ierr = PetscViewerBinaryGetMPIIODescriptor(viewer,&mfdes);CHKERRQ(ierr);
  ierr = PetscViewerBinaryGetMPIIOOffset(viewer,&off);CHKERRQ(ierr);
  ierr = MPI_File_set_view(mfdes,off,MPIU_SCALAR,view,(char *)"native",MPI_INFO_NULL);CHKERRQ(ierr);
  ierr = MPIU_File_read_all(mfdes,avec,lsizes[0],MPIU_SCALAR,MPI_STATUS_IGNORE);CHKERRQ(ierr);
  ierr = MPI_Type_get_extent(view,&ul,&ub);CHKERRQ(ierr);
  ierr = PetscViewerBinaryAddMPIIOOffset(viewer,ub);CHKERRQ(ierr);
  ierr = MPI_Type_free(&view);CHKERRQ(ierr);

  ierr = VecRestoreArray(vec,&avec);CHKERRQ(ierr);
  ierr = VecAssemblyBegin(vec);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(vec);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
#endif
    
#undef __FUNCT__  
#define __FUNCT__ "VecLoad_Binary"
PetscErrorCode VecLoad_Binary(Vec vec, PetscViewer viewer)
{
  PetscMPIInt    size,rank,tag;
  int            fd;
  PetscInt       i,rows = 0,n,*range,N,bs;
  PetscErrorCode ierr;
  PetscBool      flag;
  PetscScalar    *avec,*avecwork;
  MPI_Comm       comm;
  MPI_Request    request;
  MPI_Status     status;
#if defined(PETSC_HAVE_MPIIO)
  PetscBool      useMPIIO;
#endif

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject)viewer,&comm);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm,&size);CHKERRQ(ierr);
  
  ierr = PetscViewerBinaryGetDescriptor(viewer,&fd);CHKERRQ(ierr);
  ierr = PetscViewerBinaryReadVecHeader_Private(viewer,&rows);CHKERRQ(ierr);
  /* Set Vec sizes,blocksize,and type if not already set */
  if (vec->map->n < 0 && vec->map->N < 0) {
     ierr = VecSetSizes(vec,PETSC_DECIDE,rows);CHKERRQ(ierr);
  }
  ierr = PetscOptionsGetInt(((PetscObject)vec)->prefix, "-vecload_block_size", &bs, &flag);CHKERRQ(ierr);
  if (flag) {
    ierr = VecSetBlockSize(vec, bs);CHKERRQ(ierr);
  }

  /* If sizes and type already set,check if the vector global size is correct */
  ierr = VecGetSize(vec, &N);CHKERRQ(ierr);
  if (N != rows) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_FILE_UNEXPECTED, "Vector in file different length (%d) then input vector (%d)", rows, N);

#if defined(PETSC_HAVE_MPIIO)
  ierr = PetscViewerBinaryGetMPIIO(viewer,&useMPIIO);CHKERRQ(ierr);
  if (useMPIIO) {
    ierr = VecLoad_Binary_MPIIO(vec, viewer);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }
#endif

  ierr = VecGetLocalSize(vec,&n);CHKERRQ(ierr); 
  ierr = PetscObjectGetNewTag((PetscObject)viewer,&tag);CHKERRQ(ierr);
  ierr = VecGetArray(vec,&avec);CHKERRQ(ierr);
  if (!rank) {
    ierr = PetscBinaryRead(fd,avec,n,PETSC_SCALAR);CHKERRQ(ierr);

    if (size > 1) {
      /* read in other chuncks and send to other processors */
      /* determine maximum chunck owned by other */
      range = vec->map->range;
      n = 1;
      for (i=1; i<size; i++) {
	n = PetscMax(n,range[i+1] - range[i]);
      }
      ierr = PetscMalloc(n*sizeof(PetscScalar),&avecwork);CHKERRQ(ierr);
      for (i=1; i<size; i++) {
	n    = range[i+1] - range[i];
	ierr = PetscBinaryRead(fd,avecwork,n,PETSC_SCALAR);CHKERRQ(ierr);
	ierr = MPI_Isend(avecwork,n,MPIU_SCALAR,i,tag,comm,&request);CHKERRQ(ierr);
	ierr = MPI_Wait(&request,&status);CHKERRQ(ierr);
      }
      ierr = PetscFree(avecwork);CHKERRQ(ierr);
    }
  } else {
    ierr = MPI_Recv(avec,n,MPIU_SCALAR,0,tag,comm,&status);CHKERRQ(ierr);
  }

  ierr = VecRestoreArray(vec,&avec);CHKERRQ(ierr);
  ierr = VecAssemblyBegin(vec);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(vec);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#if defined(PETSC_HAVE_HDF5)
#undef __FUNCT__  
#define __FUNCT__ "PetscViewerHDF5OpenGroup"
PetscErrorCode PetscViewerHDF5OpenGroup(PetscViewer viewer, hid_t *fileId, hid_t *groupId) {
  hid_t          file_id, group;
  const char    *groupName = PETSC_NULL;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscViewerHDF5GetFileId(viewer, &file_id);CHKERRQ(ierr);
  ierr = PetscViewerHDF5GetGroup(viewer, &groupName);CHKERRQ(ierr);
  /* Open group */
  if (groupName) {
    PetscBool root;

    ierr = PetscStrcmp(groupName, "/", &root);CHKERRQ(ierr);
    if (!root && !H5Lexists(file_id, groupName, H5P_DEFAULT)) {
#if (H5_VERS_MAJOR * 10000 + H5_VERS_MINOR * 100 + H5_VERS_RELEASE >= 10800)
      group = H5Gcreate2(file_id, groupName, 0, H5P_DEFAULT, H5P_DEFAULT);
#else /* deprecated HDF5 1.6 API */
      group = H5Gcreate(file_id, groupName, 0);
#endif
      if (group < 0) SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_LIB, "Could not create group %s", groupName);
      ierr = H5Gclose(group);CHKERRQ(ierr);
    }
#if (H5_VERS_MAJOR * 10000 + H5_VERS_MINOR * 100 + H5_VERS_RELEASE >= 10800)
    group = H5Gopen2(file_id, groupName, H5P_DEFAULT);
#else
    group = H5Gopen(file_id, groupName);
#endif
    if (group < 0) SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_LIB, "Could not open group %s", groupName);
  } else {
    group = file_id;
  }
  *fileId  = file_id;
  *groupId = group;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "VecLoad_HDF5"
/*
     This should handle properly the cases where PetscInt is 32 or 64 and hsize_t is 32 or 64. These means properly casting with
   checks back and forth between the two types of variables.
*/
PetscErrorCode VecLoad_HDF5(Vec xin, PetscViewer viewer)
{
  hid_t          file_id, group, dset_id, filespace, memspace, plist_id;
  hsize_t        rdim, dim;
  hsize_t        dims[4], count[4], offset[4];
  herr_t         status;
  PetscInt       n, N, bs = 1, bsInd, lenInd, low, timestep;
  PetscScalar   *x;
  PetscBool      flag;
  const char    *vecname;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscViewerHDF5OpenGroup(viewer, &file_id, &group);CHKERRQ(ierr);
  ierr = PetscViewerHDF5GetTimestep(viewer, &timestep);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(((PetscObject)xin)->prefix, "-vecload_block_size", &bs, &flag);CHKERRQ(ierr);

  /* Create the dataset with default properties and close filespace */
  ierr = PetscObjectGetName((PetscObject)xin,&vecname);CHKERRQ(ierr);
#if (H5_VERS_MAJOR * 10000 + H5_VERS_MINOR * 100 + H5_VERS_RELEASE >= 10800)
  dset_id = H5Dopen2(group, vecname, H5P_DEFAULT);
#else
  dset_id = H5Dopen(group, vecname);
#endif
  if (dset_id == -1) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Could not H5Dopen() with Vec named %s",vecname);
  /* Retrieve the dataspace for the dataset */
  filespace = H5Dget_space(dset_id);
  if (filespace == -1) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_LIB,"Could not H5Dget_space()");
  dim = 0;
  if (timestep >= 0) {
    ++dim;
  }
  ++dim;
  if (bs >= 1) {
    ++dim;
  }
#if defined(PETSC_USE_COMPLEX)
  ++dim;
#endif
  rdim = H5Sget_simple_extent_dims(filespace, dims, PETSC_NULL);
#if defined(PETSC_USE_COMPLEX)
  bsInd = rdim-2;
#else
  bsInd = rdim-1;
#endif
  lenInd = timestep >= 0 ? 1 : 0;
  if (rdim != dim) {
    if (rdim == dim+1 && bs == 1) {
      bs = dims[bsInd];
      if (flag) SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_FILE_UNEXPECTED, "Block size 1 specified for vector does not match blocksize in file %d",bs);
    } else SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_FILE_UNEXPECTED, "Dimension of array in file %d not %d as expected",rdim,dim);
  } else if (bs >= 1 && bs != (PetscInt) dims[bsInd]) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_FILE_UNEXPECTED, "Block size %d specified for vector does not match blocksize in file %d",bs,dims[bsInd]);

  /* Set Vec sizes,blocksize,and type if not already set */
  if ((xin)->map-> n < 0 && (xin)->map->N < 0) {
    ierr = VecSetSizes(xin, PETSC_DECIDE, dims[lenInd]);CHKERRQ(ierr);
  }
  if (bs > 1 || flag) {
    ierr = VecSetBlockSize(xin, bs);CHKERRQ(ierr);
  }

  /* If sizes and type already set,check if the vector global size is correct */
  ierr = VecGetSize(xin, &N);CHKERRQ(ierr);
  if (N/bs != (PetscInt) dims[lenInd]) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_FILE_UNEXPECTED, "Vector in file different length (%d) then input vector (%d)", (PetscInt) dims[lenInd], N/bs);

  /* Each process defines a dataset and reads it from the hyperslab in the file */
  ierr = VecGetLocalSize(xin, &n);CHKERRQ(ierr);
  dim = 0;
  if (timestep >= 0) {
    count[dim] = 1;
    ++dim;
  }
  count[dim] = PetscHDF5IntCast(n)/bs;
  ++dim;
  if (bs >= 1) {
    count[dim] = bs;
    ++dim;
  }
#if defined(PETSC_USE_COMPLEX)
  count[dim] = 2;
  ++dim;
#endif
  memspace = H5Screate_simple(dim, count, NULL);
  if (memspace == -1) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_LIB,"Could not H5Screate_simple()");

  /* Select hyperslab in the file */
  ierr = VecGetOwnershipRange(xin, &low, PETSC_NULL);CHKERRQ(ierr);
  dim = 0;
  if (timestep >= 0) {
    offset[dim] = timestep;
    ++dim;
  }
  offset[dim] = PetscHDF5IntCast(low/bs);
  ++dim;
  if (bs >= 1) {
    offset[dim] = 0;
    ++dim;
  }
#if defined(PETSC_USE_COMPLEX)
  offset[dim] = 0;
  ++dim;
#endif
  status = H5Sselect_hyperslab(filespace, H5S_SELECT_SET, offset, NULL, count, NULL);CHKERRQ(status);

  /* Create property list for collective dataset read */
  plist_id = H5Pcreate(H5P_DATASET_XFER);
  if (plist_id == -1) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_LIB,"Could not H5Pcreate()");
#if defined(PETSC_HAVE_H5PSET_FAPL_MPIO)
  status = H5Pset_dxpl_mpio(plist_id, H5FD_MPIO_COLLECTIVE);CHKERRQ(status);
#endif
  /* To write dataset independently use H5Pset_dxpl_mpio(plist_id, H5FD_MPIO_INDEPENDENT) */

  ierr = VecGetArray(xin, &x);CHKERRQ(ierr);
  status = H5Dread(dset_id, H5T_NATIVE_DOUBLE, memspace, filespace, plist_id, x);CHKERRQ(status);
  ierr = VecRestoreArray(xin, &x);CHKERRQ(ierr);

  /* Close/release resources */
  if (group != file_id) {
    status = H5Gclose(group);CHKERRQ(status);
  }
  status = H5Pclose(plist_id);CHKERRQ(status);
  status = H5Sclose(filespace);CHKERRQ(status);
  status = H5Sclose(memspace);CHKERRQ(status);
  status = H5Dclose(dset_id);CHKERRQ(status);

  ierr = VecAssemblyBegin(xin);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(xin);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
#endif

#undef __FUNCT__
#define __FUNCT__ "VecLoad_Default"

PetscErrorCode  VecLoad_Default(Vec newvec, PetscViewer viewer)
{
  PetscErrorCode ierr;
  PetscBool      isbinary;
#if defined(PETSC_HAVE_HDF5)
  PetscBool      ishdf5;
#endif

  PetscFunctionBegin;
  ierr = PetscTypeCompare((PetscObject)viewer,PETSCVIEWERBINARY,&isbinary);CHKERRQ(ierr);
#if defined(PETSC_HAVE_HDF5)
  ierr = PetscTypeCompare((PetscObject)viewer,PETSCVIEWERHDF5,&ishdf5);CHKERRQ(ierr);
#endif

#if defined(PETSC_HAVE_HDF5)
  if (ishdf5) {
    if (!((PetscObject)newvec)->name) { 
      SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Since HDF5 format gives ASCII name for each object in file; must use VecLoad() after setting name of Vec with PetscObjectSetName()");
     ierr = PetscLogEventEnd(VEC_Load,viewer,0,0,0);CHKERRQ(ierr);
    }
    ierr = VecLoad_HDF5(newvec, viewer);CHKERRQ(ierr);
  } else
#endif
  {
    ierr = VecLoad_Binary(newvec, viewer);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

