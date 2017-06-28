
/*
   This file contains simple binary input routines for vectors.  The
   analogous output routines are within each vector implementation's
   VecView (with viewer types PETSCVIEWERBINARY)
 */

#include <petscsys.h>
#include <petscvec.h>         /*I  "petscvec.h"  I*/
#include <petsc/private/vecimpl.h>
#include <petscviewerhdf5.h>

static PetscErrorCode PetscViewerBinaryReadVecHeader_Private(PetscViewer viewer,PetscInt *rows)
{
  PetscErrorCode ierr;
  MPI_Comm       comm;
  PetscInt       tr[2],type;

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject)viewer,&comm);CHKERRQ(ierr);
  /* Read vector header */
  ierr = PetscViewerBinaryRead(viewer,tr,2,NULL,PETSC_INT);CHKERRQ(ierr);
  type = tr[0];
  if (type != VEC_FILE_CLASSID) {
    ierr = PetscLogEventEnd(VEC_Load,viewer,0,0,0);CHKERRQ(ierr);
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Not a vector next in file");
  }
  *rows = tr[1];
  PetscFunctionReturn(0);
}

#if defined(PETSC_HAVE_MPIIO)
static PetscErrorCode VecLoad_Binary_MPIIO(Vec vec, PetscViewer viewer)
{
  PetscErrorCode ierr;
  PetscMPIInt    lsize;
  PetscScalar    *avec;
  MPI_File       mfdes;
  MPI_Offset     off;

  PetscFunctionBegin;
  ierr = VecGetArray(vec,&avec);CHKERRQ(ierr);
  ierr = PetscMPIIntCast(vec->map->n,&lsize);CHKERRQ(ierr);

  ierr = PetscViewerBinaryGetMPIIODescriptor(viewer,&mfdes);CHKERRQ(ierr);
  ierr = PetscViewerBinaryGetMPIIOOffset(viewer,&off);CHKERRQ(ierr);
  off += vec->map->rstart*sizeof(PetscScalar);
  ierr = MPI_File_set_view(mfdes,off,MPIU_SCALAR,MPIU_SCALAR,(char*)"native",MPI_INFO_NULL);CHKERRQ(ierr);
  ierr = MPIU_File_read_all(mfdes,avec,lsize,MPIU_SCALAR,MPI_STATUS_IGNORE);CHKERRQ(ierr);
  ierr = PetscViewerBinaryAddMPIIOOffset(viewer,vec->map->N*sizeof(PetscScalar));CHKERRQ(ierr);

  ierr = VecRestoreArray(vec,&avec);CHKERRQ(ierr);
  ierr = VecAssemblyBegin(vec);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(vec);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
#endif

PetscErrorCode VecLoad_Binary(Vec vec, PetscViewer viewer)
{
  PetscMPIInt    size,rank,tag;
  int            fd;
  PetscInt       i,rows = 0,n,*range,N,bs;
  PetscErrorCode ierr;
  PetscBool      flag,skipheader;
  PetscScalar    *avec,*avecwork;
  MPI_Comm       comm;
  MPI_Request    request;
  MPI_Status     status;
#if defined(PETSC_HAVE_MPIIO)
  PetscBool      useMPIIO;
#endif

  PetscFunctionBegin;
  /* force binary viewer to load .info file if it has not yet done so */
  ierr = PetscViewerSetUp(viewer);CHKERRQ(ierr);
  ierr = PetscObjectGetComm((PetscObject)viewer,&comm);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm,&size);CHKERRQ(ierr);

  ierr = PetscViewerBinaryGetDescriptor(viewer,&fd);CHKERRQ(ierr);
  ierr = PetscViewerBinaryGetSkipHeader(viewer,&skipheader);CHKERRQ(ierr);
  if (!skipheader) {
    ierr = PetscViewerBinaryReadVecHeader_Private(viewer,&rows);CHKERRQ(ierr);
  } else {
    VecType vtype;
    ierr = VecGetType(vec,&vtype);CHKERRQ(ierr);
    if (!vtype) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_USER, "Vector binary file header was skipped, thus the user must specify the type and length of input vector");
    ierr = VecGetSize(vec,&N);CHKERRQ(ierr);
    if (!N) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_USER, "Vector binary file header was skipped, thus the user must specify the length of input vector");
    rows = N;
  }
  /* Set Vec sizes,blocksize,and type if not already set. Block size first so that local sizes will be compatible. */
  ierr = PetscOptionsGetInt(NULL,((PetscObject)vec)->prefix, "-vecload_block_size", &bs, &flag);CHKERRQ(ierr);
  if (flag) {
    ierr = VecSetBlockSize(vec, bs);CHKERRQ(ierr);
  }
  if (vec->map->n < 0 && vec->map->N < 0) {
    ierr = VecSetSizes(vec,PETSC_DECIDE,rows);CHKERRQ(ierr);
  }

  /* If sizes and type already set,check if the vector global size is correct */
  ierr = VecGetSize(vec, &N);CHKERRQ(ierr);
  if (N != rows) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_FILE_UNEXPECTED, "Vector in file different length (%D) then input vector (%D)", rows, N);

#if defined(PETSC_HAVE_MPIIO)
  ierr = PetscViewerBinaryGetUseMPIIO(viewer,&useMPIIO);CHKERRQ(ierr);
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
      for (i=1; i<size; i++) n = PetscMax(n,range[i+1] - range[i]);

      ierr = PetscMalloc1(n,&avecwork);CHKERRQ(ierr);
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
PetscErrorCode PetscViewerHDF5OpenGroup(PetscViewer viewer, hid_t *fileId, hid_t *groupId)
{
  hid_t          file_id, group;
  htri_t         found;
  const char     *groupName = NULL;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscViewerHDF5GetFileId(viewer, &file_id);CHKERRQ(ierr);
  ierr = PetscViewerHDF5GetGroup(viewer, &groupName);CHKERRQ(ierr);
  /* Open group */
  if (groupName) {
    PetscBool root;

    ierr = PetscStrcmp(groupName, "/", &root);CHKERRQ(ierr);
    PetscStackCall("H5Lexists",found = H5Lexists(file_id, groupName, H5P_DEFAULT));
    if (!root && (found <= 0)) {
#if (H5_VERS_MAJOR * 10000 + H5_VERS_MINOR * 100 + H5_VERS_RELEASE >= 10800)
      PetscStackCallHDF5Return(group,H5Gcreate2,(file_id, groupName, 0, H5P_DEFAULT, H5P_DEFAULT));
#else /* deprecated HDF5 1.6 API */
      PetscStackCallHDF5Return(group,H5Gcreate,(file_id, groupName, 0));
#endif
      PetscStackCallHDF5(H5Gclose,(group));
    }
#if (H5_VERS_MAJOR * 10000 + H5_VERS_MINOR * 100 + H5_VERS_RELEASE >= 10800)
    PetscStackCallHDF5Return(group,H5Gopen2,(file_id, groupName, H5P_DEFAULT));
#else
    PetscStackCallHDF5Return(group,H5Gopen,file_id, groupName));
#endif
  } else group = file_id;

  *fileId  = file_id;
  *groupId = group;
  PetscFunctionReturn(0);
}

PetscErrorCode PetscViewerHDF5ReadSizes(PetscViewer viewer, const char name[], PetscInt *bs, PetscInt *N)
{
  hid_t          file_id, group, dset_id, filespace;
  int            rdim, dim;
  hsize_t        dims[4];
  PetscInt       bsInd, lenInd, timestep;
  PetscBool      complexVal = PETSC_FALSE;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscViewerHDF5OpenGroup(viewer, &file_id, &group);CHKERRQ(ierr);
  ierr = PetscViewerHDF5GetTimestep(viewer, &timestep);CHKERRQ(ierr);
#if (H5_VERS_MAJOR * 10000 + H5_VERS_MINOR * 100 + H5_VERS_RELEASE >= 10800)
  PetscStackCallHDF5Return(dset_id,H5Dopen2,(group, name, H5P_DEFAULT));
#else
  PetscStackCallHDF5Return(dset_id,H5Dopen,(group, name));
#endif
  PetscStackCallHDF5Return(filespace,H5Dget_space,(dset_id));
  dim = 0;
  if (timestep >= 0) ++dim;
  ++dim; /* length in blocks */
  ++dim; /* block size */
  {
    const char *groupname;
    char       vecgroup[PETSC_MAX_PATH_LEN];

    ierr = PetscViewerHDF5GetGroup(viewer,&groupname);CHKERRQ(ierr);
    ierr = PetscSNPrintf(vecgroup,PETSC_MAX_PATH_LEN,"%s/%s",groupname,name);CHKERRQ(ierr);
    ierr = PetscViewerHDF5HasAttribute(viewer,vecgroup,"complex",&complexVal);CHKERRQ(ierr);
  }
  if (complexVal) ++dim;
  PetscStackCallHDF5Return(rdim,H5Sget_simple_extent_dims,(filespace, dims, NULL));
  if (complexVal) {
    bsInd = rdim-2;
  } else {
    bsInd = rdim-1;
  }
  lenInd = timestep >= 0 ? 1 : 0;
  if (rdim != dim) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_FILE_UNEXPECTED, "Dimension of array in file %d not %d as expected", rdim, dim);
  /* Close/release resources */
  PetscStackCallHDF5(H5Sclose,(filespace));
  PetscStackCallHDF5(H5Dclose,(dset_id));
  if (group != file_id) PetscStackCallHDF5(H5Gclose,(group));
  if (bs) *bs = (PetscInt) dims[bsInd];
  if (N)  *N  = (PetscInt) dims[lenInd]*dims[bsInd];
  PetscFunctionReturn(0);
}

/*
     This should handle properly the cases where PetscInt is 32 or 64 and hsize_t is 32 or 64. These means properly casting with
   checks back and forth between the two types of variables.
*/
PetscErrorCode VecLoad_HDF5(Vec xin, PetscViewer viewer)
{
  hid_t          file_id, group, dset_id, filespace, memspace, plist_id;
  int            rdim, dim;
  hsize_t        dims[4], count[4], offset[4];
  PetscInt       n, N, bs = 1, bsInd, lenInd, low, timestep;
  PetscScalar    *x;
  hid_t          scalartype; /* scalar type (H5T_NATIVE_FLOAT or H5T_NATIVE_DOUBLE) */
  const char     *vecname;
  PetscErrorCode ierr;
  PetscBool      dim2 = PETSC_FALSE;

  PetscFunctionBegin;
#if defined(PETSC_USE_REAL_SINGLE)
  scalartype = H5T_NATIVE_FLOAT;
#elif defined(PETSC_USE_REAL___FLOAT128)
#error "HDF5 output with 128 bit floats not supported."
#elif defined(PETSC_USE_REAL___FP16)
#error "HDF5 output with 16 bit floats not supported."
#else
  scalartype = H5T_NATIVE_DOUBLE;
#endif

  ierr = PetscViewerHDF5OpenGroup(viewer, &file_id, &group);CHKERRQ(ierr);
  ierr = PetscViewerHDF5GetTimestep(viewer, &timestep);CHKERRQ(ierr);
  ierr = VecGetBlockSize(xin,&bs);CHKERRQ(ierr);
  /* Create the dataset with default properties and close filespace */
  ierr = PetscObjectGetName((PetscObject)xin,&vecname);CHKERRQ(ierr);
#if (H5_VERS_MAJOR * 10000 + H5_VERS_MINOR * 100 + H5_VERS_RELEASE >= 10800)
  PetscStackCallHDF5Return(dset_id,H5Dopen2,(group, vecname, H5P_DEFAULT));
#else
  PetscStackCallHDF5Return(dset_id,H5Dopen,(group, vecname));
#endif
  /* Retrieve the dataspace for the dataset */
  PetscStackCallHDF5Return(filespace,H5Dget_space,(dset_id));
  dim = 0;
  if (timestep >= 0) ++dim;
  ++dim;
  if (bs > 1) ++dim;
#if defined(PETSC_USE_COMPLEX)
  ++dim;
#endif
  PetscStackCallHDF5Return(rdim,H5Sget_simple_extent_dims,(filespace, dims, NULL));
#if defined(PETSC_USE_COMPLEX)
  bsInd = rdim-2;
#else
  bsInd = rdim-1;
#endif
  lenInd = timestep >= 0 ? 1 : 0;
  
  if (rdim != dim) {
    /* In this case the input dataset have one extra, unexpected dimension. */
    if (rdim == dim+1)
    {
      /* In this case the block size is unset */
      if (bs == -1)
      {
        ierr = VecSetBlockSize(xin, dims[bsInd]);CHKERRQ(ierr);
        bs = dims[bsInd];
      }
    
      /* In this case the block size unity */
      else if (bs == 1 && dims[bsInd] == 1) dim2 = PETSC_TRUE;
      
      /* Special error message for the case where bs does not match the input file */
      else if (bs != (PetscInt) dims[bsInd]) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_FILE_UNEXPECTED, "Block size of array in file is %D, not %D as expected",(PetscInt)dims[bsInd],bs);
      
      /* All other cases is errors */
      else SETERRQ3(PETSC_COMM_SELF,PETSC_ERR_FILE_UNEXPECTED, "Dimension of array in file is %d, not %d as expected with bs = %D",rdim,dim,bs);
    }
    else SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_FILE_UNEXPECTED, "Dimension of array in file is %d, not %d as expected",rdim,dim);
  } else if (bs > 1 && bs != (PetscInt) dims[bsInd]) {
    ierr = VecSetBlockSize(xin, dims[bsInd]);CHKERRQ(ierr);
    bs = dims[bsInd];
  }
  
  /* Set Vec sizes,blocksize,and type if not already set */
  if ((xin)->map->n < 0 && (xin)->map->N < 0) {
    ierr = VecSetSizes(xin, PETSC_DECIDE, dims[lenInd]*bs);CHKERRQ(ierr);
  }
  /* If sizes and type already set,check if the vector global size is correct */
  ierr = VecGetSize(xin, &N);CHKERRQ(ierr);
  if (N/bs != (PetscInt) dims[lenInd]) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_FILE_UNEXPECTED, "Vector in file different length (%D) then input vector (%D)", (PetscInt) dims[lenInd], N/bs);

  /* Each process defines a dataset and reads it from the hyperslab in the file */
  ierr = VecGetLocalSize(xin, &n);CHKERRQ(ierr);
  dim  = 0;
  if (timestep >= 0) {
    count[dim] = 1;
    ++dim;
  }
  ierr = PetscHDF5IntCast(n/bs,count + dim);CHKERRQ(ierr);
  ++dim;
  if (bs > 1 || dim2) {
    count[dim] = bs;
    ++dim;
  }
#if defined(PETSC_USE_COMPLEX)
  count[dim] = 2;
  ++dim;
#endif
  PetscStackCallHDF5Return(memspace,H5Screate_simple,(dim, count, NULL));

  /* Select hyperslab in the file */
  ierr = VecGetOwnershipRange(xin, &low, NULL);CHKERRQ(ierr);
  dim  = 0;
  if (timestep >= 0) {
    offset[dim] = timestep;
    ++dim;
  }
  ierr = PetscHDF5IntCast(low/bs,offset + dim);CHKERRQ(ierr);
  ++dim;
  if (bs > 1 || dim2) {
    offset[dim] = 0;
    ++dim;
  }
#if defined(PETSC_USE_COMPLEX)
  offset[dim] = 0;
  ++dim;
#endif
  PetscStackCallHDF5(H5Sselect_hyperslab,(filespace, H5S_SELECT_SET, offset, NULL, count, NULL));

  /* Create property list for collective dataset read */
  PetscStackCallHDF5Return(plist_id,H5Pcreate,(H5P_DATASET_XFER));
#if defined(PETSC_HAVE_H5PSET_FAPL_MPIO)
  PetscStackCallHDF5(H5Pset_dxpl_mpio,(plist_id, H5FD_MPIO_COLLECTIVE));
#endif
  /* To write dataset independently use H5Pset_dxpl_mpio(plist_id, H5FD_MPIO_INDEPENDENT) */

  ierr   = VecGetArray(xin, &x);CHKERRQ(ierr);
  PetscStackCallHDF5(H5Dread,(dset_id, scalartype, memspace, filespace, plist_id, x));
  ierr   = VecRestoreArray(xin, &x);CHKERRQ(ierr);

  /* Close/release resources */
  if (group != file_id) {
    PetscStackCallHDF5(H5Gclose,(group));
  }
  PetscStackCallHDF5(H5Pclose,(plist_id));
  PetscStackCallHDF5(H5Sclose,(filespace));
  PetscStackCallHDF5(H5Sclose,(memspace));
  PetscStackCallHDF5(H5Dclose,(dset_id));

  ierr = VecAssemblyBegin(xin);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(xin);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
#endif


PetscErrorCode  VecLoad_Default(Vec newvec, PetscViewer viewer)
{
  PetscErrorCode ierr;
  PetscBool      isbinary;
#if defined(PETSC_HAVE_HDF5)
  PetscBool      ishdf5;
#endif

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERBINARY,&isbinary);CHKERRQ(ierr);
#if defined(PETSC_HAVE_HDF5)
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERHDF5,&ishdf5);CHKERRQ(ierr);
#endif

#if defined(PETSC_HAVE_HDF5)
  if (ishdf5) {
    if (!((PetscObject)newvec)->name) {
      ierr = PetscLogEventEnd(VEC_Load,viewer,0,0,0);CHKERRQ(ierr);
      SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Since HDF5 format gives ASCII name for each object in file; must use VecLoad() after setting name of Vec with PetscObjectSetName()");
    }
    ierr = VecLoad_HDF5(newvec, viewer);CHKERRQ(ierr);
  } else
#endif
  {
    ierr = VecLoad_Binary(newvec, viewer);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*@
  VecChop - Set all values in the vector with an absolute value less than the tolerance to zero

  Input Parameters:
+ v   - The vector
- tol - The zero tolerance

  Output Parameters:
. v - The chopped vector

  Level: intermediate

.seealso: VecCreate(), VecSet()
@*/
PetscErrorCode VecChop(Vec v, PetscReal tol)
{
  PetscScalar    *a;
  PetscInt       n, i;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecGetLocalSize(v, &n);CHKERRQ(ierr);
  ierr = VecGetArray(v, &a);CHKERRQ(ierr);
  for (i = 0; i < n; ++i) {
    if (PetscAbsScalar(a[i]) < tol) a[i] = 0.0;
  }
  ierr = VecRestoreArray(v, &a);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
