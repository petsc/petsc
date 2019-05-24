#include <petsc/private/viewerimpl.h>
#include <petscviewerhdf5.h>    /*I   "petscviewerhdf5.h"   I*/
#include <petsc/private/isimpl.h> /* for PetscViewerHDF5Load */
#include <petscis.h>    /*I   "petscis.h"   I*/

#if defined(PETSC_HAVE_HDF5)

struct _n_HDF5ReadCtx {
  hid_t file, group, dataset, dataspace, plist;
  PetscInt timestep;
  PetscBool complexVal, dim2, horizontal;
};
typedef struct _n_HDF5ReadCtx* HDF5ReadCtx;

static PetscErrorCode PetscViewerHDF5ReadInitialize_Private(PetscViewer viewer, const char name[], HDF5ReadCtx *ctx)
{
  HDF5ReadCtx    h=NULL;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscNew(&h);CHKERRQ(ierr);
  ierr = PetscViewerHDF5OpenGroup(viewer, &h->file, &h->group);CHKERRQ(ierr);
  PetscStackCallHDF5Return(h->dataset,H5Dopen2,(h->group, name, H5P_DEFAULT));
  PetscStackCallHDF5Return(h->dataspace,H5Dget_space,(h->dataset));
  ierr = PetscViewerHDF5GetTimestep(viewer, &h->timestep);CHKERRQ(ierr);
  ierr = PetscViewerHDF5HasAttribute(viewer,name,"complex",&h->complexVal);CHKERRQ(ierr);
  if (h->complexVal) {ierr = PetscViewerHDF5ReadAttribute(viewer,name,"complex",PETSC_BOOL,&h->complexVal);CHKERRQ(ierr);}
  /* MATLAB stores column vectors horizontally */
  ierr = PetscViewerHDF5HasAttribute(viewer,name,"MATLAB_class",&h->horizontal);CHKERRQ(ierr);
  /* Create property list for collective dataset read */
  PetscStackCallHDF5Return(h->plist,H5Pcreate,(H5P_DATASET_XFER));
#if defined(PETSC_HAVE_H5PSET_FAPL_MPIO)
  PetscStackCallHDF5(H5Pset_dxpl_mpio,(h->plist, H5FD_MPIO_COLLECTIVE));
#endif
  /* To write dataset independently use H5Pset_dxpl_mpio(plist_id, H5FD_MPIO_INDEPENDENT) */
  *ctx = h;
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscViewerHDF5ReadFinalize_Private(PetscViewer viewer, HDF5ReadCtx *ctx)
{
  HDF5ReadCtx    h;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  h = *ctx;
  PetscStackCallHDF5(H5Pclose,(h->plist));
  PetscStackCallHDF5(H5Gclose,(h->group));
  PetscStackCallHDF5(H5Sclose,(h->dataspace));
  PetscStackCallHDF5(H5Dclose,(h->dataset));
  ierr = PetscFree(*ctx);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscViewerHDF5ReadSizes_Private(PetscViewer viewer, HDF5ReadCtx ctx, PetscLayout *map_)
{
  int            rdim, dim;
  hsize_t        dims[4];
  PetscInt       bsInd, lenInd, bs, len, N;
  PetscLayout    map;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!(*map_)) {
    ierr = PetscLayoutCreate(PetscObjectComm((PetscObject)viewer),map_);CHKERRQ(ierr);
  }
  map = *map_;
  /* calculate expected number of dimensions */
  dim = 0;
  if (ctx->timestep >= 0) ++dim;
  ++dim; /* length in blocks */
  if (ctx->complexVal) ++dim;
  /* get actual number of dimensions in dataset */
  PetscStackCallHDF5Return(rdim,H5Sget_simple_extent_dims,(ctx->dataspace, dims, NULL));
  /* calculate expected dimension indices */
  lenInd = 0;
  if (ctx->timestep >= 0) ++lenInd;
  bsInd = lenInd + 1;
  ctx->dim2 = PETSC_FALSE;
  if (rdim == dim) {
    bs = 1; /* support vectors stored as 1D array */
  } else if (rdim == dim+1) {
    bs = (PetscInt) dims[bsInd];
    if (bs == 1) ctx->dim2 = PETSC_TRUE; /* vector with blocksize of 1, still stored as 2D array */
  } else {
    SETERRQ2(PetscObjectComm((PetscObject)viewer), PETSC_ERR_FILE_UNEXPECTED, "Number of dimensions %d not %d as expected", rdim, dim);
  }
  len = dims[lenInd];
  if (ctx->horizontal) {
    if (len != 1) SETERRQ(PetscObjectComm((PetscObject)viewer), PETSC_ERR_SUP, "Cannot have horizontal array with number of rows > 1. In case of MATLAB MAT-file, vectors must be saved as column vectors.");
    len = bs;
    bs = 1;
  }
  N = (PetscInt) len*bs;

  /* Set Vec sizes,blocksize,and type if not already set */
  if (map->bs < 0) {
    ierr = PetscLayoutSetBlockSize(map, bs);CHKERRQ(ierr);
  } else if (map->bs != bs) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_FILE_UNEXPECTED, "Block size of array in file is %D, not %D as expected",bs,map->bs);
  if (map->N < 0) {
    ierr = PetscLayoutSetSize(map, N);CHKERRQ(ierr);
  } else if (map->N != N) SETERRQ2(PetscObjectComm((PetscObject)viewer),PETSC_ERR_FILE_UNEXPECTED, "Global size of array in file is %D, not %D as expected",N,map->N);
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscViewerHDF5ReadSelectHyperslab_Private(PetscViewer viewer, HDF5ReadCtx ctx, PetscLayout map, hid_t *memspace)
{
  hsize_t        count[4], offset[4];
  int            dim;
  PetscInt       bs, n, low;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /* Compute local size and ownership range */
  ierr = PetscLayoutSetUp(map);CHKERRQ(ierr);
  ierr = PetscLayoutGetBlockSize(map, &bs);CHKERRQ(ierr);
  ierr = PetscLayoutGetLocalSize(map, &n);CHKERRQ(ierr);
  ierr = PetscLayoutGetRange(map, &low, NULL);CHKERRQ(ierr);

  /* Each process defines a dataset and reads it from the hyperslab in the file */
  dim  = 0;
  if (ctx->timestep >= 0) {
    count[dim]  = 1;
    offset[dim] = ctx->timestep;
    ++dim;
  }
  if (ctx->horizontal) {
    count[dim]  = 1;
    offset[dim] = 0;
    ++dim;
  }
  {
    ierr = PetscHDF5IntCast(n/bs, &count[dim]);CHKERRQ(ierr);
    ierr = PetscHDF5IntCast(low/bs, &offset[dim]);CHKERRQ(ierr);
    ++dim;
  }
  if (bs > 1 || ctx->dim2) {
    if (PetscUnlikely(ctx->horizontal)) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "cannot have horizontal array with blocksize > 1");
    count[dim]  = bs;
    offset[dim] = 0;
    ++dim;
  }
  if (ctx->complexVal) {
    count[dim]  = 2;
    offset[dim] = 0;
    ++dim;
  }
  PetscStackCallHDF5Return(*memspace,H5Screate_simple,(dim, count, NULL));
  PetscStackCallHDF5(H5Sselect_hyperslab,(ctx->dataspace, H5S_SELECT_SET, offset, NULL, count, NULL));
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscViewerHDF5ReadArray_Private(PetscViewer viewer, HDF5ReadCtx h, hid_t datatype, hid_t memspace, void *arr)
{
  PetscFunctionBegin;
  PetscStackCallHDF5(H5Dread,(h->dataset, datatype, memspace, h->dataspace, h->plist, arr));
  PetscFunctionReturn(0);
}

PetscErrorCode PetscViewerHDF5Load(PetscViewer viewer, const char *name, PetscLayout map, hid_t datatype, void **newarr)
{
  HDF5ReadCtx     h=NULL;
  hid_t           memspace=0;
  size_t          unitsize;
  void            *arr;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  ierr = PetscViewerHDF5ReadInitialize_Private(viewer, name, &h);CHKERRQ(ierr);
#if defined(PETSC_USE_COMPLEX)
  if (!h->complexVal) {
    H5T_class_t clazz = H5Tget_class(datatype);
    if (clazz == H5T_FLOAT) SETERRQ(PetscObjectComm((PetscObject)viewer),PETSC_ERR_SUP,"File contains real numbers but PETSc is configured for complex. The conversion is not yet implemented. Configure with --with-scalar-type=real.");
  }
#else
  if (h->complexVal) SETERRQ(PetscObjectComm((PetscObject)viewer),PETSC_ERR_SUP,"File contains complex numbers but PETSc not configured for them. Configure with --with-scalar-type=complex.");
#endif

  ierr = PetscViewerHDF5ReadSizes_Private(viewer, h, &map);CHKERRQ(ierr);
  ierr = PetscLayoutSetUp(map);CHKERRQ(ierr);
  ierr = PetscViewerHDF5ReadSelectHyperslab_Private(viewer, h, map, &memspace);CHKERRQ(ierr);

  unitsize = H5Tget_size(datatype);
  if (h->complexVal) unitsize *= 2;
  if (unitsize <= 0 || unitsize > PetscMax(sizeof(PetscInt),sizeof(PetscScalar))) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Sanity check failed: HDF5 function H5Tget_size(datatype) returned suspicious value %D",unitsize);
  ierr = PetscMalloc(map->n*unitsize, &arr);CHKERRQ(ierr);

  ierr = PetscViewerHDF5ReadArray_Private(viewer, h, datatype, memspace, arr);CHKERRQ(ierr);
  PetscStackCallHDF5(H5Sclose,(memspace));
  ierr = PetscViewerHDF5ReadFinalize_Private(viewer, &h);CHKERRQ(ierr);
  *newarr = arr;
  PetscFunctionReturn(0);
}

/*@C
 PetscViewerHDF5ReadSizes - Read block size and global size of a vector (Vec or IS) stored in an HDF5 file.

  Input Parameters:
+ viewer - The HDF5 viewer
- name   - The dataset name

  Output Parameter:
+ bs     - block size
- N      - global size

  Notes:
  The dataset is stored as an HDF5 dataspace with 1-4 dimensions in the order
  1) # timesteps (optional), 2) # blocks, 3) # elements per block (optional), 4) real and imaginary part (only for complex).

  The dataset can be stored as a 2D dataspace even if its blocksize is 1; see PetscViewerHDF5SetBaseDimension2().

  Level: advanced

.seealso: PetscViewerHDF5Open(), VecLoad(), ISLoad(), VecGetSize(), ISGetSize(), PetscViewerHDF5SetBaseDimension2()
@*/
PetscErrorCode PetscViewerHDF5ReadSizes(PetscViewer viewer, const char name[], PetscInt *bs, PetscInt *N)
{
  HDF5ReadCtx    h=NULL;
  PetscLayout    map=NULL;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_CLASSID,1);
  ierr = PetscViewerHDF5ReadInitialize_Private(viewer, name, &h);CHKERRQ(ierr);
  ierr = PetscViewerHDF5ReadSizes_Private(viewer, h, &map);CHKERRQ(ierr);
  ierr = PetscViewerHDF5ReadFinalize_Private(viewer, &h);CHKERRQ(ierr);
  if (bs) *bs = map->bs;
  if (N) *N = map->N;
  ierr = PetscLayoutDestroy(&map);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#endif /* defined(PETSC_HAVE_HDF5) */
