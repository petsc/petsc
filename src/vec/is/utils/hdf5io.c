#include <petsc/private/viewerhdf5impl.h>
#include <petsclayouthdf5.h>    /*I   "petsclayoutdf5.h"   I*/
#include <petscis.h>    /*I   "petscis.h"   I*/

#if defined(PETSC_HAVE_HDF5)

struct _n_HDF5ReadCtx {
  hid_t     file, group, dataset, dataspace;
  int       lenInd, bsInd, rdim;
  hsize_t   *dims;
  PetscBool complexVal, dim2, horizontal;
};
typedef struct _n_HDF5ReadCtx* HDF5ReadCtx;

PetscErrorCode PetscViewerHDF5CheckTimestepping_Internal(PetscViewer viewer, const char name[])
{
  PetscViewer_HDF5 *hdf5 = (PetscViewer_HDF5*) viewer->data;
  PetscBool        timestepping=PETSC_FALSE;
  const char       *group;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  ierr = PetscViewerHDF5GetGroup(viewer, &group);CHKERRQ(ierr);
  ierr = PetscViewerHDF5ReadAttribute(viewer,name,"timestepping",PETSC_BOOL,&timestepping,&timestepping);CHKERRQ(ierr);
  if (timestepping != hdf5->timestepping) SETERRQ4(PetscObjectComm((PetscObject)viewer),PETSC_ERR_FILE_UNEXPECTED,"Dataset %s/%s stored with timesteps? %s Timestepping pushed? %s", group, name, PetscBools[timestepping], PetscBools[hdf5->timestepping]);
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscViewerHDF5ReadInitialize_Private(PetscViewer viewer, const char name[], HDF5ReadCtx *ctx)
{
  HDF5ReadCtx    h=NULL;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscViewerHDF5CheckTimestepping_Internal(viewer, name);CHKERRQ(ierr);
  ierr = PetscNew(&h);CHKERRQ(ierr);
  ierr = PetscViewerHDF5OpenGroup(viewer, &h->file, &h->group);CHKERRQ(ierr);
  PetscStackCallHDF5Return(h->dataset,H5Dopen2,(h->group, name, H5P_DEFAULT));
  PetscStackCallHDF5Return(h->dataspace,H5Dget_space,(h->dataset));
  ierr = PetscViewerHDF5ReadAttribute(viewer,name,"complex",PETSC_BOOL,&h->complexVal,&h->complexVal);CHKERRQ(ierr);
  /* MATLAB stores column vectors horizontally */
  ierr = PetscViewerHDF5HasAttribute(viewer,name,"MATLAB_class",&h->horizontal);CHKERRQ(ierr);
  *ctx = h;
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscViewerHDF5ReadFinalize_Private(PetscViewer viewer, HDF5ReadCtx *ctx)
{
  HDF5ReadCtx    h;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  h = *ctx;
  PetscStackCallHDF5(H5Gclose,(h->group));
  PetscStackCallHDF5(H5Sclose,(h->dataspace));
  PetscStackCallHDF5(H5Dclose,(h->dataset));
  ierr = PetscFree((*ctx)->dims);CHKERRQ(ierr);
  ierr = PetscFree(*ctx);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscViewerHDF5ReadSizes_Private(PetscViewer viewer, HDF5ReadCtx ctx, PetscBool setup, PetscLayout *map_)
{
  PetscViewer_HDF5 *hdf5 = (PetscViewer_HDF5*) viewer->data;
  PetscInt         bs, len, N;
  PetscLayout      map;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  if (!(*map_)) {
    ierr = PetscLayoutCreate(PetscObjectComm((PetscObject)viewer),map_);CHKERRQ(ierr);
  }
  map = *map_;

  /* Get actual number of dimensions in dataset */
  PetscStackCallHDF5Return(ctx->rdim,H5Sget_simple_extent_dims,(ctx->dataspace, NULL, NULL));
  ierr = PetscMalloc1(ctx->rdim, &ctx->dims);CHKERRQ(ierr);
  PetscStackCallHDF5Return(ctx->rdim,H5Sget_simple_extent_dims,(ctx->dataspace, ctx->dims, NULL));

  /*
     Dimensions are in this order:
     [0]        timesteps (optional)
     [lenInd]   entries (numbers or blocks)
     ...
     [bsInd]    entries of blocks (optional)
     [bsInd+1]  real & imaginary part (optional)
      = rdim-1
   */

  /* Get entries dimension index */
  ctx->lenInd = 0;
  if (hdf5->timestepping) ++ctx->lenInd;

  /* Get block dimension index */
  ctx->bsInd = ctx->rdim-1;
  if (ctx->complexVal) --ctx->bsInd;
  if (ctx->lenInd > ctx->bsInd) SETERRQ2(PetscObjectComm((PetscObject)viewer), PETSC_ERR_PLIB, "Calculated block dimension index = %D < %D = length dimension index.",ctx->bsInd,ctx->lenInd);
  if (ctx->bsInd > ctx->rdim - 1) SETERRQ2(PetscObjectComm((PetscObject)viewer), PETSC_ERR_FILE_UNEXPECTED, "Calculated block dimension index = %D > %D = total number of dimensions - 1.",ctx->bsInd,ctx->rdim-1);

  /* Get block size */
  ctx->dim2 = PETSC_FALSE;
  if (ctx->lenInd == ctx->bsInd) {
    bs = 1; /* support vectors stored as 1D array */
  } else {
    bs = (PetscInt) ctx->dims[ctx->bsInd];
    if (bs == 1) ctx->dim2 = PETSC_TRUE; /* vector with blocksize of 1, still stored as 2D array */
  }

  /* Get global size */
  len = ctx->dims[ctx->lenInd];
  if (ctx->horizontal) {
    PetscInt t;
    /* support horizontal 1D arrays (MATLAB vectors) - swap meaning of blocks and entries */
    if (ctx->complexVal) SETERRQ(PetscObjectComm((PetscObject)viewer), PETSC_ERR_SUP, "Complex and horizontal at the same time not allowed.");
    t = len; len = bs; bs = t;
    t = ctx->lenInd; ctx->lenInd = ctx->bsInd; ctx->bsInd = t;
  }
  N = (PetscInt) len*bs;

  /* Set global size, blocksize and type if not yet set */
  if (map->bs < 0) {
    ierr = PetscLayoutSetBlockSize(map, bs);CHKERRQ(ierr);
  } else if (map->bs != bs) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_FILE_UNEXPECTED, "Block size of array in file is %D, not %D as expected",bs,map->bs);
  if (map->N < 0) {
    ierr = PetscLayoutSetSize(map, N);CHKERRQ(ierr);
  } else if (map->N != N) SETERRQ2(PetscObjectComm((PetscObject)viewer),PETSC_ERR_FILE_UNEXPECTED, "Global size of array in file is %D, not %D as expected",N,map->N);
  if (setup) {ierr = PetscLayoutSetUp(map);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscViewerHDF5ReadSelectHyperslab_Private(PetscViewer viewer, HDF5ReadCtx ctx, PetscLayout map, hid_t *memspace)
{
  PetscViewer_HDF5 *hdf5 = (PetscViewer_HDF5*) viewer->data;
  hsize_t          *count, *offset;
  PetscInt         bs, n, low;
  int              i;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  /* Compute local size and ownership range */
  ierr = PetscLayoutSetUp(map);CHKERRQ(ierr);
  ierr = PetscLayoutGetBlockSize(map, &bs);CHKERRQ(ierr);
  ierr = PetscLayoutGetLocalSize(map, &n);CHKERRQ(ierr);
  ierr = PetscLayoutGetRange(map, &low, NULL);CHKERRQ(ierr);

  /* Each process defines a dataset and reads it from the hyperslab in the file */
  ierr = PetscMalloc2(ctx->rdim, &count, ctx->rdim, &offset);CHKERRQ(ierr);
  for (i=0; i<ctx->rdim; i++) {
    /* By default, select all entries with no offset */
    offset[i] = 0;
    count[i] = ctx->dims[i];
  }
  if (hdf5->timestepping) {
    count[0]  = 1;
    offset[0] = hdf5->timestep;
  }
  {
    ierr = PetscHDF5IntCast(n/bs, &count[ctx->lenInd]);CHKERRQ(ierr);
    ierr = PetscHDF5IntCast(low/bs, &offset[ctx->lenInd]);CHKERRQ(ierr);
  }
  if (ctx->complexVal && count[ctx->bsInd+1] != 2) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_FILE_UNEXPECTED,"Complex numbers must have exactly 2 parts (%D)",count[ctx->bsInd+1]);
  PetscStackCallHDF5Return(*memspace,H5Screate_simple,(ctx->rdim, count, NULL));
  PetscStackCallHDF5(H5Sselect_hyperslab,(ctx->dataspace, H5S_SELECT_SET, offset, NULL, count, NULL));
  ierr = PetscFree2(count, offset);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscViewerHDF5ReadArray_Private(PetscViewer viewer, HDF5ReadCtx h, hid_t datatype, hid_t memspace, void *arr)
{
  PetscViewer_HDF5 *hdf5 = (PetscViewer_HDF5*) viewer->data;

  PetscFunctionBegin;
  PetscStackCallHDF5(H5Dread,(h->dataset, datatype, memspace, h->dataspace, hdf5->dxpl_id, arr));
  PetscFunctionReturn(0);
}

/*@C
  PetscViewerHDF5Load - Read a raw array from the HDF5 dataset.

  Input Parameters:
+ viewer   - The HDF5 viewer
. name     - The dataset name
. map      - The layout which specifies array partitioning
- datatype - The HDF5 datatype of the items in the dataset

  Output Parameter:
+ map      - The set up layout (with global size and blocksize according to dataset)
- newarr   - The partitioned array, a memory image of the given dataset

  Level: developer

  Notes:
  This is intended mainly for internal use; users should use higher level routines such as ISLoad(), VecLoad(), DMLoad().
  The array is partitioned according to the given PetscLayout which is converted to an HDF5 hyperslab.
  This name is relative to the current group returned by PetscViewerHDF5OpenGroup().

  Fortran Notes:
  This routine is not available in Fortran.

.seealso PetscViewerHDF5Open(), PetscViewerHDF5PushGroup(), PetscViewerHDF5OpenGroup(), PetscViewerHDF5ReadSizes(), VecLoad(), ISLoad()
@*/
PetscErrorCode PetscViewerHDF5Load(PetscViewer viewer, const char *name, PetscLayout map, hid_t datatype, void **newarr)
{
  PetscBool       has;
  HDF5ReadCtx     h=NULL;
  hid_t           memspace=0;
  size_t          unitsize;
  void            *arr;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  ierr = PetscViewerHDF5HasDataset(viewer, name, &has);CHKERRQ(ierr);
  if (!has) {
    const char *group;
    ierr = PetscViewerHDF5GetGroup(viewer, &group);CHKERRQ(ierr);
    SETERRQ2(PetscObjectComm((PetscObject)viewer), PETSC_ERR_FILE_UNEXPECTED, "Object (dataset) \"%s\" not stored in group %s", name, group ? group : "/");
  }
  ierr = PetscViewerHDF5ReadInitialize_Private(viewer, name, &h);CHKERRQ(ierr);
#if defined(PETSC_USE_COMPLEX)
  if (!h->complexVal) {
    H5T_class_t clazz = H5Tget_class(datatype);
    if (clazz == H5T_FLOAT) SETERRQ(PetscObjectComm((PetscObject)viewer),PETSC_ERR_SUP,"File contains real numbers but PETSc is configured for complex. The conversion is not yet implemented. Configure with --with-scalar-type=real.");
  }
#else
  if (h->complexVal) SETERRQ(PetscObjectComm((PetscObject)viewer),PETSC_ERR_SUP,"File contains complex numbers but PETSc not configured for them. Configure with --with-scalar-type=complex.");
#endif

  ierr = PetscViewerHDF5ReadSizes_Private(viewer, h, PETSC_TRUE, &map);CHKERRQ(ierr);
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
  ierr = PetscViewerHDF5ReadSizes_Private(viewer, h, PETSC_FALSE, &map);CHKERRQ(ierr);
  ierr = PetscViewerHDF5ReadFinalize_Private(viewer, &h);CHKERRQ(ierr);
  if (bs) *bs = map->bs;
  if (N) *N = map->N;
  ierr = PetscLayoutDestroy(&map);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#endif /* defined(PETSC_HAVE_HDF5) */
