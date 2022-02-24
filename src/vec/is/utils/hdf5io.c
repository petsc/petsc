#include <petsc/private/viewerhdf5impl.h>
#include <petsclayouthdf5.h>    /*I   "petsclayoutdf5.h"   I*/
#include <petscis.h>    /*I   "petscis.h"   I*/

#if defined(PETSC_HAVE_HDF5)

struct _n_HDF5ReadCtx {
  hid_t     file, group, dataset, dataspace;
  int       lenInd, bsInd, complexInd, rdim;
  hsize_t   *dims;
  PetscBool complexVal, dim2;
};
typedef struct _n_HDF5ReadCtx* HDF5ReadCtx;

PetscErrorCode PetscViewerHDF5CheckTimestepping_Internal(PetscViewer viewer, const char name[])
{
  PetscViewer_HDF5 *hdf5 = (PetscViewer_HDF5*) viewer->data;
  PetscBool        timestepping = PETSC_FALSE;
  const char       *group;

  PetscFunctionBegin;
  CHKERRQ(PetscViewerHDF5GetGroup(viewer, &group));
  CHKERRQ(PetscViewerHDF5ReadAttribute(viewer,name,"timestepping",PETSC_BOOL,&timestepping,&timestepping));
  PetscCheckFalse(timestepping != hdf5->timestepping,PetscObjectComm((PetscObject)viewer),PETSC_ERR_FILE_UNEXPECTED,"Dataset %s/%s stored with timesteps? %s Timestepping pushed? %s", group, name, PetscBools[timestepping], PetscBools[hdf5->timestepping]);
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscViewerHDF5ReadInitialize_Private(PetscViewer viewer, const char name[], HDF5ReadCtx *ctx)
{
  PetscViewer_HDF5 *hdf5 = (PetscViewer_HDF5*) viewer->data;
  HDF5ReadCtx      h=NULL;

  PetscFunctionBegin;
  CHKERRQ(PetscViewerHDF5CheckTimestepping_Internal(viewer, name));
  CHKERRQ(PetscNew(&h));
  CHKERRQ(PetscViewerHDF5OpenGroup(viewer, &h->file, &h->group));
  PetscStackCallHDF5Return(h->dataset,H5Dopen2,(h->group, name, H5P_DEFAULT));
  PetscStackCallHDF5Return(h->dataspace,H5Dget_space,(h->dataset));
  CHKERRQ(PetscViewerHDF5ReadAttribute(viewer,name,"complex",PETSC_BOOL,&h->complexVal,&h->complexVal));
  if (!hdf5->horizontal) {
    /* MATLAB stores column vectors horizontally */
    CHKERRQ(PetscViewerHDF5HasAttribute(viewer,name,"MATLAB_class",&hdf5->horizontal));
  }
  *ctx = h;
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscViewerHDF5ReadFinalize_Private(PetscViewer viewer, HDF5ReadCtx *ctx)
{
  HDF5ReadCtx    h;

  PetscFunctionBegin;
  h = *ctx;
  PetscStackCallHDF5(H5Gclose,(h->group));
  PetscStackCallHDF5(H5Sclose,(h->dataspace));
  PetscStackCallHDF5(H5Dclose,(h->dataset));
  CHKERRQ(PetscFree((*ctx)->dims));
  CHKERRQ(PetscFree(*ctx));
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscViewerHDF5ReadSizes_Private(PetscViewer viewer, HDF5ReadCtx ctx, PetscBool setup, PetscLayout *map_)
{
  PetscViewer_HDF5 *hdf5 = (PetscViewer_HDF5*) viewer->data;
  PetscInt         bs, len, N;
  PetscLayout      map;

  PetscFunctionBegin;
  if (!(*map_)) {
    CHKERRQ(PetscLayoutCreate(PetscObjectComm((PetscObject)viewer),map_));
  }
  map = *map_;

  /* Get actual number of dimensions in dataset */
  PetscStackCallHDF5Return(ctx->rdim,H5Sget_simple_extent_dims,(ctx->dataspace, NULL, NULL));
  CHKERRQ(PetscMalloc1(ctx->rdim, &ctx->dims));
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
  if (ctx->complexVal) {
    ctx->bsInd = ctx->rdim-2;
    ctx->complexInd = ctx->rdim-1;
  } else {
    ctx->bsInd = ctx->rdim-1;
    ctx->complexInd = -1;
  }
  PetscCheckFalse(ctx->lenInd > ctx->bsInd,PetscObjectComm((PetscObject)viewer), PETSC_ERR_PLIB, "Calculated block dimension index = %d < %d = length dimension index.",ctx->bsInd,ctx->lenInd);
  PetscCheckFalse(ctx->bsInd > ctx->rdim - 1,PetscObjectComm((PetscObject)viewer), PETSC_ERR_FILE_UNEXPECTED, "Calculated block dimension index = %d > %d = total number of dimensions - 1.",ctx->bsInd,ctx->rdim-1);
  PetscCheckFalse(ctx->complexVal && ctx->dims[ctx->complexInd] != 2,PETSC_COMM_SELF,PETSC_ERR_FILE_UNEXPECTED,"Complex numbers must have exactly 2 parts (%llu)",ctx->dims[ctx->complexInd]);

  if (hdf5->horizontal) {
    PetscInt t;
    /* support horizontal 1D arrays (MATLAB vectors) - swap meaning of blocks and entries */
    t = ctx->lenInd; ctx->lenInd = ctx->bsInd; ctx->bsInd = t;
  }

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
  N = (PetscInt) len*bs;

  /* Set global size, blocksize and type if not yet set */
  if (map->bs < 0) {
    CHKERRQ(PetscLayoutSetBlockSize(map, bs));
  } else PetscCheckFalse(map->bs != bs,PETSC_COMM_SELF,PETSC_ERR_FILE_UNEXPECTED, "Block size of array in file is %" PetscInt_FMT ", not %" PetscInt_FMT " as expected",bs,map->bs);
  if (map->N < 0) {
    CHKERRQ(PetscLayoutSetSize(map, N));
  } else PetscCheckFalse(map->N != N,PetscObjectComm((PetscObject)viewer),PETSC_ERR_FILE_UNEXPECTED, "Global size of array in file is %" PetscInt_FMT ", not %" PetscInt_FMT " as expected",N,map->N);
  if (setup) CHKERRQ(PetscLayoutSetUp(map));
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscViewerHDF5ReadSelectHyperslab_Private(PetscViewer viewer, HDF5ReadCtx ctx, PetscLayout map, hid_t *memspace)
{
  PetscViewer_HDF5 *hdf5 = (PetscViewer_HDF5*) viewer->data;
  hsize_t          *count, *offset;
  PetscInt         bs, n, low;
  int              i;

  PetscFunctionBegin;
  /* Compute local size and ownership range */
  CHKERRQ(PetscLayoutSetUp(map));
  CHKERRQ(PetscLayoutGetBlockSize(map, &bs));
  CHKERRQ(PetscLayoutGetLocalSize(map, &n));
  CHKERRQ(PetscLayoutGetRange(map, &low, NULL));

  /* Each process defines a dataset and reads it from the hyperslab in the file */
  CHKERRQ(PetscMalloc2(ctx->rdim, &count, ctx->rdim, &offset));
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
    CHKERRQ(PetscHDF5IntCast(n/bs, &count[ctx->lenInd]));
    CHKERRQ(PetscHDF5IntCast(low/bs, &offset[ctx->lenInd]));
  }
  PetscStackCallHDF5Return(*memspace,H5Screate_simple,(ctx->rdim, count, NULL));
  PetscStackCallHDF5(H5Sselect_hyperslab,(ctx->dataspace, H5S_SELECT_SET, offset, NULL, count, NULL));
  CHKERRQ(PetscFree2(count, offset));
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
- datatype - The HDF5 datatype of the items in the dataset

  Input/Output Parameter:
. map      - The layout which specifies array partitioning, on output the
             set up layout (with global size and blocksize according to dataset)

  Output Parameter:
. newarr   - The partitioned array, a memory image of the given dataset

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
  const char      *group;
  HDF5ReadCtx     h=NULL;
  hid_t           memspace=0;
  size_t          unitsize;
  void            *arr;

  PetscFunctionBegin;
  CHKERRQ(PetscViewerHDF5GetGroup(viewer, &group));
  CHKERRQ(PetscViewerHDF5HasDataset(viewer, name, &has));
  PetscCheckFalse(!has,PetscObjectComm((PetscObject)viewer), PETSC_ERR_FILE_UNEXPECTED, "Object (dataset) \"%s\" not stored in group %s", name, group ? group : "/");
  CHKERRQ(PetscViewerHDF5ReadInitialize_Private(viewer, name, &h));
#if defined(PETSC_USE_COMPLEX)
  if (!h->complexVal) {
    H5T_class_t clazz = H5Tget_class(datatype);
    PetscCheckFalse(clazz == H5T_FLOAT,PetscObjectComm((PetscObject)viewer),PETSC_ERR_SUP,"Dataset %s/%s is marked as real but PETSc is configured for complex scalars. The conversion is not yet implemented. Configure with --with-scalar-type=real to read this dataset", group ? group : "",name);
  }
#else
  PetscCheckFalse(h->complexVal,PetscObjectComm((PetscObject)viewer),PETSC_ERR_SUP,"Dataset %s/%s is marked as complex but PETSc is configured for real scalars. Configure with --with-scalar-type=complex to read this dataset", group ? group : "",name);
#endif

  CHKERRQ(PetscViewerHDF5ReadSizes_Private(viewer, h, PETSC_TRUE, &map));
  CHKERRQ(PetscViewerHDF5ReadSelectHyperslab_Private(viewer, h, map, &memspace));

  unitsize = H5Tget_size(datatype);
  if (h->complexVal) unitsize *= 2;
  /* unitsize is size_t i.e. always unsigned, so the negative check is pointless? */
  PetscCheck(unitsize > 0 && unitsize <= PetscMax(sizeof(PetscInt),sizeof(PetscScalar)),PETSC_COMM_SELF,PETSC_ERR_LIB,"Sanity check failed: HDF5 function H5Tget_size(datatype) returned suspicious value %zu",unitsize);
  CHKERRQ(PetscMalloc(map->n*unitsize, &arr));

  CHKERRQ(PetscViewerHDF5ReadArray_Private(viewer, h, datatype, memspace, arr));
  PetscStackCallHDF5(H5Sclose,(memspace));
  CHKERRQ(PetscViewerHDF5ReadFinalize_Private(viewer, &h));
  *newarr = arr;
  PetscFunctionReturn(0);
}

/*@C
 PetscViewerHDF5ReadSizes - Read block size and global size of a vector (Vec or IS) stored in an HDF5 file.

  Input Parameters:
+ viewer - The HDF5 viewer
- name   - The dataset name

  Output Parameters:
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

  PetscFunctionBegin;
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_CLASSID,1);
  CHKERRQ(PetscViewerHDF5ReadInitialize_Private(viewer, name, &h));
  CHKERRQ(PetscViewerHDF5ReadSizes_Private(viewer, h, PETSC_FALSE, &map));
  CHKERRQ(PetscViewerHDF5ReadFinalize_Private(viewer, &h));
  if (bs) *bs = map->bs;
  if (N) *N = map->N;
  CHKERRQ(PetscLayoutDestroy(&map));
  PetscFunctionReturn(0);
}

#endif /* defined(PETSC_HAVE_HDF5) */
