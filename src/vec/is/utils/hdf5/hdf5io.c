#include <petsc/private/viewerhdf5impl.h>
#include <petsclayouthdf5.h> /*I   "petsclayoutdf5.h"   I*/
#include <petscis.h>         /*I   "petscis.h"   I*/

struct _n_HDF5ReadCtx {
  const char *name;
  hid_t       file, group, dataset, dataspace;
  int         lenInd, bsInd, complexInd, rdim;
  hsize_t    *dims;
  PetscBool   complexVal, dim2;

  // Needed for compression
  PetscInt  runs;
  PetscInt *cind;
};
typedef struct _n_HDF5ReadCtx *HDF5ReadCtx;

PetscErrorCode PetscViewerHDF5CheckTimestepping_Internal(PetscViewer viewer, const char name[])
{
  PetscViewer_HDF5 *hdf5         = (PetscViewer_HDF5 *)viewer->data;
  PetscBool         timestepping = PETSC_FALSE;

  PetscFunctionBegin;
  PetscCall(PetscViewerHDF5ReadAttribute(viewer, name, "timestepping", PETSC_BOOL, &hdf5->defTimestepping, &timestepping));
  if (timestepping != hdf5->timestepping) {
    const char *group;

    PetscCall(PetscViewerHDF5GetGroup(viewer, NULL, &group));
    SETERRQ(PetscObjectComm((PetscObject)viewer), PETSC_ERR_FILE_UNEXPECTED, "Dataset %s/%s stored with timesteps? %s Timestepping pushed? %s", group, name, PetscBools[timestepping], PetscBools[hdf5->timestepping]);
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscViewerHDF5ReadInitialize_Private(PetscViewer viewer, const char name[], HDF5ReadCtx *ctx)
{
  PetscViewer_HDF5 *hdf5 = (PetscViewer_HDF5 *)viewer->data;
  HDF5ReadCtx       h    = NULL;

  PetscFunctionBegin;
  PetscCall(PetscViewerHDF5CheckTimestepping_Internal(viewer, name));
  PetscCall(PetscNew(&h));
  h->name = name;
  PetscCall(PetscViewerHDF5OpenGroup(viewer, NULL, &h->file, &h->group));
  PetscCallHDF5Return(h->dataset, H5Dopen2, (h->group, name, H5P_DEFAULT));
  PetscCallHDF5Return(h->dataspace, H5Dget_space, (h->dataset));
  PetscCall(PetscViewerHDF5ReadAttribute(viewer, name, "complex", PETSC_BOOL, &h->complexVal, &h->complexVal));
  if (!hdf5->horizontal) {
    /* MATLAB stores column vectors horizontally */
    PetscCall(PetscViewerHDF5HasAttribute(viewer, name, "MATLAB_class", &hdf5->horizontal));
  }
  h->runs = 0;
  h->cind = NULL;
  *ctx    = h;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscViewerHDF5ReadFinalize_Private(PetscViewer viewer, HDF5ReadCtx *ctx)
{
  HDF5ReadCtx h;

  PetscFunctionBegin;
  h = *ctx;
  PetscCallHDF5(H5Gclose, (h->group));
  PetscCallHDF5(H5Sclose, (h->dataspace));
  PetscCallHDF5(H5Dclose, (h->dataset));
  PetscCall(PetscFree((*ctx)->dims));
  PetscCall(PetscFree((*ctx)->cind));
  PetscCall(PetscFree(*ctx));
  PetscFunctionReturn(PETSC_SUCCESS);
}

// Need forward declaration because we have a cyclic call chain
static PetscErrorCode PetscViewerHDF5Load_Internal(PetscViewer, const char[], PetscBool, PetscLayout, hid_t, void **);

static PetscErrorCode PetscViewerHDF5ReadSizes_Private(PetscViewer viewer, HDF5ReadCtx ctx, PetscBool uncompress, PetscBool setup, PetscLayout *map_)
{
  PetscViewer_HDF5 *hdf5 = (PetscViewer_HDF5 *)viewer->data;
  PetscInt          bs, N;
  PetscLayout       map;
  PetscBool         compressed;

  PetscFunctionBegin;
  if (!*map_) PetscCall(PetscLayoutCreate(PetscObjectComm((PetscObject)viewer), map_));
  map = *map_;

  PetscCall(PetscViewerHDF5HasAttribute(viewer, ctx->name, "compressed", &compressed));
  if (compressed && uncompress) {
    hid_t           inttype;
    PetscLayout     cmap;
    PetscInt       *lcind, N = 0;
    PetscMPIInt    *counts, *displs, size, n;
    const PetscInt *range;
    MPI_Comm        comm;

#if defined(PETSC_USE_64BIT_INDICES)
    inttype = H5T_NATIVE_LLONG;
#else
    inttype = H5T_NATIVE_INT;
#endif
    PetscCall(PetscObjectGetComm((PetscObject)viewer, &comm));
    PetscCall(PetscLayoutCreate(PetscObjectComm((PetscObject)viewer), &cmap));
    cmap->bs = 3;
    PetscCall(PetscViewerHDF5Load_Internal(viewer, ctx->name, PETSC_FALSE, cmap, inttype, (void **)&lcind));
    PetscCheck(!(cmap->n % 3), PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Compressed IS must have an even number of entries, not %" PetscInt_FMT, cmap->n);
    for (PetscInt i = 0; i < cmap->n / 3; ++i) N += lcind[i * 3 + 0];
    PetscCallMPI(MPIU_Allreduce(MPI_IN_PLACE, &N, 1, MPIU_INT, MPIU_SUM, comm));
    ctx->runs = cmap->N / 3;
    PetscCall(PetscMalloc1(cmap->N, &ctx->cind));
    PetscCallMPI(MPI_Comm_size(comm, &size));
    PetscCall(PetscLayoutGetRanges(cmap, &range));
    PetscCall(PetscMalloc2(size, &counts, size, &displs));
    for (PetscInt r = 0; r < size; ++r) {
      PetscCall(PetscMPIIntCast(range[r + 1] - range[r], &counts[r]));
      PetscCall(PetscMPIIntCast(range[r], &displs[r]));
    }
    PetscCall(PetscMPIIntCast(cmap->n, &n));
    PetscCallMPI(MPI_Allgatherv(lcind, n, MPIU_INT, ctx->cind, counts, displs, MPIU_INT, comm));
    PetscCall(PetscFree2(counts, displs));
    PetscCall(PetscFree(lcind));
    PetscCall(PetscLayoutDestroy(&cmap));

    ctx->dim2   = PETSC_FALSE;
    ctx->rdim   = 1;
    ctx->lenInd = 0;
    PetscCall(PetscMalloc1(ctx->rdim, &ctx->dims));
    ctx->dims[0] = N;
    bs           = 1;
    goto layout;
  }

  /* Get actual number of dimensions in dataset */
  PetscCallHDF5Return(ctx->rdim, H5Sget_simple_extent_dims, (ctx->dataspace, NULL, NULL));
  PetscCall(PetscMalloc1(ctx->rdim, &ctx->dims));
  PetscCallHDF5Return(ctx->rdim, H5Sget_simple_extent_dims, (ctx->dataspace, ctx->dims, NULL));

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
    ctx->bsInd      = ctx->rdim - 2;
    ctx->complexInd = ctx->rdim - 1;
  } else {
    ctx->bsInd      = ctx->rdim - 1;
    ctx->complexInd = -1;
  }
  PetscCheck(ctx->lenInd <= ctx->bsInd, PetscObjectComm((PetscObject)viewer), PETSC_ERR_PLIB, "Calculated block dimension index = %d < %d = length dimension index.", ctx->bsInd, ctx->lenInd);
  PetscCheck(ctx->bsInd <= ctx->rdim - 1, PetscObjectComm((PetscObject)viewer), PETSC_ERR_FILE_UNEXPECTED, "Calculated block dimension index = %d > %d = total number of dimensions - 1.", ctx->bsInd, ctx->rdim - 1);
  PetscCheck(!ctx->complexVal || ctx->dims[ctx->complexInd] == 2, PETSC_COMM_SELF, PETSC_ERR_FILE_UNEXPECTED, "Complex numbers must have exactly 2 parts (%" PRIuHSIZE ")", ctx->dims[ctx->complexInd]);

  if (hdf5->horizontal) {
    /* support horizontal 1D arrays (MATLAB vectors) - swap meaning of blocks and entries */
    int t       = ctx->lenInd;
    ctx->lenInd = ctx->bsInd;
    ctx->bsInd  = t;
  }

  /* Get block size */
  ctx->dim2 = PETSC_FALSE;
  if (ctx->lenInd == ctx->bsInd) {
    bs = 1; /* support vectors stored as 1D array */
  } else {
    bs = (PetscInt)ctx->dims[ctx->bsInd];
    if (bs == 1) ctx->dim2 = PETSC_TRUE; /* vector with blocksize of 1, still stored as 2D array */
  }

layout:
  /* Get global size */
  PetscCall(PetscIntCast(bs * ctx->dims[ctx->lenInd], &N));

  /* Set global size, blocksize and type if not yet set */
  PetscCall(PetscLayoutSetBlockSize(map, bs));
  if (map->N < 0) PetscCall(PetscLayoutSetSize(map, N));
  else PetscCheck(map->N == N, PetscObjectComm((PetscObject)viewer), PETSC_ERR_FILE_UNEXPECTED, "Global size of array %s in file is %" PetscInt_FMT ", not %" PetscInt_FMT " as expected", ctx->name, N, map->N);
  if (setup) PetscCall(PetscLayoutSetUp(map));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscViewerHDF5ReadSelectHyperslab_Private(PetscViewer viewer, HDF5ReadCtx ctx, PetscLayout map, hid_t *memspace)
{
  PetscViewer_HDF5 *hdf5 = (PetscViewer_HDF5 *)viewer->data;
  hsize_t          *count, *offset;
  PetscInt          bs, n, low;
  int               i;

  PetscFunctionBegin;
  /* Compute local size and ownership range */
  PetscCall(PetscLayoutSetUp(map));
  PetscCall(PetscLayoutGetBlockSize(map, &bs));
  PetscCall(PetscLayoutGetLocalSize(map, &n));
  PetscCall(PetscLayoutGetRange(map, &low, NULL));

  /* Each process defines a dataset and reads it from the hyperslab in the file */
  PetscCall(PetscMalloc2(ctx->rdim, &count, ctx->rdim, &offset));
  for (i = 0; i < ctx->rdim; i++) {
    /* By default, select all entries with no offset */
    offset[i] = 0;
    count[i]  = ctx->dims[i];
  }
  if (hdf5->timestepping) {
    count[0]  = 1;
    offset[0] = hdf5->timestep;
  }
  {
    PetscCall(PetscHDF5IntCast(n / bs, &count[ctx->lenInd]));
    PetscCall(PetscHDF5IntCast(low / bs, &offset[ctx->lenInd]));
  }
  PetscCallHDF5Return(*memspace, H5Screate_simple, (ctx->rdim, count, NULL));
  PetscCallHDF5(H5Sselect_hyperslab, (ctx->dataspace, H5S_SELECT_SET, offset, NULL, count, NULL));
  PetscCall(PetscFree2(count, offset));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscViewerHDF5ReadArray_Private(PetscViewer viewer, HDF5ReadCtx h, hid_t datatype, hid_t memspace, void *arr)
{
  PetscViewer_HDF5 *hdf5 = (PetscViewer_HDF5 *)viewer->data;

  PetscFunctionBegin;
  PetscCallHDF5(H5Dread, (h->dataset, datatype, memspace, h->dataspace, hdf5->dxpl_id, arr));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscViewerHDF5Load_Internal(PetscViewer viewer, const char name[], PetscBool uncompress, PetscLayout map, hid_t datatype, void **newarr)
{
  PetscBool   has;
  const char *group;
  HDF5ReadCtx h        = NULL;
  hid_t       memspace = 0;
  size_t      unitsize;
  void       *arr;

  PetscFunctionBegin;
  PetscCall(PetscViewerHDF5GetGroup(viewer, NULL, &group));
  PetscCall(PetscViewerHDF5HasDataset(viewer, name, &has));
  PetscCheck(has, PetscObjectComm((PetscObject)viewer), PETSC_ERR_FILE_UNEXPECTED, "Object (dataset) \"%s\" not stored in group %s", name, group);
  PetscCall(PetscViewerHDF5ReadInitialize_Private(viewer, name, &h));
#if defined(PETSC_USE_COMPLEX)
  if (!h->complexVal) {
    H5T_class_t clazz = H5Tget_class(datatype);
    PetscCheck(clazz != H5T_FLOAT, PetscObjectComm((PetscObject)viewer), PETSC_ERR_SUP, "Dataset %s/%s is marked as real but PETSc is configured for complex scalars. The conversion is not yet implemented. Configure with --with-scalar-type=real to read this dataset", group ? group : "", name);
  }
#else
  PetscCheck(!h->complexVal, PetscObjectComm((PetscObject)viewer), PETSC_ERR_SUP, "Dataset %s/%s is marked as complex but PETSc is configured for real scalars. Configure with --with-scalar-type=complex to read this dataset", group, name);
#endif

  PetscCall(PetscViewerHDF5ReadSizes_Private(viewer, h, uncompress, PETSC_TRUE, &map));
  PetscCall(PetscViewerHDF5ReadSelectHyperslab_Private(viewer, h, map, &memspace));

  if (h->runs && uncompress) {
    PetscInt *ind;

    PetscCall(PetscInfo(viewer, "Read compressed object with name %s of size %" PetscInt_FMT ":%" PetscInt_FMT "\n", name, map->n, map->N));
    // Each process stores the whole compression, so skip any leading parts
    PetscCall(PetscMalloc1(map->n, &ind));
    for (PetscInt i = 0, off = 0; i < h->runs; ++i) {
      for (PetscInt j = 0, inc = 0; j < h->cind[i * 3 + 0]; ++j, ++off, inc += h->cind[i * 3 + 1]) {
        if (off >= map->rend) {
          i = h->runs;
          break;
        }
        if (off >= map->rstart) ind[off - map->rstart] = h->cind[i * 3 + 2] + inc;
      }
    }
    *newarr = ind;
    goto cleanup;
  }

  unitsize = H5Tget_size(datatype);
  if (h->complexVal) unitsize *= 2;
  /* unitsize is size_t i.e. always unsigned, so the negative check is pointless? */
  PetscCheck(unitsize > 0 && unitsize <= PetscMax(sizeof(PetscInt), sizeof(PetscScalar)), PETSC_COMM_SELF, PETSC_ERR_LIB, "Sanity check failed: HDF5 function H5Tget_size(datatype) returned suspicious value %zu", unitsize);
  PetscCall(PetscMalloc(map->n * unitsize, &arr));

  PetscCall(PetscViewerHDF5ReadArray_Private(viewer, h, datatype, memspace, arr));
  *newarr = arr;

cleanup:
  PetscCallHDF5(H5Sclose, (memspace));
  PetscCall(PetscViewerHDF5ReadFinalize_Private(viewer, &h));
  PetscCall(PetscFree(group));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PetscViewerHDF5Load - Read a raw array from the `PETSCVIEWERHDF5` dataset in parallel

  Collective; No Fortran Support

  Input Parameters:
+ viewer   - The `PETSCVIEWERHDF5` viewer
. name     - The dataset name
- datatype - The HDF5 datatype of the items in the dataset

  Input/Output Parameter:
. map - The layout which specifies array partitioning, on output the
             set up layout (with global size and blocksize according to dataset)

  Output Parameter:
. newarr - The partitioned array, a memory image of the given dataset

  Level: developer

  Notes:
  This is intended mainly for internal use; users should use higher level routines such as `ISLoad()`, `VecLoad()`, `DMLoad()`.

  The array is partitioned according to the given `PetscLayout` which is converted to an HDF5 hyperslab.

  This name is relative to the current group returned by `PetscViewerHDF5OpenGroup()`.

.seealso: `PetscViewer`, `PETSCVIEWERHDF5`, `PetscViewerHDF5Open()`, `PetscViewerHDF5PushGroup()`, `PetscViewerHDF5OpenGroup()`, `PetscViewerHDF5ReadSizes()`,
          `VecLoad()`, `ISLoad()`, `PetscLayout`
@*/
PetscErrorCode PetscViewerHDF5Load(PetscViewer viewer, const char name[], PetscLayout map, hid_t datatype, void **newarr)
{
  PetscFunctionBegin;
  PetscCall(PetscViewerHDF5Load_Internal(viewer, name, PETSC_TRUE, map, datatype, newarr));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscViewerHDF5ReadSizes - Read block size and global size of a `Vec` or `IS` stored in an HDF5 file.

  Input Parameters:
+ viewer - The `PETSCVIEWERHDF5` viewer
- name   - The dataset name

  Output Parameters:
+ bs - block size
- N  - global size

  Level: advanced

  Notes:
  The dataset is stored as an HDF5 dataspace with 1-4 dimensions in the order
  1) # timesteps (optional), 2) # blocks, 3) # elements per block (optional), 4) real and imaginary part (only for complex).

  The dataset can be stored as a 2D dataspace even if its blocksize is 1; see `PetscViewerHDF5SetBaseDimension2()`.

.seealso: `PetscViewer`, `PETSCVIEWERHDF5`, `PetscViewerHDF5Open()`, `VecLoad()`, `ISLoad()`, `VecGetSize()`, `ISGetSize()`, `PetscViewerHDF5SetBaseDimension2()`
@*/
PetscErrorCode PetscViewerHDF5ReadSizes(PetscViewer viewer, const char name[], PetscInt *bs, PetscInt *N)
{
  HDF5ReadCtx h   = NULL;
  PetscLayout map = NULL;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 1);
  PetscCall(PetscViewerHDF5ReadInitialize_Private(viewer, name, &h));
  PetscCall(PetscViewerHDF5ReadSizes_Private(viewer, h, PETSC_TRUE, PETSC_FALSE, &map));
  PetscCall(PetscViewerHDF5ReadFinalize_Private(viewer, &h));
  if (bs) *bs = map->bs;
  if (N) *N = map->N;
  PetscCall(PetscLayoutDestroy(&map));
  PetscFunctionReturn(PETSC_SUCCESS);
}
