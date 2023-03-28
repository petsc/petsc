
#include <petsc/private/isimpl.h> /*I "petscis.h"  I*/
#include <petsc/private/hashmapi.h>
#include <petscsf.h>
#include <petscviewer.h>

PetscClassId          IS_LTOGM_CLASSID;
static PetscErrorCode ISLocalToGlobalMappingGetBlockInfo_Private(ISLocalToGlobalMapping, PetscInt *, PetscInt **, PetscInt **, PetscInt ***);

typedef struct {
  PetscInt *globals;
} ISLocalToGlobalMapping_Basic;

typedef struct {
  PetscHMapI globalht;
} ISLocalToGlobalMapping_Hash;

/*@C
  ISGetPointRange - Returns a description of the points in an `IS` suitable for traversal

  Not Collective

  Input Parameter:
. pointIS - The `IS` object

  Output Parameters:
+ pStart - The first index, see notes
. pEnd   - One past the last index, see notes
- points - The indices, see notes

  Level: intermediate

  Notes:
  If the `IS` contains contiguous indices in an `ISSTRIDE`, then the indices are contained in [pStart, pEnd) and points = `NULL`.
  Otherwise, `pStart = 0`, `pEnd = numIndices`, and points is an array of the indices. This supports the following pattern
.vb
  ISGetPointRange(is, &pStart, &pEnd, &points);
  for (p = pStart; p < pEnd; ++p) {
    const PetscInt point = points ? points[p] : p;
    // use point
  }
  ISRestorePointRange(is, &pstart, &pEnd, &points);
.ve
  Hence the same code can be written for `pointIS` being a `ISSTRIDE` or `ISGENERAL`

.seealso: [](sec_scatter), `IS`, `ISRestorePointRange()`, `ISGetPointSubrange()`, `ISGetIndices()`, `ISCreateStride()`
@*/
PetscErrorCode ISGetPointRange(IS pointIS, PetscInt *pStart, PetscInt *pEnd, const PetscInt **points)
{
  PetscInt  numCells, step = 1;
  PetscBool isStride;

  PetscFunctionBeginHot;
  *pStart = 0;
  *points = NULL;
  PetscCall(ISGetLocalSize(pointIS, &numCells));
  PetscCall(PetscObjectTypeCompare((PetscObject)pointIS, ISSTRIDE, &isStride));
  if (isStride) PetscCall(ISStrideGetInfo(pointIS, pStart, &step));
  *pEnd = *pStart + numCells;
  if (!isStride || step != 1) PetscCall(ISGetIndices(pointIS, points));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  ISRestorePointRange - Destroys the traversal description created with `ISGetPointRange()`

  Not Collective

  Input Parameters:
+ pointIS - The `IS` object
. pStart  - The first index, from `ISGetPointRange()`
. pEnd    - One past the last index, from `ISGetPointRange()`
- points  - The indices, from `ISGetPointRange()`

  Level: intermediate

.seealso: [](sec_scatter), `IS`, `ISGetPointRange()`, `ISGetPointSubrange()`, `ISGetIndices()`, `ISCreateStride()`
@*/
PetscErrorCode ISRestorePointRange(IS pointIS, PetscInt *pStart, PetscInt *pEnd, const PetscInt **points)
{
  PetscInt  step = 1;
  PetscBool isStride;

  PetscFunctionBeginHot;
  PetscCall(PetscObjectTypeCompare((PetscObject)pointIS, ISSTRIDE, &isStride));
  if (isStride) PetscCall(ISStrideGetInfo(pointIS, pStart, &step));
  if (!isStride || step != 1) PetscCall(ISGetIndices(pointIS, points));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  ISGetPointSubrange - Configures the input `IS` to be a subrange for the traversal information given

  Not Collective

  Input Parameters:
+ subpointIS - The `IS` object to be configured
. pStart     - The first index of the subrange
. pEnd       - One past the last index for the subrange
- points     - The indices for the entire range, from `ISGetPointRange()`

  Output Parameters:
. subpointIS - The `IS` object now configured to be a subrange

  Level: intermediate

  Note:
  The input `IS` will now respond properly to calls to `ISGetPointRange()` and return the subrange.

.seealso: [](sec_scatter), `IS`, `ISGetPointRange()`, `ISRestorePointRange()`, `ISGetIndices()`, `ISCreateStride()`
@*/
PetscErrorCode ISGetPointSubrange(IS subpointIS, PetscInt pStart, PetscInt pEnd, const PetscInt *points)
{
  PetscFunctionBeginHot;
  if (points) {
    PetscCall(ISSetType(subpointIS, ISGENERAL));
    PetscCall(ISGeneralSetIndices(subpointIS, pEnd - pStart, &points[pStart], PETSC_USE_POINTER));
  } else {
    PetscCall(ISSetType(subpointIS, ISSTRIDE));
    PetscCall(ISStrideSetStride(subpointIS, pEnd - pStart, pStart, 1));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* -----------------------------------------------------------------------------------------*/

/*
    Creates the global mapping information in the ISLocalToGlobalMapping structure

    If the user has not selected how to handle the global to local mapping then use HASH for "large" problems
*/
static PetscErrorCode ISGlobalToLocalMappingSetUp(ISLocalToGlobalMapping mapping)
{
  PetscInt i, *idx = mapping->indices, n = mapping->n, end, start;

  PetscFunctionBegin;
  if (mapping->data) PetscFunctionReturn(PETSC_SUCCESS);
  end   = 0;
  start = PETSC_MAX_INT;

  for (i = 0; i < n; i++) {
    if (idx[i] < 0) continue;
    if (idx[i] < start) start = idx[i];
    if (idx[i] > end) end = idx[i];
  }
  if (start > end) {
    start = 0;
    end   = -1;
  }
  mapping->globalstart = start;
  mapping->globalend   = end;
  if (!((PetscObject)mapping)->type_name) {
    if ((end - start) > PetscMax(4 * n, 1000000)) {
      PetscCall(ISLocalToGlobalMappingSetType(mapping, ISLOCALTOGLOBALMAPPINGHASH));
    } else {
      PetscCall(ISLocalToGlobalMappingSetType(mapping, ISLOCALTOGLOBALMAPPINGBASIC));
    }
  }
  PetscTryTypeMethod(mapping, globaltolocalmappingsetup);
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode ISGlobalToLocalMappingSetUp_Basic(ISLocalToGlobalMapping mapping)
{
  PetscInt                      i, *idx = mapping->indices, n = mapping->n, end, start, *globals;
  ISLocalToGlobalMapping_Basic *map;

  PetscFunctionBegin;
  start = mapping->globalstart;
  end   = mapping->globalend;
  PetscCall(PetscNew(&map));
  PetscCall(PetscMalloc1(end - start + 2, &globals));
  map->globals = globals;
  for (i = 0; i < end - start + 1; i++) globals[i] = -1;
  for (i = 0; i < n; i++) {
    if (idx[i] < 0) continue;
    globals[idx[i] - start] = i;
  }
  mapping->data = (void *)map;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode ISGlobalToLocalMappingSetUp_Hash(ISLocalToGlobalMapping mapping)
{
  PetscInt                     i, *idx = mapping->indices, n = mapping->n;
  ISLocalToGlobalMapping_Hash *map;

  PetscFunctionBegin;
  PetscCall(PetscNew(&map));
  PetscCall(PetscHMapICreate(&map->globalht));
  for (i = 0; i < n; i++) {
    if (idx[i] < 0) continue;
    PetscCall(PetscHMapISet(map->globalht, idx[i], i));
  }
  mapping->data = (void *)map;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode ISLocalToGlobalMappingDestroy_Basic(ISLocalToGlobalMapping mapping)
{
  ISLocalToGlobalMapping_Basic *map = (ISLocalToGlobalMapping_Basic *)mapping->data;

  PetscFunctionBegin;
  if (!map) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(PetscFree(map->globals));
  PetscCall(PetscFree(mapping->data));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode ISLocalToGlobalMappingDestroy_Hash(ISLocalToGlobalMapping mapping)
{
  ISLocalToGlobalMapping_Hash *map = (ISLocalToGlobalMapping_Hash *)mapping->data;

  PetscFunctionBegin;
  if (!map) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(PetscHMapIDestroy(&map->globalht));
  PetscCall(PetscFree(mapping->data));
  PetscFunctionReturn(PETSC_SUCCESS);
}

#define GTOLTYPE _Basic
#define GTOLNAME _Basic
#define GTOLBS   mapping->bs
#define GTOL(g, local) \
  do { \
    local = map->globals[g / bs - start]; \
    if (local >= 0) local = bs * local + (g % bs); \
  } while (0)

#include <../src/vec/is/utils/isltog.h>

#define GTOLTYPE _Basic
#define GTOLNAME Block_Basic
#define GTOLBS   1
#define GTOL(g, local) \
  do { \
    local = map->globals[g - start]; \
  } while (0)
#include <../src/vec/is/utils/isltog.h>

#define GTOLTYPE _Hash
#define GTOLNAME _Hash
#define GTOLBS   mapping->bs
#define GTOL(g, local) \
  do { \
    (void)PetscHMapIGet(map->globalht, g / bs, &local); \
    if (local >= 0) local = bs * local + (g % bs); \
  } while (0)
#include <../src/vec/is/utils/isltog.h>

#define GTOLTYPE _Hash
#define GTOLNAME Block_Hash
#define GTOLBS   1
#define GTOL(g, local) \
  do { \
    (void)PetscHMapIGet(map->globalht, g, &local); \
  } while (0)
#include <../src/vec/is/utils/isltog.h>

/*@
    ISLocalToGlobalMappingDuplicate - Duplicates the local to global mapping object

    Not Collective

    Input Parameter:
.   ltog - local to global mapping

    Output Parameter:
.   nltog - the duplicated local to global mapping

    Level: advanced

.seealso: [](sec_scatter), `ISLocalToGlobalMapping`, `ISLocalToGlobalMappingDestroy()`, `ISLocalToGlobalMappingCreate()`
@*/
PetscErrorCode ISLocalToGlobalMappingDuplicate(ISLocalToGlobalMapping ltog, ISLocalToGlobalMapping *nltog)
{
  ISLocalToGlobalMappingType l2gtype;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ltog, IS_LTOGM_CLASSID, 1);
  PetscCall(ISLocalToGlobalMappingCreate(PetscObjectComm((PetscObject)ltog), ltog->bs, ltog->n, ltog->indices, PETSC_COPY_VALUES, nltog));
  PetscCall(ISLocalToGlobalMappingGetType(ltog, &l2gtype));
  PetscCall(ISLocalToGlobalMappingSetType(*nltog, l2gtype));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
    ISLocalToGlobalMappingGetSize - Gets the local size of a local to global mapping

    Not Collective

    Input Parameter:
.   ltog - local to global mapping

    Output Parameter:
.   n - the number of entries in the local mapping, `ISLocalToGlobalMappingGetIndices()` returns an array of this length

    Level: advanced

.seealso: [](sec_scatter), `ISLocalToGlobalMapping`, `ISLocalToGlobalMappingDestroy()`, `ISLocalToGlobalMappingCreate()`
@*/
PetscErrorCode ISLocalToGlobalMappingGetSize(ISLocalToGlobalMapping mapping, PetscInt *n)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(mapping, IS_LTOGM_CLASSID, 1);
  PetscValidIntPointer(n, 2);
  *n = mapping->bs * mapping->n;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
   ISLocalToGlobalMappingViewFromOptions - View an `ISLocalToGlobalMapping` based on values in the options database

   Collective

   Input Parameters:
+  A - the local to global mapping object
.  obj - Optional object that provides the options prefix used for the options database query
-  name - command line option

   Level: intermediate

   Note:
   See `PetscObjectViewFromOptions()` for the available `PetscViewer` and `PetscViewerFormat`

.seealso: [](sec_scatter), `PetscViewer`, ``ISLocalToGlobalMapping`, `ISLocalToGlobalMappingView`, `PetscObjectViewFromOptions()`, `ISLocalToGlobalMappingCreate()`
@*/
PetscErrorCode ISLocalToGlobalMappingViewFromOptions(ISLocalToGlobalMapping A, PetscObject obj, const char name[])
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(A, IS_LTOGM_CLASSID, 1);
  PetscCall(PetscObjectViewFromOptions((PetscObject)A, obj, name));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
    ISLocalToGlobalMappingView - View a local to global mapping

    Not Collective

    Input Parameters:
+   ltog - local to global mapping
-   viewer - viewer

    Level: advanced

.seealso: [](sec_scatter), `PetscViewer`, `ISLocalToGlobalMapping`, `ISLocalToGlobalMappingDestroy()`, `ISLocalToGlobalMappingCreate()`
@*/
PetscErrorCode ISLocalToGlobalMappingView(ISLocalToGlobalMapping mapping, PetscViewer viewer)
{
  PetscInt    i;
  PetscMPIInt rank;
  PetscBool   iascii;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mapping, IS_LTOGM_CLASSID, 1);
  if (!viewer) PetscCall(PetscViewerASCIIGetStdout(PetscObjectComm((PetscObject)mapping), &viewer));
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 2);

  PetscCallMPI(MPI_Comm_rank(PetscObjectComm((PetscObject)mapping), &rank));
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERASCII, &iascii));
  if (iascii) {
    PetscCall(PetscObjectPrintClassNamePrefixType((PetscObject)mapping, viewer));
    PetscCall(PetscViewerASCIIPushSynchronized(viewer));
    for (i = 0; i < mapping->n; i++) {
      PetscInt bs = mapping->bs, g = mapping->indices[i];
      if (bs == 1) PetscCall(PetscViewerASCIISynchronizedPrintf(viewer, "[%d] %" PetscInt_FMT " %" PetscInt_FMT "\n", rank, i, g));
      else PetscCall(PetscViewerASCIISynchronizedPrintf(viewer, "[%d] %" PetscInt_FMT ":%" PetscInt_FMT " %" PetscInt_FMT ":%" PetscInt_FMT "\n", rank, i * bs, (i + 1) * bs, g * bs, (g + 1) * bs));
    }
    PetscCall(PetscViewerFlush(viewer));
    PetscCall(PetscViewerASCIIPopSynchronized(viewer));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
    ISLocalToGlobalMappingCreateIS - Creates a mapping between a local (0 to n)
    ordering and a global parallel ordering.

    Not Collective

    Input Parameter:
.   is - index set containing the global numbers for each local number

    Output Parameter:
.   mapping - new mapping data structure

    Level: advanced

    Note:
    the block size of the `IS` determines the block size of the mapping

.seealso: [](sec_scatter), `ISLocalToGlobalMapping`, `ISLocalToGlobalMappingDestroy()`, `ISLocalToGlobalMappingCreate()`, `ISLocalToGlobalMappingSetFromOptions()`
@*/
PetscErrorCode ISLocalToGlobalMappingCreateIS(IS is, ISLocalToGlobalMapping *mapping)
{
  PetscInt        n, bs;
  const PetscInt *indices;
  MPI_Comm        comm;
  PetscBool       isblock;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(is, IS_CLASSID, 1);
  PetscValidPointer(mapping, 2);

  PetscCall(PetscObjectGetComm((PetscObject)is, &comm));
  PetscCall(ISGetLocalSize(is, &n));
  PetscCall(PetscObjectTypeCompare((PetscObject)is, ISBLOCK, &isblock));
  if (!isblock) {
    PetscCall(ISGetIndices(is, &indices));
    PetscCall(ISLocalToGlobalMappingCreate(comm, 1, n, indices, PETSC_COPY_VALUES, mapping));
    PetscCall(ISRestoreIndices(is, &indices));
  } else {
    PetscCall(ISGetBlockSize(is, &bs));
    PetscCall(ISBlockGetIndices(is, &indices));
    PetscCall(ISLocalToGlobalMappingCreate(comm, bs, n / bs, indices, PETSC_COPY_VALUES, mapping));
    PetscCall(ISBlockRestoreIndices(is, &indices));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
    ISLocalToGlobalMappingCreateSF - Creates a mapping between a local (0 to n)
    ordering and a global parallel ordering.

    Collective

    Input Parameters:
+   sf - star forest mapping contiguous local indices to (rank, offset)
-   start - first global index on this process, or `PETSC_DECIDE` to compute contiguous global numbering automatically

    Output Parameter:
.   mapping - new mapping data structure

    Level: advanced

    Note:
    If any processor calls this with `start` = `PETSC_DECIDE` then all processors must, otherwise the program will hang.

.seealso: [](sec_scatter), `PetscSF`, `ISLocalToGlobalMappingDestroy()`, `ISLocalToGlobalMappingCreate()`, `ISLocalToGlobalMappingCreateIS()`, `ISLocalToGlobalMappingSetFromOptions()`
@*/
PetscErrorCode ISLocalToGlobalMappingCreateSF(PetscSF sf, PetscInt start, ISLocalToGlobalMapping *mapping)
{
  PetscInt i, maxlocal, nroots, nleaves, *globals, *ltog;
  MPI_Comm comm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(sf, PETSCSF_CLASSID, 1);
  PetscValidPointer(mapping, 3);
  PetscCall(PetscObjectGetComm((PetscObject)sf, &comm));
  PetscCall(PetscSFGetGraph(sf, &nroots, &nleaves, NULL, NULL));
  if (start == PETSC_DECIDE) {
    start = 0;
    PetscCallMPI(MPI_Exscan(&nroots, &start, 1, MPIU_INT, MPI_SUM, comm));
  } else PetscCheck(start >= 0, comm, PETSC_ERR_ARG_OUTOFRANGE, "start must be nonnegative or PETSC_DECIDE");
  PetscCall(PetscSFGetLeafRange(sf, NULL, &maxlocal));
  ++maxlocal;
  PetscCall(PetscMalloc1(nroots, &globals));
  PetscCall(PetscMalloc1(maxlocal, &ltog));
  for (i = 0; i < nroots; i++) globals[i] = start + i;
  for (i = 0; i < maxlocal; i++) ltog[i] = -1;
  PetscCall(PetscSFBcastBegin(sf, MPIU_INT, globals, ltog, MPI_REPLACE));
  PetscCall(PetscSFBcastEnd(sf, MPIU_INT, globals, ltog, MPI_REPLACE));
  PetscCall(ISLocalToGlobalMappingCreate(comm, 1, maxlocal, ltog, PETSC_OWN_POINTER, mapping));
  PetscCall(PetscFree(globals));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
    ISLocalToGlobalMappingSetBlockSize - Sets the blocksize of the mapping

    Not Collective

    Input Parameters:
+   mapping - mapping data structure
-   bs - the blocksize

    Level: advanced

.seealso: [](sec_scatter), `ISLocalToGlobalMapping`, `ISLocalToGlobalMappingDestroy()`, `ISLocalToGlobalMappingCreateIS()`
@*/
PetscErrorCode ISLocalToGlobalMappingSetBlockSize(ISLocalToGlobalMapping mapping, PetscInt bs)
{
  PetscInt       *nid;
  const PetscInt *oid;
  PetscInt        i, cn, on, obs, nn;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mapping, IS_LTOGM_CLASSID, 1);
  PetscCheck(bs >= 1, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Invalid block size %" PetscInt_FMT, bs);
  if (bs == mapping->bs) PetscFunctionReturn(PETSC_SUCCESS);
  on  = mapping->n;
  obs = mapping->bs;
  oid = mapping->indices;
  nn  = (on * obs) / bs;
  PetscCheck((on * obs) % bs == 0, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Block size %" PetscInt_FMT " is inconsistent with block size %" PetscInt_FMT " and number of block indices %" PetscInt_FMT, bs, obs, on);

  PetscCall(PetscMalloc1(nn, &nid));
  PetscCall(ISLocalToGlobalMappingGetIndices(mapping, &oid));
  for (i = 0; i < nn; i++) {
    PetscInt j;
    for (j = 0, cn = 0; j < bs - 1; j++) {
      if (oid[i * bs + j] < 0) {
        cn++;
        continue;
      }
      PetscCheck(oid[i * bs + j] == oid[i * bs + j + 1] - 1, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Block sizes %" PetscInt_FMT " and %" PetscInt_FMT " are incompatible with the block indices: non consecutive indices %" PetscInt_FMT " %" PetscInt_FMT, bs, obs, oid[i * bs + j], oid[i * bs + j + 1]);
    }
    if (oid[i * bs + j] < 0) cn++;
    if (cn) {
      PetscCheck(cn == bs, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Block sizes %" PetscInt_FMT " and %" PetscInt_FMT " are incompatible with the block indices: invalid number of negative entries in block %" PetscInt_FMT, bs, obs, cn);
      nid[i] = -1;
    } else {
      nid[i] = oid[i * bs] / bs;
    }
  }
  PetscCall(ISLocalToGlobalMappingRestoreIndices(mapping, &oid));

  mapping->n  = nn;
  mapping->bs = bs;
  PetscCall(PetscFree(mapping->indices));
  mapping->indices     = nid;
  mapping->globalstart = 0;
  mapping->globalend   = 0;

  /* reset the cached information */
  PetscCall(PetscFree(mapping->info_procs));
  PetscCall(PetscFree(mapping->info_numprocs));
  if (mapping->info_indices) {
    PetscInt i;

    PetscCall(PetscFree((mapping->info_indices)[0]));
    for (i = 1; i < mapping->info_nproc; i++) PetscCall(PetscFree(mapping->info_indices[i]));
    PetscCall(PetscFree(mapping->info_indices));
  }
  mapping->info_cached = PETSC_FALSE;

  PetscTryTypeMethod(mapping, destroy);
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
    ISLocalToGlobalMappingGetBlockSize - Gets the blocksize of the mapping
    ordering and a global parallel ordering.

    Not Collective

    Input Parameter:
.   mapping - mapping data structure

    Output Parameter:
.   bs - the blocksize

    Level: advanced

.seealso: [](sec_scatter), `ISLocalToGlobalMapping`, `ISLocalToGlobalMappingDestroy()`, `ISLocalToGlobalMappingCreateIS()`
@*/
PetscErrorCode ISLocalToGlobalMappingGetBlockSize(ISLocalToGlobalMapping mapping, PetscInt *bs)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(mapping, IS_LTOGM_CLASSID, 1);
  *bs = mapping->bs;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
    ISLocalToGlobalMappingCreate - Creates a mapping between a local (0 to n)
    ordering and a global parallel ordering.

    Not Collective, but communicator may have more than one process

    Input Parameters:
+   comm - MPI communicator
.   bs - the block size
.   n - the number of local elements divided by the block size, or equivalently the number of block indices
.   indices - the global index for each local element, these do not need to be in increasing order (sorted), these values should not be scaled (i.e. multiplied) by the blocksize bs
-   mode - see PetscCopyMode

    Output Parameter:
.   mapping - new mapping data structure

    Level: advanced

    Notes:
    There is one integer value in indices per block and it represents the actual indices bs*idx + j, where j=0,..,bs-1

    For "small" problems when using `ISGlobalToLocalMappingApply()` and `ISGlobalToLocalMappingApplyBlock()`, the `ISLocalToGlobalMappingType`
    of `ISLOCALTOGLOBALMAPPINGBASIC` will be used; this uses more memory but is faster; this approach is not scalable for extremely large mappings.

    For large problems `ISLOCALTOGLOBALMAPPINGHASH` is used, this is scalable.
    Use `ISLocalToGlobalMappingSetType()` or call `ISLocalToGlobalMappingSetFromOptions()` with the option
    `-islocaltoglobalmapping_type` <`basic`,`hash`> to control which is used.

.seealso: [](sec_scatter), `ISLocalToGlobalMapping`, `ISLocalToGlobalMappingDestroy()`, `ISLocalToGlobalMappingCreateIS()`, `ISLocalToGlobalMappingSetFromOptions()`,
          `ISLOCALTOGLOBALMAPPINGBASIC`, `ISLOCALTOGLOBALMAPPINGHASH`
          `ISLocalToGlobalMappingSetType()`, `ISLocalToGlobalMappingType`
@*/
PetscErrorCode ISLocalToGlobalMappingCreate(MPI_Comm comm, PetscInt bs, PetscInt n, const PetscInt indices[], PetscCopyMode mode, ISLocalToGlobalMapping *mapping)
{
  PetscInt *in;

  PetscFunctionBegin;
  if (n) PetscValidIntPointer(indices, 4);
  PetscValidPointer(mapping, 6);

  *mapping = NULL;
  PetscCall(ISInitializePackage());

  PetscCall(PetscHeaderCreate(*mapping, IS_LTOGM_CLASSID, "ISLocalToGlobalMapping", "Local to global mapping", "IS", comm, ISLocalToGlobalMappingDestroy, ISLocalToGlobalMappingView));
  (*mapping)->n  = n;
  (*mapping)->bs = bs;
  if (mode == PETSC_COPY_VALUES) {
    PetscCall(PetscMalloc1(n, &in));
    PetscCall(PetscArraycpy(in, indices, n));
    (*mapping)->indices         = in;
    (*mapping)->dealloc_indices = PETSC_TRUE;
  } else if (mode == PETSC_OWN_POINTER) {
    (*mapping)->indices         = (PetscInt *)indices;
    (*mapping)->dealloc_indices = PETSC_TRUE;
  } else if (mode == PETSC_USE_POINTER) {
    (*mapping)->indices = (PetscInt *)indices;
  } else SETERRQ(comm, PETSC_ERR_ARG_OUTOFRANGE, "Invalid mode %d", mode);
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscFunctionList ISLocalToGlobalMappingList = NULL;

/*@
   ISLocalToGlobalMappingSetFromOptions - Set mapping options from the options database.

   Not Collective

   Input Parameter:
.  mapping - mapping data structure

   Options Database Key:
.  -islocaltoglobalmapping_type - <basic,hash> nonscalable and scalable versions

   Level: advanced

.seealso: [](sec_scatter), `ISLocalToGlobalMapping`, `ISLocalToGlobalMappingDestroy()`, `ISLocalToGlobalMappingCreateIS()`, `ISLocalToGlobalMappingSetFromOptions()`,
          `ISLOCALTOGLOBALMAPPINGBASIC`, `ISLOCALTOGLOBALMAPPINGHASH`
          `ISLocalToGlobalMappingSetType()`, `ISLocalToGlobalMappingType`
@*/
PetscErrorCode ISLocalToGlobalMappingSetFromOptions(ISLocalToGlobalMapping mapping)
{
  char                       type[256];
  ISLocalToGlobalMappingType defaulttype = "Not set";
  PetscBool                  flg;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mapping, IS_LTOGM_CLASSID, 1);
  PetscCall(ISLocalToGlobalMappingRegisterAll());
  PetscObjectOptionsBegin((PetscObject)mapping);
  PetscCall(PetscOptionsFList("-islocaltoglobalmapping_type", "ISLocalToGlobalMapping method", "ISLocalToGlobalMappingSetType", ISLocalToGlobalMappingList, (char *)(((PetscObject)mapping)->type_name) ? ((PetscObject)mapping)->type_name : defaulttype, type, 256, &flg));
  if (flg) PetscCall(ISLocalToGlobalMappingSetType(mapping, type));
  PetscOptionsEnd();
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
   ISLocalToGlobalMappingDestroy - Destroys a mapping between a local (0 to n)
   ordering and a global parallel ordering.

   Not Collective

   Input Parameter:
.  mapping - mapping data structure

   Level: advanced

.seealso: [](sec_scatter), `ISLocalToGlobalMapping`, `ISLocalToGlobalMappingCreate()`
@*/
PetscErrorCode ISLocalToGlobalMappingDestroy(ISLocalToGlobalMapping *mapping)
{
  PetscFunctionBegin;
  if (!*mapping) PetscFunctionReturn(PETSC_SUCCESS);
  PetscValidHeaderSpecific((*mapping), IS_LTOGM_CLASSID, 1);
  if (--((PetscObject)(*mapping))->refct > 0) {
    *mapping = NULL;
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  if ((*mapping)->dealloc_indices) PetscCall(PetscFree((*mapping)->indices));
  PetscCall(PetscFree((*mapping)->info_procs));
  PetscCall(PetscFree((*mapping)->info_numprocs));
  if ((*mapping)->info_indices) {
    PetscInt i;

    PetscCall(PetscFree(((*mapping)->info_indices)[0]));
    for (i = 1; i < (*mapping)->info_nproc; i++) PetscCall(PetscFree(((*mapping)->info_indices)[i]));
    PetscCall(PetscFree((*mapping)->info_indices));
  }
  if ((*mapping)->info_nodei) PetscCall(PetscFree(((*mapping)->info_nodei)[0]));
  PetscCall(PetscFree2((*mapping)->info_nodec, (*mapping)->info_nodei));
  if ((*mapping)->ops->destroy) PetscCall((*(*mapping)->ops->destroy)(*mapping));
  PetscCall(PetscHeaderDestroy(mapping));
  *mapping = NULL;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
    ISLocalToGlobalMappingApplyIS - Creates from an `IS` in the local numbering
    a new index set using the global numbering defined in an `ISLocalToGlobalMapping`
    context.

    Collective

    Input Parameters:
+   mapping - mapping between local and global numbering
-   is - index set in local numbering

    Output Parameter:
.   newis - index set in global numbering

    Level: advanced

    Note:
    The output `IS` will have the same communicator as the input `IS`.

.seealso: [](sec_scatter), `ISLocalToGlobalMappingApply()`, `ISLocalToGlobalMappingCreate()`,
          `ISLocalToGlobalMappingDestroy()`, `ISGlobalToLocalMappingApply()`
@*/
PetscErrorCode ISLocalToGlobalMappingApplyIS(ISLocalToGlobalMapping mapping, IS is, IS *newis)
{
  PetscInt        n, *idxout;
  const PetscInt *idxin;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mapping, IS_LTOGM_CLASSID, 1);
  PetscValidHeaderSpecific(is, IS_CLASSID, 2);
  PetscValidPointer(newis, 3);

  PetscCall(ISGetLocalSize(is, &n));
  PetscCall(ISGetIndices(is, &idxin));
  PetscCall(PetscMalloc1(n, &idxout));
  PetscCall(ISLocalToGlobalMappingApply(mapping, n, idxin, idxout));
  PetscCall(ISRestoreIndices(is, &idxin));
  PetscCall(ISCreateGeneral(PetscObjectComm((PetscObject)is), n, idxout, PETSC_OWN_POINTER, newis));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
   ISLocalToGlobalMappingApply - Takes a list of integers in a local numbering
   and converts them to the global numbering.

   Not Collective

   Input Parameters:
+  mapping - the local to global mapping context
.  N - number of integers
-  in - input indices in local numbering

   Output Parameter:
.  out - indices in global numbering

   Level: advanced

   Note:
   The `in` and `out` array parameters may be identical.

.seealso: [](sec_scatter), `ISLocalToGlobalMapping`, `ISLocalToGlobalMappingApplyBlock()`, `ISLocalToGlobalMappingCreate()`, `ISLocalToGlobalMappingDestroy()`,
          `ISLocalToGlobalMappingApplyIS()`, `AOCreateBasic()`, `AOApplicationToPetsc()`,
          `AOPetscToApplication()`, `ISGlobalToLocalMappingApply()`
@*/
PetscErrorCode ISLocalToGlobalMappingApply(ISLocalToGlobalMapping mapping, PetscInt N, const PetscInt in[], PetscInt out[])
{
  PetscInt i, bs, Nmax;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mapping, IS_LTOGM_CLASSID, 1);
  bs   = mapping->bs;
  Nmax = bs * mapping->n;
  if (bs == 1) {
    const PetscInt *idx = mapping->indices;
    for (i = 0; i < N; i++) {
      if (in[i] < 0) {
        out[i] = in[i];
        continue;
      }
      PetscCheck(in[i] < Nmax, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Local index %" PetscInt_FMT " too large %" PetscInt_FMT " (max) at %" PetscInt_FMT, in[i], Nmax - 1, i);
      out[i] = idx[in[i]];
    }
  } else {
    const PetscInt *idx = mapping->indices;
    for (i = 0; i < N; i++) {
      if (in[i] < 0) {
        out[i] = in[i];
        continue;
      }
      PetscCheck(in[i] < Nmax, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Local index %" PetscInt_FMT " too large %" PetscInt_FMT " (max) at %" PetscInt_FMT, in[i], Nmax - 1, i);
      out[i] = idx[in[i] / bs] * bs + (in[i] % bs);
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
   ISLocalToGlobalMappingApplyBlock - Takes a list of integers in a local block numbering and converts them to the global block numbering

   Not Collective

   Input Parameters:
+  mapping - the local to global mapping context
.  N - number of integers
-  in - input indices in local block numbering

   Output Parameter:
.  out - indices in global block numbering

   Example:
     If the index values are {0,1,6,7} set with a call to `ISLocalToGlobalMappingCreate`(`PETSC_COMM_SELF`,2,2,{0,3}) then the mapping applied to 0
     (the first block) would produce 0 and the mapping applied to 1 (the second block) would produce 3.

   Level: advanced

   Note:
   The `in` and `out` array parameters may be identical.

.seealso: [](sec_scatter), `ISLocalToGlobalMappingApply()`, `ISLocalToGlobalMappingCreate()`, `ISLocalToGlobalMappingDestroy()`,
          `ISLocalToGlobalMappingApplyIS()`, `AOCreateBasic()`, `AOApplicationToPetsc()`,
          `AOPetscToApplication()`, `ISGlobalToLocalMappingApply()`
@*/
PetscErrorCode ISLocalToGlobalMappingApplyBlock(ISLocalToGlobalMapping mapping, PetscInt N, const PetscInt in[], PetscInt out[])
{
  PetscInt        i, Nmax;
  const PetscInt *idx;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mapping, IS_LTOGM_CLASSID, 1);
  Nmax = mapping->n;
  idx  = mapping->indices;
  for (i = 0; i < N; i++) {
    if (in[i] < 0) {
      out[i] = in[i];
      continue;
    }
    PetscCheck(in[i] < Nmax, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Local block index %" PetscInt_FMT " too large %" PetscInt_FMT " (max) at %" PetscInt_FMT, in[i], Nmax - 1, i);
    out[i] = idx[in[i]];
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
    ISGlobalToLocalMappingApply - Provides the local numbering for a list of integers
    specified with a global numbering.

    Not Collective

    Input Parameters:
+   mapping - mapping between local and global numbering
.   type - `IS_GTOLM_MASK` - maps global indices with no local value to -1 in the output list (i.e., mask them)
           `IS_GTOLM_DROP` - drops the indices with no local value from the output list
.   n - number of global indices to map
-   idx - global indices to map

    Output Parameters:
+   nout - number of indices in output array (if type == `IS_GTOLM_MASK` then nout = n)
-   idxout - local index of each global index, one must pass in an array long enough
             to hold all the indices. You can call `ISGlobalToLocalMappingApply()` with
             idxout == NULL to determine the required length (returned in nout)
             and then allocate the required space and call `ISGlobalToLocalMappingApply()`
             a second time to set the values.

    Level: advanced

    Notes:
    Either `nout` or `idxout` may be `NULL`. `idx` and `idxout` may be identical.

    For "small" problems when using `ISGlobalToLocalMappingApply()` and `ISGlobalToLocalMappingApplyBlock()`, the `ISLocalToGlobalMappingType` of
    `ISLOCALTOGLOBALMAPPINGBASIC` will be used;
    this uses more memory but is faster; this approach is not scalable for extremely large mappings. For large problems `ISLOCALTOGLOBALMAPPINGHASH` is used, this is scalable.
    Use `ISLocalToGlobalMappingSetType()` or call `ISLocalToGlobalMappingSetFromOptions()` with the option -islocaltoglobalmapping_type <basic,hash> to control which is used.

    Developer Note:
    The manual page states that `idx` and `idxout` may be identical but the calling
    sequence declares `idx` as const so it cannot be the same as `idxout`.

.seealso: [](sec_scatter), `ISLocalToGlobalMapping`, `ISLocalToGlobalMappingApply()`, `ISGlobalToLocalMappingApplyBlock()`, `ISLocalToGlobalMappingCreate()`,
          `ISLocalToGlobalMappingDestroy()`
@*/
PetscErrorCode ISGlobalToLocalMappingApply(ISLocalToGlobalMapping mapping, ISGlobalToLocalMappingMode type, PetscInt n, const PetscInt idx[], PetscInt *nout, PetscInt idxout[])
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(mapping, IS_LTOGM_CLASSID, 1);
  if (!mapping->data) PetscCall(ISGlobalToLocalMappingSetUp(mapping));
  PetscUseTypeMethod(mapping, globaltolocalmappingapply, type, n, idx, nout, idxout);
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
    ISGlobalToLocalMappingApplyIS - Creates from an `IS` in the global numbering
    a new index set using the local numbering defined in an `ISLocalToGlobalMapping`
    context.

    Not Collective

    Input Parameters:
+   mapping - mapping between local and global numbering
.   type - `IS_GTOLM_MASK` - maps global indices with no local value to -1 in the output list (i.e., mask them)
           `IS_GTOLM_DROP` - drops the indices with no local value from the output list
-   is - index set in global numbering

    Output Parameter:
.   newis - index set in local numbering

    Level: advanced

    Note:
    The output `IS` will be sequential, as it encodes a purely local operation

.seealso: [](sec_scatter), `ISGlobalToLocalMapping`, `ISGlobalToLocalMappingApply()`, `ISLocalToGlobalMappingCreate()`,
          `ISLocalToGlobalMappingDestroy()`
@*/
PetscErrorCode ISGlobalToLocalMappingApplyIS(ISLocalToGlobalMapping mapping, ISGlobalToLocalMappingMode type, IS is, IS *newis)
{
  PetscInt        n, nout, *idxout;
  const PetscInt *idxin;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mapping, IS_LTOGM_CLASSID, 1);
  PetscValidHeaderSpecific(is, IS_CLASSID, 3);
  PetscValidPointer(newis, 4);

  PetscCall(ISGetLocalSize(is, &n));
  PetscCall(ISGetIndices(is, &idxin));
  if (type == IS_GTOLM_MASK) {
    PetscCall(PetscMalloc1(n, &idxout));
  } else {
    PetscCall(ISGlobalToLocalMappingApply(mapping, type, n, idxin, &nout, NULL));
    PetscCall(PetscMalloc1(nout, &idxout));
  }
  PetscCall(ISGlobalToLocalMappingApply(mapping, type, n, idxin, &nout, idxout));
  PetscCall(ISRestoreIndices(is, &idxin));
  PetscCall(ISCreateGeneral(PETSC_COMM_SELF, nout, idxout, PETSC_OWN_POINTER, newis));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
    ISGlobalToLocalMappingApplyBlock - Provides the local block numbering for a list of integers
    specified with a block global numbering.

    Not Collective

    Input Parameters:
+   mapping - mapping between local and global numbering
.   type - `IS_GTOLM_MASK` - maps global indices with no local value to -1 in the output list (i.e., mask them)
           `IS_GTOLM_DROP` - drops the indices with no local value from the output list
.   n - number of global indices to map
-   idx - global indices to map

    Output Parameters:
+   nout - number of indices in output array (if type == `IS_GTOLM_MASK` then nout = n)
-   idxout - local index of each global index, one must pass in an array long enough
             to hold all the indices. You can call `ISGlobalToLocalMappingApplyBlock()` with
             idxout == NULL to determine the required length (returned in nout)
             and then allocate the required space and call `ISGlobalToLocalMappingApplyBlock()`
             a second time to set the values.

    Level: advanced

    Notes:
    Either `nout` or `idxout` may be `NULL`. `idx` and `idxout` may be identical.

    For "small" problems when using `ISGlobalToLocalMappingApply()` and `ISGlobalToLocalMappingApplyBlock()`, the `ISLocalToGlobalMappingType` of
    `ISLOCALTOGLOBALMAPPINGBASIC` will be used;
    this uses more memory but is faster; this approach is not scalable for extremely large mappings. For large problems `ISLOCALTOGLOBALMAPPINGHASH` is used, this is scalable.
    Use `ISLocalToGlobalMappingSetType()` or call `ISLocalToGlobalMappingSetFromOptions()` with the option -islocaltoglobalmapping_type <basic,hash> to control which is used.

    Developer Note:
    The manual page states that `idx` and `idxout` may be identical but the calling
    sequence declares `idx` as const so it cannot be the same as `idxout`.

.seealso: [](sec_scatter), `ISLocalToGlobalMapping`, `ISLocalToGlobalMappingApply()`, `ISGlobalToLocalMappingApply()`, `ISLocalToGlobalMappingCreate()`,
          `ISLocalToGlobalMappingDestroy()`
@*/
PetscErrorCode ISGlobalToLocalMappingApplyBlock(ISLocalToGlobalMapping mapping, ISGlobalToLocalMappingMode type, PetscInt n, const PetscInt idx[], PetscInt *nout, PetscInt idxout[])
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(mapping, IS_LTOGM_CLASSID, 1);
  if (!mapping->data) PetscCall(ISGlobalToLocalMappingSetUp(mapping));
  PetscUseTypeMethod(mapping, globaltolocalmappingapplyblock, type, n, idx, nout, idxout);
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
    ISLocalToGlobalMappingGetBlockInfo - Gets the neighbor information for each processor and
     each index shared by more than one processor

    Collective

    Input Parameter:
.   mapping - the mapping from local to global indexing

    Output Parameters:
+   nproc - number of processors that are connected to this one
.   proc - neighboring processors
.   numproc - number of indices for each subdomain (processor)
-   indices - indices of nodes (in local numbering) shared with neighbors (sorted by global numbering)

    Level: advanced

    Fortran Usage:
.vb
        PetscInt indices[nproc][numprocmax],ierr)
        ISLocalToGlobalMpngGetInfoSize(ISLocalToGlobalMapping,PetscInt nproc,PetscInt numprocmax,ierr) followed by
        ISLocalToGlobalMappingGetInfo(ISLocalToGlobalMapping,PetscInt nproc, PetscInt procs[nproc],PetscInt numprocs[nproc],
.ve

   Fortran Note:
        There is no `ISLocalToGlobalMappingRestoreInfo()` in Fortran. You must make sure that `procs`[], `numprocs`[] and
        `indices`[][] are large enough arrays, either by allocating them dynamically or defining static ones large enough.

.seealso: [](sec_scatter), `ISLocalToGlobalMappingDestroy()`, `ISLocalToGlobalMappingCreateIS()`, `ISLocalToGlobalMappingCreate()`,
          `ISLocalToGlobalMappingRestoreInfo()`
@*/
PetscErrorCode ISLocalToGlobalMappingGetBlockInfo(ISLocalToGlobalMapping mapping, PetscInt *nproc, PetscInt *procs[], PetscInt *numprocs[], PetscInt **indices[])
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(mapping, IS_LTOGM_CLASSID, 1);
  if (mapping->info_cached) {
    *nproc    = mapping->info_nproc;
    *procs    = mapping->info_procs;
    *numprocs = mapping->info_numprocs;
    *indices  = mapping->info_indices;
  } else {
    PetscCall(ISLocalToGlobalMappingGetBlockInfo_Private(mapping, nproc, procs, numprocs, indices));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode ISLocalToGlobalMappingGetBlockInfo_Private(ISLocalToGlobalMapping mapping, PetscInt *nproc, PetscInt *procs[], PetscInt *numprocs[], PetscInt **indices[])
{
  PetscMPIInt  size, rank, tag1, tag2, tag3, *len, *source, imdex;
  PetscInt     i, n = mapping->n, Ng, ng, max = 0, *lindices = mapping->indices;
  PetscInt    *nprocs, *owner, nsends, *sends, j, *starts, nmax, nrecvs, *recvs, proc;
  PetscInt     cnt, scale, *ownedsenders, *nownedsenders, rstart;
  PetscInt     node, nownedm, nt, *sends2, nsends2, *starts2, *lens2, *dest, nrecvs2, *starts3, *recvs2, k, *bprocs, *tmp;
  PetscInt     first_procs, first_numprocs, *first_indices;
  MPI_Request *recv_waits, *send_waits;
  MPI_Status   recv_status, *send_status, *recv_statuses;
  MPI_Comm     comm;
  PetscBool    debug = PETSC_FALSE;

  PetscFunctionBegin;
  PetscCall(PetscObjectGetComm((PetscObject)mapping, &comm));
  PetscCallMPI(MPI_Comm_size(comm, &size));
  PetscCallMPI(MPI_Comm_rank(comm, &rank));
  if (size == 1) {
    *nproc = 0;
    *procs = NULL;
    PetscCall(PetscNew(numprocs));
    (*numprocs)[0] = 0;
    PetscCall(PetscNew(indices));
    (*indices)[0] = NULL;
    /* save info for reuse */
    mapping->info_nproc    = *nproc;
    mapping->info_procs    = *procs;
    mapping->info_numprocs = *numprocs;
    mapping->info_indices  = *indices;
    mapping->info_cached   = PETSC_TRUE;
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  PetscCall(PetscOptionsGetBool(((PetscObject)mapping)->options, NULL, "-islocaltoglobalmappinggetinfo_debug", &debug, NULL));

  /*
    Notes on ISLocalToGlobalMappingGetBlockInfo

    globally owned node - the nodes that have been assigned to this processor in global
           numbering, just for this routine.

    nontrivial globally owned node - node assigned to this processor that is on a subdomain
           boundary (i.e. is has more than one local owner)

    locally owned node - node that exists on this processors subdomain

    nontrivial locally owned node - node that is not in the interior (i.e. has more than one
           local subdomain
  */
  PetscCall(PetscObjectGetNewTag((PetscObject)mapping, &tag1));
  PetscCall(PetscObjectGetNewTag((PetscObject)mapping, &tag2));
  PetscCall(PetscObjectGetNewTag((PetscObject)mapping, &tag3));

  for (i = 0; i < n; i++) {
    if (lindices[i] > max) max = lindices[i];
  }
  PetscCall(MPIU_Allreduce(&max, &Ng, 1, MPIU_INT, MPI_MAX, comm));
  Ng++;
  PetscCallMPI(MPI_Comm_size(comm, &size));
  PetscCallMPI(MPI_Comm_rank(comm, &rank));
  scale = Ng / size + 1;
  ng    = scale;
  if (rank == size - 1) ng = Ng - scale * (size - 1);
  ng     = PetscMax(1, ng);
  rstart = scale * rank;

  /* determine ownership ranges of global indices */
  PetscCall(PetscMalloc1(2 * size, &nprocs));
  PetscCall(PetscArrayzero(nprocs, 2 * size));

  /* determine owners of each local node  */
  PetscCall(PetscMalloc1(n, &owner));
  for (i = 0; i < n; i++) {
    proc                 = lindices[i] / scale; /* processor that globally owns this index */
    nprocs[2 * proc + 1] = 1;                   /* processor globally owns at least one of ours */
    owner[i]             = proc;
    nprocs[2 * proc]++; /* count of how many that processor globally owns of ours */
  }
  nsends = 0;
  for (i = 0; i < size; i++) nsends += nprocs[2 * i + 1];
  PetscCall(PetscInfo(mapping, "Number of global owners for my local data %" PetscInt_FMT "\n", nsends));

  /* inform other processors of number of messages and max length*/
  PetscCall(PetscMaxSum(comm, nprocs, &nmax, &nrecvs));
  PetscCall(PetscInfo(mapping, "Number of local owners for my global data %" PetscInt_FMT "\n", nrecvs));

  /* post receives for owned rows */
  PetscCall(PetscMalloc1((2 * nrecvs + 1) * (nmax + 1), &recvs));
  PetscCall(PetscMalloc1(nrecvs + 1, &recv_waits));
  for (i = 0; i < nrecvs; i++) PetscCallMPI(MPI_Irecv(recvs + 2 * nmax * i, 2 * nmax, MPIU_INT, MPI_ANY_SOURCE, tag1, comm, recv_waits + i));

  /* pack messages containing lists of local nodes to owners */
  PetscCall(PetscMalloc1(2 * n + 1, &sends));
  PetscCall(PetscMalloc1(size + 1, &starts));
  starts[0] = 0;
  for (i = 1; i < size; i++) starts[i] = starts[i - 1] + 2 * nprocs[2 * i - 2];
  for (i = 0; i < n; i++) {
    sends[starts[owner[i]]++] = lindices[i];
    sends[starts[owner[i]]++] = i;
  }
  PetscCall(PetscFree(owner));
  starts[0] = 0;
  for (i = 1; i < size; i++) starts[i] = starts[i - 1] + 2 * nprocs[2 * i - 2];

  /* send the messages */
  PetscCall(PetscMalloc1(nsends + 1, &send_waits));
  PetscCall(PetscMalloc1(nsends + 1, &dest));
  cnt = 0;
  for (i = 0; i < size; i++) {
    if (nprocs[2 * i]) {
      PetscCallMPI(MPI_Isend(sends + starts[i], 2 * nprocs[2 * i], MPIU_INT, i, tag1, comm, send_waits + cnt));
      dest[cnt] = i;
      cnt++;
    }
  }
  PetscCall(PetscFree(starts));

  /* wait on receives */
  PetscCall(PetscMalloc1(nrecvs + 1, &source));
  PetscCall(PetscMalloc1(nrecvs + 1, &len));
  cnt = nrecvs;
  PetscCall(PetscCalloc1(ng + 1, &nownedsenders));
  while (cnt) {
    PetscCallMPI(MPI_Waitany(nrecvs, recv_waits, &imdex, &recv_status));
    /* unpack receives into our local space */
    PetscCallMPI(MPI_Get_count(&recv_status, MPIU_INT, &len[imdex]));
    source[imdex] = recv_status.MPI_SOURCE;
    len[imdex]    = len[imdex] / 2;
    /* count how many local owners for each of my global owned indices */
    for (i = 0; i < len[imdex]; i++) nownedsenders[recvs[2 * imdex * nmax + 2 * i] - rstart]++;
    cnt--;
  }
  PetscCall(PetscFree(recv_waits));

  /* count how many globally owned indices are on an edge multiplied by how many processors own them. */
  nownedm = 0;
  for (i = 0; i < ng; i++) {
    if (nownedsenders[i] > 1) nownedm += nownedsenders[i];
  }

  /* create single array to contain rank of all local owners of each globally owned index */
  PetscCall(PetscMalloc1(nownedm + 1, &ownedsenders));
  PetscCall(PetscMalloc1(ng + 1, &starts));
  starts[0] = 0;
  for (i = 1; i < ng; i++) {
    if (nownedsenders[i - 1] > 1) starts[i] = starts[i - 1] + nownedsenders[i - 1];
    else starts[i] = starts[i - 1];
  }

  /* for each nontrivial globally owned node list all arriving processors */
  for (i = 0; i < nrecvs; i++) {
    for (j = 0; j < len[i]; j++) {
      node = recvs[2 * i * nmax + 2 * j] - rstart;
      if (nownedsenders[node] > 1) ownedsenders[starts[node]++] = source[i];
    }
  }

  if (debug) { /* -----------------------------------  */
    starts[0] = 0;
    for (i = 1; i < ng; i++) {
      if (nownedsenders[i - 1] > 1) starts[i] = starts[i - 1] + nownedsenders[i - 1];
      else starts[i] = starts[i - 1];
    }
    for (i = 0; i < ng; i++) {
      if (nownedsenders[i] > 1) {
        PetscCall(PetscSynchronizedPrintf(comm, "[%d] global node %" PetscInt_FMT " local owner processors: ", rank, i + rstart));
        for (j = 0; j < nownedsenders[i]; j++) PetscCall(PetscSynchronizedPrintf(comm, "%" PetscInt_FMT " ", ownedsenders[starts[i] + j]));
        PetscCall(PetscSynchronizedPrintf(comm, "\n"));
      }
    }
    PetscCall(PetscSynchronizedFlush(comm, PETSC_STDOUT));
  } /* -----------------------------------  */

  /* wait on original sends */
  if (nsends) {
    PetscCall(PetscMalloc1(nsends, &send_status));
    PetscCallMPI(MPI_Waitall(nsends, send_waits, send_status));
    PetscCall(PetscFree(send_status));
  }
  PetscCall(PetscFree(send_waits));
  PetscCall(PetscFree(sends));
  PetscCall(PetscFree(nprocs));

  /* pack messages to send back to local owners */
  starts[0] = 0;
  for (i = 1; i < ng; i++) {
    if (nownedsenders[i - 1] > 1) starts[i] = starts[i - 1] + nownedsenders[i - 1];
    else starts[i] = starts[i - 1];
  }
  nsends2 = nrecvs;
  PetscCall(PetscMalloc1(nsends2 + 1, &nprocs)); /* length of each message */
  for (i = 0; i < nrecvs; i++) {
    nprocs[i] = 1;
    for (j = 0; j < len[i]; j++) {
      node = recvs[2 * i * nmax + 2 * j] - rstart;
      if (nownedsenders[node] > 1) nprocs[i] += 2 + nownedsenders[node];
    }
  }
  nt = 0;
  for (i = 0; i < nsends2; i++) nt += nprocs[i];

  PetscCall(PetscMalloc1(nt + 1, &sends2));
  PetscCall(PetscMalloc1(nsends2 + 1, &starts2));

  starts2[0] = 0;
  for (i = 1; i < nsends2; i++) starts2[i] = starts2[i - 1] + nprocs[i - 1];
  /*
     Each message is 1 + nprocs[i] long, and consists of
       (0) the number of nodes being sent back
       (1) the local node number,
       (2) the number of processors sharing it,
       (3) the processors sharing it
  */
  for (i = 0; i < nsends2; i++) {
    cnt                = 1;
    sends2[starts2[i]] = 0;
    for (j = 0; j < len[i]; j++) {
      node = recvs[2 * i * nmax + 2 * j] - rstart;
      if (nownedsenders[node] > 1) {
        sends2[starts2[i]]++;
        sends2[starts2[i] + cnt++] = recvs[2 * i * nmax + 2 * j + 1];
        sends2[starts2[i] + cnt++] = nownedsenders[node];
        PetscCall(PetscArraycpy(&sends2[starts2[i] + cnt], &ownedsenders[starts[node]], nownedsenders[node]));
        cnt += nownedsenders[node];
      }
    }
  }

  /* receive the message lengths */
  nrecvs2 = nsends;
  PetscCall(PetscMalloc1(nrecvs2 + 1, &lens2));
  PetscCall(PetscMalloc1(nrecvs2 + 1, &starts3));
  PetscCall(PetscMalloc1(nrecvs2 + 1, &recv_waits));
  for (i = 0; i < nrecvs2; i++) PetscCallMPI(MPI_Irecv(&lens2[i], 1, MPIU_INT, dest[i], tag2, comm, recv_waits + i));

  /* send the message lengths */
  for (i = 0; i < nsends2; i++) PetscCallMPI(MPI_Send(&nprocs[i], 1, MPIU_INT, source[i], tag2, comm));

  /* wait on receives of lens */
  if (nrecvs2) {
    PetscCall(PetscMalloc1(nrecvs2, &recv_statuses));
    PetscCallMPI(MPI_Waitall(nrecvs2, recv_waits, recv_statuses));
    PetscCall(PetscFree(recv_statuses));
  }
  PetscCall(PetscFree(recv_waits));

  starts3[0] = 0;
  nt         = 0;
  for (i = 0; i < nrecvs2 - 1; i++) {
    starts3[i + 1] = starts3[i] + lens2[i];
    nt += lens2[i];
  }
  if (nrecvs2) nt += lens2[nrecvs2 - 1];

  PetscCall(PetscMalloc1(nt + 1, &recvs2));
  PetscCall(PetscMalloc1(nrecvs2 + 1, &recv_waits));
  for (i = 0; i < nrecvs2; i++) PetscCallMPI(MPI_Irecv(recvs2 + starts3[i], lens2[i], MPIU_INT, dest[i], tag3, comm, recv_waits + i));

  /* send the messages */
  PetscCall(PetscMalloc1(nsends2 + 1, &send_waits));
  for (i = 0; i < nsends2; i++) PetscCallMPI(MPI_Isend(sends2 + starts2[i], nprocs[i], MPIU_INT, source[i], tag3, comm, send_waits + i));

  /* wait on receives */
  if (nrecvs2) {
    PetscCall(PetscMalloc1(nrecvs2, &recv_statuses));
    PetscCallMPI(MPI_Waitall(nrecvs2, recv_waits, recv_statuses));
    PetscCall(PetscFree(recv_statuses));
  }
  PetscCall(PetscFree(recv_waits));
  PetscCall(PetscFree(nprocs));

  if (debug) { /* -----------------------------------  */
    cnt = 0;
    for (i = 0; i < nrecvs2; i++) {
      nt = recvs2[cnt++];
      for (j = 0; j < nt; j++) {
        PetscCall(PetscSynchronizedPrintf(comm, "[%d] local node %" PetscInt_FMT " number of subdomains %" PetscInt_FMT ": ", rank, recvs2[cnt], recvs2[cnt + 1]));
        for (k = 0; k < recvs2[cnt + 1]; k++) PetscCall(PetscSynchronizedPrintf(comm, "%" PetscInt_FMT " ", recvs2[cnt + 2 + k]));
        cnt += 2 + recvs2[cnt + 1];
        PetscCall(PetscSynchronizedPrintf(comm, "\n"));
      }
    }
    PetscCall(PetscSynchronizedFlush(comm, PETSC_STDOUT));
  } /* -----------------------------------  */

  /* count number subdomains for each local node */
  PetscCall(PetscCalloc1(size, &nprocs));
  cnt = 0;
  for (i = 0; i < nrecvs2; i++) {
    nt = recvs2[cnt++];
    for (j = 0; j < nt; j++) {
      for (k = 0; k < recvs2[cnt + 1]; k++) nprocs[recvs2[cnt + 2 + k]]++;
      cnt += 2 + recvs2[cnt + 1];
    }
  }
  nt = 0;
  for (i = 0; i < size; i++) nt += (nprocs[i] > 0);
  *nproc = nt;
  PetscCall(PetscMalloc1(nt + 1, procs));
  PetscCall(PetscMalloc1(nt + 1, numprocs));
  PetscCall(PetscMalloc1(nt + 1, indices));
  for (i = 0; i < nt + 1; i++) (*indices)[i] = NULL;
  PetscCall(PetscMalloc1(size, &bprocs));
  cnt = 0;
  for (i = 0; i < size; i++) {
    if (nprocs[i] > 0) {
      bprocs[i]        = cnt;
      (*procs)[cnt]    = i;
      (*numprocs)[cnt] = nprocs[i];
      PetscCall(PetscMalloc1(nprocs[i], &(*indices)[cnt]));
      cnt++;
    }
  }

  /* make the list of subdomains for each nontrivial local node */
  PetscCall(PetscArrayzero(*numprocs, nt));
  cnt = 0;
  for (i = 0; i < nrecvs2; i++) {
    nt = recvs2[cnt++];
    for (j = 0; j < nt; j++) {
      for (k = 0; k < recvs2[cnt + 1]; k++) (*indices)[bprocs[recvs2[cnt + 2 + k]]][(*numprocs)[bprocs[recvs2[cnt + 2 + k]]]++] = recvs2[cnt];
      cnt += 2 + recvs2[cnt + 1];
    }
  }
  PetscCall(PetscFree(bprocs));
  PetscCall(PetscFree(recvs2));

  /* sort the node indexing by their global numbers */
  nt = *nproc;
  for (i = 0; i < nt; i++) {
    PetscCall(PetscMalloc1((*numprocs)[i], &tmp));
    for (j = 0; j < (*numprocs)[i]; j++) tmp[j] = lindices[(*indices)[i][j]];
    PetscCall(PetscSortIntWithArray((*numprocs)[i], tmp, (*indices)[i]));
    PetscCall(PetscFree(tmp));
  }

  if (debug) { /* -----------------------------------  */
    nt = *nproc;
    for (i = 0; i < nt; i++) {
      PetscCall(PetscSynchronizedPrintf(comm, "[%d] subdomain %" PetscInt_FMT " number of indices %" PetscInt_FMT ": ", rank, (*procs)[i], (*numprocs)[i]));
      for (j = 0; j < (*numprocs)[i]; j++) PetscCall(PetscSynchronizedPrintf(comm, "%" PetscInt_FMT " ", (*indices)[i][j]));
      PetscCall(PetscSynchronizedPrintf(comm, "\n"));
    }
    PetscCall(PetscSynchronizedFlush(comm, PETSC_STDOUT));
  } /* -----------------------------------  */

  /* wait on sends */
  if (nsends2) {
    PetscCall(PetscMalloc1(nsends2, &send_status));
    PetscCallMPI(MPI_Waitall(nsends2, send_waits, send_status));
    PetscCall(PetscFree(send_status));
  }

  PetscCall(PetscFree(starts3));
  PetscCall(PetscFree(dest));
  PetscCall(PetscFree(send_waits));

  PetscCall(PetscFree(nownedsenders));
  PetscCall(PetscFree(ownedsenders));
  PetscCall(PetscFree(starts));
  PetscCall(PetscFree(starts2));
  PetscCall(PetscFree(lens2));

  PetscCall(PetscFree(source));
  PetscCall(PetscFree(len));
  PetscCall(PetscFree(recvs));
  PetscCall(PetscFree(nprocs));
  PetscCall(PetscFree(sends2));

  /* put the information about myself as the first entry in the list */
  first_procs    = (*procs)[0];
  first_numprocs = (*numprocs)[0];
  first_indices  = (*indices)[0];
  for (i = 0; i < *nproc; i++) {
    if ((*procs)[i] == rank) {
      (*procs)[0]    = (*procs)[i];
      (*numprocs)[0] = (*numprocs)[i];
      (*indices)[0]  = (*indices)[i];
      (*procs)[i]    = first_procs;
      (*numprocs)[i] = first_numprocs;
      (*indices)[i]  = first_indices;
      break;
    }
  }

  /* save info for reuse */
  mapping->info_nproc    = *nproc;
  mapping->info_procs    = *procs;
  mapping->info_numprocs = *numprocs;
  mapping->info_indices  = *indices;
  mapping->info_cached   = PETSC_TRUE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
    ISLocalToGlobalMappingRestoreBlockInfo - Frees the memory allocated by `ISLocalToGlobalMappingGetBlockInfo()`

    Collective

    Input Parameter:
.   mapping - the mapping from local to global indexing

    Output Parameters:
+   nproc - number of processors that are connected to this one
.   proc - neighboring processors
.   numproc - number of indices for each processor
-   indices - indices of local nodes shared with neighbor (sorted by global numbering)

    Level: advanced

.seealso: [](sec_scatter), `ISLocalToGlobalMappingDestroy()`, `ISLocalToGlobalMappingCreateIS()`, `ISLocalToGlobalMappingCreate()`,
          `ISLocalToGlobalMappingGetInfo()`
@*/
PetscErrorCode ISLocalToGlobalMappingRestoreBlockInfo(ISLocalToGlobalMapping mapping, PetscInt *nproc, PetscInt *procs[], PetscInt *numprocs[], PetscInt **indices[])
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(mapping, IS_LTOGM_CLASSID, 1);
  if (mapping->info_free) {
    PetscCall(PetscFree(*numprocs));
    if (*indices) {
      PetscInt i;

      PetscCall(PetscFree((*indices)[0]));
      for (i = 1; i < *nproc; i++) PetscCall(PetscFree((*indices)[i]));
      PetscCall(PetscFree(*indices));
    }
  }
  *nproc    = 0;
  *procs    = NULL;
  *numprocs = NULL;
  *indices  = NULL;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
    ISLocalToGlobalMappingGetInfo - Gets the neighbor information for each processor and
     each index shared by more than one processor

    Collective

    Input Parameter:
.   mapping - the mapping from local to global indexing

    Output Parameters:
+   nproc - number of processors that are connected to this one
.   proc - neighboring processors
.   numproc - number of indices for each subdomain (processor)
-   indices - indices of nodes (in local numbering) shared with neighbors (sorted by global numbering)

    Level: advanced

    Note:
    The user needs to call `ISLocalToGlobalMappingRestoreInfo()` when the data is no longer needed.

    Fortran Usage:
.vb
        PetscInt indices[nproc][numprocmax],ierr)
        ISLocalToGlobalMpngGetInfoSize(ISLocalToGlobalMapping,PetscInt nproc,PetscInt numprocmax,ierr) followed by
        ISLocalToGlobalMappingGetInfo(ISLocalToGlobalMapping,PetscInt nproc, PetscInt procs[nproc],PetscInt numprocs[nproc],
.ve

    Fortran Note:
        There is no `ISLocalToGlobalMappingRestoreInfo()` in Fortran. You must make sure that `procs`[], `numprocs`[] and
        `indices`[][] are large enough arrays, either by allocating them dynamically or defining static ones large enough.

.seealso: [](sec_scatter), `ISLocalToGlobalMappingDestroy()`, `ISLocalToGlobalMappingCreateIS()`, `ISLocalToGlobalMappingCreate()`,
          `ISLocalToGlobalMappingRestoreInfo()`
@*/
PetscErrorCode ISLocalToGlobalMappingGetInfo(ISLocalToGlobalMapping mapping, PetscInt *nproc, PetscInt *procs[], PetscInt *numprocs[], PetscInt **indices[])
{
  PetscInt **bindices = NULL, *bnumprocs = NULL, bs, i, j, k;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mapping, IS_LTOGM_CLASSID, 1);
  bs = mapping->bs;
  PetscCall(ISLocalToGlobalMappingGetBlockInfo(mapping, nproc, procs, &bnumprocs, &bindices));
  if (bs > 1) { /* we need to expand the cached info */
    PetscCall(PetscCalloc1(*nproc, &*indices));
    PetscCall(PetscCalloc1(*nproc, &*numprocs));
    for (i = 0; i < *nproc; i++) {
      PetscCall(PetscMalloc1(bs * bnumprocs[i], &(*indices)[i]));
      for (j = 0; j < bnumprocs[i]; j++) {
        for (k = 0; k < bs; k++) (*indices)[i][j * bs + k] = bs * bindices[i][j] + k;
      }
      (*numprocs)[i] = bnumprocs[i] * bs;
    }
    mapping->info_free = PETSC_TRUE;
  } else {
    *numprocs = bnumprocs;
    *indices  = bindices;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
    ISLocalToGlobalMappingRestoreInfo - Frees the memory allocated by `ISLocalToGlobalMappingGetInfo()`

    Collective

    Input Parameter:
.   mapping - the mapping from local to global indexing

    Output Parameters:
+   nproc - number of processors that are connected to this one
.   proc - neighboring processors
.   numproc - number of indices for each processor
-   indices - indices of local nodes shared with neighbor (sorted by global numbering)

    Level: advanced

.seealso: [](sec_scatter), `ISLocalToGlobalMappingDestroy()`, `ISLocalToGlobalMappingCreateIS()`, `ISLocalToGlobalMappingCreate()`,
          `ISLocalToGlobalMappingGetInfo()`
@*/
PetscErrorCode ISLocalToGlobalMappingRestoreInfo(ISLocalToGlobalMapping mapping, PetscInt *nproc, PetscInt *procs[], PetscInt *numprocs[], PetscInt **indices[])
{
  PetscFunctionBegin;
  PetscCall(ISLocalToGlobalMappingRestoreBlockInfo(mapping, nproc, procs, numprocs, indices));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
    ISLocalToGlobalMappingGetNodeInfo - Gets the neighbor information for each MPI rank

    Collective

    Input Parameter:
.   mapping - the mapping from local to global indexing

    Output Parameters:
+   nnodes - number of local nodes (same `ISLocalToGlobalMappingGetSize()`)
.   count - number of neighboring processors per node
-   indices - indices of processes sharing the node (sorted)

    Level: advanced

    Note:
    The user needs to call `ISLocalToGlobalMappingRestoreInfo()` when the data is no longer needed.

.seealso: [](sec_scatter), `ISLocalToGlobalMappingDestroy()`, `ISLocalToGlobalMappingCreateIS()`, `ISLocalToGlobalMappingCreate()`,
          `ISLocalToGlobalMappingGetInfo()`, `ISLocalToGlobalMappingRestoreNodeInfo()`
@*/
PetscErrorCode ISLocalToGlobalMappingGetNodeInfo(ISLocalToGlobalMapping mapping, PetscInt *nnodes, PetscInt *count[], PetscInt **indices[])
{
  PetscInt n;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mapping, IS_LTOGM_CLASSID, 1);
  PetscCall(ISLocalToGlobalMappingGetSize(mapping, &n));
  if (!mapping->info_nodec) {
    PetscInt i, m, n_neigh, *neigh, *n_shared, **shared;

    PetscCall(PetscMalloc2(n + 1, &mapping->info_nodec, n, &mapping->info_nodei));
    PetscCall(ISLocalToGlobalMappingGetInfo(mapping, &n_neigh, &neigh, &n_shared, &shared));
    for (i = 0; i < n; i++) mapping->info_nodec[i] = 1;
    m                      = n;
    mapping->info_nodec[n] = 0;
    for (i = 1; i < n_neigh; i++) {
      PetscInt j;

      m += n_shared[i];
      for (j = 0; j < n_shared[i]; j++) mapping->info_nodec[shared[i][j]] += 1;
    }
    if (n) PetscCall(PetscMalloc1(m, &mapping->info_nodei[0]));
    for (i = 1; i < n; i++) mapping->info_nodei[i] = mapping->info_nodei[i - 1] + mapping->info_nodec[i - 1];
    PetscCall(PetscArrayzero(mapping->info_nodec, n));
    for (i = 0; i < n; i++) {
      mapping->info_nodec[i]    = 1;
      mapping->info_nodei[i][0] = neigh[0];
    }
    for (i = 1; i < n_neigh; i++) {
      PetscInt j;

      for (j = 0; j < n_shared[i]; j++) {
        PetscInt k = shared[i][j];

        mapping->info_nodei[k][mapping->info_nodec[k]] = neigh[i];
        mapping->info_nodec[k] += 1;
      }
    }
    for (i = 0; i < n; i++) PetscCall(PetscSortRemoveDupsInt(&mapping->info_nodec[i], mapping->info_nodei[i]));
    PetscCall(ISLocalToGlobalMappingRestoreInfo(mapping, &n_neigh, &neigh, &n_shared, &shared));
  }
  if (nnodes) *nnodes = n;
  if (count) *count = mapping->info_nodec;
  if (indices) *indices = mapping->info_nodei;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
    ISLocalToGlobalMappingRestoreNodeInfo - Frees the memory allocated by `ISLocalToGlobalMappingGetNodeInfo()`

    Collective

    Input Parameter:
.   mapping - the mapping from local to global indexing

    Output Parameters:
+   nnodes - number of local nodes
.   count - number of neighboring processors per node
-   indices - indices of processes sharing the node (sorted)

    Level: advanced

.seealso: [](sec_scatter), `ISLocalToGlobalMappingDestroy()`, `ISLocalToGlobalMappingCreateIS()`, `ISLocalToGlobalMappingCreate()`,
          `ISLocalToGlobalMappingGetInfo()`
@*/
PetscErrorCode ISLocalToGlobalMappingRestoreNodeInfo(ISLocalToGlobalMapping mapping, PetscInt *nnodes, PetscInt *count[], PetscInt **indices[])
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(mapping, IS_LTOGM_CLASSID, 1);
  if (nnodes) *nnodes = 0;
  if (count) *count = NULL;
  if (indices) *indices = NULL;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*MC
    ISLocalToGlobalMappingGetIndicesF90 - Get global indices for every local point that is mapped

    Synopsis:
    ISLocalToGlobalMappingGetIndicesF90(ISLocalToGlobalMapping ltog, PetscInt, pointer :: array(:)}, integer ierr)

    Not Collective

    Input Parameter:
.   A - the matrix

    Output Parameter:
.   array - array of indices, the length of this array may be obtained with `ISLocalToGlobalMappingGetSize()`

    Level: advanced

    Note:
    Use  `ISLocalToGlobalMappingGetIndicesF90()` when you no longer need access to the data

.seealso: [](sec_fortranarrays), [](sec_scatter), `ISLocalToGlobalMapping`, `ISLocalToGlobalMappingGetIndices()`, `ISLocalToGlobalMappingRestoreIndices()`,
          `ISLocalToGlobalMappingRestoreIndicesF90()`
M*/

/*MC
    ISLocalToGlobalMappingRestoreIndicesF90 - restores the global indices for every local point that is mapped obtained with `ISLocalToGlobalMappingGetIndicesF90()`

    Synopsis:
    ISLocalToGlobalMappingRestoreIndicesF90(ISLocalToGlobalMapping ltog, PetscInt, pointer :: array(:)}, integer ierr)

    Not Collective

    Input Parameters:
+   A - the matrix
-   array - array of indices, the length of this array may be obtained with `ISLocalToGlobalMappingGetSize()`

    Level: advanced

.seealso: [](sec_fortranarrays), [](sec_scatter), `ISLocalToGlobalMapping`, `ISLocalToGlobalMappingGetIndices()`, `ISLocalToGlobalMappingRestoreIndices()`,
          `ISLocalToGlobalMappingGetIndicesF90()`
M*/

/*@C
   ISLocalToGlobalMappingGetIndices - Get global indices for every local point that is mapped

   Not Collective

   Input Parameter:
. ltog - local to global mapping

   Output Parameter:
. array - array of indices, the length of this array may be obtained with `ISLocalToGlobalMappingGetSize()`

   Level: advanced

   Note:
    `ISLocalToGlobalMappingGetSize()` returns the length the this array

.seealso: [](sec_scatter), `ISLocalToGlobalMappingCreate()`, `ISLocalToGlobalMappingApply()`, `ISLocalToGlobalMappingRestoreIndices()`,
          `ISLocalToGlobalMappingGetBlockIndices()`, `ISLocalToGlobalMappingRestoreBlockIndices()`
@*/
PetscErrorCode ISLocalToGlobalMappingGetIndices(ISLocalToGlobalMapping ltog, const PetscInt **array)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ltog, IS_LTOGM_CLASSID, 1);
  PetscValidPointer(array, 2);
  if (ltog->bs == 1) {
    *array = ltog->indices;
  } else {
    PetscInt       *jj, k, i, j, n = ltog->n, bs = ltog->bs;
    const PetscInt *ii;

    PetscCall(PetscMalloc1(bs * n, &jj));
    *array = jj;
    k      = 0;
    ii     = ltog->indices;
    for (i = 0; i < n; i++)
      for (j = 0; j < bs; j++) jj[k++] = bs * ii[i] + j;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
   ISLocalToGlobalMappingRestoreIndices - Restore indices obtained with `ISLocalToGlobalMappingGetIndices()`

   Not Collective

   Input Parameters:
+ ltog - local to global mapping
- array - array of indices

   Level: advanced

.seealso: [](sec_scatter), `ISLocalToGlobalMappingCreate()`, `ISLocalToGlobalMappingApply()`, `ISLocalToGlobalMappingGetIndices()`
@*/
PetscErrorCode ISLocalToGlobalMappingRestoreIndices(ISLocalToGlobalMapping ltog, const PetscInt **array)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ltog, IS_LTOGM_CLASSID, 1);
  PetscValidPointer(array, 2);
  PetscCheck(ltog->bs != 1 || *array == ltog->indices, PETSC_COMM_SELF, PETSC_ERR_ARG_BADPTR, "Trying to return mismatched pointer");

  if (ltog->bs > 1) PetscCall(PetscFree(*(void **)array));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
   ISLocalToGlobalMappingGetBlockIndices - Get global indices for every local block

   Not Collective

   Input Parameter:
. ltog - local to global mapping

   Output Parameter:
. array - array of indices

   Level: advanced

.seealso: [](sec_scatter), `ISLocalToGlobalMappingCreate()`, `ISLocalToGlobalMappingApply()`, `ISLocalToGlobalMappingRestoreBlockIndices()`
@*/
PetscErrorCode ISLocalToGlobalMappingGetBlockIndices(ISLocalToGlobalMapping ltog, const PetscInt **array)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ltog, IS_LTOGM_CLASSID, 1);
  PetscValidPointer(array, 2);
  *array = ltog->indices;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
   ISLocalToGlobalMappingRestoreBlockIndices - Restore indices obtained with `ISLocalToGlobalMappingGetBlockIndices()`

   Not Collective

   Input Parameters:
+ ltog - local to global mapping
- array - array of indices

   Level: advanced

.seealso: [](sec_scatter), `ISLocalToGlobalMappingCreate()`, `ISLocalToGlobalMappingApply()`, `ISLocalToGlobalMappingGetIndices()`
@*/
PetscErrorCode ISLocalToGlobalMappingRestoreBlockIndices(ISLocalToGlobalMapping ltog, const PetscInt **array)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ltog, IS_LTOGM_CLASSID, 1);
  PetscValidPointer(array, 2);
  PetscCheck(*array == ltog->indices, PETSC_COMM_SELF, PETSC_ERR_ARG_BADPTR, "Trying to return mismatched pointer");
  *array = NULL;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
   ISLocalToGlobalMappingConcatenate - Create a new mapping that concatenates a list of mappings

   Not Collective

   Input Parameters:
+ comm - communicator for the new mapping, must contain the communicator of every mapping to concatenate
. n - number of mappings to concatenate
- ltogs - local to global mappings

   Output Parameter:
. ltogcat - new mapping

   Level: advanced

   Note:
   This currently always returns a mapping with block size of 1

   Developer Note:
   If all the input mapping have the same block size we could easily handle that as a special case

.seealso: [](sec_scatter), `ISLocalToGlobalMapping`, `ISLocalToGlobalMappingCreate()`
@*/
PetscErrorCode ISLocalToGlobalMappingConcatenate(MPI_Comm comm, PetscInt n, const ISLocalToGlobalMapping ltogs[], ISLocalToGlobalMapping *ltogcat)
{
  PetscInt i, cnt, m, *idx;

  PetscFunctionBegin;
  PetscCheck(n >= 0, comm, PETSC_ERR_ARG_OUTOFRANGE, "Must have a non-negative number of mappings, given %" PetscInt_FMT, n);
  if (n > 0) PetscValidPointer(ltogs, 3);
  for (i = 0; i < n; i++) PetscValidHeaderSpecific(ltogs[i], IS_LTOGM_CLASSID, 3);
  PetscValidPointer(ltogcat, 4);
  for (cnt = 0, i = 0; i < n; i++) {
    PetscCall(ISLocalToGlobalMappingGetSize(ltogs[i], &m));
    cnt += m;
  }
  PetscCall(PetscMalloc1(cnt, &idx));
  for (cnt = 0, i = 0; i < n; i++) {
    const PetscInt *subidx;
    PetscCall(ISLocalToGlobalMappingGetSize(ltogs[i], &m));
    PetscCall(ISLocalToGlobalMappingGetIndices(ltogs[i], &subidx));
    PetscCall(PetscArraycpy(&idx[cnt], subidx, m));
    PetscCall(ISLocalToGlobalMappingRestoreIndices(ltogs[i], &subidx));
    cnt += m;
  }
  PetscCall(ISLocalToGlobalMappingCreate(comm, 1, cnt, idx, PETSC_OWN_POINTER, ltogcat));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*MC
      ISLOCALTOGLOBALMAPPINGBASIC - basic implementation of the `ISLocalToGlobalMapping` object. When `ISGlobalToLocalMappingApply()` is
                                    used this is good for only small and moderate size problems.

   Options Database Key:
.   -islocaltoglobalmapping_type basic - select this method

   Level: beginner

   Developer Note:
   This stores all the mapping information on each MPI rank.

.seealso: [](sec_scatter), `ISLocalToGlobalMappingCreate()`, `ISLocalToGlobalMappingSetType()`, `ISLOCALTOGLOBALMAPPINGHASH`
M*/
PETSC_EXTERN PetscErrorCode ISLocalToGlobalMappingCreate_Basic(ISLocalToGlobalMapping ltog)
{
  PetscFunctionBegin;
  ltog->ops->globaltolocalmappingapply      = ISGlobalToLocalMappingApply_Basic;
  ltog->ops->globaltolocalmappingsetup      = ISGlobalToLocalMappingSetUp_Basic;
  ltog->ops->globaltolocalmappingapplyblock = ISGlobalToLocalMappingApplyBlock_Basic;
  ltog->ops->destroy                        = ISLocalToGlobalMappingDestroy_Basic;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*MC
      ISLOCALTOGLOBALMAPPINGHASH - hash implementation of the `ISLocalToGlobalMapping` object. When `ISGlobalToLocalMappingApply()` is
                                    used this is good for large memory problems.

   Options Database Key:
.   -islocaltoglobalmapping_type hash - select this method

   Level: beginner

   Note:
    This is selected automatically for large problems if the user does not set the type.

.seealso: [](sec_scatter), `ISLocalToGlobalMappingCreate()`, `ISLocalToGlobalMappingSetType()`, `ISLOCALTOGLOBALMAPPINGBASIC`
M*/
PETSC_EXTERN PetscErrorCode ISLocalToGlobalMappingCreate_Hash(ISLocalToGlobalMapping ltog)
{
  PetscFunctionBegin;
  ltog->ops->globaltolocalmappingapply      = ISGlobalToLocalMappingApply_Hash;
  ltog->ops->globaltolocalmappingsetup      = ISGlobalToLocalMappingSetUp_Hash;
  ltog->ops->globaltolocalmappingapplyblock = ISGlobalToLocalMappingApplyBlock_Hash;
  ltog->ops->destroy                        = ISLocalToGlobalMappingDestroy_Hash;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
    ISLocalToGlobalMappingRegister -  Registers a method for applying a global to local mapping with an `ISLocalToGlobalMapping`

   Not Collective

   Input Parameters:
+  sname - name of a new method
-  function - routine to create method context

   Sample usage:
.vb
   ISLocalToGlobalMappingRegister("my_mapper",MyCreate);
.ve

   Then, your mapping can be chosen with the procedural interface via
$     ISLocalToGlobalMappingSetType(ltog,"my_mapper")
   or at runtime via the option
$     -islocaltoglobalmapping_type my_mapper

   Level: advanced

   Note:
   `ISLocalToGlobalMappingRegister()` may be called multiple times to add several user-defined mappings.

.seealso: [](sec_scatter), `ISLocalToGlobalMappingRegisterAll()`, `ISLocalToGlobalMappingRegisterDestroy()`, `ISLOCALTOGLOBALMAPPINGBASIC`,
          `ISLOCALTOGLOBALMAPPINGHASH`, `ISLocalToGlobalMapping`, `ISLocalToGlobalMappingApply()`
@*/
PetscErrorCode ISLocalToGlobalMappingRegister(const char sname[], PetscErrorCode (*function)(ISLocalToGlobalMapping))
{
  PetscFunctionBegin;
  PetscCall(ISInitializePackage());
  PetscCall(PetscFunctionListAdd(&ISLocalToGlobalMappingList, sname, function));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
   ISLocalToGlobalMappingSetType - Sets the implementation type `ISLocalToGlobalMapping` will use

   Logically Collective

   Input Parameters:
+  ltog - the `ISLocalToGlobalMapping` object
-  type - a known method

   Options Database Key:
.  -islocaltoglobalmapping_type  <method> - Sets the method; use -help for a list of available methods (for instance, basic or hash)

  Level: intermediate

   Notes:
   See `ISLocalToGlobalMappingType` for available methods

  Normally, it is best to use the `ISLocalToGlobalMappingSetFromOptions()` command and
  then set the `ISLocalToGlobalMappingType` from the options database rather than by using
  this routine.

  Developer Note:
  `ISLocalToGlobalMappingRegister()` is used to add new types to `ISLocalToGlobalMappingList` from which they
  are accessed by `ISLocalToGlobalMappingSetType()`.

.seealso: [](sec_scatter), `ISLocalToGlobalMappingType`, `ISLocalToGlobalMappingType`, `ISLocalToGlobalMappingRegister()`, `ISLocalToGlobalMappingCreate()`, `ISLocalToGlobalMappingGetType()`
@*/
PetscErrorCode ISLocalToGlobalMappingSetType(ISLocalToGlobalMapping ltog, ISLocalToGlobalMappingType type)
{
  PetscBool match;
  PetscErrorCode (*r)(ISLocalToGlobalMapping) = NULL;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ltog, IS_LTOGM_CLASSID, 1);
  if (type) PetscValidCharPointer(type, 2);

  PetscCall(PetscObjectTypeCompare((PetscObject)ltog, type, &match));
  if (match) PetscFunctionReturn(PETSC_SUCCESS);

  /* L2G maps defer type setup at globaltolocal calls, allow passing NULL here */
  if (type) {
    PetscCall(PetscFunctionListFind(ISLocalToGlobalMappingList, type, &r));
    PetscCheck(r, PetscObjectComm((PetscObject)ltog), PETSC_ERR_ARG_UNKNOWN_TYPE, "Unable to find requested ISLocalToGlobalMapping type %s", type);
  }
  /* Destroy the previous private LTOG context */
  PetscTryTypeMethod(ltog, destroy);
  ltog->ops->destroy = NULL;

  PetscCall(PetscObjectChangeTypeName((PetscObject)ltog, type));
  if (r) PetscCall((*r)(ltog));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
   ISLocalToGlobalMappingGetType - Get the type of the `ISLocalToGlobalMapping`

   Not Collective

   Input Parameter:
.  ltog - the `ISLocalToGlobalMapping` object

   Output Parameter:
.  type - the type

   Level: intermediate

.seealso: [](sec_scatter), `ISLocalToGlobalMappingType`, `ISLocalToGlobalMappingType`, `ISLocalToGlobalMappingRegister()`, `ISLocalToGlobalMappingCreate()`, `ISLocalToGlobalMappingSetType()`
@*/
PetscErrorCode ISLocalToGlobalMappingGetType(ISLocalToGlobalMapping ltog, ISLocalToGlobalMappingType *type)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ltog, IS_LTOGM_CLASSID, 1);
  PetscValidPointer(type, 2);
  *type = ((PetscObject)ltog)->type_name;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscBool ISLocalToGlobalMappingRegisterAllCalled = PETSC_FALSE;

/*@C
  ISLocalToGlobalMappingRegisterAll - Registers all of the local to global mapping components in the `IS` package.

  Not Collective

  Level: advanced

.seealso: [](sec_scatter), `ISRegister()`, `ISLocalToGlobalRegister()`
@*/
PetscErrorCode ISLocalToGlobalMappingRegisterAll(void)
{
  PetscFunctionBegin;
  if (ISLocalToGlobalMappingRegisterAllCalled) PetscFunctionReturn(PETSC_SUCCESS);
  ISLocalToGlobalMappingRegisterAllCalled = PETSC_TRUE;
  PetscCall(ISLocalToGlobalMappingRegister(ISLOCALTOGLOBALMAPPINGBASIC, ISLocalToGlobalMappingCreate_Basic));
  PetscCall(ISLocalToGlobalMappingRegister(ISLOCALTOGLOBALMAPPINGHASH, ISLocalToGlobalMappingCreate_Hash));
  PetscFunctionReturn(PETSC_SUCCESS);
}
