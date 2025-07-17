#include <petsc/private/isimpl.h> /*I "petscis.h"  I*/
#include <petsc/private/hashmapi.h>
#include <petscsf.h>
#include <petscviewer.h>
#include <petscbt.h>

PetscClassId          IS_LTOGM_CLASSID;
static PetscErrorCode ISLocalToGlobalMappingSetUpBlockInfo_Private(ISLocalToGlobalMapping);

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
PetscErrorCode ISGetPointRange(IS pointIS, PetscInt *pStart, PetscInt *pEnd, const PetscInt *points[])
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
PetscErrorCode ISRestorePointRange(IS pointIS, PetscInt *pStart, PetscInt *pEnd, const PetscInt *points[])
{
  PetscInt  step = 1;
  PetscBool isStride;

  PetscFunctionBeginHot;
  PetscCall(PetscObjectTypeCompare((PetscObject)pointIS, ISSTRIDE, &isStride));
  if (isStride) PetscCall(ISStrideGetInfo(pointIS, pStart, &step));
  if (!isStride || step != 1) PetscCall(ISGetIndices(pointIS, points));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
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
PetscErrorCode ISGetPointSubrange(IS subpointIS, PetscInt pStart, PetscInt pEnd, const PetscInt points[])
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
  start = PETSC_INT_MAX;

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

static PetscErrorCode ISLocalToGlobalMappingResetBlockInfo_Private(ISLocalToGlobalMapping mapping)
{
  PetscFunctionBegin;
  PetscCall(PetscFree(mapping->info_procs));
  PetscCall(PetscFree(mapping->info_numprocs));
  if (mapping->info_indices) {
    for (PetscInt i = 0; i < mapping->info_nproc; i++) PetscCall(PetscFree(mapping->info_indices[i]));
    PetscCall(PetscFree(mapping->info_indices));
  }
  if (mapping->info_nodei) PetscCall(PetscFree(mapping->info_nodei[0]));
  PetscCall(PetscFree2(mapping->info_nodec, mapping->info_nodei));
  PetscCall(PetscSFDestroy(&mapping->multileaves_sf));
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
. ltog - local to global mapping

  Output Parameter:
. nltog - the duplicated local to global mapping

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
. mapping - local to global mapping

  Output Parameter:
. n - the number of entries in the local mapping, `ISLocalToGlobalMappingGetIndices()` returns an array of this length

  Level: advanced

.seealso: [](sec_scatter), `ISLocalToGlobalMapping`, `ISLocalToGlobalMappingDestroy()`, `ISLocalToGlobalMappingCreate()`
@*/
PetscErrorCode ISLocalToGlobalMappingGetSize(ISLocalToGlobalMapping mapping, PetscInt *n)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(mapping, IS_LTOGM_CLASSID, 1);
  PetscAssertPointer(n, 2);
  *n = mapping->bs * mapping->n;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  ISLocalToGlobalMappingViewFromOptions - View an `ISLocalToGlobalMapping` based on values in the options database

  Collective

  Input Parameters:
+ A    - the local to global mapping object
. obj  - Optional object that provides the options prefix used for the options database query
- name - command line option

  Level: intermediate

  Note:
  See `PetscObjectViewFromOptions()` for the available `PetscViewer` and `PetscViewerFormat`

.seealso: [](sec_scatter), `PetscViewer`, `ISLocalToGlobalMapping`, `ISLocalToGlobalMappingView`, `PetscObjectViewFromOptions()`, `ISLocalToGlobalMappingCreate()`
@*/
PetscErrorCode ISLocalToGlobalMappingViewFromOptions(ISLocalToGlobalMapping A, PetscObject obj, const char name[])
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(A, IS_LTOGM_CLASSID, 1);
  PetscCall(PetscObjectViewFromOptions((PetscObject)A, obj, name));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  ISLocalToGlobalMappingView - View a local to global mapping

  Collective on viewer

  Input Parameters:
+ mapping - local to global mapping
- viewer  - viewer

  Level: intermediate

.seealso: [](sec_scatter), `PetscViewer`, `ISLocalToGlobalMapping`, `ISLocalToGlobalMappingDestroy()`, `ISLocalToGlobalMappingCreate()`
@*/
PetscErrorCode ISLocalToGlobalMappingView(ISLocalToGlobalMapping mapping, PetscViewer viewer)
{
  PetscBool         isascii, isbinary;
  PetscViewerFormat format;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mapping, IS_LTOGM_CLASSID, 1);
  if (!viewer) PetscCall(PetscViewerASCIIGetStdout(PetscObjectComm((PetscObject)mapping), &viewer));
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 2);

  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERASCII, &isascii));
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERBINARY, &isbinary));
  PetscCall(PetscViewerGetFormat(viewer, &format));
  if (isascii) {
    if (format == PETSC_VIEWER_ASCII_MATLAB) {
      const PetscInt *idxs;
      IS              is;
      const char     *name = ((PetscObject)mapping)->name;
      char            iname[PETSC_MAX_PATH_LEN];

      PetscCall(PetscSNPrintf(iname, sizeof(iname), "%sl2g", name ? name : ""));
      PetscCall(ISLocalToGlobalMappingGetIndices(mapping, &idxs));
      PetscCall(ISCreateGeneral(PetscObjectComm((PetscObject)viewer), mapping->n * mapping->bs, idxs, PETSC_USE_POINTER, &is));
      PetscCall(PetscObjectSetName((PetscObject)is, iname));
      PetscCall(ISView(is, viewer));
      PetscCall(ISLocalToGlobalMappingRestoreIndices(mapping, &idxs));
      PetscCall(ISDestroy(&is));
    } else {
      PetscMPIInt rank;

      PetscCallMPI(MPI_Comm_rank(PetscObjectComm((PetscObject)mapping), &rank));
      PetscCall(PetscObjectPrintClassNamePrefixType((PetscObject)mapping, viewer));
      PetscCall(PetscViewerASCIIPushSynchronized(viewer));
      for (PetscInt i = 0; i < mapping->n; i++) {
        PetscInt bs = mapping->bs, g = mapping->indices[i];
        if (bs == 1) PetscCall(PetscViewerASCIISynchronizedPrintf(viewer, "[%d] %" PetscInt_FMT " %" PetscInt_FMT "\n", rank, i, g));
        else PetscCall(PetscViewerASCIISynchronizedPrintf(viewer, "[%d] %" PetscInt_FMT ":%" PetscInt_FMT " %" PetscInt_FMT ":%" PetscInt_FMT "\n", rank, i * bs, (i + 1) * bs, g * bs, (g + 1) * bs));
      }
      PetscCall(PetscViewerFlush(viewer));
      PetscCall(PetscViewerASCIIPopSynchronized(viewer));
    }
  } else if (isbinary) {
    PetscBool skipHeader;

    PetscCall(PetscViewerSetUp(viewer));
    PetscCall(PetscViewerBinaryGetSkipHeader(viewer, &skipHeader));
    if (!skipHeader) {
      PetscMPIInt size;
      PetscInt    tr[3];

      PetscCallMPI(MPI_Comm_size(PetscObjectComm((PetscObject)viewer), &size));
      tr[0] = IS_LTOGM_FILE_CLASSID;
      tr[1] = mapping->bs;
      tr[2] = size;
      PetscCall(PetscViewerBinaryWrite(viewer, tr, 3, PETSC_INT));
      PetscCall(PetscViewerBinaryWriteAll(viewer, &mapping->n, 1, PETSC_DETERMINE, PETSC_DETERMINE, PETSC_INT));
    }
    /* write block indices */
    PetscCall(PetscViewerBinaryWriteAll(viewer, mapping->indices, mapping->n, PETSC_DETERMINE, PETSC_DETERMINE, PETSC_INT));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  ISLocalToGlobalMappingLoad - Loads a local-to-global mapping that has been stored in binary format.

  Collective on viewer

  Input Parameters:
+ mapping - the newly loaded map, this needs to have been created with `ISLocalToGlobalMappingCreate()` or some related function before a call to `ISLocalToGlobalMappingLoad()`
- viewer  - binary file viewer, obtained from `PetscViewerBinaryOpen()`

  Level: intermediate

.seealso: [](sec_scatter), `PetscViewer`, `ISLocalToGlobalMapping`, `ISLocalToGlobalMappingView()`, `ISLocalToGlobalMappingCreate()`
@*/
PetscErrorCode ISLocalToGlobalMappingLoad(ISLocalToGlobalMapping mapping, PetscViewer viewer)
{
  PetscBool isbinary, skipHeader;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mapping, IS_LTOGM_CLASSID, 1);
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 2);
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERBINARY, &isbinary));
  PetscCheck(isbinary, PetscObjectComm((PetscObject)viewer), PETSC_ERR_SUP, "Invalid viewer of type %s", ((PetscObject)viewer)->type_name);

  /* reset previous data */
  PetscCall(ISLocalToGlobalMappingResetBlockInfo_Private(mapping));

  PetscCall(PetscViewerSetUp(viewer));
  PetscCall(PetscViewerBinaryGetSkipHeader(viewer, &skipHeader));

  /* When skipping header, it assumes bs and n have been already set */
  if (!skipHeader) {
    MPI_Comm comm = PetscObjectComm((PetscObject)viewer);
    PetscInt tr[3], nold = mapping->n, *sizes, nmaps = PETSC_DECIDE, st = 0;

    PetscCall(PetscViewerBinaryRead(viewer, tr, 3, NULL, PETSC_INT));
    PetscCheck(tr[0] == IS_LTOGM_FILE_CLASSID, PETSC_COMM_SELF, PETSC_ERR_FILE_UNEXPECTED, "Not a local-to-global map next in file");

    mapping->bs = tr[1];
    PetscCall(PetscMalloc1(tr[2], &sizes));
    PetscCall(PetscViewerBinaryRead(viewer, sizes, tr[2], NULL, PETSC_INT));

    /* consume the input, read multiple maps per process if needed */
    PetscCall(PetscSplitOwnership(comm, &nmaps, &tr[2]));
    PetscCallMPI(MPI_Exscan(&nmaps, &st, 1, MPIU_INT, MPI_SUM, comm));
    mapping->n = 0;
    for (PetscInt i = st; i < st + nmaps; i++) mapping->n += sizes[i];
    PetscCall(PetscFree(sizes));

    if (nold != mapping->n) {
      if (mapping->dealloc_indices) PetscCall(PetscFree(mapping->indices));
      mapping->indices = NULL;
    }
  }

  /* read indices */
  if (mapping->n && !mapping->indices) {
    PetscCall(PetscMalloc1(mapping->n, &mapping->indices));
    mapping->dealloc_indices = PETSC_TRUE;
  }
  PetscCall(PetscViewerBinaryReadAll(viewer, mapping->indices, mapping->n, PETSC_DETERMINE, PETSC_DETERMINE, PETSC_INT));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  ISLocalToGlobalMappingCreateIS - Creates a mapping between a local (0 to n)
  ordering and a global parallel ordering.

  Not Collective

  Input Parameter:
. is - index set containing the global numbers for each local number

  Output Parameter:
. mapping - new mapping data structure

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
  PetscAssertPointer(mapping, 2);

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

/*@
  ISLocalToGlobalMappingCreateSF - Creates a mapping between a local (0 to n) ordering and a global parallel ordering induced by a star forest.

  Collective

  Input Parameters:
+ sf    - star forest mapping contiguous local indices to (rank, offset)
- start - first global index on this process, or `PETSC_DECIDE` to compute contiguous global numbering automatically

  Output Parameter:
. mapping - new mapping data structure

  Level: advanced

  Note:
  If a process calls this function with `start` = `PETSC_DECIDE` then all processes must, otherwise the program will hang.

.seealso: [](sec_scatter), `PetscSF`, `ISLocalToGlobalMappingDestroy()`, `ISLocalToGlobalMappingCreate()`, `ISLocalToGlobalMappingCreateIS()`, `ISLocalToGlobalMappingSetFromOptions()`
@*/
PetscErrorCode ISLocalToGlobalMappingCreateSF(PetscSF sf, PetscInt start, ISLocalToGlobalMapping *mapping)
{
  PetscInt i, maxlocal, nroots, nleaves, *globals, *ltog;
  MPI_Comm comm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(sf, PETSCSF_CLASSID, 1);
  PetscAssertPointer(mapping, 3);
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
+ mapping - mapping data structure
- bs      - the blocksize

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
  PetscCall(ISLocalToGlobalMappingResetBlockInfo_Private(mapping));
  PetscTryTypeMethod(mapping, destroy);
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  ISLocalToGlobalMappingGetBlockSize - Gets the blocksize of the mapping
  ordering and a global parallel ordering.

  Not Collective

  Input Parameter:
. mapping - mapping data structure

  Output Parameter:
. bs - the blocksize

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
+ comm    - MPI communicator
. bs      - the block size
. n       - the number of local elements divided by the block size, or equivalently the number of block indices
. indices - the global index for each local element, these do not need to be in increasing order (sorted), these values should not be scaled (i.e. multiplied) by the blocksize bs
- mode    - see PetscCopyMode

  Output Parameter:
. mapping - new mapping data structure

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
  if (n) PetscAssertPointer(indices, 4);
  PetscAssertPointer(mapping, 6);

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
. mapping - mapping data structure

  Options Database Key:
. -islocaltoglobalmapping_type - <basic,hash> nonscalable and scalable versions

  Level: advanced

.seealso: [](sec_scatter), `ISLocalToGlobalMapping`, `ISLocalToGlobalMappingDestroy()`,
`ISLocalToGlobalMappingCreateIS()`, `ISLOCALTOGLOBALMAPPINGBASIC`,
`ISLOCALTOGLOBALMAPPINGHASH`, `ISLocalToGlobalMappingSetType()`, `ISLocalToGlobalMappingType`
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
  PetscCall(PetscOptionsFList("-islocaltoglobalmapping_type", "ISLocalToGlobalMapping method", "ISLocalToGlobalMappingSetType", ISLocalToGlobalMappingList, ((PetscObject)mapping)->type_name ? ((PetscObject)mapping)->type_name : defaulttype, type, 256, &flg));
  if (flg) PetscCall(ISLocalToGlobalMappingSetType(mapping, type));
  PetscOptionsEnd();
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  ISLocalToGlobalMappingDestroy - Destroys a mapping between a local (0 to n)
  ordering and a global parallel ordering.

  Not Collective

  Input Parameter:
. mapping - mapping data structure

  Level: advanced

.seealso: [](sec_scatter), `ISLocalToGlobalMapping`, `ISLocalToGlobalMappingCreate()`
@*/
PetscErrorCode ISLocalToGlobalMappingDestroy(ISLocalToGlobalMapping *mapping)
{
  PetscFunctionBegin;
  if (!*mapping) PetscFunctionReturn(PETSC_SUCCESS);
  PetscValidHeaderSpecific(*mapping, IS_LTOGM_CLASSID, 1);
  if (--((PetscObject)*mapping)->refct > 0) {
    *mapping = NULL;
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  if ((*mapping)->dealloc_indices) PetscCall(PetscFree((*mapping)->indices));
  PetscCall(ISLocalToGlobalMappingResetBlockInfo_Private(*mapping));
  PetscTryTypeMethod(*mapping, destroy);
  PetscCall(PetscHeaderDestroy(mapping));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  ISLocalToGlobalMappingApplyIS - Creates from an `IS` in the local numbering
  a new index set using the global numbering defined in an `ISLocalToGlobalMapping`
  context.

  Collective

  Input Parameters:
+ mapping - mapping between local and global numbering
- is      - index set in local numbering

  Output Parameter:
. newis - index set in global numbering

  Level: advanced

  Note:
  The output `IS` will have the same communicator as the input `IS` as well as the same block size.

.seealso: [](sec_scatter), `ISLocalToGlobalMappingApply()`, `ISLocalToGlobalMappingCreate()`,
          `ISLocalToGlobalMappingDestroy()`, `ISGlobalToLocalMappingApply()`
@*/
PetscErrorCode ISLocalToGlobalMappingApplyIS(ISLocalToGlobalMapping mapping, IS is, IS *newis)
{
  PetscInt        n, *idxout, bs;
  const PetscInt *idxin;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mapping, IS_LTOGM_CLASSID, 1);
  PetscValidHeaderSpecific(is, IS_CLASSID, 2);
  PetscAssertPointer(newis, 3);

  PetscCall(ISGetLocalSize(is, &n));
  PetscCall(ISGetBlockSize(is, &bs));
  PetscCall(ISGetIndices(is, &idxin));
  PetscCall(PetscMalloc1(n, &idxout));
  PetscCall(ISLocalToGlobalMappingApply(mapping, n, idxin, idxout));
  PetscCall(ISRestoreIndices(is, &idxin));
  PetscCall(ISCreateGeneral(PetscObjectComm((PetscObject)is), n, idxout, PETSC_OWN_POINTER, newis));
  PetscCall(ISSetBlockSize(*newis, bs));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  ISLocalToGlobalMappingApply - Takes a list of integers in a local numbering
  and converts them to the global numbering.

  Not Collective

  Input Parameters:
+ mapping - the local to global mapping context
. N       - number of integers
- in      - input indices in local numbering

  Output Parameter:
. out - indices in global numbering

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
+ mapping - the local to global mapping context
. N       - number of integers
- in      - input indices in local block numbering

  Output Parameter:
. out - indices in global block numbering

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
+ mapping - mapping between local and global numbering
. type    - `IS_GTOLM_MASK` - maps global indices with no local value to -1 in the output list (i.e., mask them)
           `IS_GTOLM_DROP` - drops the indices with no local value from the output list
. n       - number of global indices to map
- idx     - global indices to map

  Output Parameters:
+ nout   - number of indices in output array (if type == `IS_GTOLM_MASK` then nout = n)
- idxout - local index of each global index, one must pass in an array long enough
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

  Developer Notes:
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
+ mapping - mapping between local and global numbering
. type    - `IS_GTOLM_MASK` - maps global indices with no local value to -1 in the output list (i.e., mask them)
           `IS_GTOLM_DROP` - drops the indices with no local value from the output list
- is      - index set in global numbering

  Output Parameter:
. newis - index set in local numbering

  Level: advanced

  Notes:
  The output `IS` will be sequential, as it encodes a purely local operation

  If `type` is `IS_GTOLM_MASK`, `newis` will have the same block size as `is`

.seealso: [](sec_scatter), `ISGlobalToLocalMapping`, `ISGlobalToLocalMappingApply()`, `ISLocalToGlobalMappingCreate()`,
          `ISLocalToGlobalMappingDestroy()`
@*/
PetscErrorCode ISGlobalToLocalMappingApplyIS(ISLocalToGlobalMapping mapping, ISGlobalToLocalMappingMode type, IS is, IS *newis)
{
  PetscInt        n, nout, *idxout, bs;
  const PetscInt *idxin;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mapping, IS_LTOGM_CLASSID, 1);
  PetscValidHeaderSpecific(is, IS_CLASSID, 3);
  PetscAssertPointer(newis, 4);

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
  if (type == IS_GTOLM_MASK) {
    PetscCall(ISGetBlockSize(is, &bs));
    PetscCall(ISSetBlockSize(*newis, bs));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  ISGlobalToLocalMappingApplyBlock - Provides the local block numbering for a list of integers
  specified with a block global numbering.

  Not Collective

  Input Parameters:
+ mapping - mapping between local and global numbering
. type    - `IS_GTOLM_MASK` - maps global indices with no local value to -1 in the output list (i.e., mask them)
           `IS_GTOLM_DROP` - drops the indices with no local value from the output list
. n       - number of global indices to map
- idx     - global indices to map

  Output Parameters:
+ nout   - number of indices in output array (if type == `IS_GTOLM_MASK` then nout = n)
- idxout - local index of each global index, one must pass in an array long enough
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

  Developer Notes:
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
  ISLocalToGlobalMappingGetBlockInfo - Gets the neighbor information

  Collective the first time it is called

  Input Parameter:
. mapping - the mapping from local to global indexing

  Output Parameters:
+ nproc    - number of processes that are connected to the calling process
. procs    - neighboring processes
. numprocs - number of block indices for each process
- indices  - block indices (in local numbering) shared with neighbors (sorted by global numbering)

  Level: advanced

.seealso: [](sec_scatter), `ISLocalToGlobalMappingDestroy()`, `ISLocalToGlobalMappingCreateIS()`, `ISLocalToGlobalMappingCreate()`,
          `ISLocalToGlobalMappingRestoreBlockInfo()`, `ISLocalToGlobalMappingGetBlockMultiLeavesSF()`
@*/
PetscErrorCode ISLocalToGlobalMappingGetBlockInfo(ISLocalToGlobalMapping mapping, PetscInt *nproc, PetscInt *procs[], PetscInt *numprocs[], PetscInt **indices[])
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(mapping, IS_LTOGM_CLASSID, 1);
  PetscCall(ISLocalToGlobalMappingSetUpBlockInfo_Private(mapping));
  if (nproc) *nproc = mapping->info_nproc;
  if (procs) *procs = mapping->info_procs;
  if (numprocs) *numprocs = mapping->info_numprocs;
  if (indices) *indices = mapping->info_indices;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  ISLocalToGlobalMappingGetBlockNodeInfo - Gets the neighbor information for each local block index

  Collective the first time it is called

  Input Parameter:
. mapping - the mapping from local to global indexing

  Output Parameters:
+ n       - number of local block nodes
. n_procs - an array storing the number of processes for each local block node (including self)
- procs   - the processes' rank for each local block node (sorted, self is first)

  Level: advanced

  Notes:
  The user needs to call `ISLocalToGlobalMappingRestoreBlockNodeInfo()` when the data is no longer needed.
  The information returned by this function complements that of `ISLocalToGlobalMappingGetBlockInfo()`.
  The latter only provides local information, and the neighboring information
  cannot be inferred in the general case, unless the mapping is locally one-to-one on each process.

.seealso: `ISLocalToGlobalMappingDestroy()`, `ISLocalToGlobalMappingCreateIS()`, `ISLocalToGlobalMappingCreate()`,
          `ISLocalToGlobalMappingGetBlockInfo()`, `ISLocalToGlobalMappingRestoreBlockNodeInfo()`, `ISLocalToGlobalMappingGetNodeInfo()`
@*/
PetscErrorCode ISLocalToGlobalMappingGetBlockNodeInfo(ISLocalToGlobalMapping mapping, PetscInt *n, PetscInt *n_procs[], PetscInt **procs[])
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(mapping, IS_LTOGM_CLASSID, 1);
  PetscCall(ISLocalToGlobalMappingSetUpBlockInfo_Private(mapping));
  if (n) *n = mapping->n;
  if (n_procs) *n_procs = mapping->info_nodec;
  if (procs) *procs = mapping->info_nodei;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  ISLocalToGlobalMappingRestoreBlockNodeInfo - Frees the memory allocated by `ISLocalToGlobalMappingGetBlockNodeInfo()`

  Not Collective

  Input Parameters:
+ mapping - the mapping from local to global indexing
. n       - number of local block nodes
. n_procs - an array storing the number of processes for each local block nodes (including self)
- procs   - the processes' rank for each local block node (sorted, self is first)

  Level: advanced

.seealso: `ISLocalToGlobalMappingDestroy()`, `ISLocalToGlobalMappingCreateIS()`, `ISLocalToGlobalMappingCreate()`,
          `ISLocalToGlobalMappingGetBlockNodeInfo()`
@*/
PetscErrorCode ISLocalToGlobalMappingRestoreBlockNodeInfo(ISLocalToGlobalMapping mapping, PetscInt *n, PetscInt *n_procs[], PetscInt **procs[])
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(mapping, IS_LTOGM_CLASSID, 1);
  if (n) *n = 0;
  if (n_procs) *n_procs = NULL;
  if (procs) *procs = NULL;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  ISLocalToGlobalMappingGetBlockMultiLeavesSF - Get the star-forest to communicate multi-leaf block data

  Collective the first time it is called

  Input Parameter:
. mapping - the mapping from local to global indexing

  Output Parameter:
. mlsf - the `PetscSF`

  Level: advanced

  Notes:
  The returned star forest is suitable to exchange local information with other processes sharing the same global block index.
  For example, suppose a mapping with two processes has been created with
.vb
    rank 0 global block indices: [0, 1, 2]
    rank 1 global block indices: [2, 3, 4]
.ve
  and we want to share the local information
.vb
    rank 0 data: [-1, -2, -3]
    rank 1 data: [1, 2, 3]
.ve
  then, the broadcasting action of `mlsf` will allow to collect
.vb
    rank 0 mlleafdata: [-1, -2, -3, 3]
    rank 1 mlleafdata: [-3, 3, 1, 2]
.ve
  Use ``ISLocalToGlobalMappingGetBlockNodeInfo()`` to index into the multi-leaf data.

.seealso: [](sec_scatter), `ISLocalToGlobalMappingGetBlockNodeInfo()`, `PetscSF`
@*/
PetscErrorCode ISLocalToGlobalMappingGetBlockMultiLeavesSF(ISLocalToGlobalMapping mapping, PetscSF *mlsf)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(mapping, IS_LTOGM_CLASSID, 1);
  PetscAssertPointer(mlsf, 2);
  PetscCall(ISLocalToGlobalMappingSetUpBlockInfo_Private(mapping));
  *mlsf = mapping->multileaves_sf;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode ISLocalToGlobalMappingSetUpBlockInfo_Private(ISLocalToGlobalMapping mapping)
{
  PetscSF            sf, sf2, imsf, msf;
  MPI_Comm           comm;
  const PetscSFNode *sfnode;
  PetscSFNode       *newsfnode;
  PetscLayout        layout;
  PetscHMapI         neighs;
  PetscHashIter      iter;
  PetscBool          missing;
  const PetscInt    *gidxs, *rootdegree;
  PetscInt          *mask, *mrootdata, *leafdata, *newleafdata, *leafrd, *tmpg;
  PetscInt           nroots, nleaves, newnleaves, bs, i, j, m, mnroots, p;
  PetscMPIInt        rank, size;

  PetscFunctionBegin;
  if (mapping->multileaves_sf) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(PetscObjectGetComm((PetscObject)mapping, &comm));
  PetscCallMPI(MPI_Comm_size(comm, &size));
  PetscCallMPI(MPI_Comm_rank(comm, &rank));

  /* Get mapping indices */
  PetscCall(ISLocalToGlobalMappingGetBlockSize(mapping, &bs));
  PetscCall(ISLocalToGlobalMappingGetBlockIndices(mapping, &gidxs));
  PetscCall(ISLocalToGlobalMappingGetSize(mapping, &nleaves));
  nleaves /= bs;

  /* Create layout for global indices */
  for (i = 0, m = 0; i < nleaves; i++) m = PetscMax(m, gidxs[i]);
  PetscCallMPI(MPIU_Allreduce(MPI_IN_PLACE, &m, 1, MPIU_INT, MPI_MAX, comm));
  PetscCall(PetscLayoutCreate(comm, &layout));
  PetscCall(PetscLayoutSetSize(layout, m + 1));
  PetscCall(PetscLayoutSetUp(layout));

  /* Create SF to share global indices */
  PetscCall(PetscSFCreate(comm, &sf));
  PetscCall(PetscSFSetGraphLayout(sf, layout, nleaves, NULL, PETSC_OWN_POINTER, gidxs));
  PetscCall(PetscSFSetUp(sf));
  PetscCall(PetscLayoutDestroy(&layout));

  /* communicate root degree to leaves */
  PetscCall(PetscSFGetGraph(sf, &nroots, NULL, NULL, &sfnode));
  PetscCall(PetscSFComputeDegreeBegin(sf, &rootdegree));
  PetscCall(PetscSFComputeDegreeEnd(sf, &rootdegree));
  for (i = 0, mnroots = 0; i < nroots; i++) mnroots += rootdegree[i];
  PetscCall(PetscMalloc3(2 * PetscMax(mnroots, nroots), &mrootdata, 2 * nleaves, &leafdata, nleaves, &leafrd));
  for (i = 0, m = 0; i < nroots; i++) {
    mrootdata[2 * i + 0] = rootdegree[i];
    mrootdata[2 * i + 1] = m;
    m += rootdegree[i];
  }
  PetscCall(PetscSFBcastBegin(sf, MPIU_2INT, mrootdata, leafdata, MPI_REPLACE));
  PetscCall(PetscSFBcastEnd(sf, MPIU_2INT, mrootdata, leafdata, MPI_REPLACE));

  /* allocate enough space to store ranks */
  for (i = 0, newnleaves = 0; i < nleaves; i++) {
    newnleaves += leafdata[2 * i];
    leafrd[i] = leafdata[2 * i];
  }

  /* create new SF nodes to collect multi-root data at leaves */
  PetscCall(PetscMalloc1(newnleaves, &newsfnode));
  for (i = 0, m = 0; i < nleaves; i++) {
    for (j = 0; j < leafrd[i]; j++) {
      newsfnode[m].rank  = sfnode[i].rank;
      newsfnode[m].index = leafdata[2 * i + 1] + j;
      m++;
    }
  }

  /* gather ranks at multi roots */
  for (i = 0; i < mnroots; i++) mrootdata[i] = -1;
  for (i = 0; i < nleaves; i++) leafdata[i] = rank;

  PetscCall(PetscSFGatherBegin(sf, MPIU_INT, leafdata, mrootdata));
  PetscCall(PetscSFGatherEnd(sf, MPIU_INT, leafdata, mrootdata));

  /* from multi-roots to multi-leaves */
  PetscCall(PetscSFCreate(comm, &sf2));
  PetscCall(PetscSFSetGraph(sf2, mnroots, newnleaves, NULL, PETSC_OWN_POINTER, newsfnode, PETSC_OWN_POINTER));
  PetscCall(PetscSFSetUp(sf2));

  /* broadcast multi-root data to multi-leaves */
  PetscCall(PetscMalloc1(newnleaves, &newleafdata));
  PetscCall(PetscSFBcastBegin(sf2, MPIU_INT, mrootdata, newleafdata, MPI_REPLACE));
  PetscCall(PetscSFBcastEnd(sf2, MPIU_INT, mrootdata, newleafdata, MPI_REPLACE));

  /* sort sharing ranks */
  for (i = 0, m = 0; i < nleaves; i++) {
    PetscCall(PetscSortInt(leafrd[i], newleafdata + m));
    m += leafrd[i];
  }

  /* Number of neighbors and their ranks */
  PetscCall(PetscHMapICreate(&neighs));
  for (i = 0; i < newnleaves; i++) PetscCall(PetscHMapIPut(neighs, newleafdata[i], &iter, &missing));
  PetscCall(PetscHMapIGetSize(neighs, &mapping->info_nproc));
  PetscCall(PetscMalloc1(mapping->info_nproc + 1, &mapping->info_procs));
  PetscCall(PetscHMapIGetKeys(neighs, (i = 0, &i), mapping->info_procs));
  for (i = 0; i < mapping->info_nproc; i++) { /* put info for self first */
    if (mapping->info_procs[i] == rank) {
      PetscInt newr = mapping->info_procs[0];

      mapping->info_procs[0] = rank;
      mapping->info_procs[i] = newr;
      break;
    }
  }
  if (mapping->info_nproc) PetscCall(PetscSortInt(mapping->info_nproc - 1, mapping->info_procs + 1));
  PetscCall(PetscHMapIDestroy(&neighs));

  /* collect info data */
  PetscCall(PetscMalloc1(mapping->info_nproc, &mapping->info_numprocs));
  PetscCall(PetscMalloc1(mapping->info_nproc, &mapping->info_indices));
  for (i = 0; i < mapping->info_nproc; i++) mapping->info_indices[i] = NULL;

  PetscCall(PetscMalloc1(nleaves, &mask));
  PetscCall(PetscMalloc1(nleaves, &tmpg));
  for (p = 0; p < mapping->info_nproc; p++) {
    PetscInt *tmp, trank = mapping->info_procs[p];

    PetscCall(PetscMemzero(mask, nleaves * sizeof(*mask)));
    for (i = 0, m = 0; i < nleaves; i++) {
      for (j = 0; j < leafrd[i]; j++) {
        if (newleafdata[m] == trank) mask[i]++;
        if (!p && newleafdata[m] != rank) mask[i]++;
        m++;
      }
    }
    for (i = 0, m = 0; i < nleaves; i++)
      if (mask[i] > (!p ? 1 : 0)) m++;

    PetscCall(PetscMalloc1(m, &tmp));
    for (i = 0, m = 0; i < nleaves; i++)
      if (mask[i] > (!p ? 1 : 0)) {
        tmp[m]  = i;
        tmpg[m] = gidxs[i];
        m++;
      }
    PetscCall(PetscSortIntWithArray(m, tmpg, tmp));
    mapping->info_indices[p]  = tmp;
    mapping->info_numprocs[p] = m;
  }

  /* Node info */
  PetscCall(PetscMalloc2(nleaves, &mapping->info_nodec, nleaves + 1, &mapping->info_nodei));
  PetscCall(PetscArraycpy(mapping->info_nodec, leafrd, nleaves));
  PetscCall(PetscMalloc1(newnleaves, &mapping->info_nodei[0]));
  for (i = 0; i < nleaves - 1; i++) mapping->info_nodei[i + 1] = mapping->info_nodei[i] + mapping->info_nodec[i];
  PetscCall(PetscArraycpy(mapping->info_nodei[0], newleafdata, newnleaves));

  /* Create SF from leaves to multi-leaves */
  PetscCall(PetscSFGetMultiSF(sf, &msf));
  PetscCall(PetscSFCreateInverseSF(msf, &imsf));
  PetscCall(PetscSFCompose(imsf, sf2, &mapping->multileaves_sf));
  PetscCall(PetscSFDestroy(&imsf));
  PetscCall(PetscSFDestroy(&sf));
  PetscCall(PetscSFDestroy(&sf2));

  PetscCall(ISLocalToGlobalMappingRestoreBlockIndices(mapping, &gidxs));
  PetscCall(PetscFree(tmpg));
  PetscCall(PetscFree(mask));
  PetscCall(PetscFree3(mrootdata, leafdata, leafrd));
  PetscCall(PetscFree(newleafdata));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  ISLocalToGlobalMappingRestoreBlockInfo - Frees the memory allocated by `ISLocalToGlobalMappingGetBlockInfo()`

  Not Collective

  Input Parameters:
+ mapping  - the mapping from local to global indexing
. nproc    - number of processes that are connected to the calling process
. procs    - neighboring processes
. numprocs - number of block indices for each process
- indices  - block indices (in local numbering) shared with neighbors (sorted by global numbering)

  Level: advanced

.seealso: [](sec_scatter), `ISLocalToGlobalMappingDestroy()`, `ISLocalToGlobalMappingCreateIS()`, `ISLocalToGlobalMappingCreate()`,
          `ISLocalToGlobalMappingGetInfo()`
@*/
PetscErrorCode ISLocalToGlobalMappingRestoreBlockInfo(ISLocalToGlobalMapping mapping, PetscInt *nproc, PetscInt *procs[], PetscInt *numprocs[], PetscInt **indices[])
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(mapping, IS_LTOGM_CLASSID, 1);
  if (nproc) *nproc = 0;
  if (procs) *procs = NULL;
  if (numprocs) *numprocs = NULL;
  if (indices) *indices = NULL;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  ISLocalToGlobalMappingGetInfo - Gets the neighbor information for each process

  Collective the first time it is called

  Input Parameter:
. mapping - the mapping from local to global indexing

  Output Parameters:
+ nproc    - number of processes that are connected to the calling process
. procs    - neighboring processes
. numprocs - number of indices for each process
- indices  - indices (in local numbering) shared with neighbors (sorted by global numbering)

  Level: advanced

  Note:
  The user needs to call `ISLocalToGlobalMappingRestoreInfo()` when the data is no longer needed.

  Fortran Notes:
  There is no `ISLocalToGlobalMappingRestoreInfo()` in Fortran. You must make sure that
  `procs`[], `numprocs`[] and `indices`[][] are large enough arrays, either by allocating them
  dynamically or defining static ones large enough.

.seealso: [](sec_scatter), `ISLocalToGlobalMappingDestroy()`, `ISLocalToGlobalMappingCreateIS()`, `ISLocalToGlobalMappingCreate()`,
          `ISLocalToGlobalMappingRestoreInfo()`, `ISLocalToGlobalMappingGetNodeInfo()`
@*/
PetscErrorCode ISLocalToGlobalMappingGetInfo(ISLocalToGlobalMapping mapping, PetscInt *nproc, PetscInt *procs[], PetscInt *numprocs[], PetscInt **indices[])
{
  PetscInt **bindices = NULL, *bnumprocs = NULL, bs, i, j, k, n, *bprocs;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mapping, IS_LTOGM_CLASSID, 1);
  bs = mapping->bs;
  PetscCall(ISLocalToGlobalMappingGetBlockInfo(mapping, &n, &bprocs, &bnumprocs, &bindices));
  if (bs > 1) { /* we need to expand the cached info */
    if (indices) PetscCall(PetscCalloc1(n, indices));
    if (numprocs) PetscCall(PetscCalloc1(n, numprocs));
    if (indices || numprocs) {
      for (i = 0; i < n; i++) {
        if (indices) {
          PetscCall(PetscMalloc1(bs * bnumprocs[i], &(*indices)[i]));
          for (j = 0; j < bnumprocs[i]; j++) {
            for (k = 0; k < bs; k++) (*indices)[i][j * bs + k] = bs * bindices[i][j] + k;
          }
        }
        if (numprocs) (*numprocs)[i] = bnumprocs[i] * bs;
      }
    }
  } else {
    if (numprocs) *numprocs = bnumprocs;
    if (indices) *indices = bindices;
  }
  if (nproc) *nproc = n;
  if (procs) *procs = bprocs;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  ISLocalToGlobalMappingRestoreInfo - Frees the memory allocated by `ISLocalToGlobalMappingGetInfo()`

  Not Collective

  Input Parameters:
+ mapping  - the mapping from local to global indexing
. nproc    - number of processes that are connected to the calling process
. procs    - neighboring processes
. numprocs - number of indices for each process
- indices  - indices (in local numbering) shared with neighbors (sorted by global numbering)

  Level: advanced

.seealso: [](sec_scatter), `ISLocalToGlobalMappingDestroy()`, `ISLocalToGlobalMappingCreateIS()`, `ISLocalToGlobalMappingCreate()`,
          `ISLocalToGlobalMappingGetInfo()`
@*/
PetscErrorCode ISLocalToGlobalMappingRestoreInfo(ISLocalToGlobalMapping mapping, PetscInt *nproc, PetscInt *procs[], PetscInt *numprocs[], PetscInt **indices[])
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(mapping, IS_LTOGM_CLASSID, 1);
  if (mapping->bs > 1) {
    if (numprocs) PetscCall(PetscFree(*numprocs));
    if (indices) {
      if (*indices)
        for (PetscInt i = 0; i < *nproc; i++) PetscCall(PetscFree((*indices)[i]));
      PetscCall(PetscFree(*indices));
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  ISLocalToGlobalMappingGetNodeInfo - Gets the neighbor information of local nodes

  Collective the first time it is called

  Input Parameter:
. mapping - the mapping from local to global indexing

  Output Parameters:
+ n       - number of local nodes
. n_procs - an array storing the number of processes for each local node (including self)
- procs   - the processes' rank for each local node (sorted, self is first)

  Level: advanced

  Note:
  The user needs to call `ISLocalToGlobalMappingRestoreNodeInfo()` when the data is no longer needed.

.seealso: [](sec_scatter), `ISLocalToGlobalMappingDestroy()`, `ISLocalToGlobalMappingCreateIS()`, `ISLocalToGlobalMappingCreate()`,
          `ISLocalToGlobalMappingGetInfo()`, `ISLocalToGlobalMappingRestoreNodeInfo()`, `ISLocalToGlobalMappingGetBlockNodeInfo()`
@*/
PetscErrorCode ISLocalToGlobalMappingGetNodeInfo(ISLocalToGlobalMapping mapping, PetscInt *n, PetscInt *n_procs[], PetscInt **procs[])
{
  PetscInt **bprocs = NULL, *bn_procs = NULL, bs, i, j, k, bn;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mapping, IS_LTOGM_CLASSID, 1);
  bs = mapping->bs;
  PetscCall(ISLocalToGlobalMappingGetBlockNodeInfo(mapping, &bn, &bn_procs, &bprocs));
  if (bs > 1) { /* we need to expand the cached info */
    PetscInt *tn_procs;
    PetscInt  c;

    PetscCall(PetscMalloc1(bn * bs, &tn_procs));
    for (i = 0, c = 0; i < bn; i++) {
      for (k = 0; k < bs; k++) tn_procs[i * bs + k] = bn_procs[i];
      c += bs * bn_procs[i];
    }
    if (n) *n = bn * bs;
    if (procs) {
      PetscInt **tprocs;
      PetscInt   tn = bn * bs;

      PetscCall(PetscMalloc1(tn, &tprocs));
      if (tn) PetscCall(PetscMalloc1(c, &tprocs[0]));
      for (i = 0; i < tn - 1; i++) tprocs[i + 1] = tprocs[i] + tn_procs[i];
      for (i = 0; i < bn; i++) {
        for (k = 0; k < bs; k++) {
          for (j = 0; j < bn_procs[i]; j++) tprocs[i * bs + k][j] = bprocs[i][j];
        }
      }
      *procs = tprocs;
    }
    if (n_procs) *n_procs = tn_procs;
    else PetscCall(PetscFree(tn_procs));
  } else {
    if (n) *n = bn;
    if (n_procs) *n_procs = bn_procs;
    if (procs) *procs = bprocs;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  ISLocalToGlobalMappingRestoreNodeInfo - Frees the memory allocated by `ISLocalToGlobalMappingGetNodeInfo()`

  Not Collective

  Input Parameters:
+ mapping - the mapping from local to global indexing
. n       - number of local nodes
. n_procs - an array storing the number of processes for each local node (including self)
- procs   - the processes' rank for each local node (sorted, self is first)

  Level: advanced

.seealso: [](sec_scatter), `ISLocalToGlobalMappingDestroy()`, `ISLocalToGlobalMappingCreateIS()`, `ISLocalToGlobalMappingCreate()`,
          `ISLocalToGlobalMappingGetInfo()`
@*/
PetscErrorCode ISLocalToGlobalMappingRestoreNodeInfo(ISLocalToGlobalMapping mapping, PetscInt *n, PetscInt *n_procs[], PetscInt **procs[])
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(mapping, IS_LTOGM_CLASSID, 1);
  if (mapping->bs > 1) {
    if (n_procs) PetscCall(PetscFree(*n_procs));
    if (procs) {
      if (*procs) PetscCall(PetscFree((*procs)[0]));
      PetscCall(PetscFree(*procs));
    }
  }
  PetscCall(ISLocalToGlobalMappingRestoreBlockNodeInfo(mapping, n, n_procs, procs));
  PetscFunctionReturn(PETSC_SUCCESS);
}

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
PetscErrorCode ISLocalToGlobalMappingGetIndices(ISLocalToGlobalMapping ltog, const PetscInt *array[])
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ltog, IS_LTOGM_CLASSID, 1);
  PetscAssertPointer(array, 2);
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
+ ltog  - local to global mapping
- array - array of indices

  Level: advanced

.seealso: [](sec_scatter), `ISLocalToGlobalMappingCreate()`, `ISLocalToGlobalMappingApply()`, `ISLocalToGlobalMappingGetIndices()`
@*/
PetscErrorCode ISLocalToGlobalMappingRestoreIndices(ISLocalToGlobalMapping ltog, const PetscInt *array[])
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ltog, IS_LTOGM_CLASSID, 1);
  PetscAssertPointer(array, 2);
  PetscCheck(ltog->bs != 1 || *array == ltog->indices, PETSC_COMM_SELF, PETSC_ERR_ARG_BADPTR, "Trying to return mismatched pointer");
  if (ltog->bs > 1) PetscCall(PetscFree(*(void **)array));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  ISLocalToGlobalMappingGetBlockIndices - Get global indices for every local block in a `ISLocalToGlobalMapping`

  Not Collective

  Input Parameter:
. ltog - local to global mapping

  Output Parameter:
. array - array of indices

  Level: advanced

.seealso: [](sec_scatter), `ISLocalToGlobalMapping`, `ISLocalToGlobalMappingCreate()`, `ISLocalToGlobalMappingApply()`,
          `ISLocalToGlobalMappingRestoreBlockIndices()`
@*/
PetscErrorCode ISLocalToGlobalMappingGetBlockIndices(ISLocalToGlobalMapping ltog, const PetscInt *array[])
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ltog, IS_LTOGM_CLASSID, 1);
  PetscAssertPointer(array, 2);
  *array = ltog->indices;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  ISLocalToGlobalMappingRestoreBlockIndices - Restore indices obtained with `ISLocalToGlobalMappingGetBlockIndices()`

  Not Collective

  Input Parameters:
+ ltog  - local to global mapping
- array - array of indices

  Level: advanced

.seealso: [](sec_scatter), `ISLocalToGlobalMappingCreate()`, `ISLocalToGlobalMappingApply()`, `ISLocalToGlobalMappingGetIndices()`
@*/
PetscErrorCode ISLocalToGlobalMappingRestoreBlockIndices(ISLocalToGlobalMapping ltog, const PetscInt *array[])
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ltog, IS_LTOGM_CLASSID, 1);
  PetscAssertPointer(array, 2);
  PetscCheck(*array == ltog->indices, PETSC_COMM_SELF, PETSC_ERR_ARG_BADPTR, "Trying to return mismatched pointer");
  *array = NULL;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  ISLocalToGlobalMappingConcatenate - Create a new mapping that concatenates a list of mappings

  Not Collective

  Input Parameters:
+ comm  - communicator for the new mapping, must contain the communicator of every mapping to concatenate
. n     - number of mappings to concatenate
- ltogs - local to global mappings

  Output Parameter:
. ltogcat - new mapping

  Level: advanced

  Note:
  This currently always returns a mapping with block size of 1

  Developer Notes:
  If all the input mapping have the same block size we could easily handle that as a special case

.seealso: [](sec_scatter), `ISLocalToGlobalMapping`, `ISLocalToGlobalMappingCreate()`
@*/
PetscErrorCode ISLocalToGlobalMappingConcatenate(MPI_Comm comm, PetscInt n, const ISLocalToGlobalMapping ltogs[], ISLocalToGlobalMapping *ltogcat)
{
  PetscInt i, cnt, m, *idx;

  PetscFunctionBegin;
  PetscCheck(n >= 0, comm, PETSC_ERR_ARG_OUTOFRANGE, "Must have a non-negative number of mappings, given %" PetscInt_FMT, n);
  if (n > 0) PetscAssertPointer(ltogs, 3);
  for (i = 0; i < n; i++) PetscValidHeaderSpecific(ltogs[i], IS_LTOGM_CLASSID, 3);
  PetscAssertPointer(ltogcat, 4);
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

  Not Collective, No Fortran Support

  Input Parameters:
+ sname    - name of a new method
- function - routine to create method context

  Example Usage:
.vb
   ISLocalToGlobalMappingRegister("my_mapper", MyCreate);
.ve

  Then, your mapping can be chosen with the procedural interface via
.vb
  ISLocalToGlobalMappingSetType(ltog, "my_mapper")
.ve
  or at runtime via the option
.vb
  -islocaltoglobalmapping_type my_mapper
.ve

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

/*@
  ISLocalToGlobalMappingSetType - Sets the implementation type `ISLocalToGlobalMapping` will use

  Logically Collective

  Input Parameters:
+ ltog - the `ISLocalToGlobalMapping` object
- type - a known method

  Options Database Key:
. -islocaltoglobalmapping_type  <method> - Sets the method; use -help for a list of available methods (for instance, basic or hash)

  Level: intermediate

  Notes:
  See `ISLocalToGlobalMappingType` for available methods

  Normally, it is best to use the `ISLocalToGlobalMappingSetFromOptions()` command and
  then set the `ISLocalToGlobalMappingType` from the options database rather than by using
  this routine.

  Developer Notes:
  `ISLocalToGlobalMappingRegister()` is used to add new types to `ISLocalToGlobalMappingList` from which they
  are accessed by `ISLocalToGlobalMappingSetType()`.

.seealso: [](sec_scatter), `ISLocalToGlobalMappingType`, `ISLocalToGlobalMappingRegister()`, `ISLocalToGlobalMappingCreate()`, `ISLocalToGlobalMappingGetType()`
@*/
PetscErrorCode ISLocalToGlobalMappingSetType(ISLocalToGlobalMapping ltog, ISLocalToGlobalMappingType type)
{
  PetscBool match;
  PetscErrorCode (*r)(ISLocalToGlobalMapping) = NULL;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ltog, IS_LTOGM_CLASSID, 1);
  if (type) PetscAssertPointer(type, 2);

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

/*@
  ISLocalToGlobalMappingGetType - Get the type of the `ISLocalToGlobalMapping`

  Not Collective

  Input Parameter:
. ltog - the `ISLocalToGlobalMapping` object

  Output Parameter:
. type - the type

  Level: intermediate

.seealso: [](sec_scatter), `ISLocalToGlobalMappingType`, `ISLocalToGlobalMappingRegister()`, `ISLocalToGlobalMappingCreate()`, `ISLocalToGlobalMappingSetType()`
@*/
PetscErrorCode ISLocalToGlobalMappingGetType(ISLocalToGlobalMapping ltog, ISLocalToGlobalMappingType *type)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ltog, IS_LTOGM_CLASSID, 1);
  PetscAssertPointer(type, 2);
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
