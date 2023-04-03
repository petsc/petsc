
#include <petsc/private/isimpl.h> /*I "petscis.h"  I*/
#include <petscviewer.h>
#include <petscsf.h>

const char *const ISColoringTypes[] = {"global", "ghosted", "ISColoringType", "IS_COLORING_", NULL};

PetscErrorCode ISColoringReference(ISColoring coloring)
{
  PetscFunctionBegin;
  coloring->refct++;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
   ISColoringSetType - indicates if the coloring is for the local representation (including ghost points) or the global representation of a `Mat`

   Collective

   Input Parameters:
+    coloring - the coloring object
-    type - either `IS_COLORING_LOCAL` or `IS_COLORING_GLOBAL`

   Level: intermediate

   Notes:
   `IS_COLORING_LOCAL` can lead to faster computations since parallel ghost point updates are not needed for each color

   With `IS_COLORING_LOCAL` the coloring is in the numbering of the local vector, for `IS_COLORING_GLOBAL` it is in the numbering of the global vector

.seealso: `MatFDColoringCreate()`, `ISColoring`, `ISColoringType`, `ISColoringCreate()`, `IS_COLORING_LOCAL`, `IS_COLORING_GLOBAL`, `ISColoringGetType()`
@*/
PetscErrorCode ISColoringSetType(ISColoring coloring, ISColoringType type)
{
  PetscFunctionBegin;
  coloring->ctype = type;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C

    ISColoringGetType - gets if the coloring is for the local representation (including ghost points) or the global representation

   Collective

   Input Parameter:
.   coloring - the coloring object

   Output Parameter:
.    type - either `IS_COLORING_LOCAL` or `IS_COLORING_GLOBAL`

   Level: intermediate

.seealso: `MatFDColoringCreate()`, `ISColoring`, `ISColoringType`, `ISColoringCreate()`, `IS_COLORING_LOCAL`, `IS_COLORING_GLOBAL`, `ISColoringSetType()`
@*/
PetscErrorCode ISColoringGetType(ISColoring coloring, ISColoringType *type)
{
  PetscFunctionBegin;
  *type = coloring->ctype;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
   ISColoringDestroy - Destroys an `ISColoring` coloring context.

   Collective

   Input Parameter:
.  iscoloring - the coloring context

   Level: advanced

.seealso: `ISColoring`, `ISColoringView()`, `MatColoring`
@*/
PetscErrorCode ISColoringDestroy(ISColoring *iscoloring)
{
  PetscInt i;

  PetscFunctionBegin;
  if (!*iscoloring) PetscFunctionReturn(PETSC_SUCCESS);
  PetscValidPointer((*iscoloring), 1);
  if (--(*iscoloring)->refct > 0) {
    *iscoloring = NULL;
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  if ((*iscoloring)->is) {
    for (i = 0; i < (*iscoloring)->n; i++) PetscCall(ISDestroy(&(*iscoloring)->is[i]));
    PetscCall(PetscFree((*iscoloring)->is));
  }
  if ((*iscoloring)->allocated) PetscCall(PetscFree((*iscoloring)->colors));
  PetscCall(PetscCommDestroy(&(*iscoloring)->comm));
  PetscCall(PetscFree((*iscoloring)));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  ISColoringViewFromOptions - Processes command line options to determine if/how an `ISColoring` object is to be viewed.

  Collective

  Input Parameters:
+ obj   - the `ISColoring` object
. prefix - prefix to use for viewing, or `NULL` to use prefix of `mat`
- optionname - option to activate viewing

  Level: intermediate

  Developer Note:
  This cannot use `PetscObjectViewFromOptions()` because `ISColoring` is not a `PetscObject`

.seealso: `ISColoring`, `ISColoringView()`
@*/
PetscErrorCode ISColoringViewFromOptions(ISColoring obj, PetscObject bobj, const char optionname[])
{
  PetscViewer       viewer;
  PetscBool         flg;
  PetscViewerFormat format;
  char             *prefix;

  PetscFunctionBegin;
  prefix = bobj ? bobj->prefix : NULL;
  PetscCall(PetscOptionsGetViewer(obj->comm, NULL, prefix, optionname, &viewer, &format, &flg));
  if (flg) {
    PetscCall(PetscViewerPushFormat(viewer, format));
    PetscCall(ISColoringView(obj, viewer));
    PetscCall(PetscViewerPopFormat(viewer));
    PetscCall(PetscViewerDestroy(&viewer));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
   ISColoringView - Views an `ISColoring` coloring context.

   Collective

   Input Parameters:
+  iscoloring - the coloring context
-  viewer - the viewer

   Level: advanced

.seealso: `ISColoring()`, `ISColoringViewFromOptions()`, `ISColoringDestroy()`, `ISColoringGetIS()`, `MatColoring`
@*/
PetscErrorCode ISColoringView(ISColoring iscoloring, PetscViewer viewer)
{
  PetscInt  i;
  PetscBool iascii;
  IS       *is;

  PetscFunctionBegin;
  PetscValidPointer(iscoloring, 1);
  if (!viewer) PetscCall(PetscViewerASCIIGetStdout(iscoloring->comm, &viewer));
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 2);

  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERASCII, &iascii));
  if (iascii) {
    MPI_Comm    comm;
    PetscMPIInt size, rank;

    PetscCall(PetscObjectGetComm((PetscObject)viewer, &comm));
    PetscCallMPI(MPI_Comm_size(comm, &size));
    PetscCallMPI(MPI_Comm_rank(comm, &rank));
    PetscCall(PetscViewerASCIIPrintf(viewer, "ISColoring Object: %d MPI processes\n", size));
    PetscCall(PetscViewerASCIIPrintf(viewer, "ISColoringType: %s\n", ISColoringTypes[iscoloring->ctype]));
    PetscCall(PetscViewerASCIIPushSynchronized(viewer));
    PetscCall(PetscViewerASCIISynchronizedPrintf(viewer, "[%d] Number of colors %" PetscInt_FMT "\n", rank, iscoloring->n));
    PetscCall(PetscViewerFlush(viewer));
    PetscCall(PetscViewerASCIIPopSynchronized(viewer));
  }

  PetscCall(ISColoringGetIS(iscoloring, PETSC_USE_POINTER, PETSC_IGNORE, &is));
  for (i = 0; i < iscoloring->n; i++) PetscCall(ISView(iscoloring->is[i], viewer));
  PetscCall(ISColoringRestoreIS(iscoloring, PETSC_USE_POINTER, &is));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
   ISColoringGetColors - Returns an array with the color for each local node

   Not Collective

   Input Parameter:
.  iscoloring - the coloring context

   Output Parameters:
+  n - number of nodes
.  nc - number of colors
-  colors - color for each node

   Level: advanced

   Notes:
   Do not free the `colors` array.

   The `colors` array will only be valid for the lifetime of the `ISColoring`

.seealso: `ISColoring`, `ISColoringValue`, `ISColoringRestoreIS()`, `ISColoringView()`, `ISColoringGetIS()`
@*/
PetscErrorCode ISColoringGetColors(ISColoring iscoloring, PetscInt *n, PetscInt *nc, const ISColoringValue **colors)
{
  PetscFunctionBegin;
  PetscValidPointer(iscoloring, 1);

  if (n) *n = iscoloring->N;
  if (nc) *nc = iscoloring->n;
  if (colors) *colors = iscoloring->colors;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
   ISColoringGetIS - Extracts index sets from the coloring context. Each is contains the nodes of one color

   Collective

   Input Parameters:
+  iscoloring - the coloring context
-  mode - if this value is `PETSC_OWN_POINTER` then the caller owns the pointer and must free the array of `IS` and each `IS` in the array

   Output Parameters:
+  nn - number of index sets in the coloring context
-  is - array of index sets

   Level: advanced

   Note:
   If mode is `PETSC_USE_POINTER` then `ISColoringRestoreIS()` must be called when the `IS` are no longer needed

.seealso: `ISColoring`, `IS`, `ISColoringRestoreIS()`, `ISColoringView()`, `ISColoringGetColoring()`, `ISColoringGetColors()`
@*/
PetscErrorCode ISColoringGetIS(ISColoring iscoloring, PetscCopyMode mode, PetscInt *nn, IS *isis[])
{
  PetscFunctionBegin;
  PetscValidPointer(iscoloring, 1);

  if (nn) *nn = iscoloring->n;
  if (isis) {
    if (!iscoloring->is) {
      PetscInt        *mcolors, **ii, nc = iscoloring->n, i, base, n = iscoloring->N;
      ISColoringValue *colors = iscoloring->colors;
      IS              *is;

      if (PetscDefined(USE_DEBUG)) {
        for (i = 0; i < n; i++) PetscCheck(((PetscInt)colors[i]) < nc, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Coloring is our of range index %d value %d number colors %d", (int)i, (int)colors[i], (int)nc);
      }

      /* generate the lists of nodes for each color */
      PetscCall(PetscCalloc1(nc, &mcolors));
      for (i = 0; i < n; i++) mcolors[colors[i]]++;

      PetscCall(PetscMalloc1(nc, &ii));
      PetscCall(PetscMalloc1(n, &ii[0]));
      for (i = 1; i < nc; i++) ii[i] = ii[i - 1] + mcolors[i - 1];
      PetscCall(PetscArrayzero(mcolors, nc));

      if (iscoloring->ctype == IS_COLORING_GLOBAL) {
        PetscCallMPI(MPI_Scan(&iscoloring->N, &base, 1, MPIU_INT, MPI_SUM, iscoloring->comm));
        base -= iscoloring->N;
        for (i = 0; i < n; i++) ii[colors[i]][mcolors[colors[i]]++] = i + base; /* global idx */
      } else if (iscoloring->ctype == IS_COLORING_LOCAL) {
        for (i = 0; i < n; i++) ii[colors[i]][mcolors[colors[i]]++] = i; /* local idx */
      } else SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Not provided for this ISColoringType type");

      PetscCall(PetscMalloc1(nc, &is));
      for (i = 0; i < nc; i++) PetscCall(ISCreateGeneral(iscoloring->comm, mcolors[i], ii[i], PETSC_COPY_VALUES, is + i));

      if (mode != PETSC_OWN_POINTER) iscoloring->is = is;
      *isis = is;
      PetscCall(PetscFree(ii[0]));
      PetscCall(PetscFree(ii));
      PetscCall(PetscFree(mcolors));
    } else {
      *isis = iscoloring->is;
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
   ISColoringRestoreIS - Restores the index sets extracted from the coloring context with `ISColoringGetIS()` using `PETSC_USE_POINTER`

   Collective

   Input Parameters:
+  iscoloring - the coloring context
.  mode - who retains ownership of the is
-  is - array of index sets

   Level: advanced

.seealso: `ISColoring()`, `IS`, `ISColoringGetIS()`, `ISColoringView()`, `PetscCopyMode`
@*/
PetscErrorCode ISColoringRestoreIS(ISColoring iscoloring, PetscCopyMode mode, IS *is[])
{
  PetscFunctionBegin;
  PetscValidPointer(iscoloring, 1);

  /* currently nothing is done here */
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
    ISColoringCreate - Generates an `ISColoring` context from lists (provided by each MPI process) of colors for each node.

    Collective

    Input Parameters:
+   comm - communicator for the processors creating the coloring
.   ncolors - max color value
.   n - number of nodes on this processor
.   colors - array containing the colors for this MPI rank, color numbers begin at 0, for each local node
-   mode - see `PetscCopyMode` for meaning of this flag.

    Output Parameter:
.   iscoloring - the resulting coloring data structure

    Options Database Key:
.   -is_coloring_view - Activates `ISColoringView()`

   Level: advanced

    Notes:
    By default sets coloring type to  `IS_COLORING_GLOBAL`

.seealso: `ISColoring`, `ISColoringValue`, `MatColoringCreate()`, `ISColoringView()`, `ISColoringDestroy()`, `ISColoringSetType()`
@*/
PetscErrorCode ISColoringCreate(MPI_Comm comm, PetscInt ncolors, PetscInt n, const ISColoringValue colors[], PetscCopyMode mode, ISColoring *iscoloring)
{
  PetscMPIInt size, rank, tag;
  PetscInt    base, top, i;
  PetscInt    nc, ncwork;
  MPI_Status  status;

  PetscFunctionBegin;
  if (ncolors != PETSC_DECIDE && ncolors > IS_COLORING_MAX) {
    PetscCheck(ncolors <= PETSC_MAX_UINT16, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Max color value exceeds %d limit. This number is unrealistic. Perhaps a bug in code?\nCurrent max: %d user requested: %" PetscInt_FMT, PETSC_MAX_UINT16, PETSC_IS_COLORING_MAX, ncolors);
    SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Max color value exceeds limit. Perhaps reconfigure PETSc with --with-is-color-value-type=short?\n Current max: %d user requested: %" PetscInt_FMT, PETSC_IS_COLORING_MAX, ncolors);
  }
  PetscCall(PetscNew(iscoloring));
  PetscCall(PetscCommDuplicate(comm, &(*iscoloring)->comm, &tag));
  comm = (*iscoloring)->comm;

  /* compute the number of the first node on my processor */
  PetscCallMPI(MPI_Comm_size(comm, &size));

  /* should use MPI_Scan() */
  PetscCallMPI(MPI_Comm_rank(comm, &rank));
  if (rank == 0) {
    base = 0;
    top  = n;
  } else {
    PetscCallMPI(MPI_Recv(&base, 1, MPIU_INT, rank - 1, tag, comm, &status));
    top = base + n;
  }
  if (rank < size - 1) PetscCallMPI(MPI_Send(&top, 1, MPIU_INT, rank + 1, tag, comm));

  /* compute the total number of colors */
  ncwork = 0;
  for (i = 0; i < n; i++) {
    if (ncwork < colors[i]) ncwork = colors[i];
  }
  ncwork++;
  PetscCall(MPIU_Allreduce(&ncwork, &nc, 1, MPIU_INT, MPI_MAX, comm));
  PetscCheck(nc <= ncolors, PETSC_COMM_SELF, PETSC_ERR_ARG_INCOMP, "Number of colors passed in %" PetscInt_FMT " is less then the actual number of colors in array %" PetscInt_FMT, ncolors, nc);
  (*iscoloring)->n     = nc;
  (*iscoloring)->is    = NULL;
  (*iscoloring)->N     = n;
  (*iscoloring)->refct = 1;
  (*iscoloring)->ctype = IS_COLORING_GLOBAL;
  if (mode == PETSC_COPY_VALUES) {
    PetscCall(PetscMalloc1(n, &(*iscoloring)->colors));
    PetscCall(PetscArraycpy((*iscoloring)->colors, colors, n));
    (*iscoloring)->allocated = PETSC_TRUE;
  } else if (mode == PETSC_OWN_POINTER) {
    (*iscoloring)->colors    = (ISColoringValue *)colors;
    (*iscoloring)->allocated = PETSC_TRUE;
  } else {
    (*iscoloring)->colors    = (ISColoringValue *)colors;
    (*iscoloring)->allocated = PETSC_FALSE;
  }
  PetscCall(ISColoringViewFromOptions(*iscoloring, NULL, "-is_coloring_view"));
  PetscCall(PetscInfo(0, "Number of colors %" PetscInt_FMT "\n", nc));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
    ISBuildTwoSided - Takes an `IS` that describes where each element will be mapped globally over all ranks.
    Generates an `IS` that contains new numbers from remote or local on the `IS`.

    Collective

    Input Parameters:
+   ito - an `IS` describes where each entry will be mapped. Negative target rank will be ignored
-   toindx - an `IS` describes what indices should send. `NULL` means sending natural numbering

    Output Parameter:
.   rows - contains new numbers from remote or local

   Level: advanced

   Developer Note:
   This manual page is incomprehensible and still needs to be fixed

.seealso: [](sec_scatter), `IS`, `MatPartitioningCreate()`, `ISPartitioningToNumbering()`, `ISPartitioningCount()`
@*/
PetscErrorCode ISBuildTwoSided(IS ito, IS toindx, IS *rows)
{
  const PetscInt *ito_indices, *toindx_indices;
  PetscInt       *send_indices, rstart, *recv_indices, nrecvs, nsends;
  PetscInt       *tosizes, *fromsizes, i, j, *tosizes_tmp, *tooffsets_tmp, ito_ln;
  PetscMPIInt    *toranks, *fromranks, size, target_rank, *fromperm_newtoold, nto, nfrom;
  PetscLayout     isrmap;
  MPI_Comm        comm;
  PetscSF         sf;
  PetscSFNode    *iremote;

  PetscFunctionBegin;
  PetscCall(PetscObjectGetComm((PetscObject)ito, &comm));
  PetscCallMPI(MPI_Comm_size(comm, &size));
  PetscCall(ISGetLocalSize(ito, &ito_ln));
  PetscCall(ISGetLayout(ito, &isrmap));
  PetscCall(PetscLayoutGetRange(isrmap, &rstart, NULL));
  PetscCall(ISGetIndices(ito, &ito_indices));
  PetscCall(PetscCalloc2(size, &tosizes_tmp, size + 1, &tooffsets_tmp));
  for (i = 0; i < ito_ln; i++) {
    if (ito_indices[i] < 0) continue;
    else PetscCheck(ito_indices[i] < size, comm, PETSC_ERR_ARG_OUTOFRANGE, "target rank %" PetscInt_FMT " is larger than communicator size %d ", ito_indices[i], size);
    tosizes_tmp[ito_indices[i]]++;
  }
  nto = 0;
  for (i = 0; i < size; i++) {
    tooffsets_tmp[i + 1] = tooffsets_tmp[i] + tosizes_tmp[i];
    if (tosizes_tmp[i] > 0) nto++;
  }
  PetscCall(PetscCalloc2(nto, &toranks, 2 * nto, &tosizes));
  nto = 0;
  for (i = 0; i < size; i++) {
    if (tosizes_tmp[i] > 0) {
      toranks[nto]         = i;
      tosizes[2 * nto]     = tosizes_tmp[i];   /* size */
      tosizes[2 * nto + 1] = tooffsets_tmp[i]; /* offset */
      nto++;
    }
  }
  nsends = tooffsets_tmp[size];
  PetscCall(PetscCalloc1(nsends, &send_indices));
  if (toindx) PetscCall(ISGetIndices(toindx, &toindx_indices));
  for (i = 0; i < ito_ln; i++) {
    if (ito_indices[i] < 0) continue;
    target_rank                              = ito_indices[i];
    send_indices[tooffsets_tmp[target_rank]] = toindx ? toindx_indices[i] : (i + rstart);
    tooffsets_tmp[target_rank]++;
  }
  if (toindx) PetscCall(ISRestoreIndices(toindx, &toindx_indices));
  PetscCall(ISRestoreIndices(ito, &ito_indices));
  PetscCall(PetscFree2(tosizes_tmp, tooffsets_tmp));
  PetscCall(PetscCommBuildTwoSided(comm, 2, MPIU_INT, nto, toranks, tosizes, &nfrom, &fromranks, &fromsizes));
  PetscCall(PetscFree2(toranks, tosizes));
  PetscCall(PetscMalloc1(nfrom, &fromperm_newtoold));
  for (i = 0; i < nfrom; i++) fromperm_newtoold[i] = i;
  PetscCall(PetscSortMPIIntWithArray(nfrom, fromranks, fromperm_newtoold));
  nrecvs = 0;
  for (i = 0; i < nfrom; i++) nrecvs += fromsizes[i * 2];
  PetscCall(PetscCalloc1(nrecvs, &recv_indices));
  PetscCall(PetscMalloc1(nrecvs, &iremote));
  nrecvs = 0;
  for (i = 0; i < nfrom; i++) {
    for (j = 0; j < fromsizes[2 * fromperm_newtoold[i]]; j++) {
      iremote[nrecvs].rank    = fromranks[i];
      iremote[nrecvs++].index = fromsizes[2 * fromperm_newtoold[i] + 1] + j;
    }
  }
  PetscCall(PetscSFCreate(comm, &sf));
  PetscCall(PetscSFSetGraph(sf, nsends, nrecvs, NULL, PETSC_OWN_POINTER, iremote, PETSC_OWN_POINTER));
  PetscCall(PetscSFSetType(sf, PETSCSFBASIC));
  /* how to put a prefix ? */
  PetscCall(PetscSFSetFromOptions(sf));
  PetscCall(PetscSFBcastBegin(sf, MPIU_INT, send_indices, recv_indices, MPI_REPLACE));
  PetscCall(PetscSFBcastEnd(sf, MPIU_INT, send_indices, recv_indices, MPI_REPLACE));
  PetscCall(PetscSFDestroy(&sf));
  PetscCall(PetscFree(fromranks));
  PetscCall(PetscFree(fromsizes));
  PetscCall(PetscFree(fromperm_newtoold));
  PetscCall(PetscFree(send_indices));
  if (rows) {
    PetscCall(PetscSortInt(nrecvs, recv_indices));
    PetscCall(ISCreateGeneral(comm, nrecvs, recv_indices, PETSC_OWN_POINTER, rows));
  } else {
    PetscCall(PetscFree(recv_indices));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
    ISPartitioningToNumbering - Takes an `IS' that represents a partitioning (the MPI rank that each local entry belongs to) and on each MPI process
    generates an `IS` that contains a new global node number in the new ordering for each entry

    Collective

    Input Parameter:
.   partitioning - a partitioning as generated by `MatPartitioningApply()` or `MatPartitioningApplyND()`

    Output Parameter:
.   is - on each processor the index set that defines the global numbers
         (in the new numbering) for all the nodes currently (before the partitioning)
         on that processor

   Level: advanced

   Note:
   The resulting `IS` tells where each local entry is mapped to in a new global ordering

.seealso: [](sec_scatter), `IS`, `MatPartitioningCreate()`, `AOCreateBasic()`, `ISPartitioningCount()`
@*/
PetscErrorCode ISPartitioningToNumbering(IS part, IS *is)
{
  MPI_Comm        comm;
  IS              ndorder;
  PetscInt        i, np, npt, n, *starts = NULL, *sums = NULL, *lsizes = NULL, *newi = NULL;
  const PetscInt *indices = NULL;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(part, IS_CLASSID, 1);
  PetscValidPointer(is, 2);
  /* see if the partitioning comes from nested dissection */
  PetscCall(PetscObjectQuery((PetscObject)part, "_petsc_matpartitioning_ndorder", (PetscObject *)&ndorder));
  if (ndorder) {
    PetscCall(PetscObjectReference((PetscObject)ndorder));
    *is = ndorder;
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  PetscCall(PetscObjectGetComm((PetscObject)part, &comm));
  /* count the number of partitions, i.e., virtual processors */
  PetscCall(ISGetLocalSize(part, &n));
  PetscCall(ISGetIndices(part, &indices));
  np = 0;
  for (i = 0; i < n; i++) np = PetscMax(np, indices[i]);
  PetscCall(MPIU_Allreduce(&np, &npt, 1, MPIU_INT, MPI_MAX, comm));
  np = npt + 1; /* so that it looks like a MPI_Comm_size output */

  /*
        lsizes - number of elements of each partition on this particular processor
        sums - total number of "previous" nodes for any particular partition
        starts - global number of first element in each partition on this processor
  */
  PetscCall(PetscMalloc3(np, &lsizes, np, &starts, np, &sums));
  PetscCall(PetscArrayzero(lsizes, np));
  for (i = 0; i < n; i++) lsizes[indices[i]]++;
  PetscCall(MPIU_Allreduce(lsizes, sums, np, MPIU_INT, MPI_SUM, comm));
  PetscCallMPI(MPI_Scan(lsizes, starts, np, MPIU_INT, MPI_SUM, comm));
  for (i = 0; i < np; i++) starts[i] -= lsizes[i];
  for (i = 1; i < np; i++) {
    sums[i] += sums[i - 1];
    starts[i] += sums[i - 1];
  }

  /*
      For each local index give it the new global number
  */
  PetscCall(PetscMalloc1(n, &newi));
  for (i = 0; i < n; i++) newi[i] = starts[indices[i]]++;
  PetscCall(PetscFree3(lsizes, starts, sums));

  PetscCall(ISRestoreIndices(part, &indices));
  PetscCall(ISCreateGeneral(comm, n, newi, PETSC_OWN_POINTER, is));
  PetscCall(ISSetPermutation(*is));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
    ISPartitioningCount - Takes a `IS` that represents a partitioning (the MPI rank that each local entry belongs to) and determines the number of
    resulting elements on each (partition) rank

    Collective

    Input Parameters:
+   partitioning - a partitioning as generated by `MatPartitioningApply()` or `MatPartitioningApplyND()`
-   len - length of the array count, this is the total number of partitions

    Output Parameter:
.   count - array of length size, to contain the number of elements assigned
        to each partition, where size is the number of partitions generated
         (see notes below).

   Level: advanced

    Notes:
    By default the number of partitions generated (and thus the length
    of count) is the size of the communicator associated with `IS`,
    but it can be set by `MatPartitioningSetNParts()`.

    The resulting array of lengths can for instance serve as input of `PCBJacobiSetTotalBlocks()`.

    If the partitioning has been obtained by `MatPartitioningApplyND()`, the returned count does not include the separators.

.seealso: [](sec_scatter), `IS`, `MatPartitioningCreate()`, `AOCreateBasic()`, `ISPartitioningToNumbering()`,
          `MatPartitioningSetNParts()`, `MatPartitioningApply()`, `MatPartitioningApplyND()`
@*/
PetscErrorCode ISPartitioningCount(IS part, PetscInt len, PetscInt count[])
{
  MPI_Comm        comm;
  PetscInt        i, n, *lsizes;
  const PetscInt *indices;
  PetscMPIInt     npp;

  PetscFunctionBegin;
  PetscCall(PetscObjectGetComm((PetscObject)part, &comm));
  if (len == PETSC_DEFAULT) {
    PetscMPIInt size;
    PetscCallMPI(MPI_Comm_size(comm, &size));
    len = (PetscInt)size;
  }

  /* count the number of partitions */
  PetscCall(ISGetLocalSize(part, &n));
  PetscCall(ISGetIndices(part, &indices));
  if (PetscDefined(USE_DEBUG)) {
    PetscInt np = 0, npt;
    for (i = 0; i < n; i++) np = PetscMax(np, indices[i]);
    PetscCall(MPIU_Allreduce(&np, &npt, 1, MPIU_INT, MPI_MAX, comm));
    np = npt + 1; /* so that it looks like a MPI_Comm_size output */
    PetscCheck(np <= len, PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "Length of count array %" PetscInt_FMT " is less than number of partitions %" PetscInt_FMT, len, np);
  }

  /*
        lsizes - number of elements of each partition on this particular processor
        sums - total number of "previous" nodes for any particular partition
        starts - global number of first element in each partition on this processor
  */
  PetscCall(PetscCalloc1(len, &lsizes));
  for (i = 0; i < n; i++) {
    if (indices[i] > -1) lsizes[indices[i]]++;
  }
  PetscCall(ISRestoreIndices(part, &indices));
  PetscCall(PetscMPIIntCast(len, &npp));
  PetscCall(MPIU_Allreduce(lsizes, count, npp, MPIU_INT, MPI_SUM, comm));
  PetscCall(PetscFree(lsizes));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
    ISAllGather - Given an index set `IS` on each processor, generates a large
    index set (same on each processor) by concatenating together each
    processors index set.

    Collective

    Input Parameter:
.   is - the distributed index set

    Output Parameter:
.   isout - the concatenated index set (same on all processors)

    Level: intermediate

    Notes:
    `ISAllGather()` is clearly not scalable for large index sets.

    The `IS` created on each processor must be created with a common
    communicator (e.g., `PETSC_COMM_WORLD`). If the index sets were created
    with `PETSC_COMM_SELF`, this routine will not work as expected, since
    each process will generate its own new `IS` that consists only of
    itself.

    The communicator for this new `IS` is `PETSC_COMM_SELF`

.seealso: [](sec_scatter), `IS`, `ISCreateGeneral()`, `ISCreateStride()`, `ISCreateBlock()`
@*/
PetscErrorCode ISAllGather(IS is, IS *isout)
{
  PetscInt       *indices, n, i, N, step, first;
  const PetscInt *lindices;
  MPI_Comm        comm;
  PetscMPIInt     size, *sizes = NULL, *offsets = NULL, nn;
  PetscBool       stride;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(is, IS_CLASSID, 1);
  PetscValidPointer(isout, 2);

  PetscCall(PetscObjectGetComm((PetscObject)is, &comm));
  PetscCallMPI(MPI_Comm_size(comm, &size));
  PetscCall(ISGetLocalSize(is, &n));
  PetscCall(PetscObjectTypeCompare((PetscObject)is, ISSTRIDE, &stride));
  if (size == 1 && stride) { /* should handle parallel ISStride also */
    PetscCall(ISStrideGetInfo(is, &first, &step));
    PetscCall(ISCreateStride(PETSC_COMM_SELF, n, first, step, isout));
  } else {
    PetscCall(PetscMalloc2(size, &sizes, size, &offsets));

    PetscCall(PetscMPIIntCast(n, &nn));
    PetscCallMPI(MPI_Allgather(&nn, 1, MPI_INT, sizes, 1, MPI_INT, comm));
    offsets[0] = 0;
    for (i = 1; i < size; i++) {
      PetscInt s = offsets[i - 1] + sizes[i - 1];
      PetscCall(PetscMPIIntCast(s, &offsets[i]));
    }
    N = offsets[size - 1] + sizes[size - 1];

    PetscCall(PetscMalloc1(N, &indices));
    PetscCall(ISGetIndices(is, &lindices));
    PetscCallMPI(MPI_Allgatherv((void *)lindices, nn, MPIU_INT, indices, sizes, offsets, MPIU_INT, comm));
    PetscCall(ISRestoreIndices(is, &lindices));
    PetscCall(PetscFree2(sizes, offsets));

    PetscCall(ISCreateGeneral(PETSC_COMM_SELF, N, indices, PETSC_OWN_POINTER, isout));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
    ISAllGatherColors - Given a a set of colors on each processor, generates a large
    set (same on each processor) by concatenating together each processors colors

    Collective

    Input Parameters:
+   comm - communicator to share the indices
.   n - local size of set
-   lindices - local colors

    Output Parameters:
+   outN - total number of indices
-   outindices - all of the colors

    Level: intermediate

    Note:
    `ISAllGatherColors()` is clearly not scalable for large index sets.

.seealso: `ISCOloringValue`, `ISColoring()`, `ISCreateGeneral()`, `ISCreateStride()`, `ISCreateBlock()`, `ISAllGather()`
@*/
PetscErrorCode ISAllGatherColors(MPI_Comm comm, PetscInt n, ISColoringValue *lindices, PetscInt *outN, ISColoringValue *outindices[])
{
  ISColoringValue *indices;
  PetscInt         i, N;
  PetscMPIInt      size, *offsets = NULL, *sizes = NULL, nn = n;

  PetscFunctionBegin;
  PetscCallMPI(MPI_Comm_size(comm, &size));
  PetscCall(PetscMalloc2(size, &sizes, size, &offsets));

  PetscCallMPI(MPI_Allgather(&nn, 1, MPI_INT, sizes, 1, MPI_INT, comm));
  offsets[0] = 0;
  for (i = 1; i < size; i++) offsets[i] = offsets[i - 1] + sizes[i - 1];
  N = offsets[size - 1] + sizes[size - 1];
  PetscCall(PetscFree2(sizes, offsets));

  PetscCall(PetscMalloc1(N + 1, &indices));
  PetscCallMPI(MPI_Allgatherv(lindices, (PetscMPIInt)n, MPIU_COLORING_VALUE, indices, sizes, offsets, MPIU_COLORING_VALUE, comm));

  *outindices = indices;
  if (outN) *outN = N;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
    ISComplement - Given an index set `IS` generates the complement index set. That is
       all indices that are NOT in the given set.

    Collective

    Input Parameters:
+   is - the index set
.   nmin - the first index desired in the local part of the complement
-   nmax - the largest index desired in the local part of the complement (note that all indices in is must be greater or equal to nmin and less than nmax)

    Output Parameter:
.   isout - the complement

    Level: intermediate

    Notes:
    The communicator for `isout` is the same as for the input `is`

    For a parallel `is`, this will generate the local part of the complement on each process

    To generate the entire complement (on each process) of a parallel `IS`, first call `ISAllGather()` and then
    call this routine.

.seealso: [](sec_scatter), `IS`, `ISCreateGeneral()`, `ISCreateStride()`, `ISCreateBlock()`, `ISAllGather()`
@*/
PetscErrorCode ISComplement(IS is, PetscInt nmin, PetscInt nmax, IS *isout)
{
  const PetscInt *indices;
  PetscInt        n, i, j, unique, cnt, *nindices;
  PetscBool       sorted;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(is, IS_CLASSID, 1);
  PetscValidPointer(isout, 4);
  PetscCheck(nmin >= 0, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "nmin %" PetscInt_FMT " cannot be negative", nmin);
  PetscCheck(nmin <= nmax, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "nmin %" PetscInt_FMT " cannot be greater than nmax %" PetscInt_FMT, nmin, nmax);
  PetscCall(ISSorted(is, &sorted));
  PetscCheck(sorted, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Index set must be sorted");

  PetscCall(ISGetLocalSize(is, &n));
  PetscCall(ISGetIndices(is, &indices));
  if (PetscDefined(USE_DEBUG)) {
    for (i = 0; i < n; i++) {
      PetscCheck(indices[i] >= nmin, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Index %" PetscInt_FMT "'s value %" PetscInt_FMT " is smaller than minimum given %" PetscInt_FMT, i, indices[i], nmin);
      PetscCheck(indices[i] < nmax, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Index %" PetscInt_FMT "'s value %" PetscInt_FMT " is larger than maximum given %" PetscInt_FMT, i, indices[i], nmax);
    }
  }
  /* Count number of unique entries */
  unique = (n > 0);
  for (i = 0; i < n - 1; i++) {
    if (indices[i + 1] != indices[i]) unique++;
  }
  PetscCall(PetscMalloc1(nmax - nmin - unique, &nindices));
  cnt = 0;
  for (i = nmin, j = 0; i < nmax; i++) {
    if (j < n && i == indices[j]) do {
        j++;
      } while (j < n && i == indices[j]);
    else nindices[cnt++] = i;
  }
  PetscCheck(cnt == nmax - nmin - unique, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Number of entries found in complement %" PetscInt_FMT " does not match expected %" PetscInt_FMT, cnt, nmax - nmin - unique);
  PetscCall(ISCreateGeneral(PetscObjectComm((PetscObject)is), cnt, nindices, PETSC_OWN_POINTER, isout));
  PetscCall(ISRestoreIndices(is, &indices));
  PetscFunctionReturn(PETSC_SUCCESS);
}
