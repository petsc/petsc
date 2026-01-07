#include <petscdmda.h>                 /*I  "petscdmda.h"      I*/
#include <petscdmplex.h>               /*I  "petscdmplex.h"    I*/
#include <petsc/private/dmswarmimpl.h> /*I  "petscdmswarm.h"   I*/

PetscClassId DMSWARMSORT_CLASSID;

static int sort_CompareSwarmPoint(const void *dataA, const void *dataB)
{
  SwarmPoint *pointA = (SwarmPoint *)dataA;
  SwarmPoint *pointB = (SwarmPoint *)dataB;

  if (pointA->cell_index < pointB->cell_index) {
    return -1;
  } else if (pointA->cell_index > pointB->cell_index) {
    return 1;
  } else {
    return 0;
  }
}

static PetscErrorCode DMSwarmSortApplyCellIndexSort(DMSwarmSort ctx)
{
  PetscFunctionBegin;
  if (ctx->list) qsort(ctx->list, ctx->npoints, sizeof(SwarmPoint), sort_CompareSwarmPoint);
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DMSwarmSortCreate(DMSwarmSort *sort)
{
  DMSwarmSort s;

  PetscFunctionBegin;
  PetscCall(DMInitializePackage());
  PetscCall(PetscHeaderCreate(s, DMSWARMSORT_CLASSID, "DMSwarmSort", "Sort context for a DMSwarm", "DM", PETSC_COMM_SELF, DMSwarmSortDestroy, DMSwarmSortView));
  PetscCall(PetscObjectSetName((PetscObject)s, "Sort"));
  s->isvalid = PETSC_FALSE;
  s->ncells  = 0;
  s->npoints = 0;
  PetscCall(PetscMalloc1(1, &s->pcell_offsets));
  PetscCall(PetscMalloc1(1, &s->list));
  *sort = s;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DMSwarmSortSetup(DMSwarmSort ctx, DM dm, PetscInt ncells)
{
  DMSwarmCellDM celldm;
  PetscInt     *swarm_cellid;
  PetscInt      p, npoints;
  PetscInt      tmp, c, count;
  const char   *cellid;

  PetscFunctionBegin;
  if (!ctx) PetscFunctionReturn(PETSC_SUCCESS);
  if (ctx->isvalid) PetscFunctionReturn(PETSC_SUCCESS);

  PetscCall(PetscLogEventBegin(DMSWARM_Sort, 0, 0, 0, 0));
  /* check the number of cells */
  if (ncells != ctx->ncells) {
    PetscCall(PetscRealloc(sizeof(PetscInt) * (ncells + 1), &ctx->pcell_offsets));
    ctx->ncells = ncells;
  }
  PetscCall(PetscArrayzero(ctx->pcell_offsets, ctx->ncells + 1));

  /* get the number of points */
  PetscCall(DMSwarmGetLocalSize(dm, &npoints));
  if (npoints != ctx->npoints) {
    PetscCall(PetscRealloc(sizeof(SwarmPoint) * npoints, &ctx->list));
    ctx->npoints = npoints;
  }
  PetscCall(PetscArrayzero(ctx->list, npoints));

  PetscCall(DMSwarmGetCellDMActive(dm, &celldm));
  PetscCall(DMSwarmCellDMGetCellID(celldm, &cellid));
  PetscCall(DMSwarmGetField(dm, cellid, NULL, NULL, (void **)&swarm_cellid));
  for (p = 0; p < ctx->npoints; p++) {
    ctx->list[p].point_index = p;
    ctx->list[p].cell_index  = swarm_cellid[p];
    PetscCheck(ctx->list[p].cell_index >= 0 && ctx->list[p].cell_index < ctx->ncells, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Cell index %" PetscInt_FMT " is not in [0, %" PetscInt_FMT ")", swarm_cellid[p], ctx->ncells);
  }
  PetscCall(DMSwarmRestoreField(dm, cellid, NULL, NULL, (void **)&swarm_cellid));
  PetscCall(DMSwarmSortApplyCellIndexSort(ctx));

  /* sum points per cell */
  for (p = 0; p < ctx->npoints; p++) ctx->pcell_offsets[ctx->list[p].cell_index]++;

  /* create offset list */
  count = 0;
  for (c = 0; c < ctx->ncells; c++) {
    tmp                   = ctx->pcell_offsets[c];
    ctx->pcell_offsets[c] = count;
    count                 = count + tmp;
  }
  ctx->pcell_offsets[c] = count;

  ctx->isvalid = PETSC_TRUE;
  PetscCall(PetscLogEventEnd(DMSWARM_Sort, 0, 0, 0, 0));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMSwarmSortDestroy - destroy a `DMSwarmSort`

  Collective

  Input Parameter:
. sort - address of `DMSwarmSort`

  Level: advanced

.seealso: `DMSwarmSort`, `DMSwarmSortCreate()`
@*/
PetscErrorCode DMSwarmSortDestroy(DMSwarmSort *sort)
{
  PetscFunctionBegin;
  if (!sort) PetscFunctionReturn(PETSC_SUCCESS);
  if (!*sort) PetscFunctionReturn(PETSC_SUCCESS);
  PetscValidHeaderSpecific(*sort, DMSWARMSORT_CLASSID, 1);
  if (--((PetscObject)*sort)->refct > 0) {
    *sort = NULL;
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  PetscTryTypeMethod(*sort, destroy);
  PetscCall(PetscFree((*sort)->list));
  PetscCall(PetscFree((*sort)->pcell_offsets));
  PetscCall(PetscHeaderDestroy(sort));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMSwarmSortView - view a `DMSwarmSort`

  Collective

  Input Parameters:
+ sort   - `DMSwarmSort`
- viewer - viewer to display sort context, for example `PETSC_VIEWER_STDOUT_WORLD`

  Level: advanced

.seealso: `DMSwarmSort`, `DMSwarmSortCreate()`, `PetscViewer`
@*/
PetscErrorCode DMSwarmSortView(DMSwarmSort sort, PetscViewer viewer)
{
  PetscBool iascii;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(sort, DMSWARMSORT_CLASSID, 1);
  if (!viewer) PetscCall(PetscViewerASCIIGetStdout(PetscObjectComm((PetscObject)sort), &viewer));
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 2);
  PetscCheckSameComm(sort, 1, viewer, 2);
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERASCII, &iascii));
  if (iascii) {
    PetscCall(PetscObjectPrintClassNamePrefixType((PetscObject)sort, viewer));
    PetscCall(PetscViewerASCIIPushTab(viewer));
  }
  PetscTryTypeMethod(sort, view, viewer);
  if (iascii) PetscCall(PetscViewerASCIIPopTab(viewer));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMSwarmSortGetNumberOfPointsPerCell - Returns the number of points in a cell

  Not Collective

  Input Parameters:
+ sw   - a `DMSWARM` objects
- cell - the cell number in the cell `DM`

  Output Parameter:
. npoints - the number of points in the cell

  Level: advanced

  Notes:
  You must call `DMSwarmSortGetAccess()` before you can call `DMSwarmSortGetNumberOfPointsPerCell()`

.seealso: `DMSWARM`, `DMSwarmSetType()`, `DMSwarmSortGetAccess()`, `DMSwarmSortGetPointsPerCell()`
@*/
PetscErrorCode DMSwarmSortGetNumberOfPointsPerCell(DM sw, PetscInt cell, PetscInt *npoints)
{
  DMSwarmCellDM celldm;
  DMSwarmSort   ctx;

  PetscFunctionBegin;
  PetscCall(DMSwarmGetCellDMActive(sw, &celldm));
  PetscCall(DMSwarmCellDMGetSort(celldm, &ctx));
  PetscCheck(ctx, PetscObjectComm((PetscObject)sw), PETSC_ERR_USER, "The DMSwarmSort context has not been created. Must call DMSwarmSortGetAccess() first");
  PetscCheck(ctx->isvalid, PETSC_COMM_SELF, PETSC_ERR_USER, "SwarmPointSort container is not valid. Must call DMSwarmSortGetAccess() first");
  PetscCheck(cell < ctx->ncells, PETSC_COMM_SELF, PETSC_ERR_USER, "Cell index (%" PetscInt_FMT ") is greater than max number of local cells (%" PetscInt_FMT ")", cell, ctx->ncells);
  PetscCheck(cell >= 0, PETSC_COMM_SELF, PETSC_ERR_USER, "Cell index (%" PetscInt_FMT ") cannot be negative", cell);
  *npoints = ctx->pcell_offsets[cell + 1] - ctx->pcell_offsets[cell];
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  DMSwarmSortGetPointsPerCell - Creates an array of point indices for all points in a cell

  Not Collective

  Input Parameters:
+ sw      - a `DMSWARM` object
. cell    - the cell number in the cell `DM`
. npoints - the number of points in the cell
- pidlist - array of the indices identifying all points in cell e

  Level: advanced

  Note:
  You must call `DMSwarmSortGetAccess()` before you can call `DMSwarmSortGetPointsPerCell()`, and call `DMSwarmRestorePointsPerCell()` afterwards

.seealso: `DMSWARM`, `DMSwarmSetType()`, `DMSwarmRestorePointsPerCell()`, `DMSwarmSortGetAccess()`, `DMSwarmSortGetNumberOfPointsPerCell()`
@*/
PetscErrorCode DMSwarmSortGetPointsPerCell(DM sw, PetscInt cell, PetscInt *npoints, PetscInt **pidlist)
{
  DMSwarmCellDM celldm;
  PetscInt      pid, pid_unsorted;
  DMSwarmSort   ctx;

  PetscFunctionBegin;
  PetscCall(DMSwarmGetCellDMActive(sw, &celldm));
  PetscCall(DMSwarmCellDMGetSort(celldm, &ctx));
  PetscCheck(ctx, PetscObjectComm((PetscObject)sw), PETSC_ERR_USER, "The DMSwarmSort context has not been created. Must call DMSwarmSortGetAccess() first");
  PetscCall(DMSwarmSortGetNumberOfPointsPerCell(sw, cell, npoints));
  PetscCall(DMGetWorkArray(sw, *npoints, MPIU_SCALAR, pidlist));
  for (PetscInt p = 0; p < *npoints; ++p) {
    pid           = ctx->pcell_offsets[cell] + p;
    pid_unsorted  = ctx->list[pid].point_index;
    (*pidlist)[p] = pid_unsorted;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  DMSwarmSortRestorePointsPerCell - Restores an array of point indices for all points in a cell

  Not Collective

  Input Parameters:
+ dm      - a `DMSWARM` object
. e       - the index of the cell
. npoints - the number of points in the cell
- pidlist - array of the indices identifying all points in cell e

  Level: advanced

  Note:
  You must call `DMSwarmSortGetAccess()` and `DMSwarmSortGetPointsPerCell()` before you can call `DMSwarmSortRestorePointsPerCell()`

.seealso: `DMSWARM`, `DMSwarmSetType()`, `DMSwarmSortGetPointsPerCell()`, `DMSwarmSortGetAccess()`, `DMSwarmSortGetNumberOfPointsPerCell()`
@*/
PetscErrorCode DMSwarmSortRestorePointsPerCell(DM dm, PetscInt e, PetscInt *npoints, PetscInt **pidlist)
{
  PetscFunctionBegin;
  PetscCall(DMRestoreWorkArray(dm, *npoints, MPIU_SCALAR, pidlist));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMSwarmSortGetAccess - Setups up a `DMSWARM` point sort context for efficient traversal of points within a cell

  Not Collective

  Input Parameter:
. sw - a `DMSWARM` object

  Level: advanced

  Notes:
  Calling `DMSwarmSortGetAccess()` creates a list which enables easy identification of all points contained in a
  given cell. This method does not explicitly sort the data within the `DMSWARM` based on the cell index associated
  with a `DMSWARM` point.

  The sort context is valid only for the `DMSWARM` points defined at the time when `DMSwarmSortGetAccess()` was called.
  For example, suppose the swarm contained NP points when `DMSwarmSortGetAccess()` was called. If the user subsequently
  adds 10 additional points to the swarm, the sort context is still valid, but only for the first NP points.
  The indices associated with the 10 new additional points will not be contained within the sort context.
  This means that the user can still safely perform queries via `DMSwarmSortGetPointsPerCell()` and
  `DMSwarmSortGetPointsPerCell()`, however the results return will be based on the first NP points.

  If any` DMSWARM` re-sizing method is called after `DMSwarmSortGetAccess()` which modifies any of the first NP entries
  in the `DMSWARM`, the sort context will become invalid. Currently there are no guards to prevent the user from
  invalidating the sort context. For this reason, we highly recommend you do not use `DMSwarmRemovePointAtIndex()` in
  between calls to `DMSwarmSortGetAccess()` and `DMSwarmSortRestoreAccess()`.

  To facilitate safe removal of points using the sort context, we suggest a "two pass" strategy in which the
  first pass "marks" points for removal, and the second pass actually removes the points from the `DMSWARM`

  You must call `DMSwarmSortGetAccess()` before you can call `DMSwarmSortGetPointsPerCell()` or `DMSwarmSortGetNumberOfPointsPerCell()`

  The sort context may become invalid if any re-sizing methods are applied which alter the first NP points
  within swarm at the time `DMSwarmSortGetAccess()` was called.

  You must call `DMSwarmSortRestoreAccess()` when you no longer need access to the sort context

.seealso: `DMSWARM`, `DMSwarmSetType()`, `DMSwarmSortRestoreAccess()`
@*/
PetscErrorCode DMSwarmSortGetAccess(DM sw)
{
  DM            dm;
  DMSwarmCellDM celldm;
  DMSwarmSort   ctx;
  PetscInt      ncells = 0;
  PetscBool     isda, isplex, isshell;

  PetscFunctionBegin;
  PetscCall(DMSwarmGetCellDMActive(sw, &celldm));
  PetscCall(DMSwarmCellDMGetSort(celldm, &ctx));
  if (!ctx) {
    PetscCall(DMSwarmSortCreate(&ctx));
    PetscCall(DMSwarmCellDMSetSort(celldm, ctx));
    PetscCall(DMSwarmSortDestroy(&ctx));
    PetscCall(DMSwarmCellDMGetSort(celldm, &ctx));
  }

  /* get the number of cells */
  PetscCall(DMSwarmGetCellDM(sw, &dm));
  PetscCall(PetscObjectTypeCompare((PetscObject)dm, DMDA, &isda));
  PetscCall(PetscObjectTypeCompare((PetscObject)dm, DMPLEX, &isplex));
  PetscCall(PetscObjectTypeCompare((PetscObject)dm, DMSHELL, &isshell));
  if (isda) {
    const PetscInt *element;
    PetscInt        nel, npe;

    PetscCall(DMDAGetElements(dm, &nel, &npe, &element));
    ncells = nel;
    PetscCall(DMDARestoreElements(dm, &nel, &npe, &element));
  } else if (isplex) {
    PetscInt ps, pe;

    PetscCall(DMPlexGetHeightStratum(dm, 0, &ps, &pe));
    ncells = pe - ps;
  } else if (isshell) {
    PetscErrorCode (*method_DMShellGetNumberOfCells)(DM, PetscInt *);

    PetscCall(PetscObjectQueryFunction((PetscObject)dm, "DMGetNumberOfCells_C", &method_DMShellGetNumberOfCells));
    if (method_DMShellGetNumberOfCells) {
      PetscCall(method_DMShellGetNumberOfCells(dm, &ncells));
    } else
      SETERRQ(PetscObjectComm((PetscObject)sw), PETSC_ERR_SUP, "Cannot determine the number of cells for the DMSHELL object. User must provide a method via PetscObjectComposeFunction( (PetscObject)shelldm, \"DMGetNumberOfCells_C\", your_function_to_compute_number_of_cells);");
  } else SETERRQ(PetscObjectComm((PetscObject)sw), PETSC_ERR_SUP, "Cannot determine the number of cells for a DM not of type DA, PLEX or SHELL");

  /* setup */
  PetscCall(DMSwarmSortSetup(ctx, sw, ncells));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMSwarmSortRestoreAccess - Invalidates the `DMSWARM` point sorting context previously computed with `DMSwarmSortGetAccess()`

  Not Collective

  Input Parameter:
. sw - a `DMSWARM` object

  Level: advanced

  Note:
  You must call `DMSwarmSortGetAccess()` before calling `DMSwarmSortRestoreAccess()`

.seealso: `DMSWARM`, `DMSwarmSetType()`, `DMSwarmSortGetAccess()`
@*/
PetscErrorCode DMSwarmSortRestoreAccess(DM sw)
{
  DMSwarmCellDM celldm;
  DMSwarmSort   ctx;

  PetscFunctionBegin;
  PetscCall(DMSwarmGetCellDMActive(sw, &celldm));
  PetscCall(DMSwarmCellDMGetSort(celldm, &ctx));
  if (!ctx) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCheck(ctx->isvalid, PetscObjectComm((PetscObject)sw), PETSC_ERR_SUP, "You must call DMSwarmSortGetAccess() before calling DMSwarmSortRestoreAccess()");
  ctx->isvalid = PETSC_FALSE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMSwarmSortGetIsValid - Gets the isvalid flag associated with a `DMSWARM` point sorting context

  Not Collective

  Input Parameter:
. sw - a `DMSWARM` object

  Output Parameter:
. isvalid - flag indicating whether the sort context is up-to-date

  Level: advanced

.seealso: `DMSWARM`, `DMSwarmSetType()`, `DMSwarmSortGetAccess()`
@*/
PetscErrorCode DMSwarmSortGetIsValid(DM sw, PetscBool *isvalid)
{
  DMSwarmCellDM celldm;
  DMSwarmSort   ctx;

  PetscFunctionBegin;
  PetscCall(DMSwarmGetCellDMActive(sw, &celldm));
  PetscCall(DMSwarmCellDMGetSort(celldm, &ctx));
  if (!ctx) {
    *isvalid = PETSC_FALSE;
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  *isvalid = ctx->isvalid;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMSwarmSortGetSizes - Gets the sizes associated with a `DMSWARM` point sorting context

  Not Collective

  Input Parameter:
. sw - a `DMSWARM` object

  Output Parameters:
+ ncells  - number of cells within the sort context (pass `NULL` to ignore)
- npoints - number of points used to create the sort context (pass `NULL` to ignore)

  Level: advanced

.seealso: `DMSWARM`, `DMSwarmSetType()`, `DMSwarmSortGetAccess()`
@*/
PetscErrorCode DMSwarmSortGetSizes(DM sw, PetscInt *ncells, PetscInt *npoints)
{
  DMSwarmCellDM celldm;
  DMSwarmSort   ctx;

  PetscFunctionBegin;
  PetscCall(DMSwarmGetCellDMActive(sw, &celldm));
  PetscCall(DMSwarmCellDMGetSort(celldm, &ctx));
  if (!ctx) {
    if (ncells) *ncells = 0;
    if (npoints) *npoints = 0;
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  if (ncells) *ncells = ctx->ncells;
  if (npoints) *npoints = ctx->npoints;
  PetscFunctionReturn(PETSC_SUCCESS);
}
