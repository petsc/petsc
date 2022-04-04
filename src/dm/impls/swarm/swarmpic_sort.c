#include <petscdm.h>
#include <petscdmda.h>
#include <petscdmplex.h>
#include <petscdmswarm.h>
#include <petsc/private/dmswarmimpl.h>

int sort_CompareSwarmPoint(const void *dataA,const void *dataB)
{
  SwarmPoint *pointA = (SwarmPoint*)dataA;
  SwarmPoint *pointB = (SwarmPoint*)dataB;

  if (pointA->cell_index < pointB->cell_index) {
    return -1;
  } else if (pointA->cell_index > pointB->cell_index) {
    return 1;
  } else {
    return 0;
  }
}

PetscErrorCode DMSwarmSortApplyCellIndexSort(DMSwarmSort ctx)
{
  PetscFunctionBegin;
  qsort(ctx->list,ctx->npoints,sizeof(SwarmPoint),sort_CompareSwarmPoint);
  PetscFunctionReturn(0);
}

PetscErrorCode DMSwarmSortCreate(DMSwarmSort *_ctx)
{
  DMSwarmSort    ctx;

  PetscFunctionBegin;
  PetscCall(PetscNew(&ctx));
  ctx->isvalid = PETSC_FALSE;
  ctx->ncells  = 0;
  ctx->npoints = 0;
  PetscCall(PetscMalloc1(1,&ctx->pcell_offsets));
  PetscCall(PetscMalloc1(1,&ctx->list));
  *_ctx = ctx;
  PetscFunctionReturn(0);
}

PetscErrorCode DMSwarmSortSetup(DMSwarmSort ctx,DM dm,PetscInt ncells)
{
  PetscInt        *swarm_cellid;
  PetscInt        p,npoints;
  PetscInt        tmp,c,count;

  PetscFunctionBegin;
  if (!ctx) PetscFunctionReturn(0);
  if (ctx->isvalid) PetscFunctionReturn(0);

  PetscCall(PetscLogEventBegin(DMSWARM_Sort,0,0,0,0));
  /* check the number of cells */
  if (ncells != ctx->ncells) {
    PetscCall(PetscRealloc(sizeof(PetscInt)*(ncells + 1),&ctx->pcell_offsets));
    ctx->ncells = ncells;
  }
  PetscCall(PetscArrayzero(ctx->pcell_offsets,ctx->ncells + 1));

  /* get the number of points */
  PetscCall(DMSwarmGetLocalSize(dm,&npoints));
  if (npoints != ctx->npoints) {
    PetscCall(PetscRealloc(sizeof(SwarmPoint)*npoints,&ctx->list));
    ctx->npoints = npoints;
  }
  PetscCall(PetscArrayzero(ctx->list,npoints));

  PetscCall(DMSwarmGetField(dm,DMSwarmPICField_cellid,NULL,NULL,(void**)&swarm_cellid));
  for (p=0; p<ctx->npoints; p++) {
    ctx->list[p].point_index = p;
    ctx->list[p].cell_index  = swarm_cellid[p];
  }
  PetscCall(DMSwarmRestoreField(dm,DMSwarmPICField_cellid,NULL,NULL,(void**)&swarm_cellid));
  PetscCall(DMSwarmSortApplyCellIndexSort(ctx));

  /* sum points per cell */
  for (p=0; p<ctx->npoints; p++) {
    ctx->pcell_offsets[ ctx->list[p].cell_index ]++;
  }

  /* create offset list */
  count = 0;
  for (c=0; c<ctx->ncells; c++) {
    tmp = ctx->pcell_offsets[c];
    ctx->pcell_offsets[c] = count;
    count = count + tmp;
  }
  ctx->pcell_offsets[c] = count;

  ctx->isvalid = PETSC_TRUE;
  PetscCall(PetscLogEventEnd(DMSWARM_Sort,0,0,0,0));
  PetscFunctionReturn(0);
}

PetscErrorCode DMSwarmSortDestroy(DMSwarmSort *_ctx)
{
  DMSwarmSort     ctx;

  PetscFunctionBegin;
  if (!_ctx) PetscFunctionReturn(0);
  if (!*_ctx) PetscFunctionReturn(0);
  ctx = *_ctx;
  if (ctx->list)      {
    PetscCall(PetscFree(ctx->list));
  }
  if (ctx->pcell_offsets) {
    PetscCall(PetscFree(ctx->pcell_offsets));
  }
  PetscCall(PetscFree(ctx));
  *_ctx = NULL;
  PetscFunctionReturn(0);
}

/*@C
   DMSwarmSortGetNumberOfPointsPerCell - Returns the number of points in a cell

   Not collective

   Input parameters:
+  dm - a DMSwarm objects
.  e - the index of the cell
-  npoints - the number of points in the cell

   Level: advanced

   Notes:
   You must call DMSwarmSortGetAccess() before you can call DMSwarmSortGetNumberOfPointsPerCell()

.seealso: DMSwarmSetType(), DMSwarmSortGetAccess(), DMSwarmSortGetPointsPerCell()
@*/
PetscErrorCode DMSwarmSortGetNumberOfPointsPerCell(DM dm,PetscInt e,PetscInt *npoints)
{
  DM_Swarm     *swarm = (DM_Swarm*)dm->data;
  PetscInt     points_per_cell;
  DMSwarmSort  ctx;

  PetscFunctionBegin;
  ctx = swarm->sort_context;
  PetscCheck(ctx,PetscObjectComm((PetscObject)dm),PETSC_ERR_USER,"The DMSwarmSort context has not been created. Must call DMSwarmSortGetAccess() first");
  PetscCheck(ctx->isvalid,PETSC_COMM_SELF,PETSC_ERR_USER,"SwarmPointSort container is not valid. Must call DMSwarmSortGetAccess() first");
  PetscCheck(e < ctx->ncells,PETSC_COMM_SELF,PETSC_ERR_USER,"Cell index (%D) is greater than max number of local cells (%D)",e,ctx->ncells);
  PetscCheck(e >= 0,PETSC_COMM_SELF,PETSC_ERR_USER,"Cell index (%D) cannot be negative",e);
  points_per_cell = ctx->pcell_offsets[e+1] - ctx->pcell_offsets[e];
  *npoints = points_per_cell;
  PetscFunctionReturn(0);
}

/*@C
   DMSwarmSortGetPointsPerCell - Creates an array of point indices for all points in a cell

   Not collective

   Input parameters:
+  dm - a DMSwarm object
.  e - the index of the cell
.  npoints - the number of points in the cell
-  pidlist - array of the indices indentifying all points in cell e

   Level: advanced

   Notes:
     You must call DMSwarmSortGetAccess() before you can call DMSwarmSortGetPointsPerCell()

     The array pidlist is internally created and must be free'd by the user

.seealso: DMSwarmSetType(), DMSwarmSortGetAccess(), DMSwarmSortGetNumberOfPointsPerCell()
@*/
PETSC_EXTERN PetscErrorCode DMSwarmSortGetPointsPerCell(DM dm,PetscInt e,PetscInt *npoints,PetscInt **pidlist)
{
  DM_Swarm       *swarm = (DM_Swarm*)dm->data;
  PetscInt       points_per_cell;
  PetscInt       p,pid,pid_unsorted;
  PetscInt       *plist;
  DMSwarmSort    ctx;

  PetscFunctionBegin;
  ctx = swarm->sort_context;
  PetscCheck(ctx,PetscObjectComm((PetscObject)dm),PETSC_ERR_USER,"The DMSwarmSort context has not been created. Must call DMSwarmSortGetAccess() first");
  PetscCall(DMSwarmSortGetNumberOfPointsPerCell(dm,e,&points_per_cell));
  PetscCall(PetscMalloc1(points_per_cell,&plist));
  for (p=0; p<points_per_cell; p++) {
    pid = ctx->pcell_offsets[e] + p;
    pid_unsorted = ctx->list[pid].point_index;
    plist[p] = pid_unsorted;
  }
  *npoints = points_per_cell;
  *pidlist = plist;

  PetscFunctionReturn(0);
}

/*@C
   DMSwarmSortGetAccess - Setups up a DMSwarm point sort context for efficient traversal of points within a cell

   Not collective

   Input parameter:
.  dm - a DMSwarm object

   Calling DMSwarmSortGetAccess() creates a list which enables easy identification of all points contained in a
   given cell. This method does not explicitly sort the data within the DMSwarm based on the cell index associated
   with a DMSwarm point.

   The sort context is valid only for the DMSwarm points defined at the time when DMSwarmSortGetAccess() was called.
   For example, suppose the swarm contained NP points when DMSwarmSortGetAccess() was called. If the user subsequently
   adds 10 additional points to the swarm, the sort context is still valid, but only for the first NP points.
   The indices associated with the 10 new additional points will not be contained within the sort context.
   This means that the user can still safely perform queries via DMSwarmSortGetPointsPerCell() and
   DMSwarmSortGetPointsPerCell(), however the results return will be based on the first NP points.

   If any DMSwam re-sizing method is called after DMSwarmSortGetAccess() which modifies any of the first NP entries
   in the DMSwarm, the sort context will become invalid. Currently there are no guards to prevent the user from
   invalidating the sort context. For this reason, we highly recommend you do not use DMSwarmRemovePointAtIndex() in
   between calls to DMSwarmSortGetAccess() and DMSwarmSortRestoreAccess().

   To facilitate safe removal of points using the sort context, we suggest a "two pass" strategy in which the
   first pass "marks" points for removal, and the second pass actually removes the points from the DMSwarm.

   Notes:
     You must call DMSwarmSortGetAccess() before you can call DMSwarmSortGetPointsPerCell() or DMSwarmSortGetNumberOfPointsPerCell()

     The sort context may become invalid if any re-sizing methods are applied which alter the first NP points
     within swarm at the time DMSwarmSortGetAccess() was called.

     You must call DMSwarmSortRestoreAccess() when you no longer need access to the sort context

   Level: advanced

.seealso: DMSwarmSetType(), DMSwarmSortRestoreAccess()
@*/
PETSC_EXTERN PetscErrorCode DMSwarmSortGetAccess(DM dm)
{
  DM_Swarm       *swarm = (DM_Swarm*)dm->data;
  PetscInt       ncells;
  DM             celldm;
  PetscBool      isda,isplex,isshell;

  PetscFunctionBegin;
  if (!swarm->sort_context) {
    PetscCall(DMSwarmSortCreate(&swarm->sort_context));
  }

  /* get the number of cells */
  PetscCall(DMSwarmGetCellDM(dm,&celldm));
  PetscCall(PetscObjectTypeCompare((PetscObject)celldm,DMDA,&isda));
  PetscCall(PetscObjectTypeCompare((PetscObject)celldm,DMPLEX,&isplex));
  PetscCall(PetscObjectTypeCompare((PetscObject)celldm,DMSHELL,&isshell));
  ncells = 0;
  if (isda) {
    PetscInt       nel,npe;
    const PetscInt *element;

    PetscCall(DMDAGetElements(celldm,&nel,&npe,&element));
    ncells = nel;
    PetscCall(DMDARestoreElements(celldm,&nel,&npe,&element));
  } else if (isplex) {
    PetscInt ps,pe;

    PetscCall(DMPlexGetHeightStratum(celldm,0,&ps,&pe));
    ncells = pe - ps;
  } else if (isshell) {
    PetscErrorCode (*method_DMShellGetNumberOfCells)(DM,PetscInt*);

    PetscCall(PetscObjectQueryFunction((PetscObject)celldm,"DMGetNumberOfCells_C",&method_DMShellGetNumberOfCells));
    if (method_DMShellGetNumberOfCells) {
      PetscCall(method_DMShellGetNumberOfCells(celldm,&ncells));
    } else SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_SUP,"Cannot determine the number of cells for the DMSHELL object. User must provide a method via PetscObjectComposeFunction( (PetscObject)shelldm, \"DMGetNumberOfCells_C\", your_function_to_compute_number_of_cells);");
  } else SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_SUP,"Cannot determine the number of cells for a DM not of type DA, PLEX or SHELL");

  /* setup */
  PetscCall(DMSwarmSortSetup(swarm->sort_context,dm,ncells));
  PetscFunctionReturn(0);
}

/*@C
   DMSwarmSortRestoreAccess - Invalidates the DMSwarm point sorting context

   Not collective

   Input parameter:
.  dm - a DMSwarm object

   Level: advanced

   Note:
   You must call DMSwarmSortGetAccess() before calling DMSwarmSortRestoreAccess()

.seealso: DMSwarmSetType(), DMSwarmSortGetAccess()
@*/
PETSC_EXTERN PetscErrorCode DMSwarmSortRestoreAccess(DM dm)
{
  DM_Swarm *swarm = (DM_Swarm*)dm->data;

  PetscFunctionBegin;
  if (!swarm->sort_context) PetscFunctionReturn(0);
  PetscCheck(swarm->sort_context->isvalid,PetscObjectComm((PetscObject)dm),PETSC_ERR_SUP,"You must call DMSwarmSortGetAccess() before calling DMSwarmSortRestoreAccess()");
  swarm->sort_context->isvalid = PETSC_FALSE;
  PetscFunctionReturn(0);
}

/*@C
   DMSwarmSortGetIsValid - Gets the isvalid flag associated with a DMSwarm point sorting context

   Not collective

   Input parameter:
.  dm - a DMSwarm object

   Output parameter:
.  isvalid - flag indicating whether the sort context is up-to-date

 Level: advanced

.seealso: DMSwarmSetType(), DMSwarmSortGetAccess()
@*/
PETSC_EXTERN PetscErrorCode DMSwarmSortGetIsValid(DM dm,PetscBool *isvalid)
{
  DM_Swarm *swarm = (DM_Swarm*)dm->data;

  PetscFunctionBegin;
  if (!swarm->sort_context) {
    *isvalid = PETSC_FALSE;
    PetscFunctionReturn(0);
  }
  *isvalid = swarm->sort_context->isvalid;
  PetscFunctionReturn(0);
}

/*@C
   DMSwarmSortGetSizes - Gets the sizes associated with a DMSwarm point sorting context

   Not collective

   Input parameter:
.  dm - a DMSwarm object

   Output parameters:
+  ncells - number of cells within the sort context (pass NULL to ignore)
-  npoints - number of points used to create the sort context (pass NULL to ignore)

   Level: advanced

.seealso: DMSwarmSetType(), DMSwarmSortGetAccess()
@*/
PETSC_EXTERN PetscErrorCode DMSwarmSortGetSizes(DM dm,PetscInt *ncells,PetscInt *npoints)
{
  DM_Swarm *swarm = (DM_Swarm*)dm->data;

  PetscFunctionBegin;
  if (!swarm->sort_context) {
    if (ncells)  *ncells  = 0;
    if (npoints) *npoints = 0;
    PetscFunctionReturn(0);
  }
  if (ncells)  *ncells = swarm->sort_context->ncells;
  if (npoints) *npoints = swarm->sort_context->npoints;
  PetscFunctionReturn(0);
}
