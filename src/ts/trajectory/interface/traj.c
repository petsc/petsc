#include <petsc/private/tsimpl.h>        /*I "petscts.h"  I*/
#include <petsc/private/tshistoryimpl.h>
#include <petscdm.h>

PetscFunctionList TSTrajectoryList              = NULL;
PetscBool         TSTrajectoryRegisterAllCalled = PETSC_FALSE;
PetscClassId      TSTRAJECTORY_CLASSID;
PetscLogEvent     TSTrajectory_Set, TSTrajectory_Get, TSTrajectory_GetVecs, TSTrajectory_SetUp;

/*@C
  TSTrajectoryRegister - Adds a way of storing trajectories to the TS package

  Not Collective

  Input Parameters:
+ name        - the name of a new user-defined creation routine
- create_func - the creation routine itself

  Notes:
  TSTrajectoryRegister() may be called multiple times to add several user-defined tses.

  Level: developer

.seealso: TSTrajectoryRegisterAll()
@*/
PetscErrorCode TSTrajectoryRegister(const char sname[],PetscErrorCode (*function)(TSTrajectory,TS))
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFunctionListAdd(&TSTrajectoryList,sname,function);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
  TSTrajectorySet - Sets a vector of state in the trajectory object

  Collective on TSTrajectory

  Input Parameters:
+ tj      - the trajectory object
. ts      - the time stepper object (optional)
. stepnum - the step number
. time    - the current time
- X       - the current solution

  Level: developer

  Notes: Usually one does not call this routine, it is called automatically during TSSolve()

.seealso: TSTrajectorySetUp(), TSTrajectoryDestroy(), TSTrajectorySetType(), TSTrajectorySetVariableNames(), TSGetTrajectory(), TSTrajectoryGet(), TSTrajectoryGetVecs()
@*/
PetscErrorCode TSTrajectorySet(TSTrajectory tj,TS ts,PetscInt stepnum,PetscReal time,Vec X)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!tj) PetscFunctionReturn(0);
  PetscValidHeaderSpecific(tj,TSTRAJECTORY_CLASSID,1);
  if (ts) PetscValidHeaderSpecific(ts,TS_CLASSID,2);
  PetscValidLogicalCollectiveInt(tj,stepnum,3);
  PetscValidLogicalCollectiveReal(tj,time,4);
  PetscValidHeaderSpecific(X,VEC_CLASSID,5);
  PetscCheckFalse(!tj->ops->set,PetscObjectComm((PetscObject)tj),PETSC_ERR_SUP,"TSTrajectory type %s",((PetscObject)tj)->type_name);
  PetscCheckFalse(!tj->setupcalled,PetscObjectComm((PetscObject)tj),PETSC_ERR_ORDER,"TSTrajectorySetUp should be called first");
  if (tj->monitor) {
    ierr = PetscViewerASCIIPrintf(tj->monitor,"TSTrajectorySet: stepnum %D, time %g (stages %D)\n",stepnum,(double)time,(PetscInt)!tj->solution_only);CHKERRQ(ierr);
  }
  ierr = PetscLogEventBegin(TSTrajectory_Set,tj,ts,0,0);CHKERRQ(ierr);
  ierr = (*tj->ops->set)(tj,ts,stepnum,time,X);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(TSTrajectory_Set,tj,ts,0,0);CHKERRQ(ierr);
  if (tj->usehistory) {
    ierr = TSHistoryUpdate(tj->tsh,stepnum,time);CHKERRQ(ierr);
  }
  if (tj->lag.caching) tj->lag.Udotcached.time = PETSC_MIN_REAL;
  PetscFunctionReturn(0);
}

/*@
  TSTrajectoryGetNumSteps - Return the number of steps registered in the TSTrajectory via TSTrajectorySet().

  Not collective.

  Input Parameters:
. tj - the trajectory object

  Output Parameter:
. steps - the number of steps

  Level: developer

.seealso: TSTrajectorySet()
@*/
PetscErrorCode TSTrajectoryGetNumSteps(TSTrajectory tj, PetscInt *steps)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(tj,TSTRAJECTORY_CLASSID,1);
  PetscValidIntPointer(steps,2);
  ierr = TSHistoryGetNumSteps(tj->tsh,steps);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
  TSTrajectoryGet - Updates the solution vector of a time stepper object by inquiring the TSTrajectory

  Collective on TS

  Input Parameters:
+ tj      - the trajectory object
. ts      - the time stepper object
- stepnum - the step number

  Output Parameter:
. time    - the time associated with the step number

  Level: developer

  Notes: Usually one does not call this routine, it is called automatically during TSSolve()

.seealso: TSTrajectorySetUp(), TSTrajectoryDestroy(), TSTrajectorySetType(), TSTrajectorySetVariableNames(), TSGetTrajectory(), TSTrajectorySet(), TSTrajectoryGetVecs(), TSGetSolution()
@*/
PetscErrorCode TSTrajectoryGet(TSTrajectory tj,TS ts,PetscInt stepnum,PetscReal *time)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscCheckFalse(!tj,PetscObjectComm((PetscObject)ts),PETSC_ERR_ARG_WRONGSTATE,"TS solver did not save trajectory");
  PetscValidHeaderSpecific(tj,TSTRAJECTORY_CLASSID,1);
  PetscValidHeaderSpecific(ts,TS_CLASSID,2);
  PetscValidLogicalCollectiveInt(tj,stepnum,3);
  PetscValidPointer(time,4);
  PetscCheckFalse(!tj->ops->get,PetscObjectComm((PetscObject)tj),PETSC_ERR_SUP,"TSTrajectory type %s",((PetscObject)tj)->type_name);
  PetscCheckFalse(!tj->setupcalled,PetscObjectComm((PetscObject)tj),PETSC_ERR_ORDER,"TSTrajectorySetUp should be called first");
  PetscCheckFalse(stepnum < 0,PetscObjectComm((PetscObject)tj),PETSC_ERR_PLIB,"Requesting negative step number");
  if (tj->monitor) {
    ierr = PetscViewerASCIIPrintf(tj->monitor,"TSTrajectoryGet: stepnum %D, stages %D\n",stepnum,(PetscInt)!tj->solution_only);CHKERRQ(ierr);
    ierr = PetscViewerFlush(tj->monitor);CHKERRQ(ierr);
  }
  ierr = PetscLogEventBegin(TSTrajectory_Get,tj,ts,0,0);CHKERRQ(ierr);
  ierr = (*tj->ops->get)(tj,ts,stepnum,time);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(TSTrajectory_Get,tj,ts,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
  TSTrajectoryGetVecs - Reconstructs the vector of state and its time derivative using information from the TSTrajectory and, possibly, from the TS

  Collective on TS

  Input Parameters:
+ tj      - the trajectory object
. ts      - the time stepper object (optional)
- stepnum - the requested step number

  Input/Output Parameter:
. time - the time associated with the step number

  Output Parameters:
+ U       - state vector (can be NULL)
- Udot    - time derivative of state vector (can be NULL)

  Level: developer

  Notes: If the step number is PETSC_DECIDE, the time argument is used to inquire the trajectory.
         If the requested time does not match any in the trajectory, Lagrangian interpolations are returned.

.seealso: TSTrajectorySetUp(), TSTrajectoryDestroy(), TSTrajectorySetType(), TSTrajectorySetVariableNames(), TSGetTrajectory(), TSTrajectorySet(), TSTrajectoryGet()
@*/
PetscErrorCode TSTrajectoryGetVecs(TSTrajectory tj,TS ts,PetscInt stepnum,PetscReal *time,Vec U,Vec Udot)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscCheckFalse(!tj,PetscObjectComm((PetscObject)ts),PETSC_ERR_ARG_WRONGSTATE,"TS solver did not save trajectory");
  PetscValidHeaderSpecific(tj,TSTRAJECTORY_CLASSID,1);
  if (ts) PetscValidHeaderSpecific(ts,TS_CLASSID,2);
  PetscValidLogicalCollectiveInt(tj,stepnum,3);
  PetscValidPointer(time,4);
  if (U) PetscValidHeaderSpecific(U,VEC_CLASSID,5);
  if (Udot) PetscValidHeaderSpecific(Udot,VEC_CLASSID,6);
  if (!U && !Udot) PetscFunctionReturn(0);
  PetscCheckFalse(!tj->setupcalled,PetscObjectComm((PetscObject)tj),PETSC_ERR_ORDER,"TSTrajectorySetUp should be called first");
  ierr = PetscLogEventBegin(TSTrajectory_GetVecs,tj,ts,0,0);CHKERRQ(ierr);
  if (tj->monitor) {
    PetscInt pU,pUdot;
    pU    = U ? 1 : 0;
    pUdot = Udot ? 1 : 0;
    ierr  = PetscViewerASCIIPrintf(tj->monitor,"Requested by GetVecs %D %D: stepnum %D, time %g\n",pU,pUdot,stepnum,(double)*time);CHKERRQ(ierr);
    ierr = PetscViewerFlush(tj->monitor);CHKERRQ(ierr);
  }
  if (U && tj->lag.caching) {
    PetscObjectId    id;
    PetscObjectState state;

    ierr = PetscObjectStateGet((PetscObject)U,&state);CHKERRQ(ierr);
    ierr = PetscObjectGetId((PetscObject)U,&id);CHKERRQ(ierr);
    if (stepnum == PETSC_DECIDE) {
      if (id == tj->lag.Ucached.id && *time == tj->lag.Ucached.time && state == tj->lag.Ucached.state) U = NULL;
    } else {
      if (id == tj->lag.Ucached.id && stepnum == tj->lag.Ucached.step && state == tj->lag.Ucached.state) U = NULL;
    }
    if (tj->monitor && !U) {
      ierr = PetscViewerASCIIPushTab(tj->monitor);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(tj->monitor,"State vector cached\n");CHKERRQ(ierr);
      ierr = PetscViewerASCIIPopTab(tj->monitor);CHKERRQ(ierr);
      ierr = PetscViewerFlush(tj->monitor);CHKERRQ(ierr);
    }
  }
  if (Udot && tj->lag.caching) {
    PetscObjectId    id;
    PetscObjectState state;

    ierr = PetscObjectStateGet((PetscObject)Udot,&state);CHKERRQ(ierr);
    ierr = PetscObjectGetId((PetscObject)Udot,&id);CHKERRQ(ierr);
    if (stepnum == PETSC_DECIDE) {
      if (id == tj->lag.Udotcached.id && *time == tj->lag.Udotcached.time && state == tj->lag.Udotcached.state) Udot = NULL;
    } else {
      if (id == tj->lag.Udotcached.id && stepnum == tj->lag.Udotcached.step && state == tj->lag.Udotcached.state) Udot = NULL;
    }
    if (tj->monitor && !Udot) {
      ierr = PetscViewerASCIIPushTab(tj->monitor);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(tj->monitor,"Derivative vector cached\n");CHKERRQ(ierr);
      ierr = PetscViewerASCIIPopTab(tj->monitor);CHKERRQ(ierr);
      ierr = PetscViewerFlush(tj->monitor);CHKERRQ(ierr);
    }
  }
  if (!U && !Udot) {
    ierr = PetscLogEventEnd(TSTrajectory_GetVecs,tj,ts,0,0);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }

  if (stepnum == PETSC_DECIDE || Udot) { /* reverse search for requested time in TSHistory */
    if (tj->monitor) {
      ierr = PetscViewerASCIIPushTab(tj->monitor);CHKERRQ(ierr);
    }
    /* cached states will be updated in the function */
    ierr = TSTrajectoryReconstruct_Private(tj,ts,*time,U,Udot);CHKERRQ(ierr);
    if (tj->monitor) {
      ierr = PetscViewerASCIIPopTab(tj->monitor);CHKERRQ(ierr);
      ierr = PetscViewerFlush(tj->monitor);CHKERRQ(ierr);
    }
  } else if (U) { /* we were asked to load from stepnum, use TSTrajectoryGet */
    TS  fakets = ts;
    Vec U2;

    /* use a fake TS if ts is missing */
    if (!ts) {
      ierr = PetscObjectQuery((PetscObject)tj,"__fake_ts",(PetscObject*)&fakets);CHKERRQ(ierr);
      if (!fakets) {
        ierr = TSCreate(PetscObjectComm((PetscObject)tj),&fakets);CHKERRQ(ierr);
        ierr = PetscObjectCompose((PetscObject)tj,"__fake_ts",(PetscObject)fakets);CHKERRQ(ierr);
        ierr = PetscObjectDereference((PetscObject)fakets);CHKERRQ(ierr);
        ierr = VecDuplicate(U,&U2);CHKERRQ(ierr);
        ierr = TSSetSolution(fakets,U2);CHKERRQ(ierr);
        ierr = PetscObjectDereference((PetscObject)U2);CHKERRQ(ierr);
      }
    }
    ierr = TSTrajectoryGet(tj,fakets,stepnum,time);CHKERRQ(ierr);
    ierr = TSGetSolution(fakets,&U2);CHKERRQ(ierr);
    ierr = VecCopy(U2,U);CHKERRQ(ierr);
    ierr = PetscObjectStateGet((PetscObject)U,&tj->lag.Ucached.state);CHKERRQ(ierr);
    ierr = PetscObjectGetId((PetscObject)U,&tj->lag.Ucached.id);CHKERRQ(ierr);
    tj->lag.Ucached.time = *time;
    tj->lag.Ucached.step = stepnum;
  }
  ierr = PetscLogEventEnd(TSTrajectory_GetVecs,tj,ts,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
   TSTrajectoryViewFromOptions - View from Options

   Collective on TSTrajectory

   Input Parameters:
+  A - the TSTrajectory context
.  obj - Optional object
-  name - command line option

   Level: intermediate
.seealso:  TSTrajectory, TSTrajectoryView, PetscObjectViewFromOptions(), TSTrajectoryCreate()
@*/
PetscErrorCode  TSTrajectoryViewFromOptions(TSTrajectory A,PetscObject obj,const char name[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(A,TSTRAJECTORY_CLASSID,1);
  ierr = PetscObjectViewFromOptions((PetscObject)A,obj,name);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
    TSTrajectoryView - Prints information about the trajectory object

    Collective on TSTrajectory

    Input Parameters:
+   tj - the TSTrajectory context obtained from TSTrajectoryCreate()
-   viewer - visualization context

    Options Database Key:
.   -ts_trajectory_view - calls TSTrajectoryView() at end of TSAdjointStep()

    Notes:
    The available visualization contexts include
+     PETSC_VIEWER_STDOUT_SELF - standard output (default)
-     PETSC_VIEWER_STDOUT_WORLD - synchronized standard
         output where only the first processor opens
         the file.  All other processors send their
         data to the first processor to print.

    The user can open an alternative visualization context with
    PetscViewerASCIIOpen() - output to a specified file.

    Level: developer

.seealso: PetscViewerASCIIOpen()
@*/
PetscErrorCode  TSTrajectoryView(TSTrajectory tj,PetscViewer viewer)
{
  PetscErrorCode ierr;
  PetscBool      iascii;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(tj,TSTRAJECTORY_CLASSID,1);
  if (!viewer) {
    ierr = PetscViewerASCIIGetStdout(PetscObjectComm((PetscObject)tj),&viewer);CHKERRQ(ierr);
  }
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_CLASSID,2);
  PetscCheckSameComm(tj,1,viewer,2);

  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii);CHKERRQ(ierr);
  if (iascii) {
    ierr = PetscObjectPrintClassNamePrefixType((PetscObject)tj,viewer);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  total number of recomputations for adjoint calculation = %D\n",tj->recomps);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  disk checkpoint reads = %D\n",tj->diskreads);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  disk checkpoint writes = %D\n",tj->diskwrites);CHKERRQ(ierr);
    if (tj->ops->view) {
      ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);
      ierr = (*tj->ops->view)(tj,viewer);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPopTab(viewer);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

/*@C
   TSTrajectorySetVariableNames - Sets the name of each component in the solution vector so that it may be saved with the trajectory

   Collective on TSTrajectory

   Input Parameters:
+  tr - the trajectory context
-  names - the names of the components, final string must be NULL

   Level: intermediate

   Note: Fortran interface is not possible because of the string array argument

.seealso: TSTrajectory, TSGetTrajectory()
@*/
PetscErrorCode  TSTrajectorySetVariableNames(TSTrajectory ctx,const char * const *names)
{
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ctx,TSTRAJECTORY_CLASSID,1);
  PetscValidPointer(names,2);
  ierr = PetscStrArrayDestroy(&ctx->names);CHKERRQ(ierr);
  ierr = PetscStrArrayallocpy(names,&ctx->names);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
   TSTrajectorySetTransform - Solution vector will be transformed by provided function before being saved to disk

   Collective on TSLGCtx

   Input Parameters:
+  tj - the TSTrajectory context
.  transform - the transform function
.  destroy - function to destroy the optional context
-  ctx - optional context used by transform function

   Level: intermediate

.seealso:  TSTrajectorySetVariableNames(), TSTrajectory, TSMonitorLGSetTransform()
@*/
PetscErrorCode  TSTrajectorySetTransform(TSTrajectory tj,PetscErrorCode (*transform)(void*,Vec,Vec*),PetscErrorCode (*destroy)(void*),void *tctx)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(tj,TSTRAJECTORY_CLASSID,1);
  tj->transform        = transform;
  tj->transformdestroy = destroy;
  tj->transformctx     = tctx;
  PetscFunctionReturn(0);
}

/*@
  TSTrajectoryCreate - This function creates an empty trajectory object used to store the time dependent solution of an ODE/DAE

  Collective

  Input Parameter:
. comm - the communicator

  Output Parameter:
. tj   - the trajectory object

  Level: developer

  Notes:
    Usually one does not call this routine, it is called automatically when one calls TSSetSaveTrajectory().

.seealso: TSTrajectorySetUp(), TSTrajectoryDestroy(), TSTrajectorySetType(), TSTrajectorySetVariableNames(), TSGetTrajectory(), TSTrajectorySetKeepFiles()
@*/
PetscErrorCode  TSTrajectoryCreate(MPI_Comm comm,TSTrajectory *tj)
{
  TSTrajectory   t;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidPointer(tj,2);
  *tj = NULL;
  ierr = TSInitializePackage();CHKERRQ(ierr);

  ierr = PetscHeaderCreate(t,TSTRAJECTORY_CLASSID,"TSTrajectory","Time stepping","TS",comm,TSTrajectoryDestroy,TSTrajectoryView);CHKERRQ(ierr);
  t->setupcalled = PETSC_FALSE;
  ierr = TSHistoryCreate(comm,&t->tsh);CHKERRQ(ierr);

  t->lag.order            = 1;
  t->lag.L                = NULL;
  t->lag.T                = NULL;
  t->lag.W                = NULL;
  t->lag.WW               = NULL;
  t->lag.TW               = NULL;
  t->lag.TT               = NULL;
  t->lag.caching          = PETSC_TRUE;
  t->lag.Ucached.id       = 0;
  t->lag.Ucached.state    = -1;
  t->lag.Ucached.time     = PETSC_MIN_REAL;
  t->lag.Ucached.step     = PETSC_MAX_INT;
  t->lag.Udotcached.id    = 0;
  t->lag.Udotcached.state = -1;
  t->lag.Udotcached.time  = PETSC_MIN_REAL;
  t->lag.Udotcached.step  = PETSC_MAX_INT;
  t->adjoint_solve_mode   = PETSC_TRUE;
  t->solution_only        = PETSC_FALSE;
  t->keepfiles            = PETSC_FALSE;
  t->usehistory           = PETSC_TRUE;
  *tj  = t;
  ierr = TSTrajectorySetFiletemplate(t,"TS-%06D.bin");CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
  TSTrajectorySetType - Sets the storage method to be used as in a trajectory

  Collective on TS

  Input Parameters:
+ tj   - the TSTrajectory context
. ts   - the TS context
- type - a known method

  Options Database Command:
. -ts_trajectory_type <type> - Sets the method; use -help for a list of available methods (for instance, basic)

   Level: developer

.seealso: TS, TSTrajectoryCreate(), TSTrajectorySetFromOptions(), TSTrajectoryDestroy(), TSTrajectoryGetType()

@*/
PetscErrorCode  TSTrajectorySetType(TSTrajectory tj,TS ts,TSTrajectoryType type)
{
  PetscErrorCode (*r)(TSTrajectory,TS);
  PetscBool      match;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(tj,TSTRAJECTORY_CLASSID,1);
  ierr = PetscObjectTypeCompare((PetscObject)tj,type,&match);CHKERRQ(ierr);
  if (match) PetscFunctionReturn(0);

  ierr = PetscFunctionListFind(TSTrajectoryList,type,&r);CHKERRQ(ierr);
  PetscCheckFalse(!r,PETSC_COMM_SELF,PETSC_ERR_ARG_UNKNOWN_TYPE,"Unknown TSTrajectory type: %s",type);
  if (tj->ops->destroy) {
    ierr = (*(tj)->ops->destroy)(tj);CHKERRQ(ierr);

    tj->ops->destroy = NULL;
  }
  ierr = PetscMemzero(tj->ops,sizeof(*tj->ops));CHKERRQ(ierr);

  ierr = PetscObjectChangeTypeName((PetscObject)tj,type);CHKERRQ(ierr);
  ierr = (*r)(tj,ts);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
  TSTrajectoryGetType - Gets the trajectory type

  Collective on TS

  Input Parameters:
+ tj   - the TSTrajectory context
- ts   - the TS context

  Output Parameters:
. type - a known method

  Level: developer

.seealso: TS, TSTrajectoryCreate(), TSTrajectorySetFromOptions(), TSTrajectoryDestroy(), TSTrajectorySetType()

@*/
PetscErrorCode TSTrajectoryGetType(TSTrajectory tj,TS ts,TSTrajectoryType *type)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(tj,TSTRAJECTORY_CLASSID,1);
  if (type) *type = ((PetscObject)tj)->type_name;
  PetscFunctionReturn(0);
}

PETSC_EXTERN PetscErrorCode TSTrajectoryCreate_Basic(TSTrajectory,TS);
PETSC_EXTERN PetscErrorCode TSTrajectoryCreate_Singlefile(TSTrajectory,TS);
PETSC_EXTERN PetscErrorCode TSTrajectoryCreate_Memory(TSTrajectory,TS);
PETSC_EXTERN PetscErrorCode TSTrajectoryCreate_Visualization(TSTrajectory,TS);

/*@C
  TSTrajectoryRegisterAll - Registers all of the trajectory storage schecmes in the TS package.

  Not Collective

  Level: developer

.seealso: TSTrajectoryRegister()
@*/
PetscErrorCode  TSTrajectoryRegisterAll(void)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (TSTrajectoryRegisterAllCalled) PetscFunctionReturn(0);
  TSTrajectoryRegisterAllCalled = PETSC_TRUE;

  ierr = TSTrajectoryRegister(TSTRAJECTORYBASIC,TSTrajectoryCreate_Basic);CHKERRQ(ierr);
  ierr = TSTrajectoryRegister(TSTRAJECTORYSINGLEFILE,TSTrajectoryCreate_Singlefile);CHKERRQ(ierr);
  ierr = TSTrajectoryRegister(TSTRAJECTORYMEMORY,TSTrajectoryCreate_Memory);CHKERRQ(ierr);
  ierr = TSTrajectoryRegister(TSTRAJECTORYVISUALIZATION,TSTrajectoryCreate_Visualization);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
   TSTrajectoryReset - Resets a trajectory context

   Collective on TSTrajectory

   Input Parameter:
.  tj - the TSTrajectory context obtained from TSTrajectoryCreate()

   Level: developer

.seealso: TSTrajectoryCreate(), TSTrajectorySetUp()
@*/
PetscErrorCode TSTrajectoryReset(TSTrajectory tj)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!tj) PetscFunctionReturn(0);
  PetscValidHeaderSpecific(tj,TSTRAJECTORY_CLASSID,1);
  if (tj->ops->reset) {
    ierr = (*tj->ops->reset)(tj);CHKERRQ(ierr);
  }
  ierr = PetscFree(tj->dirfiletemplate);CHKERRQ(ierr);
  ierr = TSHistoryDestroy(&tj->tsh);CHKERRQ(ierr);
  ierr = TSHistoryCreate(PetscObjectComm((PetscObject)tj),&tj->tsh);CHKERRQ(ierr);
  tj->setupcalled = PETSC_FALSE;
  PetscFunctionReturn(0);
}

/*@
   TSTrajectoryDestroy - Destroys a trajectory context

   Collective on TSTrajectory

   Input Parameter:
.  tj - the TSTrajectory context obtained from TSTrajectoryCreate()

   Level: developer

.seealso: TSTrajectoryCreate(), TSTrajectorySetUp()
@*/
PetscErrorCode TSTrajectoryDestroy(TSTrajectory *tj)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!*tj) PetscFunctionReturn(0);
  PetscValidHeaderSpecific((*tj),TSTRAJECTORY_CLASSID,1);
  if (--((PetscObject)(*tj))->refct > 0) {*tj = NULL; PetscFunctionReturn(0);}

  ierr = TSTrajectoryReset(*tj);CHKERRQ(ierr);
  ierr = TSHistoryDestroy(&(*tj)->tsh);CHKERRQ(ierr);
  ierr = VecDestroyVecs((*tj)->lag.order+1,&(*tj)->lag.W);CHKERRQ(ierr);
  ierr = PetscFree5((*tj)->lag.L,(*tj)->lag.T,(*tj)->lag.WW,(*tj)->lag.TT,(*tj)->lag.TW);CHKERRQ(ierr);
  ierr = VecDestroy(&(*tj)->U);CHKERRQ(ierr);
  ierr = VecDestroy(&(*tj)->Udot);CHKERRQ(ierr);

  if ((*tj)->transformdestroy) {ierr = (*(*tj)->transformdestroy)((*tj)->transformctx);CHKERRQ(ierr);}
  if ((*tj)->ops->destroy) {ierr = (*(*tj)->ops->destroy)((*tj));CHKERRQ(ierr);}
  if (!((*tj)->keepfiles)) {
    PetscMPIInt rank;
    MPI_Comm    comm;

    ierr = PetscObjectGetComm((PetscObject)(*tj),&comm);CHKERRQ(ierr);
    ierr = MPI_Comm_rank(comm,&rank);CHKERRMPI(ierr);
    if (rank == 0 && (*tj)->dirname) { /* we own the directory, so we run PetscRMTree on it */
      ierr = PetscRMTree((*tj)->dirname);CHKERRQ(ierr);
    }
  }
  ierr = PetscStrArrayDestroy(&(*tj)->names);CHKERRQ(ierr);
  ierr = PetscFree((*tj)->dirname);CHKERRQ(ierr);
  ierr = PetscFree((*tj)->filetemplate);CHKERRQ(ierr);
  ierr = PetscHeaderDestroy(tj);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
  TSTrajectorySetTypeFromOptions_Private - Sets the type of ts from user options.

  Collective on TSTrajectory

  Input Parameter:
+ tj - the TSTrajectory context
- ts - the TS context

  Options Database Keys:
. -ts_trajectory_type <type> - TSTRAJECTORYBASIC, TSTRAJECTORYMEMORY, TSTRAJECTORYSINGLEFILE, TSTRAJECTORYVISUALIZATION

  Level: developer

.seealso: TSTrajectorySetFromOptions(), TSTrajectorySetType()
*/
static PetscErrorCode TSTrajectorySetTypeFromOptions_Private(PetscOptionItems *PetscOptionsObject,TSTrajectory tj,TS ts)
{
  PetscBool      opt;
  const char     *defaultType;
  char           typeName[256];
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (((PetscObject)tj)->type_name) defaultType = ((PetscObject)tj)->type_name;
  else defaultType = TSTRAJECTORYBASIC;

  ierr = TSTrajectoryRegisterAll();CHKERRQ(ierr);
  ierr = PetscOptionsFList("-ts_trajectory_type","TSTrajectory method","TSTrajectorySetType",TSTrajectoryList,defaultType,typeName,256,&opt);CHKERRQ(ierr);
  if (opt) {
    ierr = TSTrajectorySetType(tj,ts,typeName);CHKERRQ(ierr);
  } else {
    ierr = TSTrajectorySetType(tj,ts,defaultType);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*@
   TSTrajectorySetUseHistory - Use TSHistory in TSTrajectory

   Collective on TSTrajectory

   Input Parameters:
+  tj - the TSTrajectory context
-  flg - PETSC_TRUE to save, PETSC_FALSE to disable

   Options Database Keys:
.  -ts_trajectory_use_history - have it use TSHistory

   Level: advanced

.seealso: TSTrajectoryCreate(), TSTrajectoryDestroy(), TSTrajectorySetUp()
@*/
PetscErrorCode TSTrajectorySetUseHistory(TSTrajectory tj,PetscBool flg)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(tj,TSTRAJECTORY_CLASSID,1);
  PetscValidLogicalCollectiveBool(tj,flg,2);
  tj->usehistory = flg;
  PetscFunctionReturn(0);
}

/*@
   TSTrajectorySetMonitor - Monitor the schedules generated by the checkpointing controller

   Collective on TSTrajectory

   Input Parameters:
+  tj - the TSTrajectory context
-  flg - PETSC_TRUE to active a monitor, PETSC_FALSE to disable

   Options Database Keys:
.  -ts_trajectory_monitor - print TSTrajectory information

   Level: developer

.seealso: TSTrajectoryCreate(), TSTrajectoryDestroy(), TSTrajectorySetUp()
@*/
PetscErrorCode TSTrajectorySetMonitor(TSTrajectory tj,PetscBool flg)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(tj,TSTRAJECTORY_CLASSID,1);
  PetscValidLogicalCollectiveBool(tj,flg,2);
  if (flg) tj->monitor = PETSC_VIEWER_STDOUT_(PetscObjectComm((PetscObject)tj));
  else tj->monitor = NULL;
  PetscFunctionReturn(0);
}

/*@
   TSTrajectorySetKeepFiles - Keep the files generated by the TSTrajectory

   Collective on TSTrajectory

   Input Parameters:
+  tj - the TSTrajectory context
-  flg - PETSC_TRUE to save, PETSC_FALSE to disable

   Options Database Keys:
.  -ts_trajectory_keep_files - have it keep the files

   Notes:
    By default the TSTrajectory used for adjoint computations, TSTRAJECTORYBASIC, removes the files it generates at the end of the run. This causes the files to be kept.

   Level: advanced

.seealso: TSTrajectoryCreate(), TSTrajectoryDestroy(), TSTrajectorySetUp(), TSTrajectorySetMonitor()
@*/
PetscErrorCode TSTrajectorySetKeepFiles(TSTrajectory tj,PetscBool flg)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(tj,TSTRAJECTORY_CLASSID,1);
  PetscValidLogicalCollectiveBool(tj,flg,2);
  tj->keepfiles = flg;
  PetscFunctionReturn(0);
}

/*@C
   TSTrajectorySetDirname - Specify the name of the directory where disk checkpoints are stored.

   Collective on TSTrajectory

   Input Parameters:
+  tj      - the TSTrajectory context
-  dirname - the directory name

   Options Database Keys:
.  -ts_trajectory_dirname - set the directory name

   Notes:
    The final location of the files is determined by dirname/filetemplate where filetemplate was provided by TSTrajectorySetFiletemplate()

   Level: developer

.seealso: TSTrajectorySetFiletemplate(),TSTrajectorySetUp()
@*/
PetscErrorCode TSTrajectorySetDirname(TSTrajectory tj,const char dirname[])
{
  PetscErrorCode ierr;
  PetscBool      flg;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(tj,TSTRAJECTORY_CLASSID,1);
  ierr = PetscStrcmp(tj->dirname,dirname,&flg);CHKERRQ(ierr);
  if (!flg && tj->dirfiletemplate) {
    SETERRQ(PetscObjectComm((PetscObject)tj),PETSC_ERR_ARG_WRONGSTATE,"Cannot set directoryname after TSTrajectory has been setup");
  }
  ierr = PetscFree(tj->dirname);CHKERRQ(ierr);
  ierr = PetscStrallocpy(dirname,&tj->dirname);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
   TSTrajectorySetFiletemplate - Specify the name template for the files storing checkpoints.

   Collective on TSTrajectory

   Input Parameters:
+  tj      - the TSTrajectory context
-  filetemplate - the template

   Options Database Keys:
.  -ts_trajectory_file_template - set the file name template

   Notes:
    The name template should be of the form, for example filename-%06D.bin It should not begin with a leading /

   The final location of the files is determined by dirname/filetemplate where dirname was provided by TSTrajectorySetDirname(). The %06D is replaced by the
   timestep counter

   Level: developer

.seealso: TSTrajectorySetDirname(),TSTrajectorySetUp()
@*/
PetscErrorCode TSTrajectorySetFiletemplate(TSTrajectory tj,const char filetemplate[])
{
  PetscErrorCode ierr;
  const char     *ptr,*ptr2;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(tj,TSTRAJECTORY_CLASSID,1);
  PetscCheckFalse(tj->dirfiletemplate,PetscObjectComm((PetscObject)tj),PETSC_ERR_ARG_WRONGSTATE,"Cannot set filetemplate after TSTrajectory has been setup");

  PetscCheckFalse(!filetemplate[0],PetscObjectComm((PetscObject)tj),PETSC_ERR_USER,"-ts_trajectory_file_template requires a file name template, e.g. filename-%%06D.bin");
  /* Do some cursory validation of the input. */
  ierr = PetscStrstr(filetemplate,"%",(char**)&ptr);CHKERRQ(ierr);
  PetscCheckFalse(!ptr,PetscObjectComm((PetscObject)tj),PETSC_ERR_USER,"-ts_trajectory_file_template requires a file name template, e.g. filename-%%06D.bin");
  for (ptr++; ptr && *ptr; ptr++) {
    ierr = PetscStrchr("DdiouxX",*ptr,(char**)&ptr2);CHKERRQ(ierr);
    PetscCheckFalse(!ptr2 && (*ptr < '0' || '9' < *ptr),PetscObjectComm((PetscObject)tj),PETSC_ERR_USER,"Invalid file template argument to -ts_trajectory_file_template, should look like filename-%%06D.bin");
    if (ptr2) break;
  }
  ierr = PetscFree(tj->filetemplate);CHKERRQ(ierr);
  ierr = PetscStrallocpy(filetemplate,&tj->filetemplate);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
   TSTrajectorySetFromOptions - Sets various TSTrajectory parameters from user options.

   Collective on TSTrajectory

   Input Parameters:
+  tj - the TSTrajectory context obtained from TSTrajectoryCreate()
-  ts - the TS context

   Options Database Keys:
+  -ts_trajectory_type <type> - TSTRAJECTORYBASIC, TSTRAJECTORYMEMORY, TSTRAJECTORYSINGLEFILE, TSTRAJECTORYVISUALIZATION
.  -ts_trajectory_keep_files <true,false> - keep the files generated by the code after the program ends. This is true by default for TSTRAJECTORYSINGLEFILE, TSTRAJECTORYVISUALIZATION
-  -ts_trajectory_monitor - print TSTrajectory information

   Level: developer

   Notes:
    This is not normally called directly by users

.seealso: TSSetSaveTrajectory(), TSTrajectorySetUp()
@*/
PetscErrorCode  TSTrajectorySetFromOptions(TSTrajectory tj,TS ts)
{
  PetscBool      set,flg;
  char           dirname[PETSC_MAX_PATH_LEN],filetemplate[PETSC_MAX_PATH_LEN];
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(tj,TSTRAJECTORY_CLASSID,1);
  if (ts) PetscValidHeaderSpecific(ts,TS_CLASSID,2);
  ierr = PetscObjectOptionsBegin((PetscObject)tj);CHKERRQ(ierr);
  ierr = TSTrajectorySetTypeFromOptions_Private(PetscOptionsObject,tj,ts);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-ts_trajectory_use_history","Turn on/off usage of TSHistory",NULL,tj->usehistory,&tj->usehistory,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-ts_trajectory_monitor","Print checkpointing schedules","TSTrajectorySetMonitor",tj->monitor ? PETSC_TRUE:PETSC_FALSE,&flg,&set);CHKERRQ(ierr);
  if (set) {ierr = TSTrajectorySetMonitor(tj,flg);CHKERRQ(ierr);}
  ierr = PetscOptionsInt("-ts_trajectory_reconstruction_order","Interpolation order for reconstruction",NULL,tj->lag.order,&tj->lag.order,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-ts_trajectory_reconstruction_caching","Turn on/off caching of TSTrajectoryGetVecs input",NULL,tj->lag.caching,&tj->lag.caching,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-ts_trajectory_adjointmode","Instruct the trajectory that will be used in a TSAdjointSolve()",NULL,tj->adjoint_solve_mode,&tj->adjoint_solve_mode,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-ts_trajectory_solution_only","Checkpoint solution only","TSTrajectorySetSolutionOnly",tj->solution_only,&tj->solution_only,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-ts_trajectory_keep_files","Keep any trajectory files generated during the run","TSTrajectorySetKeepFiles",tj->keepfiles,&flg,&set);CHKERRQ(ierr);
  if (set) {ierr = TSTrajectorySetKeepFiles(tj,flg);CHKERRQ(ierr);}

  ierr = PetscOptionsString("-ts_trajectory_dirname","Directory name for TSTrajectory file","TSTrajectorySetDirname",NULL,dirname,sizeof(dirname)-14,&set);CHKERRQ(ierr);
  if (set) {
    ierr = TSTrajectorySetDirname(tj,dirname);CHKERRQ(ierr);
  }

  ierr = PetscOptionsString("-ts_trajectory_file_template","Template for TSTrajectory file name, use filename-%06D.bin","TSTrajectorySetFiletemplate",NULL,filetemplate,sizeof(filetemplate),&set);CHKERRQ(ierr);
  if (set) {
    ierr = TSTrajectorySetFiletemplate(tj,filetemplate);CHKERRQ(ierr);
  }

  /* Handle specific TSTrajectory options */
  if (tj->ops->setfromoptions) {
    ierr = (*tj->ops->setfromoptions)(PetscOptionsObject,tj);CHKERRQ(ierr);
  }
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
   TSTrajectorySetUp - Sets up the internal data structures, e.g. stacks, for the later use
   of a TS trajectory.

   Collective on TS

   Input Parameters:
+  ts - the TS context obtained from TSCreate()
-  tj - the TS trajectory context

   Level: developer

.seealso: TSSetSaveTrajectory(), TSTrajectoryCreate(), TSTrajectoryDestroy()
@*/
PetscErrorCode  TSTrajectorySetUp(TSTrajectory tj,TS ts)
{
  PetscErrorCode ierr;
  size_t         s1,s2;

  PetscFunctionBegin;
  if (!tj) PetscFunctionReturn(0);
  PetscValidHeaderSpecific(tj,TSTRAJECTORY_CLASSID,1);
  if (ts) PetscValidHeaderSpecific(ts,TS_CLASSID,2);
  if (tj->setupcalled) PetscFunctionReturn(0);

  ierr = PetscLogEventBegin(TSTrajectory_SetUp,tj,ts,0,0);CHKERRQ(ierr);
  if (!((PetscObject)tj)->type_name) {
    ierr = TSTrajectorySetType(tj,ts,TSTRAJECTORYBASIC);CHKERRQ(ierr);
  }
  if (tj->ops->setup) {
    ierr = (*tj->ops->setup)(tj,ts);CHKERRQ(ierr);
  }

  tj->setupcalled = PETSC_TRUE;

  /* Set the counters to zero */
  tj->recomps    = 0;
  tj->diskreads  = 0;
  tj->diskwrites = 0;
  ierr = PetscStrlen(tj->dirname,&s1);CHKERRQ(ierr);
  ierr = PetscStrlen(tj->filetemplate,&s2);CHKERRQ(ierr);
  ierr = PetscFree(tj->dirfiletemplate);CHKERRQ(ierr);
  ierr = PetscMalloc((s1 + s2 + 10)*sizeof(char),&tj->dirfiletemplate);CHKERRQ(ierr);
  ierr = PetscSNPrintf(tj->dirfiletemplate,s1+s2+10,"%s/%s",tj->dirname,tj->filetemplate);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(TSTrajectory_SetUp,tj,ts,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
   TSTrajectorySetSolutionOnly - Tells the trajectory to store just the solution, and not any intermediate stage also.

   Collective on TSTrajectory

   Input Parameters:
+  tj  - the TS trajectory context
-  flg - the boolean flag

   Level: developer

.seealso: TSSetSaveTrajectory(), TSTrajectoryCreate(), TSTrajectoryDestroy(), TSTrajectoryGetSolutionOnly()
@*/
PetscErrorCode TSTrajectorySetSolutionOnly(TSTrajectory tj,PetscBool solution_only)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(tj,TSTRAJECTORY_CLASSID,1);
  PetscValidLogicalCollectiveBool(tj,solution_only,2);
  tj->solution_only = solution_only;
  PetscFunctionReturn(0);
}

/*@
   TSTrajectoryGetSolutionOnly - Gets the value set with TSTrajectorySetSolutionOnly.

   Logically collective on TSTrajectory

   Input Parameter:
.  tj  - the TS trajectory context

   Output Parameter:
.  flg - the boolean flag

   Level: developer

.seealso: TSSetSaveTrajectory(), TSTrajectoryCreate(), TSTrajectoryDestroy(), TSTrajectorySetSolutionOnly()
@*/
PetscErrorCode TSTrajectoryGetSolutionOnly(TSTrajectory tj,PetscBool *solution_only)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(tj,TSTRAJECTORY_CLASSID,1);
  PetscValidPointer(solution_only,2);
  *solution_only = tj->solution_only;
  PetscFunctionReturn(0);
}

/*@
   TSTrajectoryGetUpdatedHistoryVecs - Get updated state and time-derivative history vectors.

   Collective on TSTrajectory

   Input Parameters:
+  tj   - the TS trajectory context
.  ts   - the TS solver context
-  time - the requested time

   Output Parameters:
+  U    - state vector at given time (can be interpolated)
-  Udot - time-derivative vector at given time (can be interpolated)

   Level: developer

   Notes: The vectors are interpolated if time does not match any time step stored in the TSTrajectory(). Pass NULL to not request a vector.
          This function differs from TSTrajectoryGetVecs since the vectors obtained cannot be modified, and they need to be returned by
          calling TSTrajectoryRestoreUpdatedHistoryVecs().

.seealso: TSSetSaveTrajectory(), TSTrajectoryCreate(), TSTrajectoryDestroy(), TSTrajectoryRestoreUpdatedHistoryVecs(), TSTrajectoryGetVecs()
@*/
PetscErrorCode TSTrajectoryGetUpdatedHistoryVecs(TSTrajectory tj, TS ts, PetscReal time, Vec *U, Vec *Udot)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(tj,TSTRAJECTORY_CLASSID,1);
  PetscValidHeaderSpecific(ts,TS_CLASSID,2);
  PetscValidLogicalCollectiveReal(tj,time,3);
  if (U) PetscValidPointer(U,4);
  if (Udot) PetscValidPointer(Udot,5);
  if (U && !tj->U) {
    DM dm;

    ierr = TSGetDM(ts,&dm);CHKERRQ(ierr);
    ierr = DMCreateGlobalVector(dm,&tj->U);CHKERRQ(ierr);
  }
  if (Udot && !tj->Udot) {
    DM dm;

    ierr = TSGetDM(ts,&dm);CHKERRQ(ierr);
    ierr = DMCreateGlobalVector(dm,&tj->Udot);CHKERRQ(ierr);
  }
  ierr = TSTrajectoryGetVecs(tj,ts,PETSC_DECIDE,&time,U ? tj->U : NULL,Udot ? tj->Udot : NULL);CHKERRQ(ierr);
  if (U) {
    ierr = VecLockReadPush(tj->U);CHKERRQ(ierr);
    *U   = tj->U;
  }
  if (Udot) {
    ierr  = VecLockReadPush(tj->Udot);CHKERRQ(ierr);
    *Udot = tj->Udot;
  }
  PetscFunctionReturn(0);
}

/*@
   TSTrajectoryRestoreUpdatedHistoryVecs - Restores updated state and time-derivative history vectors obtained with TSTrajectoryGetUpdatedHistoryVecs().

   Collective on TSTrajectory

   Input Parameters:
+  tj   - the TS trajectory context
.  U    - state vector at given time (can be interpolated)
-  Udot - time-derivative vector at given time (can be interpolated)

   Level: developer

.seealso: TSTrajectoryGetUpdatedHistoryVecs()
@*/
PetscErrorCode TSTrajectoryRestoreUpdatedHistoryVecs(TSTrajectory tj, Vec *U, Vec *Udot)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(tj,TSTRAJECTORY_CLASSID,1);
  if (U) PetscValidHeaderSpecific(*U,VEC_CLASSID,2);
  if (Udot) PetscValidHeaderSpecific(*Udot,VEC_CLASSID,3);
  PetscCheckFalse(U && *U != tj->U,PetscObjectComm((PetscObject)*U),PETSC_ERR_USER,"U was not obtained from TSTrajectoryGetUpdatedHistoryVecs()");
  PetscCheckFalse(Udot && *Udot != tj->Udot,PetscObjectComm((PetscObject)*Udot),PETSC_ERR_USER,"Udot was not obtained from TSTrajectoryGetUpdatedHistoryVecs()");
  if (U) {
    ierr = VecLockReadPop(tj->U);CHKERRQ(ierr);
    *U   = NULL;
  }
  if (Udot) {
    ierr  = VecLockReadPop(tj->Udot);CHKERRQ(ierr);
    *Udot = NULL;
  }
  PetscFunctionReturn(0);
}
