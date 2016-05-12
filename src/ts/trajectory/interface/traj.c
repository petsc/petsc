
#include <petsc/private/tsimpl.h>        /*I "petscts.h"  I*/

PetscFunctionList TSTrajectoryList              = NULL;
PetscBool         TSTrajectoryRegisterAllCalled = PETSC_FALSE;
PetscClassId      TSTRAJECTORY_CLASSID;
PetscLogEvent     TSTrajectory_Set, TSTrajectory_Get;

#undef __FUNCT__
#define __FUNCT__ "TSTrajectoryRegister"
/*@C
  TSTrajectoryRegister - Adds a way of storing trajectories to the TS package

  Not Collective

  Input Parameters:
+ name        - The name of a new user-defined creation routine
- create_func - The creation routine itself

  Notes:
  TSTrajectoryRegister() may be called multiple times to add several user-defined tses.

  Level: advanced

.keywords: TS, register

.seealso: TSTrajectoryRegisterAll(), TSTrajectoryRegisterDestroy()
@*/
PetscErrorCode TSTrajectoryRegister(const char sname[],PetscErrorCode (*function)(TSTrajectory,TS))
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFunctionListAdd(&TSTrajectoryList,sname,function);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSTrajectorySet"
PetscErrorCode TSTrajectorySet(TSTrajectory tj,TS ts,PetscInt stepnum,PetscReal time,Vec X)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!tj) PetscFunctionReturn(0);
  ierr = PetscLogEventBegin(TSTrajectory_Set,tj,ts,0,0);CHKERRQ(ierr);
  ierr = (*tj->ops->set)(tj,ts,stepnum,time,X);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(TSTrajectory_Set,tj,ts,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSTrajectoryGet"
PetscErrorCode TSTrajectoryGet(TSTrajectory tj,TS ts,PetscInt stepnum,PetscReal *time)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!tj) SETERRQ(PetscObjectComm((PetscObject)ts),PETSC_ERR_ARG_WRONGSTATE,"TS solver did not save trajectory");
  ierr = PetscLogEventBegin(TSTrajectory_Get,tj,ts,0,0);CHKERRQ(ierr);
  ierr = (*tj->ops->get)(tj,ts,stepnum,time);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(TSTrajectory_Get,tj,ts,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSTrajectoryView"
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

    Level: beginner

.keywords: TS, timestep, view

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
      ierr = (*tj->ops->view)(tj,viewer);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "TSTrajectoryCreate"
/*@C
  TSTrajectoryCreate - This function creates an empty trajectory object used to store the time dependent solution of an ODE/DAE

  Collective on MPI_Comm

  Input Parameter:
. comm - The communicator

  Output Parameter:
. tj   - The trajectory object

  Level: advanced

  Notes: Usually one does not call this routine, it is called automatically when one calls TSSetSaveTrajectory(). One can call
   TSGetTrajectory() to access the created trajectory.

.keywords: TS, create
.seealso: TSSetType(), TSSetUp(), TSDestroy(), TSSetProblemType(), TSGetTrajectory()
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
  *tj = t;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSTrajectorySetType"
/*@C
  TSTrajectorySetType - Sets the storage method to be used as in a trajectory

  Collective on TS

  Input Parameters:
+ ts   - The TS context
- type - A known method

  Options Database Command:
. -ts_trajectory_type <type> - Sets the method; use -help for a list of available methods (for instance, basic)

   Level: intermediate

.keywords: TS, set, type

.seealso: TS, TSSolve(), TSCreate(), TSSetFromOptions(), TSDestroy(), TSType

@*/
PetscErrorCode  TSTrajectorySetType(TSTrajectory tj,TS ts,const TSTrajectoryType type)
{
  PetscErrorCode (*r)(TSTrajectory,TS);
  PetscBool      match;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(tj,TSTRAJECTORY_CLASSID,1);
  ierr = PetscObjectTypeCompare((PetscObject)tj,type,&match);CHKERRQ(ierr);
  if (match) PetscFunctionReturn(0);

  ierr = PetscFunctionListFind(TSTrajectoryList,type,&r);CHKERRQ(ierr);
  if (!r) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_UNKNOWN_TYPE,"Unknown TSTrajectory type: %s",type);
  if (tj->ops->destroy) {
    ierr = (*(tj)->ops->destroy)(tj);CHKERRQ(ierr);

    tj->ops->destroy = NULL;
  }
  ierr = PetscMemzero(tj->ops,sizeof(*tj->ops));CHKERRQ(ierr);

  ierr = PetscObjectChangeTypeName((PetscObject)tj,type);CHKERRQ(ierr);
  ierr = (*r)(tj,ts);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PETSC_EXTERN PetscErrorCode TSTrajectoryCreate_Basic(TSTrajectory,TS);
PETSC_EXTERN PetscErrorCode TSTrajectoryCreate_Singlefile(TSTrajectory,TS);
PETSC_EXTERN PetscErrorCode TSTrajectoryCreate_Memory(TSTrajectory,TS);
PETSC_EXTERN PetscErrorCode TSTrajectoryCreate_Visualization(TSTrajectory,TS);

#undef __FUNCT__
#define __FUNCT__ "TSTrajectoryRegisterAll"
/*@C
  TSTrajectoryRegisterAll - Registers all of the trajectory storage schecmes in the TS package.

  Not Collective

  Level: advanced

.keywords: TS, timestepper, register, all
.seealso: TSCreate(), TSRegister(), TSRegisterDestroy()
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

#undef __FUNCT__
#define __FUNCT__ "TSTrajectoryDestroy"
/*@
   TSTrajectoryDestroy - Destroys a trajectory context

   Collective on TSTrajectory

   Input Parameter:
.  ts - the TSTrajectory context obtained from TSTrajectoryCreate()

   Level: advanced

.keywords: TS, timestepper, destroy

.seealso: TSCreate(), TSSetUp(), TSSolve()
@*/
PetscErrorCode  TSTrajectoryDestroy(TSTrajectory *tj)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!*tj) PetscFunctionReturn(0);
  PetscValidHeaderSpecific((*tj),TSTRAJECTORY_CLASSID,1);
  if (--((PetscObject)(*tj))->refct > 0) {*tj = 0; PetscFunctionReturn(0);}

  if ((*tj)->ops->destroy) {ierr = (*(*tj)->ops->destroy)((*tj));CHKERRQ(ierr);}
  ierr = PetscHeaderDestroy(tj);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSTrajectorySetTypeFromOptions_Private"
/*
  TSTrajectorySetTypeFromOptions_Private - Sets the type of ts from user options.

  Collective on TSTrajectory

  Input Parameter:
. tj - TSTrajectory

  Level: intermediate

.keywords: TS, set, options, database, type
.seealso: TSSetFromOptions(), TSSetType()
*/
static PetscErrorCode TSTrajectorySetTypeFromOptions_Private(PetscOptionItems *PetscOptionsObject,TSTrajectory tj,TS ts)
{
  PetscBool      opt;
  const char     *defaultType;
  char           typeName[256];
  PetscBool      flg;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (((PetscObject)tj)->type_name) defaultType = ((PetscObject)tj)->type_name;
  else defaultType = TSTRAJECTORYBASIC;

  ierr = TSTrajectoryRegisterAll();CHKERRQ(ierr);
  ierr = PetscOptionsFList("-ts_trajectory_type","TSTrajectory method"," TSTrajectorySetType",TSTrajectoryList,defaultType,typeName,256,&opt);CHKERRQ(ierr);
  if (opt) {
    ierr = PetscStrcmp(typeName,TSTRAJECTORYMEMORY,&flg);CHKERRQ(ierr);
    ierr = TSTrajectorySetType(tj,ts,typeName);CHKERRQ(ierr);
  } else {
    ierr = TSTrajectorySetType(tj,ts,defaultType);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSTrajectorySetFromOptions"
/*@
   TSTrajectorySetFromOptions - Sets various TSTrajectory parameters from user options.

   Collective on TSTrajectory

   Input Parameter:
.  tj - the TSTrajectory context obtained from TSTrajectoryCreate()

   Options Database Keys:
.  -ts_trajectory_type <type> - TSTRAJECTORYBASIC
.  -ts_trajectory_max_cps <int>

   Level: advanced

   Notes: This is not normally called directly by users

.keywords: TS, timestep, set, options, database, trajectory

.seealso: TSGetType(), TSSetSaveTrajectory(), TSGetTrajectory()
@*/
PetscErrorCode  TSTrajectorySetFromOptions(TSTrajectory tj,TS ts)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(tj,TSTRAJECTORY_CLASSID,1);
  PetscValidHeaderSpecific(ts,TS_CLASSID,2);
  ierr = PetscObjectOptionsBegin((PetscObject)tj);CHKERRQ(ierr);
  ierr = TSTrajectorySetTypeFromOptions_Private(PetscOptionsObject,tj,ts);CHKERRQ(ierr);
    /* Handle specific TS options */
  if (tj->ops->setfromoptions) {
    ierr = (*tj->ops->setfromoptions)(PetscOptionsObject,tj);CHKERRQ(ierr);
  }
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSTrajectorySetUp"
/*@
   TSTrajectorySetUp - Sets up the internal data structures, e.g. stacks, for the later use
   of a TS trajectory.

   Collective on TS

   Input Parameter:
.  ts - the TS context obtained from TSCreate()
.  tj - the TS trajectory context

   Level: advanced

.keywords: TS, setup, checkpoint

.seealso: TSSetSaveTrajectory(), TSTrajectoryCreate(), TSTrajectoryDestroy()
@*/
PetscErrorCode  TSTrajectorySetUp(TSTrajectory tj,TS ts)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!tj) PetscFunctionReturn(0);
  PetscValidHeaderSpecific(tj,TSTRAJECTORY_CLASSID,1);
  PetscValidHeaderSpecific(ts,TS_CLASSID,2);
  if (tj->setupcalled) PetscFunctionReturn(0);

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
  PetscFunctionReturn(0);
}
