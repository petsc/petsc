
#include <petsc/private/tsimpl.h>        /*I "petscts.h"  I*/

PetscFunctionList TSTrajectoryList              = NULL;
PetscBool         TSTrajectoryRegisterAllCalled = PETSC_FALSE;
PetscClassId      TSTRAJECTORY_CLASSID;

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
PetscErrorCode  TSTrajectoryRegister(const char sname[], PetscErrorCode (*function)(TSTrajectory))
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
  ierr = (*tj->ops->set)(tj,ts,stepnum,time,X);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSTrajectoryGet"
PetscErrorCode TSTrajectoryGet(TSTrajectory tj,TS ts,PetscInt stepnum,PetscReal *time)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!tj) SETERRQ(PetscObjectComm((PetscObject)ts),PETSC_ERR_ARG_WRONGSTATE,"TS solver did not save trajectory");
  ierr = (*tj->ops->get)(tj,ts,stepnum,time);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSTrajectoryView"
/*@C
    TSTrajectoryView - Prints information about the trajectory object

    Collective on TSTrajectory

    Input Parameters:
+   ts - the TSTrajectory context obtained from TSTrajectoryCreate()
-   viewer - visualization context

    Options Database Key:
.   -ts_view - calls TSView() at end of TSStep()

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
PetscErrorCode  TSTrajectoryView(TSTrajectory ts,PetscViewer viewer)
{
  PetscErrorCode ierr;
  PetscBool      iascii;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  if (!viewer) {
    ierr = PetscViewerASCIIGetStdout(PetscObjectComm((PetscObject)ts),&viewer);CHKERRQ(ierr);
  }
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_CLASSID,2);
  PetscCheckSameComm(ts,1,viewer,2);

  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii);CHKERRQ(ierr);
  if (iascii) {
    ierr = PetscObjectPrintClassNamePrefixType((PetscObject)ts,viewer);CHKERRQ(ierr);
    if (ts->ops->view) {
      ierr = (*ts->ops->view)(ts,viewer);CHKERRQ(ierr);
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
. tstra   - The trajectory object

  Level: advanced

  Notes: Usually one does not call this routine, it is called automatically when one calls TSSetSaveTrajectory(). One can call
   TSGetTrajectory() to access the created trajectory.

.keywords: TS, create
.seealso: TSSetType(), TSSetUp(), TSDestroy(), TSSetProblemType(), TSGetTrajectory()
@*/
PetscErrorCode  TSTrajectoryCreate(MPI_Comm comm, TSTrajectory *tstra)
{
  TSTrajectory   t;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidPointer(tstra,1);
  *tstra = NULL;
  ierr = TSInitializePackage();CHKERRQ(ierr);

  ierr = PetscHeaderCreate(t, TSTRAJECTORY_CLASSID, "TSTrajectory", "Time stepping", "TS", comm, TSTrajectoryDestroy, TSTrajectoryView);CHKERRQ(ierr);
  *tstra = t;
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
. -tstrajectory_type <type> - Sets the method; use -help for a list of available methods (for instance, basic)

   Level: intermediate

.keywords: TS, set, type

.seealso: TS, TSSolve(), TSCreate(), TSSetFromOptions(), TSDestroy(), TSType

@*/
PetscErrorCode  TSTrajectorySetType(TSTrajectory ts,const TSTrajectoryType type)
{
  PetscErrorCode (*r)(TSTrajectory);
  PetscBool      match;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts, TSTRAJECTORY_CLASSID,1);
  ierr = PetscObjectTypeCompare((PetscObject) ts, type, &match);CHKERRQ(ierr);
  if (match) PetscFunctionReturn(0);

  ierr = PetscFunctionListFind(TSTrajectoryList,type,&r);CHKERRQ(ierr);
  if (!r) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_UNKNOWN_TYPE, "Unknown TSTrajectory type: %s", type);
  if (ts->ops->destroy) {
    ierr = (*(ts)->ops->destroy)(ts);CHKERRQ(ierr);

    ts->ops->destroy = NULL;
  }
  ierr = PetscMemzero(ts->ops,sizeof(*ts->ops));CHKERRQ(ierr);

  ierr = PetscObjectChangeTypeName((PetscObject)ts, type);CHKERRQ(ierr);
  ierr = (*r)(ts);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PETSC_EXTERN PetscErrorCode TSTrajectoryCreate_Basic(TSTrajectory);
PETSC_EXTERN PetscErrorCode TSTrajectoryCreate_Singlefile(TSTrajectory);

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
  TSTrajectoryRegisterAllCalled = PETSC_TRUE;

  ierr = TSTrajectoryRegister(TSTRAJECTORYBASIC,TSTrajectoryCreate_Basic);CHKERRQ(ierr);
  ierr = TSTrajectoryRegister(TSTRAJECTORYSINGLEFILE,TSTrajectoryCreate_Singlefile);CHKERRQ(ierr);  
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
PetscErrorCode  TSTrajectoryDestroy(TSTrajectory *ts)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!*ts) PetscFunctionReturn(0);
  PetscValidHeaderSpecific((*ts),TSTRAJECTORY_CLASSID,1);
  if (--((PetscObject)(*ts))->refct > 0) {*ts = 0; PetscFunctionReturn(0);}

  if ((*ts)->ops->destroy) {ierr = (*(*ts)->ops->destroy)((*ts));CHKERRQ(ierr);}
  ierr = PetscHeaderDestroy(ts);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSTrajectorySetTypeFromOptions_Private"
/*
  TSTrajectorySetTypeFromOptions_Private - Sets the type of ts from user options.

  Collective on TSTrajectory

  Input Parameter:
. ts - The ts

  Level: intermediate

.keywords: TS, set, options, database, type
.seealso: TSSetFromOptions(), TSSetType()
*/
static PetscErrorCode TSTrajectorySetTypeFromOptions_Private(PetscOptionItems *PetscOptionsObject,TSTrajectory ts)
{
  PetscBool      opt;
  const char     *defaultType;
  char           typeName[256];
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (((PetscObject)ts)->type_name) defaultType = ((PetscObject)ts)->type_name;
  else defaultType = TSTRAJECTORYBASIC;

  if (!TSRegisterAllCalled) {ierr = TSTrajectoryRegisterAll();CHKERRQ(ierr);}
  ierr = PetscOptionsFList("-tstrajectory_type", "TSTrajectory method"," TSTrajectorySetType", TSTrajectoryList, defaultType, typeName, 256, &opt);CHKERRQ(ierr);
  if (opt) {
    ierr = TSTrajectorySetType(ts, typeName);CHKERRQ(ierr);
  } else {
    ierr = TSTrajectorySetType(ts, defaultType);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSTrajectorySetFromOptions"
/*@
   TSTrajectorySetFromOptions - Sets various TSTrajectory parameters from user options.

   Collective on TSTrajectory

   Input Parameter:
.  ts - the TSTrajectory context obtained from TSTrajectoryCreate()

   Options Database Keys:
.  -tstrajectory_type <type> - TSTRAJECTORYBASIC

   Level: advanced

   Notes: This is not normally called directly by users, instead it is called by TSSetFromOptions() after a call to 
   TSSetSaveTrajectory()

.keywords: TS, timestep, set, options, database

.seealso: TSGetType(), TSSetSaveTrajectory(), TSGetTrajectory()
@*/
PetscErrorCode  TSTrajectorySetFromOptions(TSTrajectory ts)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts, TSTRAJECTORY_CLASSID,1);
  ierr = PetscObjectOptionsBegin((PetscObject)ts);CHKERRQ(ierr);
  ierr = TSTrajectorySetTypeFromOptions_Private(PetscOptionsObject,ts);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
