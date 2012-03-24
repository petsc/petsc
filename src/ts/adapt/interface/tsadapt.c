
#include <petsc-private/tsimpl.h> /*I  "petscts.h" I*/

static PetscFList TSAdaptList;
static PetscBool  TSAdaptPackageInitialized;
static PetscBool  TSAdaptRegisterAllCalled;
static PetscClassId TSADAPT_CLASSID;

EXTERN_C_BEGIN
PetscErrorCode  TSAdaptCreate_Basic(TSAdapt);
PetscErrorCode  TSAdaptCreate_None(TSAdapt);
PetscErrorCode  TSAdaptCreate_CFL(TSAdapt);
EXTERN_C_END

#undef __FUNCT__
#define __FUNCT__ "TSAdaptRegister"
/*@C
   TSAdaptRegister - see TSAdaptRegisterDynamic()

   Level: advanced
@*/
PetscErrorCode  TSAdaptRegister(const char sname[],const char path[],const char name[],PetscErrorCode (*function)(TSAdapt))
{
  PetscErrorCode ierr;
  char           fullname[PETSC_MAX_PATH_LEN];

  PetscFunctionBegin;
  ierr = PetscFListConcat(path,name,fullname);CHKERRQ(ierr);
  ierr = PetscFListAdd(&TSAdaptList,sname,fullname,(void(*)(void))function);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSAdaptRegisterAll"
/*@C
  TSAdaptRegisterAll - Registers all of the adaptivity schemes in TSAdapt

  Not Collective

  Level: advanced

.keywords: TSAdapt, register, all

.seealso: TSAdaptRegisterDestroy()
@*/
PetscErrorCode  TSAdaptRegisterAll(const char path[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = TSAdaptRegisterDynamic(TSADAPTBASIC,path,"TSAdaptCreate_Basic",TSAdaptCreate_Basic);CHKERRQ(ierr);
  ierr = TSAdaptRegisterDynamic(TSADAPTNONE, path,"TSAdaptCreate_None", TSAdaptCreate_None);CHKERRQ(ierr);
  ierr = TSAdaptRegisterDynamic(TSADAPTCFL,  path,"TSAdaptCreate_CFL",  TSAdaptCreate_CFL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSAdaptFinalizePackage"
/*@C
  TSFinalizePackage - This function destroys everything in the TS package. It is
  called from PetscFinalize().

  Level: developer

.keywords: Petsc, destroy, package
.seealso: PetscFinalize()
@*/
PetscErrorCode  TSAdaptFinalizePackage(void)
{
  PetscFunctionBegin;
  TSAdaptPackageInitialized = PETSC_FALSE;
  TSAdaptRegisterAllCalled  = PETSC_FALSE;
  TSAdaptList               = PETSC_NULL;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSAdaptInitializePackage"
/*@C
  TSAdaptInitializePackage - This function initializes everything in the TSAdapt package. It is
  called from PetscDLLibraryRegister() when using dynamic libraries, and on the first call to
  TSCreate_GL() when using static libraries.

  Input Parameter:
  path - The dynamic library path, or PETSC_NULL

  Level: developer

.keywords: TSAdapt, initialize, package
.seealso: PetscInitialize()
@*/
PetscErrorCode  TSAdaptInitializePackage(const char path[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (TSAdaptPackageInitialized) PetscFunctionReturn(0);
  TSAdaptPackageInitialized = PETSC_TRUE;
  ierr = PetscClassIdRegister("TSAdapt",&TSADAPT_CLASSID);CHKERRQ(ierr);
  ierr = TSAdaptRegisterAll(path);CHKERRQ(ierr);
  ierr = PetscRegisterFinalize(TSAdaptFinalizePackage);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSAdaptRegisterDestroy"
/*@C
   TSAdaptRegisterDestroy - Frees the list of adaptivity schemes that were registered by TSAdaptRegister()/TSAdaptRegisterDynamic().

   Not Collective

   Level: advanced

.keywords: TSAdapt, register, destroy
.seealso: TSAdaptRegister(), TSAdaptRegisterAll(), TSAdaptRegisterDynamic()
@*/
PetscErrorCode  TSAdaptRegisterDestroy(void)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFListDestroy(&TSAdaptList);CHKERRQ(ierr);
  TSAdaptRegisterAllCalled = PETSC_FALSE;
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "TSAdaptSetType"
PetscErrorCode  TSAdaptSetType(TSAdapt adapt,const TSAdaptType type)
{
  PetscErrorCode ierr,(*r)(TSAdapt);

  PetscFunctionBegin;
  ierr = PetscFListFind(TSAdaptList,((PetscObject)adapt)->comm,type,PETSC_TRUE,(void(**)(void))&r);CHKERRQ(ierr);
  if (!r) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_UNKNOWN_TYPE,"Unknown TSAdapt type \"%s\" given",type);
  if (((PetscObject)adapt)->type_name) {ierr = (*adapt->ops->destroy)(adapt);CHKERRQ(ierr);}
  ierr = (*r)(adapt);CHKERRQ(ierr);
  ierr = PetscObjectChangeTypeName((PetscObject)adapt,type);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSAdaptSetOptionsPrefix"
PetscErrorCode  TSAdaptSetOptionsPrefix(TSAdapt adapt,const char prefix[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscObjectSetOptionsPrefix((PetscObject)adapt,prefix);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSAdaptView"
PetscErrorCode  TSAdaptView(TSAdapt adapt,PetscViewer viewer)
{
  PetscErrorCode ierr;
  PetscBool      iascii;

  PetscFunctionBegin;
  ierr = PetscTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii);CHKERRQ(ierr);
  if (iascii) {
    ierr = PetscObjectPrintClassNamePrefixType((PetscObject)adapt,viewer,"TSAdapt Object");CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"number of candidates %D\n",adapt->candidates.n);CHKERRQ(ierr);
    if (adapt->ops->view) {
      ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);
      ierr = (*adapt->ops->view)(adapt,viewer);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPopTab(viewer);CHKERRQ(ierr);
    }
  } else {
    SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_SUP,"Viewer type %s not supported",((PetscObject)viewer)->type_name);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSAdaptDestroy"
PetscErrorCode  TSAdaptDestroy(TSAdapt *adapt)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!*adapt) PetscFunctionReturn(0);
  PetscValidHeaderSpecific(*adapt,TSADAPT_CLASSID,1);
  if (--((PetscObject)(*adapt))->refct > 0) {*adapt = 0; PetscFunctionReturn(0);}
  if ((*adapt)->ops->destroy) {ierr = (*(*adapt)->ops->destroy)(*adapt);CHKERRQ(ierr);}
  ierr = PetscViewerDestroy(&(*adapt)->monitor);CHKERRQ(ierr);
  ierr = PetscHeaderDestroy(adapt);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSAdaptSetMonitor"
/*@
   TSAdaptSetMonitor - Monitor the choices made by the adaptive controller

   Collective on TSAdapt

   Input Arguments:
+  adapt - adaptive controller context
-  flg - PETSC_TRUE to active a monitor, PETSC_FALSE to disable

   Level: intermediate

.seealso: TSAdaptChoose()
@*/
PetscErrorCode TSAdaptSetMonitor(TSAdapt adapt,PetscBool flg)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (flg) {
    if (!adapt->monitor) {ierr = PetscViewerASCIIOpen(((PetscObject)adapt)->comm,"stdout",&adapt->monitor);CHKERRQ(ierr);}
  } else {
    ierr = PetscViewerDestroy(&adapt->monitor);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSAdaptSetStepLimits"
/*@
   TSAdaptSetStepLimits - Set minimum and maximum step sizes to be considered by the controller

   Logically Collective

   Input Arguments:
+  adapt - time step adaptivity context, usually gotten with TSGetAdapt()
.  hmin - minimum time step
-  hmax - maximum time step

   Options Database Keys:
+  -ts_adapt_dt_min - minimum time step
-  -ts_adapt_dt_max - maximum time step

   Level: intermediate

.seealso: TSAdapt
@*/
PetscErrorCode TSAdaptSetStepLimits(TSAdapt adapt,PetscReal hmin,PetscReal hmax)
{

  PetscFunctionBegin;
  if (hmin != PETSC_DECIDE) adapt->dt_min = hmin;
  if (hmax != PETSC_DECIDE) adapt->dt_max = hmax;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSAdaptSetFromOptions"
/*@
   TSAdaptSetFromOptions - Sets various TSAdapt parameters from user options.

   Collective on TSAdapt

   Input Parameter:
.  adapt - the TSAdapt context

   Options Database Keys:
.  -ts_adapt_type <type> - basic

   Level: advanced

   Notes:
   This function is automatically called by TSSetFromOptions()

.keywords: TS, TSGetAdapt(), TSAdaptSetType()

.seealso: TSGetType()
@*/
PetscErrorCode  TSAdaptSetFromOptions(TSAdapt adapt)
{
  PetscErrorCode ierr;
  char           type[256] = TSADAPTBASIC;
  PetscBool      set,flg;

  PetscFunctionBegin;
  /* This should use PetscOptionsBegin() if/when this becomes an object used outside of TS, but currently this
  * function can only be called from inside TSSetFromOptions_GL()  */
  ierr = PetscOptionsHead("TS Adaptivity options");CHKERRQ(ierr);
  ierr = PetscOptionsList("-ts_adapt_type","Algorithm to use for adaptivity","TSAdaptSetType",TSAdaptList,
                          ((PetscObject)adapt)->type_name?((PetscObject)adapt)->type_name:type,type,sizeof type,&flg);CHKERRQ(ierr);
  if (flg || !((PetscObject)adapt)->type_name) {
    ierr = TSAdaptSetType(adapt,type);CHKERRQ(ierr);
  }
  if (adapt->ops->setfromoptions) {ierr = (*adapt->ops->setfromoptions)(adapt);CHKERRQ(ierr);}
  ierr = PetscOptionsReal("-ts_adapt_dt_min","Minimum time step considered","TSAdaptSetStepLimits",adapt->dt_min,&adapt->dt_min,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-ts_adapt_dt_max","Maximum time step considered","TSAdaptSetStepLimits",adapt->dt_max,&adapt->dt_max,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-ts_adapt_scale_solve_failed","Scale step by this factor if solve fails","",adapt->scale_solve_failed,&adapt->scale_solve_failed,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-ts_adapt_monitor","Print choices made by adaptive controller","TSAdaptSetMonitor",adapt->monitor ? PETSC_TRUE : PETSC_FALSE,&flg,&set);CHKERRQ(ierr);
  if (set) {ierr = TSAdaptSetMonitor(adapt,flg);CHKERRQ(ierr);}
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "TSAdaptCandidatesClear"
/*@
   TSAdaptCandidatesClear - clear any previously set candidate schemes

   Logically Collective

   Input Argument:
.  adapt - adaptive controller

   Level: developer

.seealso: TSAdapt, TSAdaptCreate(), TSAdaptCandidateAdd(), TSAdaptChoose()
@*/
PetscErrorCode TSAdaptCandidatesClear(TSAdapt adapt)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscMemzero(&adapt->candidates,sizeof(adapt->candidates));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "TSAdaptCandidateAdd"
/*@C
   TSAdaptCandidateAdd - add a candidate scheme for the adaptive controller to select from

   Logically Collective

   Input Arguments:
+  adapt - time step adaptivity context, obtained with TSGetAdapt() or TSAdaptCreate()
.  name - name of the candidate scheme to add
.  order - order of the candidate scheme
.  stageorder - stage order of the candidate scheme
.  ccfl - stability coefficient relative to explicit Euler, used for CFL constraints
.  cost - relative measure of the amount of work required for the candidate scheme
-  inuse - indicates that this scheme is the one currently in use, this flag can only be set for one scheme

   Note:
   This routine is not available in Fortran.

   Level: developer

.seealso: TSAdaptCandidatesClear(), TSAdaptChoose()
@*/
PetscErrorCode TSAdaptCandidateAdd(TSAdapt adapt,const char name[],PetscInt order,PetscInt stageorder,PetscReal ccfl,PetscReal cost,PetscBool inuse)
{
  PetscInt c;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(adapt,TSADAPT_CLASSID,1);
  if (order < 1) SETERRQ1(((PetscObject)adapt)->comm,PETSC_ERR_ARG_OUTOFRANGE,"Classical order %D must be a positive integer",order);
  if (inuse) {
    if (adapt->candidates.inuse_set) SETERRQ(((PetscObject)adapt)->comm,PETSC_ERR_ARG_WRONGSTATE,"Cannot set the inuse method twice, maybe forgot to call TSAdaptCandidatesClear()");
    adapt->candidates.inuse_set = PETSC_TRUE;
  }
  /* first slot if this is the current scheme, otherwise the next available slot */
  c = inuse ? 0 : !adapt->candidates.inuse_set + adapt->candidates.n;
  adapt->candidates.name[c]         = name;
  adapt->candidates.order[c]        = order;
  adapt->candidates.stageorder[c]   = stageorder;
  adapt->candidates.ccfl[c]         = ccfl;
  adapt->candidates.cost[c]         = cost;
  adapt->candidates.n++;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSAdaptCandidatesGet"
/*@C
   TSAdaptCandidatesGet - Get the list of candidate orders of accuracy and cost

   Not Collective

   Input Arguments:
.  adapt - time step adaptivity context

   Output Arguments:
+  n - number of candidate schemes, always at least 1
.  order - the order of each candidate scheme
.  stageorder - the stage order of each candidate scheme
.  ccfl - the CFL coefficient of each scheme
-  cost - the relative cost of each scheme

   Level: developer

   Note:
   The current scheme is always returned in the first slot

.seealso: TSAdaptCandidatesClear(), TSAdaptCandidateAdd(), TSAdaptChoose()
@*/
PetscErrorCode TSAdaptCandidatesGet(TSAdapt adapt,PetscInt *n,const PetscInt **order,const PetscInt **stageorder,const PetscReal **ccfl,const PetscReal **cost)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(adapt,TSADAPT_CLASSID,1);
  if (n) *n = adapt->candidates.n;
  if (order) *order = adapt->candidates.order;
  if (stageorder) *stageorder = adapt->candidates.stageorder;
  if (ccfl) *ccfl = adapt->candidates.ccfl;
  if (cost) *cost = adapt->candidates.cost;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSAdaptChoose"
/*@C
   TSAdaptChoose - choose which method and step size to use for the next step

   Logically Collective

   Input Arguments:
+  adapt - adaptive contoller
-  h - current step size

   Output Arguments:
+  next_sc - scheme to use for the next step
.  next_h - step size to use for the next step
-  accept - PETSC_TRUE to accept the current step, PETSC_FALSE to repeat the current step with the new step size

   Note:
   The input value of parameter accept is retained from the last time step, so it will be PETSC_FALSE if the step is
   being retried after an initial rejection.

   Level: developer

.seealso: TSAdapt, TSAdaptCandidatesClear(), TSAdaptCandidateAdd()
@*/
PetscErrorCode TSAdaptChoose(TSAdapt adapt,TS ts,PetscReal h,PetscInt *next_sc,PetscReal *next_h,PetscBool *accept)
{
  PetscErrorCode ierr;
  PetscReal wlte = -1.0;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(adapt,TSADAPT_CLASSID,1);
  PetscValidHeaderSpecific(ts,TS_CLASSID,2);
  PetscValidIntPointer(next_sc,4);
  PetscValidPointer(next_h,5);
  PetscValidIntPointer(accept,6);
  if (adapt->candidates.n < 1) SETERRQ1(((PetscObject)adapt)->comm,PETSC_ERR_ARG_WRONGSTATE,"%D candidates have been registered",adapt->candidates.n);
  if (!adapt->candidates.inuse_set) SETERRQ1(((PetscObject)adapt)->comm,PETSC_ERR_ARG_WRONGSTATE,"The current in-use scheme is not among the %D candidates",adapt->candidates.n);
  ierr = (*adapt->ops->choose)(adapt,ts,h,next_sc,next_h,accept,&wlte);CHKERRQ(ierr);
  if (*next_sc < 0 || adapt->candidates.n <= *next_sc) SETERRQ2(((PetscObject)adapt)->comm,PETSC_ERR_ARG_OUTOFRANGE,"Chosen scheme %D not in valid range 0..%D",*next_sc,adapt->candidates.n-1);
  if (!(*next_h > 0.)) SETERRQ1(((PetscObject)adapt)->comm,PETSC_ERR_ARG_OUTOFRANGE,"Computed step size %G must be positive",*next_h);

  if (adapt->monitor) {
    ierr = PetscViewerASCIIAddTab(adapt->monitor,((PetscObject)adapt)->tablevel);CHKERRQ(ierr);
    if (wlte < 0) {
      ierr = PetscViewerASCIIPrintf(adapt->monitor,"    TSAdapt '%s': step %3D %s t=%-11g+%10.3e family='%s' scheme=%D:'%s' dt=%-10g\n",((PetscObject)adapt)->type_name,ts->steps,*accept?"accepted":"rejected",(double)ts->ptime,h,((PetscObject)ts)->type_name,*next_sc,adapt->candidates.name[*next_sc],(double)*next_h);CHKERRQ(ierr);
    } else {
      ierr = PetscViewerASCIIPrintf(adapt->monitor,"    TSAdapt '%s': step %3D %s t=%-11g+%10.3e wlte=%5.3g family='%s' scheme=%D:'%s' dt=%-10.3e\n",((PetscObject)adapt)->type_name,ts->steps,*accept?"accepted":"rejected",(double)ts->ptime,h,(double)wlte,((PetscObject)ts)->type_name,*next_sc,adapt->candidates.name[*next_sc],(double)*next_h);CHKERRQ(ierr);
    }
    ierr = PetscViewerASCIISubtractTab(adapt->monitor,((PetscObject)adapt)->tablevel);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSAdaptCheckStage"
/*@
   TSAdaptCheckStage - checks whether to accept a stage, (e.g. reject and change time step size if nonlinear solve fails)

   Collective

   Input Arguments:
+  adapt - adaptive controller context
-  ts - time stepper

   Output Arguments:
.  accept - PETSC_TRUE to accept the stage, PETSC_FALSE to reject

   Level: developer

.seealso:
@*/
PetscErrorCode TSAdaptCheckStage(TSAdapt adapt,TS ts,PetscBool *accept)
{
  PetscErrorCode      ierr;
  SNES                snes;
  SNESConvergedReason snesreason;

  PetscFunctionBegin;
  *accept = PETSC_TRUE;
  ierr = TSGetSNES(ts,&snes);CHKERRQ(ierr);
  ierr = SNESGetConvergedReason(snes,&snesreason);CHKERRQ(ierr);
  if (snesreason < 0) {
    PetscReal dt,new_dt;
    *accept = PETSC_FALSE;
    ierr = TSGetTimeStep(ts,&dt);CHKERRQ(ierr);
    if (ts->max_snes_failures > 0 && ++ts->num_snes_failures >= ts->max_snes_failures) {
      ts->reason = TS_DIVERGED_NONLINEAR_SOLVE;
      ierr = PetscInfo2(ts,"Step=%D, nonlinear solve solve failures %D greater than current TS allowed, stopping solve\n",ts->steps,ts->num_snes_failures);CHKERRQ(ierr);
      if (adapt->monitor) {
        ierr = PetscViewerASCIIAddTab(adapt->monitor,((PetscObject)adapt)->tablevel);CHKERRQ(ierr);
        ierr = PetscViewerASCIIPrintf(adapt->monitor,"    TSAdapt '%s': step %3D stage rejected t=%-11g+%10.3e, %D failures exceeds current TS allowed\n",((PetscObject)adapt)->type_name,ts->steps,(double)ts->ptime,dt,ts->num_snes_failures);CHKERRQ(ierr);
        ierr = PetscViewerASCIISubtractTab(adapt->monitor,((PetscObject)adapt)->tablevel);CHKERRQ(ierr);
      }
    } else {
      new_dt = dt*adapt->scale_solve_failed;
      ierr = TSSetTimeStep(ts,new_dt);CHKERRQ(ierr);
      if (adapt->monitor) {
        ierr = PetscViewerASCIIAddTab(adapt->monitor,((PetscObject)adapt)->tablevel);CHKERRQ(ierr);
        ierr = PetscViewerASCIIPrintf(adapt->monitor,"    TSAdapt '%s': step %3D stage rejected t=%-11g+%10.3e retrying with dt=%-10.3e\n",((PetscObject)adapt)->type_name,ts->steps,(double)ts->ptime,(double)dt,(double)new_dt);CHKERRQ(ierr);
        ierr = PetscViewerASCIISubtractTab(adapt->monitor,((PetscObject)adapt)->tablevel);CHKERRQ(ierr);
      }
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSAdaptCreate"
/*@
  TSAdaptCreate - create an adaptive controller context for time stepping

  Collective on MPI_Comm

  Input Parameter:
. comm - The communicator

  Output Parameter:
. adapt - new TSAdapt object

  Level: developer

  Notes:
  TSAdapt creation is handled by TS, so users should not need to call this function.

.keywords: TSAdapt, create
.seealso: TSGetAdapt(), TSAdaptSetType(), TSAdaptDestroy()
@*/
PetscErrorCode  TSAdaptCreate(MPI_Comm comm,TSAdapt *inadapt)
{
  PetscErrorCode ierr;
  TSAdapt adapt;

  PetscFunctionBegin;
  *inadapt = 0;
  ierr = PetscHeaderCreate(adapt,_p_TSAdapt,struct _TSAdaptOps,TSADAPT_CLASSID,0,"TSAdapt","General Linear adaptivity","TS",comm,TSAdaptDestroy,TSAdaptView);CHKERRQ(ierr);

  adapt->dt_min             = 1e-20;
  adapt->dt_max             = 1e50;
  adapt->scale_solve_failed = 0.25;

  *inadapt = adapt;
  PetscFunctionReturn(0);
}
