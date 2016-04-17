
#include <petsc/private/tsimpl.h> /*I  "petscts.h" I*/

PetscClassId TSADAPT_CLASSID;

static PetscFunctionList TSAdaptList;
static PetscBool         TSAdaptPackageInitialized;
static PetscBool         TSAdaptRegisterAllCalled;

PETSC_EXTERN PetscErrorCode TSAdaptCreate_None(TSAdapt);
PETSC_EXTERN PetscErrorCode TSAdaptCreate_Basic(TSAdapt);
PETSC_EXTERN PetscErrorCode TSAdaptCreate_CFL(TSAdapt);

#undef __FUNCT__
#define __FUNCT__ "TSAdaptRegister"
/*@C
   TSAdaptRegister -  adds a TSAdapt implementation

   Not Collective

   Input Parameters:
+  name_scheme - name of user-defined adaptivity scheme
-  routine_create - routine to create method context

   Notes:
   TSAdaptRegister() may be called multiple times to add several user-defined families.

   Sample usage:
.vb
   TSAdaptRegister("my_scheme",MySchemeCreate);
.ve

   Then, your scheme can be chosen with the procedural interface via
$     TSAdaptSetType(ts,"my_scheme")
   or at runtime via the option
$     -ts_adapt_type my_scheme

   Level: advanced

.keywords: TSAdapt, register

.seealso: TSAdaptRegisterAll()
@*/
PetscErrorCode  TSAdaptRegister(const char sname[],PetscErrorCode (*function)(TSAdapt))
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFunctionListAdd(&TSAdaptList,sname,function);CHKERRQ(ierr);
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
PetscErrorCode  TSAdaptRegisterAll(void)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (TSAdaptRegisterAllCalled) PetscFunctionReturn(0);
  TSAdaptRegisterAllCalled = PETSC_TRUE;
  ierr = TSAdaptRegister(TSADAPTNONE, TSAdaptCreate_None);CHKERRQ(ierr);
  ierr = TSAdaptRegister(TSADAPTBASIC,TSAdaptCreate_Basic);CHKERRQ(ierr);
  ierr = TSAdaptRegister(TSADAPTCFL,  TSAdaptCreate_CFL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSAdaptFinalizePackage"
/*@C
  TSAdaptFinalizePackage - This function destroys everything in the TS package. It is
  called from PetscFinalize().

  Level: developer

.keywords: Petsc, destroy, package
.seealso: PetscFinalize()
@*/
PetscErrorCode  TSAdaptFinalizePackage(void)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFunctionListDestroy(&TSAdaptList);CHKERRQ(ierr);
  TSAdaptPackageInitialized = PETSC_FALSE;
  TSAdaptRegisterAllCalled  = PETSC_FALSE;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSAdaptInitializePackage"
/*@C
  TSAdaptInitializePackage - This function initializes everything in the TSAdapt package. It is
  called from PetscDLLibraryRegister() when using dynamic libraries, and on the first call to
  TSCreate_GL() when using static libraries.

  Level: developer

.keywords: TSAdapt, initialize, package
.seealso: PetscInitialize()
@*/
PetscErrorCode  TSAdaptInitializePackage(void)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (TSAdaptPackageInitialized) PetscFunctionReturn(0);
  TSAdaptPackageInitialized = PETSC_TRUE;
  ierr = PetscClassIdRegister("TSAdapt",&TSADAPT_CLASSID);CHKERRQ(ierr);
  ierr = TSAdaptRegisterAll();CHKERRQ(ierr);
  ierr = PetscRegisterFinalize(TSAdaptFinalizePackage);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSAdaptSetType"
PetscErrorCode  TSAdaptSetType(TSAdapt adapt,TSAdaptType type)
{
  PetscBool      match;
  PetscErrorCode ierr,(*r)(TSAdapt);

  PetscFunctionBegin;
  PetscValidHeaderSpecific(adapt,TSADAPT_CLASSID,1);
  ierr = PetscObjectTypeCompare((PetscObject)adapt,type,&match);CHKERRQ(ierr);
  if (match) PetscFunctionReturn(0);
  ierr = PetscFunctionListFind(TSAdaptList,type,&r);CHKERRQ(ierr);
  if (!r) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_UNKNOWN_TYPE,"Unknown TSAdapt type \"%s\" given",type);
  if (adapt->ops->destroy) {ierr = (*adapt->ops->destroy)(adapt);CHKERRQ(ierr);}
  ierr = PetscMemzero(adapt->ops,sizeof(struct _TSAdaptOps));CHKERRQ(ierr);
  ierr = PetscObjectChangeTypeName((PetscObject)adapt,type);CHKERRQ(ierr);
  ierr = (*r)(adapt);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSAdaptSetOptionsPrefix"
PetscErrorCode  TSAdaptSetOptionsPrefix(TSAdapt adapt,const char prefix[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(adapt,TSADAPT_CLASSID,1);
  ierr = PetscObjectSetOptionsPrefix((PetscObject)adapt,prefix);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSAdaptLoad"
/*@C
  TSAdaptLoad - Loads a TSAdapt that has been stored in binary  with TSAdaptView().

  Collective on PetscViewer

  Input Parameters:
+ newdm - the newly loaded TSAdapt, this needs to have been created with TSAdaptCreate() or
           some related function before a call to TSAdaptLoad().
- viewer - binary file viewer, obtained from PetscViewerBinaryOpen() or
           HDF5 file viewer, obtained from PetscViewerHDF5Open()

   Level: intermediate

  Notes:
   The type is determined by the data in the file, any type set into the TSAdapt before this call is ignored.

  Notes for advanced users:
  Most users should not need to know the details of the binary storage
  format, since TSAdaptLoad() and TSAdaptView() completely hide these details.
  But for anyone who's interested, the standard binary matrix storage
  format is
.vb
     has not yet been determined
.ve

.seealso: PetscViewerBinaryOpen(), TSAdaptView(), MatLoad(), VecLoad()
@*/
PetscErrorCode  TSAdaptLoad(TSAdapt adapt,PetscViewer viewer)
{
  PetscErrorCode ierr;
  PetscBool      isbinary;
  char           type[256];

  PetscFunctionBegin;
  PetscValidHeaderSpecific(adapt,TSADAPT_CLASSID,1);
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_CLASSID,2);
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERBINARY,&isbinary);CHKERRQ(ierr);
  if (!isbinary) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Invalid viewer; open viewer with PetscViewerBinaryOpen()");

  ierr = PetscViewerBinaryRead(viewer,type,256,NULL,PETSC_CHAR);CHKERRQ(ierr);
  ierr = TSAdaptSetType(adapt,type);CHKERRQ(ierr);
  if (adapt->ops->load) {
    ierr = (*adapt->ops->load)(adapt,viewer);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSAdaptView"
PetscErrorCode  TSAdaptView(TSAdapt adapt,PetscViewer viewer)
{
  PetscErrorCode ierr;
  PetscBool      iascii,isbinary;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(adapt,TSADAPT_CLASSID,1);
  if (!viewer) {ierr = PetscViewerASCIIGetStdout(PetscObjectComm((PetscObject)adapt),&viewer);CHKERRQ(ierr);}
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_CLASSID,2);
  PetscCheckSameComm(adapt,1,viewer,2);
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERBINARY,&isbinary);CHKERRQ(ierr);
  if (iascii) {
    ierr = PetscObjectPrintClassNamePrefixType((PetscObject)adapt,viewer);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  number of candidates %D\n",adapt->candidates.n);CHKERRQ(ierr);
    if (adapt->ops->view) {
      ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);
      ierr = (*adapt->ops->view)(adapt,viewer);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPopTab(viewer);CHKERRQ(ierr);
    }
  } else if (isbinary) {
    char type[256];

    /* need to save FILE_CLASS_ID for adapt class */
    ierr = PetscStrncpy(type,((PetscObject)adapt)->type_name,256);CHKERRQ(ierr);
    ierr = PetscViewerBinaryWrite(viewer,type,256,PETSC_CHAR,PETSC_FALSE);CHKERRQ(ierr);
  } else if (adapt->ops->view) {
    ierr = (*adapt->ops->view)(adapt,viewer);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSAdaptReset"
/*@
   TSAdaptReset - Resets a TSAdapt context.

   Collective on TS

   Input Parameter:
.  adapt - the TSAdapt context obtained from TSAdaptCreate()

   Level: developer

.seealso: TSAdaptCreate(), TSAdaptDestroy()
@*/
PetscErrorCode  TSAdaptReset(TSAdapt adapt)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(adapt,TSADAPT_CLASSID,1);
  if (adapt->ops->reset) {ierr = (*adapt->ops->reset)(adapt);CHKERRQ(ierr);}
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
  if (--((PetscObject)(*adapt))->refct > 0) {*adapt = NULL; PetscFunctionReturn(0);}

  ierr = TSAdaptReset(*adapt);CHKERRQ(ierr);

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
  PetscValidHeaderSpecific(adapt,TSADAPT_CLASSID,1);
  PetscValidLogicalCollectiveBool(adapt,flg,2);
  if (flg) {
    if (!adapt->monitor) {ierr = PetscViewerASCIIOpen(PetscObjectComm((PetscObject)adapt),"stdout",&adapt->monitor);CHKERRQ(ierr);}
  } else {
    ierr = PetscViewerDestroy(&adapt->monitor);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSAdaptSetCheckStage"
/*@C
   TSAdaptSetCheckStage - set a callback to check convergence for a stage

   Logically collective on TSAdapt

   Input Arguments:
+  adapt - adaptive controller context
-  func - stage check function

   Arguments of func:
$  PetscErrorCode func(TSAdapt adapt,TS ts,PetscBool *accept)

+  adapt - adaptive controller context
.  ts - time stepping context
-  accept - pending choice of whether to accept, can be modified by this routine

   Level: advanced

.seealso: TSAdaptChoose()
@*/
PetscErrorCode TSAdaptSetCheckStage(TSAdapt adapt,PetscErrorCode (*func)(TSAdapt,TS,PetscReal,Vec,PetscBool*))
{

  PetscFunctionBegin;
  PetscValidHeaderSpecific(adapt,TSADAPT_CLASSID,1);
  adapt->checkstage = func;
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
  PetscValidHeaderSpecific(adapt,TSADAPT_CLASSID,1);
  PetscValidLogicalCollectiveReal(adapt,hmin,2);
  PetscValidLogicalCollectiveReal(adapt,hmax,3);
  if (hmin != PETSC_DEFAULT && hmin < 0) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Minimum time step %g must be non negative",(double)hmin);
  if (hmax != PETSC_DEFAULT && hmax < 0) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Minimum time step %g must be non negative",(double)hmax);
  if (hmin != PETSC_DEFAULT) adapt->dt_min = hmin;
  if (hmax != PETSC_DEFAULT) adapt->dt_max = hmax;
#if defined(PETSC_USE_DEBUG)
  hmin = adapt->dt_min;
  hmax = adapt->dt_max;
  if (hmax <= hmin) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Maximum time step %g must geather than minimum time step %g",(double)hmax,(double)hmin);
#endif
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSAdaptSetFromOptions"
/*
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
*/
PetscErrorCode  TSAdaptSetFromOptions(PetscOptionItems *PetscOptionsObject,TSAdapt adapt)
{
  PetscErrorCode ierr;
  char           type[256] = TSADAPTBASIC;
  PetscBool      set,flg;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(adapt,TSADAPT_CLASSID,1);
  /* This should use PetscOptionsBegin() if/when this becomes an object used outside of TS, but currently this
   * function can only be called from inside TSSetFromOptions()  */
  ierr = PetscOptionsHead(PetscOptionsObject,"TS Adaptivity options");CHKERRQ(ierr);
  ierr = PetscOptionsFList("-ts_adapt_type","Algorithm to use for adaptivity","TSAdaptSetType",TSAdaptList,
                          ((PetscObject)adapt)->type_name ? ((PetscObject)adapt)->type_name : type,type,sizeof(type),&flg);CHKERRQ(ierr);
  if (flg || !((PetscObject)adapt)->type_name) {
    ierr = TSAdaptSetType(adapt,type);CHKERRQ(ierr);
  }
  ierr = PetscOptionsReal("-ts_adapt_dt_min","Minimum time step considered","TSAdaptSetStepLimits",adapt->dt_min,&adapt->dt_min,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-ts_adapt_dt_max","Maximum time step considered","TSAdaptSetStepLimits",adapt->dt_max,&adapt->dt_max,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-ts_adapt_scale_solve_failed","Scale step by this factor if solve fails","",adapt->scale_solve_failed,&adapt->scale_solve_failed,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-ts_adapt_monitor","Print choices made by adaptive controller","TSAdaptSetMonitor",adapt->monitor ? PETSC_TRUE : PETSC_FALSE,&flg,&set);CHKERRQ(ierr);
  ierr = PetscOptionsEnum("-ts_adapt_wnormtype","Type of norm computed for error estimation","",NormTypes,(PetscEnum)adapt->wnormtype,(PetscEnum*)&adapt->wnormtype,NULL);CHKERRQ(ierr);
  if (adapt->wnormtype != NORM_2 && adapt->wnormtype != NORM_INFINITY) SETERRQ(PetscObjectComm((PetscObject)adapt),PETSC_ERR_SUP,"Only 2-norm and infinite norm supported");
  if (set) {ierr = TSAdaptSetMonitor(adapt,flg);CHKERRQ(ierr);}
  if (adapt->ops->setfromoptions) {ierr = (*adapt->ops->setfromoptions)(PetscOptionsObject,adapt);CHKERRQ(ierr);}
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
  PetscValidHeaderSpecific(adapt,TSADAPT_CLASSID,1);
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
  if (order < 1) SETERRQ1(PetscObjectComm((PetscObject)adapt),PETSC_ERR_ARG_OUTOFRANGE,"Classical order %D must be a positive integer",order);
  if (inuse) {
    if (adapt->candidates.inuse_set) SETERRQ(PetscObjectComm((PetscObject)adapt),PETSC_ERR_ARG_WRONGSTATE,"Cannot set the inuse method twice, maybe forgot to call TSAdaptCandidatesClear()");
    adapt->candidates.inuse_set = PETSC_TRUE;
  }
  /* first slot if this is the current scheme, otherwise the next available slot */
  c = inuse ? 0 : !adapt->candidates.inuse_set + adapt->candidates.n;

  adapt->candidates.name[c]       = name;
  adapt->candidates.order[c]      = order;
  adapt->candidates.stageorder[c] = stageorder;
  adapt->candidates.ccfl[c]       = ccfl;
  adapt->candidates.cost[c]       = cost;
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
+  next_sc - optional, scheme to use for the next step
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
  PetscInt       ncandidates = adapt->candidates.n;
  PetscInt       scheme = 0;
  PetscReal      wlte = -1.0;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(adapt,TSADAPT_CLASSID,1);
  PetscValidHeaderSpecific(ts,TS_CLASSID,2);
  if (next_sc) PetscValidIntPointer(next_sc,4);
  PetscValidPointer(next_h,5);
  PetscValidIntPointer(accept,6);
  if (next_sc) *next_sc = 0;

  /* Do not mess with adaptivity while handling events*/
  if (ts->event && ts->event->status != TSEVENT_NONE) {
    *next_h = h;
    *accept = PETSC_TRUE;
    PetscFunctionReturn(0);
  }

  ierr = (*adapt->ops->choose)(adapt,ts,h,&scheme,next_h,accept,&wlte);CHKERRQ(ierr);
  if (scheme < 0 || (ncandidates > 0 && scheme >= ncandidates)) SETERRQ2(PetscObjectComm((PetscObject)adapt),PETSC_ERR_ARG_OUTOFRANGE,"Chosen scheme %D not in valid range 0..%D",scheme,ncandidates-1);
  if (*next_h < 0) SETERRQ1(PetscObjectComm((PetscObject)adapt),PETSC_ERR_ARG_OUTOFRANGE,"Computed step size %g must be positive",(double)*next_h);
  if (next_sc) *next_sc = scheme;

  if (*accept && ts->exact_final_time == TS_EXACTFINALTIME_MATCHSTEP) {
    /* Reduce time step if it overshoots max time */
    if (ts->ptime + ts->time_step + *next_h >= ts->max_time) {
      PetscReal next_dt = ts->max_time - (ts->ptime + ts->time_step);
      if (next_dt > PETSC_SMALL) *next_h = next_dt;
      else ts->reason = TS_CONVERGED_TIME;
    }
  }

  if (adapt->monitor) {
    const char *sc_name = (scheme < ncandidates) ? adapt->candidates.name[scheme] : "";
    ierr = PetscViewerASCIIAddTab(adapt->monitor,((PetscObject)adapt)->tablevel);CHKERRQ(ierr);
    if (wlte < 0) {
      ierr = PetscViewerASCIIPrintf(adapt->monitor,"    TSAdapt '%s': step %3D %s t=%-11g+%10.3e family='%s' scheme=%D:'%s' dt=%-10.3e\n",((PetscObject)adapt)->type_name,ts->steps,*accept ? "accepted" : "rejected",(double)ts->ptime,(double)h,((PetscObject)ts)->type_name,scheme,sc_name,(double)*next_h);CHKERRQ(ierr);
    } else {
      ierr = PetscViewerASCIIPrintf(adapt->monitor,"    TSAdapt '%s': step %3D %s t=%-11g+%10.3e wlte=%5.3g family='%s' scheme=%D:'%s' dt=%-10.3e\n",((PetscObject)adapt)->type_name,ts->steps,*accept ? "accepted" : "rejected",(double)ts->ptime,(double)h,(double)wlte,((PetscObject)ts)->type_name,scheme,sc_name,(double)*next_h);CHKERRQ(ierr);
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
.  ts - time stepper
.  t - Current simulation time
-  Y - Current solution vector

   Output Arguments:
.  accept - PETSC_TRUE to accept the stage, PETSC_FALSE to reject

   Level: developer

.seealso:
@*/
PetscErrorCode TSAdaptCheckStage(TSAdapt adapt,TS ts,PetscReal t,Vec Y,PetscBool *accept)
{
  PetscErrorCode      ierr;
  SNESConvergedReason snesreason = SNES_CONVERGED_ITERATING;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(adapt,TSADAPT_CLASSID,1);
  PetscValidHeaderSpecific(ts,TS_CLASSID,2);
  PetscValidIntPointer(accept,3);

  if (ts->snes) {ierr = SNESGetConvergedReason(ts->snes,&snesreason);CHKERRQ(ierr);}
  if (snesreason < 0) {
    *accept = PETSC_FALSE;
    if (++ts->num_snes_failures >= ts->max_snes_failures && ts->max_snes_failures > 0) {
      ts->reason = TS_DIVERGED_NONLINEAR_SOLVE;
      ierr = PetscInfo2(ts,"Step=%D, nonlinear solve failures %D greater than current TS allowed, stopping solve\n",ts->steps,ts->num_snes_failures);CHKERRQ(ierr);
      if (adapt->monitor) {
        ierr = PetscViewerASCIIAddTab(adapt->monitor,((PetscObject)adapt)->tablevel);CHKERRQ(ierr);
        ierr = PetscViewerASCIIPrintf(adapt->monitor,"    TSAdapt '%s': step %3D stage rejected t=%-11g+%10.3e, nonlinear solve failures %D greater than current TS allowed\n",((PetscObject)adapt)->type_name,ts->steps,(double)ts->ptime,(double)ts->time_step,ts->num_snes_failures);CHKERRQ(ierr);
        ierr = PetscViewerASCIISubtractTab(adapt->monitor,((PetscObject)adapt)->tablevel);CHKERRQ(ierr);
      }
    }
  } else {
    *accept = PETSC_TRUE;
    ierr = TSFunctionDomainError(ts,t,Y,accept);CHKERRQ(ierr);
    if(*accept && adapt->checkstage) {
      ierr = (*adapt->checkstage)(adapt,ts,t,Y,accept);CHKERRQ(ierr);
    }
  }

  if(!(*accept) && !ts->reason) {
    PetscReal dt,new_dt;
    ierr = TSGetTimeStep(ts,&dt);CHKERRQ(ierr);
    new_dt = dt * adapt->scale_solve_failed;
    ierr = TSSetTimeStep(ts,new_dt);CHKERRQ(ierr);
    if (adapt->monitor) {
      ierr = PetscViewerASCIIAddTab(adapt->monitor,((PetscObject)adapt)->tablevel);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(adapt->monitor,"    TSAdapt '%s': step %3D stage rejected t=%-11g+%10.3e retrying with dt=%-10.3e\n",((PetscObject)adapt)->type_name,ts->steps,(double)ts->ptime,(double)dt,(double)new_dt);CHKERRQ(ierr);
      ierr = PetscViewerASCIISubtractTab(adapt->monitor,((PetscObject)adapt)->tablevel);CHKERRQ(ierr);
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
  TSAdapt        adapt;

  PetscFunctionBegin;
  PetscValidPointer(inadapt,1);
  *inadapt = NULL;
  ierr = TSAdaptInitializePackage();CHKERRQ(ierr);

  ierr = PetscHeaderCreate(adapt,TSADAPT_CLASSID,"TSAdapt","Time stepping adaptivity","TS",comm,TSAdaptDestroy,TSAdaptView);CHKERRQ(ierr);

  adapt->dt_min             = 1e-20;
  adapt->dt_max             = 1e50;
  adapt->scale_solve_failed = 0.25;
  adapt->wnormtype          = NORM_2;

  *inadapt = adapt;
  PetscFunctionReturn(0);
}
