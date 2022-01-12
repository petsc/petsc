
#include <petsc/private/tsimpl.h> /*I  "petscts.h" I*/

PetscClassId TSADAPT_CLASSID;

static PetscFunctionList TSAdaptList;
static PetscBool         TSAdaptPackageInitialized;
static PetscBool         TSAdaptRegisterAllCalled;

PETSC_EXTERN PetscErrorCode TSAdaptCreate_None(TSAdapt);
PETSC_EXTERN PetscErrorCode TSAdaptCreate_Basic(TSAdapt);
PETSC_EXTERN PetscErrorCode TSAdaptCreate_DSP(TSAdapt);
PETSC_EXTERN PetscErrorCode TSAdaptCreate_CFL(TSAdapt);
PETSC_EXTERN PetscErrorCode TSAdaptCreate_GLEE(TSAdapt);
PETSC_EXTERN PetscErrorCode TSAdaptCreate_History(TSAdapt);

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

.seealso: TSAdaptRegisterAll()
@*/
PetscErrorCode  TSAdaptRegister(const char sname[],PetscErrorCode (*function)(TSAdapt))
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = TSAdaptInitializePackage();CHKERRQ(ierr);
  ierr = PetscFunctionListAdd(&TSAdaptList,sname,function);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
  TSAdaptRegisterAll - Registers all of the adaptivity schemes in TSAdapt

  Not Collective

  Level: advanced

.seealso: TSAdaptRegisterDestroy()
@*/
PetscErrorCode  TSAdaptRegisterAll(void)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (TSAdaptRegisterAllCalled) PetscFunctionReturn(0);
  TSAdaptRegisterAllCalled = PETSC_TRUE;
  ierr = TSAdaptRegister(TSADAPTNONE,   TSAdaptCreate_None);CHKERRQ(ierr);
  ierr = TSAdaptRegister(TSADAPTBASIC,  TSAdaptCreate_Basic);CHKERRQ(ierr);
  ierr = TSAdaptRegister(TSADAPTDSP,    TSAdaptCreate_DSP);CHKERRQ(ierr);
  ierr = TSAdaptRegister(TSADAPTCFL,    TSAdaptCreate_CFL);CHKERRQ(ierr);
  ierr = TSAdaptRegister(TSADAPTGLEE,   TSAdaptCreate_GLEE);CHKERRQ(ierr);
  ierr = TSAdaptRegister(TSADAPTHISTORY,TSAdaptCreate_History);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
  TSAdaptFinalizePackage - This function destroys everything in the TS package. It is
  called from PetscFinalize().

  Level: developer

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

/*@C
  TSAdaptInitializePackage - This function initializes everything in the TSAdapt package. It is
  called from TSInitializePackage().

  Level: developer

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

/*@C
  TSAdaptSetType - sets the approach used for the error adapter, currently there is only TSADAPTBASIC and TSADAPTNONE

  Logicially Collective on TSAdapt

  Input Parameters:
+ adapt - the TS adapter, most likely obtained with TSGetAdapt()
- type - either  TSADAPTBASIC or TSADAPTNONE

  Options Database:
. -ts_adapt_type <basic or dsp or none> - to set the adapter type

  Level: intermediate

.seealso: TSGetAdapt(), TSAdaptDestroy(), TSAdaptType, TSAdaptGetType()
@*/
PetscErrorCode  TSAdaptSetType(TSAdapt adapt,TSAdaptType type)
{
  PetscBool      match;
  PetscErrorCode ierr,(*r)(TSAdapt);

  PetscFunctionBegin;
  PetscValidHeaderSpecific(adapt,TSADAPT_CLASSID,1);
  PetscValidCharPointer(type,2);
  ierr = PetscObjectTypeCompare((PetscObject)adapt,type,&match);CHKERRQ(ierr);
  if (match) PetscFunctionReturn(0);
  ierr = PetscFunctionListFind(TSAdaptList,type,&r);CHKERRQ(ierr);
  if (!r) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_UNKNOWN_TYPE,"Unknown TSAdapt type \"%s\" given",type);
  if (adapt->ops->destroy) {ierr = (*adapt->ops->destroy)(adapt);CHKERRQ(ierr);}
  ierr = PetscMemzero(adapt->ops,sizeof(struct _TSAdaptOps));CHKERRQ(ierr);
  ierr = PetscObjectChangeTypeName((PetscObject)adapt,type);CHKERRQ(ierr);
  ierr = (*r)(adapt);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
  TSAdaptGetType - gets the TS adapter method type (as a string).

  Not Collective

  Input Parameter:
. adapt - The TS adapter, most likely obtained with TSGetAdapt()

  Output Parameter:
. type - The name of TS adapter method

  Level: intermediate

.seealso TSAdaptSetType()
@*/
PetscErrorCode TSAdaptGetType(TSAdapt adapt,TSAdaptType *type)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(adapt,TSADAPT_CLASSID,1);
  PetscValidPointer(type,2);
  *type = ((PetscObject)adapt)->type_name;
  PetscFunctionReturn(0);
}

PetscErrorCode  TSAdaptSetOptionsPrefix(TSAdapt adapt,const char prefix[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(adapt,TSADAPT_CLASSID,1);
  ierr = PetscObjectSetOptionsPrefix((PetscObject)adapt,prefix);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

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

PetscErrorCode  TSAdaptView(TSAdapt adapt,PetscViewer viewer)
{
  PetscErrorCode ierr;
  PetscBool      iascii,isbinary,isnone,isglee;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(adapt,TSADAPT_CLASSID,1);
  if (!viewer) {ierr = PetscViewerASCIIGetStdout(PetscObjectComm((PetscObject)adapt),&viewer);CHKERRQ(ierr);}
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_CLASSID,2);
  PetscCheckSameComm(adapt,1,viewer,2);
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERBINARY,&isbinary);CHKERRQ(ierr);
  if (iascii) {
    ierr = PetscObjectPrintClassNamePrefixType((PetscObject)adapt,viewer);CHKERRQ(ierr);
    ierr = PetscObjectTypeCompare((PetscObject)adapt,TSADAPTNONE,&isnone);CHKERRQ(ierr);
    ierr = PetscObjectTypeCompare((PetscObject)adapt,TSADAPTGLEE,&isglee);CHKERRQ(ierr);
    if (!isnone) {
      if (adapt->always_accept) {ierr = PetscViewerASCIIPrintf(viewer,"  always accepting steps\n");CHKERRQ(ierr);}
      ierr = PetscViewerASCIIPrintf(viewer,"  safety factor %g\n",(double)adapt->safety);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer,"  extra safety factor after step rejection %g\n",(double)adapt->reject_safety);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer,"  clip fastest increase %g\n",(double)adapt->clip[1]);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer,"  clip fastest decrease %g\n",(double)adapt->clip[0]);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer,"  maximum allowed timestep %g\n",(double)adapt->dt_max);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer,"  minimum allowed timestep %g\n",(double)adapt->dt_min);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer,"  maximum solution absolute value to be ignored %g\n",(double)adapt->ignore_max);CHKERRQ(ierr);
    }
    if (isglee) {
      if (adapt->glee_use_local) {
        ierr = PetscViewerASCIIPrintf(viewer,"  GLEE uses local error control\n");CHKERRQ(ierr);
      } else {
        ierr = PetscViewerASCIIPrintf(viewer,"  GLEE uses global error control\n");CHKERRQ(ierr);
      }
    }
    if (adapt->ops->view) {
      ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);
      ierr = (*adapt->ops->view)(adapt,viewer);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPopTab(viewer);CHKERRQ(ierr);
    }
  } else if (isbinary) {
    char type[256];

    /* need to save FILE_CLASS_ID for adapt class */
    ierr = PetscStrncpy(type,((PetscObject)adapt)->type_name,256);CHKERRQ(ierr);
    ierr = PetscViewerBinaryWrite(viewer,type,256,PETSC_CHAR);CHKERRQ(ierr);
  } else if (adapt->ops->view) {
    ierr = (*adapt->ops->view)(adapt,viewer);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

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

/*@
   TSAdaptSetMonitor - Monitor the choices made by the adaptive controller

   Collective on TSAdapt

   Input Parameters:
+  adapt - adaptive controller context
-  flg - PETSC_TRUE to active a monitor, PETSC_FALSE to disable

   Options Database Keys:
.  -ts_adapt_monitor - to turn on monitoring

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

/*@C
   TSAdaptSetCheckStage - Set a callback to check convergence for a stage

   Logically collective on TSAdapt

   Input Parameters:
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

/*@
   TSAdaptSetAlwaysAccept - Set whether to always accept steps regardless of
   any error or stability condition not meeting the prescribed goal.

   Logically collective on TSAdapt

   Input Parameters:
+  adapt - time step adaptivity context, usually gotten with TSGetAdapt()
-  flag - whether to always accept steps

   Options Database Keys:
.  -ts_adapt_always_accept - to always accept steps

   Level: intermediate

.seealso: TSAdapt, TSAdaptChoose()
@*/
PetscErrorCode TSAdaptSetAlwaysAccept(TSAdapt adapt,PetscBool flag)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(adapt,TSADAPT_CLASSID,1);
  PetscValidLogicalCollectiveBool(adapt,flag,2);
  adapt->always_accept = flag;
  PetscFunctionReturn(0);
}

/*@
   TSAdaptSetSafety - Set safety factors

   Logically collective on TSAdapt

   Input Parameters:
+  adapt - adaptive controller context
.  safety - safety factor relative to target error/stability goal
-  reject_safety - extra safety factor to apply if the last step was rejected

   Options Database Keys:
+  -ts_adapt_safety <safety> - to set safety factor
-  -ts_adapt_reject_safety <reject_safety> - to set reject safety factor

   Level: intermediate

.seealso: TSAdapt, TSAdaptGetSafety(), TSAdaptChoose()
@*/
PetscErrorCode TSAdaptSetSafety(TSAdapt adapt,PetscReal safety,PetscReal reject_safety)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(adapt,TSADAPT_CLASSID,1);
  PetscValidLogicalCollectiveReal(adapt,safety,2);
  PetscValidLogicalCollectiveReal(adapt,reject_safety,3);
  if (safety != PETSC_DEFAULT && safety < 0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Safety factor %g must be non negative",(double)safety);
  if (safety != PETSC_DEFAULT && safety > 1) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Safety factor %g must be less than one",(double)safety);
  if (reject_safety != PETSC_DEFAULT && reject_safety < 0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Reject safety factor %g must be non negative",(double)reject_safety);
  if (reject_safety != PETSC_DEFAULT && reject_safety > 1) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Reject safety factor %g must be less than one",(double)reject_safety);
  if (safety != PETSC_DEFAULT) adapt->safety = safety;
  if (reject_safety != PETSC_DEFAULT) adapt->reject_safety = reject_safety;
  PetscFunctionReturn(0);
}

/*@
   TSAdaptGetSafety - Get safety factors

   Not Collective

   Input Parameter:
.  adapt - adaptive controller context

   Output Parameters:
.  safety - safety factor relative to target error/stability goal
+  reject_safety - extra safety factor to apply if the last step was rejected

   Level: intermediate

.seealso: TSAdapt, TSAdaptSetSafety(), TSAdaptChoose()
@*/
PetscErrorCode TSAdaptGetSafety(TSAdapt adapt,PetscReal *safety,PetscReal *reject_safety)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(adapt,TSADAPT_CLASSID,1);
  if (safety)        PetscValidRealPointer(safety,2);
  if (reject_safety) PetscValidRealPointer(reject_safety,3);
  if (safety)        *safety        = adapt->safety;
  if (reject_safety) *reject_safety = adapt->reject_safety;
  PetscFunctionReturn(0);
}

/*@
   TSAdaptSetMaxIgnore - Set error estimation threshold. Solution components below this threshold value will not be considered when computing error norms for time step adaptivity (in absolute value). A negative value (default) of the threshold leads to considering all solution components.

   Logically collective on TSAdapt

   Input Parameters:
+  adapt - adaptive controller context
-  max_ignore - threshold for solution components that are ignored during error estimation

   Options Database Keys:
.  -ts_adapt_max_ignore <max_ignore> - to set the threshold

   Level: intermediate

.seealso: TSAdapt, TSAdaptGetMaxIgnore(), TSAdaptChoose()
@*/
PetscErrorCode TSAdaptSetMaxIgnore(TSAdapt adapt,PetscReal max_ignore)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(adapt,TSADAPT_CLASSID,1);
  PetscValidLogicalCollectiveReal(adapt,max_ignore,2);
  adapt->ignore_max = max_ignore;
  PetscFunctionReturn(0);
}

/*@
   TSAdaptGetMaxIgnore - Get error estimation threshold. Solution components below this threshold value will not be considered when computing error norms for time step adaptivity (in absolute value).

   Not Collective

   Input Parameter:
.  adapt - adaptive controller context

   Output Parameter:
.  max_ignore - threshold for solution components that are ignored during error estimation

   Level: intermediate

.seealso: TSAdapt, TSAdaptSetMaxIgnore(), TSAdaptChoose()
@*/
PetscErrorCode TSAdaptGetMaxIgnore(TSAdapt adapt,PetscReal *max_ignore)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(adapt,TSADAPT_CLASSID,1);
  PetscValidRealPointer(max_ignore,2);
  *max_ignore = adapt->ignore_max;
  PetscFunctionReturn(0);
}

/*@
   TSAdaptSetClip - Sets the admissible decrease/increase factor in step size

   Logically collective on TSAdapt

   Input Parameters:
+  adapt - adaptive controller context
.  low - admissible decrease factor
-  high - admissible increase factor

   Options Database Keys:
.  -ts_adapt_clip <low>,<high> - to set admissible time step decrease and increase factors

   Level: intermediate

.seealso: TSAdaptChoose(), TSAdaptGetClip(), TSAdaptSetScaleSolveFailed()
@*/
PetscErrorCode TSAdaptSetClip(TSAdapt adapt,PetscReal low,PetscReal high)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(adapt,TSADAPT_CLASSID,1);
  PetscValidLogicalCollectiveReal(adapt,low,2);
  PetscValidLogicalCollectiveReal(adapt,high,3);
  if (low  != PETSC_DEFAULT && low  < 0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Decrease factor %g must be non negative",(double)low);
  if (low  != PETSC_DEFAULT && low  > 1) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Decrease factor %g must be less than one",(double)low);
  if (high != PETSC_DEFAULT && high < 1) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Increase factor %g must be greater than one",(double)high);
  if (low  != PETSC_DEFAULT) adapt->clip[0] = low;
  if (high != PETSC_DEFAULT) adapt->clip[1] = high;
  PetscFunctionReturn(0);
}

/*@
   TSAdaptGetClip - Gets the admissible decrease/increase factor in step size

   Not Collective

   Input Parameter:
.  adapt - adaptive controller context

   Output Parameters:
+  low - optional, admissible decrease factor
-  high - optional, admissible increase factor

   Level: intermediate

.seealso: TSAdaptChoose(), TSAdaptSetClip(), TSAdaptSetScaleSolveFailed()
@*/
PetscErrorCode TSAdaptGetClip(TSAdapt adapt,PetscReal *low,PetscReal *high)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(adapt,TSADAPT_CLASSID,1);
  if (low)  PetscValidRealPointer(low,2);
  if (high) PetscValidRealPointer(high,3);
  if (low)  *low  = adapt->clip[0];
  if (high) *high = adapt->clip[1];
  PetscFunctionReturn(0);
}

/*@
   TSAdaptSetScaleSolveFailed - Scale step by this factor if solve fails

   Logically collective on TSAdapt

   Input Parameters:
+  adapt - adaptive controller context
-  scale - scale

   Options Database Keys:
.  -ts_adapt_scale_solve_failed <scale> - to set scale step by this factor if solve fails

   Level: intermediate

.seealso: TSAdaptChoose(), TSAdaptGetScaleSolveFailed(), TSAdaptGetClip()
@*/
PetscErrorCode TSAdaptSetScaleSolveFailed(TSAdapt adapt,PetscReal scale)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(adapt,TSADAPT_CLASSID,1);
  PetscValidLogicalCollectiveReal(adapt,scale,2);
  if (scale != PETSC_DEFAULT && scale <= 0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Scale factor %g must be positive",(double)scale);
  if (scale != PETSC_DEFAULT && scale  > 1) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Scale factor %g must be less than one",(double)scale);
  if (scale != PETSC_DEFAULT) adapt->scale_solve_failed = scale;
  PetscFunctionReturn(0);
}

/*@
   TSAdaptGetScaleSolveFailed - Gets the admissible decrease/increase factor in step size

   Not Collective

   Input Parameter:
.  adapt - adaptive controller context

   Output Parameter:
.  scale - scale factor

   Level: intermediate

.seealso: TSAdaptChoose(), TSAdaptSetScaleSolveFailed(), TSAdaptSetClip()
@*/
PetscErrorCode TSAdaptGetScaleSolveFailed(TSAdapt adapt,PetscReal *scale)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(adapt,TSADAPT_CLASSID,1);
  if (scale)  PetscValidRealPointer(scale,2);
  if (scale)  *scale  = adapt->scale_solve_failed;
  PetscFunctionReturn(0);
}

/*@
   TSAdaptSetStepLimits - Set the minimum and maximum step sizes to be considered by the controller

   Logically collective on TSAdapt

   Input Parameters:
+  adapt - time step adaptivity context, usually gotten with TSGetAdapt()
.  hmin - minimum time step
-  hmax - maximum time step

   Options Database Keys:
+  -ts_adapt_dt_min <min> - to set minimum time step
-  -ts_adapt_dt_max <max> - to set maximum time step

   Level: intermediate

.seealso: TSAdapt, TSAdaptGetStepLimits(), TSAdaptChoose()
@*/
PetscErrorCode TSAdaptSetStepLimits(TSAdapt adapt,PetscReal hmin,PetscReal hmax)
{

  PetscFunctionBegin;
  PetscValidHeaderSpecific(adapt,TSADAPT_CLASSID,1);
  PetscValidLogicalCollectiveReal(adapt,hmin,2);
  PetscValidLogicalCollectiveReal(adapt,hmax,3);
  if (hmin != PETSC_DEFAULT && hmin < 0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Minimum time step %g must be non negative",(double)hmin);
  if (hmax != PETSC_DEFAULT && hmax < 0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Minimum time step %g must be non negative",(double)hmax);
  if (hmin != PETSC_DEFAULT) adapt->dt_min = hmin;
  if (hmax != PETSC_DEFAULT) adapt->dt_max = hmax;
  hmin = adapt->dt_min;
  hmax = adapt->dt_max;
  if (hmax <= hmin) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Maximum time step %g must greater than minimum time step %g",(double)hmax,(double)hmin);
  PetscFunctionReturn(0);
}

/*@
   TSAdaptGetStepLimits - Get the minimum and maximum step sizes to be considered by the controller

   Not Collective

   Input Parameter:
.  adapt - time step adaptivity context, usually gotten with TSGetAdapt()

   Output Parameters:
+  hmin - minimum time step
-  hmax - maximum time step

   Level: intermediate

.seealso: TSAdapt, TSAdaptSetStepLimits(), TSAdaptChoose()
@*/
PetscErrorCode TSAdaptGetStepLimits(TSAdapt adapt,PetscReal *hmin,PetscReal *hmax)
{

  PetscFunctionBegin;
  PetscValidHeaderSpecific(adapt,TSADAPT_CLASSID,1);
  if (hmin) PetscValidRealPointer(hmin,2);
  if (hmax) PetscValidRealPointer(hmax,3);
  if (hmin) *hmin = adapt->dt_min;
  if (hmax) *hmax = adapt->dt_max;
  PetscFunctionReturn(0);
}

/*
   TSAdaptSetFromOptions - Sets various TSAdapt parameters from user options.

   Collective on TSAdapt

   Input Parameter:
.  adapt - the TSAdapt context

   Options Database Keys:
+  -ts_adapt_type <type> - algorithm to use for adaptivity
.  -ts_adapt_always_accept - always accept steps regardless of error/stability goals
.  -ts_adapt_safety <safety> - safety factor relative to target error/stability goal
.  -ts_adapt_reject_safety <safety> - extra safety factor to apply if the last step was rejected
.  -ts_adapt_clip <low,high> - admissible time step decrease and increase factors
.  -ts_adapt_dt_min <min> - minimum timestep to use
.  -ts_adapt_dt_max <max> - maximum timestep to use
.  -ts_adapt_scale_solve_failed <scale> - scale timestep by this factor if a solve fails
.  -ts_adapt_wnormtype <2 or infinity> - type of norm for computing error estimates
-  -ts_adapt_time_step_increase_delay - number of timesteps to delay increasing the time step after it has been decreased due to failed solver

   Level: advanced

   Notes:
   This function is automatically called by TSSetFromOptions()

.seealso: TSGetAdapt(), TSAdaptSetType(), TSAdaptSetAlwaysAccept(), TSAdaptSetSafety(),
          TSAdaptSetClip(), TSAdaptSetScaleSolveFailed(), TSAdaptSetStepLimits(), TSAdaptSetMonitor()
*/
PetscErrorCode  TSAdaptSetFromOptions(PetscOptionItems *PetscOptionsObject,TSAdapt adapt)
{
  PetscErrorCode ierr;
  char           type[256] = TSADAPTBASIC;
  PetscReal      safety,reject_safety,clip[2],scale,hmin,hmax;
  PetscBool      set,flg;
  PetscInt       two;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(adapt,TSADAPT_CLASSID,2);
  /* This should use PetscOptionsBegin() if/when this becomes an object used outside of TS, but currently this
   * function can only be called from inside TSSetFromOptions()  */
  ierr = PetscOptionsHead(PetscOptionsObject,"TS Adaptivity options");CHKERRQ(ierr);
  ierr = PetscOptionsFList("-ts_adapt_type","Algorithm to use for adaptivity","TSAdaptSetType",TSAdaptList,((PetscObject)adapt)->type_name ? ((PetscObject)adapt)->type_name : type,type,sizeof(type),&flg);CHKERRQ(ierr);
  if (flg || !((PetscObject)adapt)->type_name) {
    ierr = TSAdaptSetType(adapt,type);CHKERRQ(ierr);
  }

  ierr = PetscOptionsBool("-ts_adapt_always_accept","Always accept the step","TSAdaptSetAlwaysAccept",adapt->always_accept,&flg,&set);CHKERRQ(ierr);
  if (set) {ierr = TSAdaptSetAlwaysAccept(adapt,flg);CHKERRQ(ierr);}

  safety = adapt->safety; reject_safety = adapt->reject_safety;
  ierr = PetscOptionsReal("-ts_adapt_safety","Safety factor relative to target error/stability goal","TSAdaptSetSafety",safety,&safety,&set);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-ts_adapt_reject_safety","Extra safety factor to apply if the last step was rejected","TSAdaptSetSafety",reject_safety,&reject_safety,&flg);CHKERRQ(ierr);
  if (set || flg) {ierr = TSAdaptSetSafety(adapt,safety,reject_safety);CHKERRQ(ierr);}

  two = 2; clip[0] = adapt->clip[0]; clip[1] = adapt->clip[1];
  ierr = PetscOptionsRealArray("-ts_adapt_clip","Admissible decrease/increase factor in step size","TSAdaptSetClip",clip,&two,&set);CHKERRQ(ierr);
  if (set && (two != 2)) SETERRQ(PetscObjectComm((PetscObject)adapt),PETSC_ERR_ARG_OUTOFRANGE,"Must give exactly two values to -ts_adapt_clip");
  if (set) {ierr = TSAdaptSetClip(adapt,clip[0],clip[1]);CHKERRQ(ierr);}

  hmin = adapt->dt_min; hmax = adapt->dt_max;
  ierr = PetscOptionsReal("-ts_adapt_dt_min","Minimum time step considered","TSAdaptSetStepLimits",hmin,&hmin,&set);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-ts_adapt_dt_max","Maximum time step considered","TSAdaptSetStepLimits",hmax,&hmax,&flg);CHKERRQ(ierr);
  if (set || flg) {ierr = TSAdaptSetStepLimits(adapt,hmin,hmax);CHKERRQ(ierr);}

  ierr = PetscOptionsReal("-ts_adapt_max_ignore","Adaptor ignores (absolute) solution values smaller than this value","",adapt->ignore_max,&adapt->ignore_max,&set);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-ts_adapt_glee_use_local","GLEE adaptor uses local error estimation for step control","",adapt->glee_use_local,&adapt->glee_use_local,&set);CHKERRQ(ierr);

  ierr = PetscOptionsReal("-ts_adapt_scale_solve_failed","Scale step by this factor if solve fails","TSAdaptSetScaleSolveFailed",adapt->scale_solve_failed,&scale,&set);CHKERRQ(ierr);
  if (set) {ierr = TSAdaptSetScaleSolveFailed(adapt,scale);CHKERRQ(ierr);}

  ierr = PetscOptionsEnum("-ts_adapt_wnormtype","Type of norm computed for error estimation","",NormTypes,(PetscEnum)adapt->wnormtype,(PetscEnum*)&adapt->wnormtype,NULL);CHKERRQ(ierr);
  if (adapt->wnormtype != NORM_2 && adapt->wnormtype != NORM_INFINITY) SETERRQ(PetscObjectComm((PetscObject)adapt),PETSC_ERR_SUP,"Only 2-norm and infinite norm supported");

  ierr = PetscOptionsInt("-ts_adapt_time_step_increase_delay","Number of timesteps to delay increasing the time step after it has been decreased due to failed solver","TSAdaptSetTimeStepIncreaseDelay",adapt->timestepjustdecreased_delay,&adapt->timestepjustdecreased_delay,NULL);CHKERRQ(ierr);

  ierr = PetscOptionsBool("-ts_adapt_monitor","Print choices made by adaptive controller","TSAdaptSetMonitor",adapt->monitor ? PETSC_TRUE : PETSC_FALSE,&flg,&set);CHKERRQ(ierr);
  if (set) {ierr = TSAdaptSetMonitor(adapt,flg);CHKERRQ(ierr);}

  if (adapt->ops->setfromoptions) {ierr = (*adapt->ops->setfromoptions)(PetscOptionsObject,adapt);CHKERRQ(ierr);}
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
   TSAdaptCandidatesClear - clear any previously set candidate schemes

   Logically collective on TSAdapt

   Input Parameter:
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

/*@C
   TSAdaptCandidateAdd - add a candidate scheme for the adaptive controller to select from

   Logically collective on TSAdapt

   Input Parameters:
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
  if (order < 1) SETERRQ(PetscObjectComm((PetscObject)adapt),PETSC_ERR_ARG_OUTOFRANGE,"Classical order %D must be a positive integer",order);
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

/*@C
   TSAdaptCandidatesGet - Get the list of candidate orders of accuracy and cost

   Not Collective

   Input Parameter:
.  adapt - time step adaptivity context

   Output Parameters:
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

/*@C
   TSAdaptChoose - choose which method and step size to use for the next step

   Collective on TSAdapt

   Input Parameters:
+  adapt - adaptive contoller
.  ts - time stepper
-  h - current step size

   Output Parameters:
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
  PetscReal      wltea = -1.0;
  PetscReal      wlter = -1.0;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(adapt,TSADAPT_CLASSID,1);
  PetscValidHeaderSpecific(ts,TS_CLASSID,2);
  if (next_sc) PetscValidIntPointer(next_sc,4);
  PetscValidPointer(next_h,5);
  PetscValidBoolPointer(accept,6);
  if (next_sc) *next_sc = 0;

  /* Do not mess with adaptivity while handling events*/
  if (ts->event && ts->event->status != TSEVENT_NONE) {
    *next_h = h;
    *accept = PETSC_TRUE;
    PetscFunctionReturn(0);
  }

  ierr = (*adapt->ops->choose)(adapt,ts,h,&scheme,next_h,accept,&wlte,&wltea,&wlter);CHKERRQ(ierr);
  if (scheme < 0 || (ncandidates > 0 && scheme >= ncandidates)) SETERRQ(PetscObjectComm((PetscObject)adapt),PETSC_ERR_ARG_OUTOFRANGE,"Chosen scheme %D not in valid range 0..%D",scheme,ncandidates-1);
  if (*next_h < 0) SETERRQ(PetscObjectComm((PetscObject)adapt),PETSC_ERR_ARG_OUTOFRANGE,"Computed step size %g must be positive",(double)*next_h);
  if (next_sc) *next_sc = scheme;

  if (*accept && ts->exact_final_time == TS_EXACTFINALTIME_MATCHSTEP) {
    /* Increase/reduce step size if end time of next step is close to or overshoots max time */
    PetscReal t = ts->ptime + ts->time_step, h = *next_h;
    PetscReal tend = t + h, tmax = ts->max_time, hmax = tmax - t;
    PetscReal a = (PetscReal)(1.0 + adapt->matchstepfac[0]);
    PetscReal b = adapt->matchstepfac[1];
    if (t < tmax && tend > tmax) *next_h = hmax;
    if (t < tmax && tend < tmax && h*b > hmax) *next_h = hmax/2;
    if (t < tmax && tend < tmax && h*a > hmax) *next_h = hmax;
  }

  if (adapt->monitor) {
    const char *sc_name = (scheme < ncandidates) ? adapt->candidates.name[scheme] : "";
    ierr = PetscViewerASCIIAddTab(adapt->monitor,((PetscObject)adapt)->tablevel);CHKERRQ(ierr);
    if (wlte < 0) {
      ierr = PetscViewerASCIIPrintf(adapt->monitor,"    TSAdapt %s %s %D:%s step %3D %s t=%-11g+%10.3e dt=%-10.3e\n",((PetscObject)adapt)->type_name,((PetscObject)ts)->type_name,scheme,sc_name,ts->steps,*accept ? "accepted" : "rejected",(double)ts->ptime,(double)h,(double)*next_h);CHKERRQ(ierr);
    } else {
      ierr = PetscViewerASCIIPrintf(adapt->monitor,"    TSAdapt %s %s %D:%s step %3D %s t=%-11g+%10.3e dt=%-10.3e wlte=%5.3g  wltea=%5.3g wlter=%5.3g\n",((PetscObject)adapt)->type_name,((PetscObject)ts)->type_name,scheme,sc_name,ts->steps,*accept ? "accepted" : "rejected",(double)ts->ptime,(double)h,(double)*next_h,(double)wlte,(double)wltea,(double)wlter);CHKERRQ(ierr);
    }
    ierr = PetscViewerASCIISubtractTab(adapt->monitor,((PetscObject)adapt)->tablevel);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*@
   TSAdaptSetTimeStepIncreaseDelay - The number of timesteps to wait after a decrease in the timestep due to failed solver
                                     before increasing the time step.

   Logicially Collective on TSAdapt

   Input Parameters:
+  adapt - adaptive controller context
-  cnt - the number of timesteps

   Options Database Key:
.  -ts_adapt_time_step_increase_delay cnt - number of steps to delay the increase

   Notes: This is to prevent an adaptor from bouncing back and forth between two nearby timesteps. The default is 0.
          The successful use of this option is problem dependent

   Developer Note: there is no theory to support this option

   Level: advanced

.seealso:
@*/
PetscErrorCode TSAdaptSetTimeStepIncreaseDelay(TSAdapt adapt,PetscInt cnt)
{
  PetscFunctionBegin;
  adapt->timestepjustdecreased_delay = cnt;
  PetscFunctionReturn(0);
}

/*@
   TSAdaptCheckStage - checks whether to accept a stage, (e.g. reject and change time step size if nonlinear solve fails or solution vector is infeasible)

   Collective on TSAdapt

   Input Parameters:
+  adapt - adaptive controller context
.  ts - time stepper
.  t - Current simulation time
-  Y - Current solution vector

   Output Parameter:
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
  PetscValidBoolPointer(accept,5);

  if (ts->snes) {ierr = SNESGetConvergedReason(ts->snes,&snesreason);CHKERRQ(ierr);}
  if (snesreason < 0) {
    *accept = PETSC_FALSE;
    if (++ts->num_snes_failures >= ts->max_snes_failures && ts->max_snes_failures > 0) {
      ts->reason = TS_DIVERGED_NONLINEAR_SOLVE;
      ierr = PetscInfo(ts,"Step=%D, nonlinear solve failures %D greater than current TS allowed, stopping solve\n",ts->steps,ts->num_snes_failures);CHKERRQ(ierr);
      if (adapt->monitor) {
        ierr = PetscViewerASCIIAddTab(adapt->monitor,((PetscObject)adapt)->tablevel);CHKERRQ(ierr);
        ierr = PetscViewerASCIIPrintf(adapt->monitor,"    TSAdapt %s step %3D stage rejected t=%-11g+%10.3e, nonlinear solve failures %D greater than current TS allowed\n",((PetscObject)adapt)->type_name,ts->steps,(double)ts->ptime,(double)ts->time_step,ts->num_snes_failures);CHKERRQ(ierr);
        ierr = PetscViewerASCIISubtractTab(adapt->monitor,((PetscObject)adapt)->tablevel);CHKERRQ(ierr);
      }
    }
  } else {
    *accept = PETSC_TRUE;
    ierr = TSFunctionDomainError(ts,t,Y,accept);CHKERRQ(ierr);
    if (*accept && adapt->checkstage) {
      ierr = (*adapt->checkstage)(adapt,ts,t,Y,accept);CHKERRQ(ierr);
      if (!*accept) {
        ierr = PetscInfo(ts,"Step=%D, solution rejected by user function provided by TSSetFunctionDomainError()\n",ts->steps);CHKERRQ(ierr);
        if (adapt->monitor) {
          ierr = PetscViewerASCIIAddTab(adapt->monitor,((PetscObject)adapt)->tablevel);CHKERRQ(ierr);
          ierr = PetscViewerASCIIPrintf(adapt->monitor,"    TSAdapt %s step %3D stage rejected by user function provided by TSSetFunctionDomainError()\n",((PetscObject)adapt)->type_name,ts->steps);CHKERRQ(ierr);
          ierr = PetscViewerASCIISubtractTab(adapt->monitor,((PetscObject)adapt)->tablevel);CHKERRQ(ierr);
        }
      }
    }
  }

  if (!(*accept) && !ts->reason) {
    PetscReal dt,new_dt;
    ierr = TSGetTimeStep(ts,&dt);CHKERRQ(ierr);
    new_dt = dt * adapt->scale_solve_failed;
    ierr = TSSetTimeStep(ts,new_dt);CHKERRQ(ierr);
    adapt->timestepjustdecreased += adapt->timestepjustdecreased_delay;
    if (adapt->monitor) {
      ierr = PetscViewerASCIIAddTab(adapt->monitor,((PetscObject)adapt)->tablevel);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(adapt->monitor,"    TSAdapt %s step %3D stage rejected (%s) t=%-11g+%10.3e retrying with dt=%-10.3e\n",((PetscObject)adapt)->type_name,ts->steps,SNESConvergedReasons[snesreason],(double)ts->ptime,(double)dt,(double)new_dt);CHKERRQ(ierr);
      ierr = PetscViewerASCIISubtractTab(adapt->monitor,((PetscObject)adapt)->tablevel);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

/*@
  TSAdaptCreate - create an adaptive controller context for time stepping

  Collective

  Input Parameter:
. comm - The communicator

  Output Parameter:
. adapt - new TSAdapt object

  Level: developer

  Notes:
  TSAdapt creation is handled by TS, so users should not need to call this function.

.seealso: TSGetAdapt(), TSAdaptSetType(), TSAdaptDestroy()
@*/
PetscErrorCode  TSAdaptCreate(MPI_Comm comm,TSAdapt *inadapt)
{
  PetscErrorCode ierr;
  TSAdapt        adapt;

  PetscFunctionBegin;
  PetscValidPointer(inadapt,2);
  *inadapt = NULL;
  ierr = TSAdaptInitializePackage();CHKERRQ(ierr);

  ierr = PetscHeaderCreate(adapt,TSADAPT_CLASSID,"TSAdapt","Time stepping adaptivity","TS",comm,TSAdaptDestroy,TSAdaptView);CHKERRQ(ierr);

  adapt->always_accept      = PETSC_FALSE;
  adapt->safety             = 0.9;
  adapt->reject_safety      = 0.5;
  adapt->clip[0]            = 0.1;
  adapt->clip[1]            = 10.;
  adapt->dt_min             = 1e-20;
  adapt->dt_max             = 1e+20;
  adapt->ignore_max         = -1.0;
  adapt->glee_use_local     = PETSC_TRUE;
  adapt->scale_solve_failed = 0.25;
  /* these two safety factors are not public, and they are used only in the TS_EXACTFINALTIME_MATCHSTEP case
     to prevent from situations were unreasonably small time steps are taken in order to match the final time */
  adapt->matchstepfac[0]    = 0.01; /* allow 1% step size increase in the last step */
  adapt->matchstepfac[1]    = 2.0;  /* halve last step if it is greater than what remains divided this factor */
  adapt->wnormtype          = NORM_2;
  adapt->timestepjustdecreased_delay = 0;

  *inadapt = adapt;
  PetscFunctionReturn(0);
}
