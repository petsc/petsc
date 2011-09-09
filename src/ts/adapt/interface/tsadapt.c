
#include <private/tsimpl.h> /*I  "petscts.h" I*/

static PetscFList TSAdaptList;
static PetscBool  TSAdaptPackageInitialized;
static PetscBool  TSAdaptRegisterAllCalled;
static PetscClassId TSADAPT_CLASSID;

EXTERN_C_BEGIN
PetscErrorCode  TSAdaptCreate_Basic(TSAdapt);
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
  ierr = PetscHeaderDestroy(adapt);CHKERRQ(ierr);
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
  PetscBool      flg;

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
.  leadingerror - leading error coefficient of the candidate scheme
.  cost - relative measure of the amount of work required for the candidate scheme
-  inuse - indicates that this scheme is the one currently in use, this flag can only be set for one scheme

   Note:
   This routine is not available in Fortran.

   Level: developer

.seealso: TSAdaptCandidatesClear(), TSAdaptChoose()
@*/
PetscErrorCode TSAdaptCandidateAdd(TSAdapt adapt,const char name[],PetscInt order,PetscInt stageorder,PetscReal leadingerror,PetscReal cost,PetscBool inuse)
{
  PetscInt c;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(adapt,TSADAPT_CLASSID,1);
  if (order < 1) SETERRQ1(((PetscObject)adapt)->comm,PETSC_ERR_ARG_OUTOFRANGE,"Classical order %D must be a positive integer",order);
  if (inuse) {
    if (adapt->candidates.inuse_set) SETERRQ(((PetscObject)adapt)->comm,PETSC_ERR_ARG_WRONGSTATE,"Cannot set the inuse method twice, maybe forgot to call TSAdaptCandidatesClear()");
    adapt->candidates.inuse_set = PETSC_TRUE;
  }
  c = !adapt->candidates.order[0] + adapt->candidates.n;
  adapt->candidates.name[c]         = name;
  adapt->candidates.order[c]        = order;
  adapt->candidates.stageorder[c]   = stageorder;
  adapt->candidates.leadingerror[c] = leadingerror;
  adapt->candidates.cost[c]         = cost;
  adapt->candidates.n++;
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

   Level: developer

.seealso: TSAdapt, TSAdaptCandidatesClear(), TSAdaptCandidateAdd()
@*/
PetscErrorCode TSAdaptChoose(TSAdapt adapt,TS ts,PetscReal h,PetscInt *next_sc,PetscReal *next_h,PetscBool *accept)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(adapt,TSADAPT_CLASSID,1);
  PetscValidHeaderSpecific(ts,TS_CLASSID,2);
  PetscValidIntPointer(next_sc,4);
  PetscValidPointer(next_h,5);
  PetscValidIntPointer(accept,6);
  if (adapt->candidates.n < 1) SETERRQ1(((PetscObject)adapt)->comm,PETSC_ERR_ARG_WRONGSTATE,"%D candidates have been registered",adapt->candidates.n);
  if (!adapt->candidates.inuse_set) SETERRQ1(((PetscObject)adapt)->comm,PETSC_ERR_ARG_WRONGSTATE,"The current in-use scheme is not among the %D candidates",adapt->candidates.n);
  ierr = (*adapt->ops->choose)(adapt,ts,h,next_sc,next_h,accept);CHKERRQ(ierr);
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
  *inadapt = adapt;
  PetscFunctionReturn(0);
}
