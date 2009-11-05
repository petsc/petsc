#define PETSCTS_DLL

#include "gl.h" /*I  "petscts.h" I*/

static PetscFList TSGLAdaptList;
static PetscTruth TSGLAdaptPackageInitialized;
static PetscTruth TSGLAdaptRegisterAllCalled;
static PetscCookie TSGLADAPT_COOKIE;

struct _TSGLAdaptOps {
  PetscErrorCode (*choose)(TSGLAdapt,PetscInt,const PetscInt[],const PetscReal[],const PetscReal[],PetscInt,PetscReal,PetscReal,PetscInt*,PetscReal*,PetscTruth*);
  PetscErrorCode (*destroy)(TSGLAdapt);
  PetscErrorCode (*view)(TSGLAdapt,PetscViewer);
  PetscErrorCode (*setfromoptions)(TSGLAdapt);
};

struct _p_TSGLAdapt {
  PETSCHEADER(struct _TSGLAdaptOps);
  void *data;
};

static PetscErrorCode TSGLAdaptCreate_None(TSGLAdapt);
static PetscErrorCode TSGLAdaptCreate_Size(TSGLAdapt);
static PetscErrorCode TSGLAdaptCreate_Both(TSGLAdapt);

#undef __FUNCT__  
#define __FUNCT__ "TSGLAdaptRegister"
/*@C
   TSGLAdaptRegister - see TSGLAdaptRegisterDynamic()

   Level: advanced
@*/
PetscErrorCode PETSCTS_DLLEXPORT TSGLAdaptRegister(const char sname[],const char path[],const char name[],PetscErrorCode (*function)(TSGLAdapt))
{
  PetscErrorCode ierr;
  char           fullname[PETSC_MAX_PATH_LEN];

  PetscFunctionBegin;
  ierr = PetscFListConcat(path,name,fullname);CHKERRQ(ierr);
  ierr = PetscFListAdd(&TSGLAdaptList,sname,fullname,(void(*)(void))function);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "TSGLAdaptRegisterAll"
/*@C
  TSGLAdaptRegisterAll - Registers all of the adaptivity schemes in TSGLAdapt

  Not Collective

  Level: advanced

.keywords: TSGLAdapt, register, all

.seealso: TSGLAdaptRegisterDestroy()
@*/
PetscErrorCode PETSCTS_DLLEXPORT TSGLAdaptRegisterAll(const char path[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = TSGLAdaptRegisterDynamic(TSGLADAPT_NONE,path,"TSGLAdaptCreate_None",TSGLAdaptCreate_None);CHKERRQ(ierr);
  ierr = TSGLAdaptRegisterDynamic(TSGLADAPT_SIZE,path,"TSGLAdaptCreate_Size",TSGLAdaptCreate_Size);CHKERRQ(ierr);
  ierr = TSGLAdaptRegisterDynamic(TSGLADAPT_BOTH,path,"TSGLAdaptCreate_Both",TSGLAdaptCreate_Both);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "TSGLFinalizePackage"
/*@C
  TSGLFinalizePackage - This function destroys everything in the TSGL package. It is
  called from PetscFinalize().

  Level: developer

.keywords: Petsc, destroy, package
.seealso: PetscFinalize()
@*/
PetscErrorCode PETSCTS_DLLEXPORT TSGLAdaptFinalizePackage(void) 
{
  PetscFunctionBegin;
  TSGLAdaptPackageInitialized = PETSC_FALSE;
  TSGLAdaptRegisterAllCalled  = PETSC_FALSE;
  TSGLAdaptList               = PETSC_NULL;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "TSGLInitializePackage"
/*@C
  TSGLAdaptInitializePackage - This function initializes everything in the TSGLAdapt package. It is
  called from PetscDLLibraryRegister() when using dynamic libraries, and on the first call to
  TSCreate_GL() when using static libraries.

  Input Parameter:
  path - The dynamic library path, or PETSC_NULL

  Level: developer

.keywords: TSGLAdapt, initialize, package
.seealso: PetscInitialize()
@*/
PetscErrorCode PETSCTS_DLLEXPORT TSGLAdaptInitializePackage(const char path[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (TSGLAdaptPackageInitialized) PetscFunctionReturn(0);
  TSGLAdaptPackageInitialized = PETSC_TRUE;
  ierr = PetscCookieRegister("TSGLAdapt",&TSGLADAPT_COOKIE);CHKERRQ(ierr);
  ierr = TSGLAdaptRegisterAll(path);CHKERRQ(ierr);
  ierr = PetscRegisterFinalize(TSGLAdaptFinalizePackage);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "TSGLAdaptRegisterDestroy"
/*@C
   TSGLAdaptRegisterDestroy - Frees the list of adaptivity schemes that were registered by TSGLAdaptRegister()/TSGLAdaptRegisterDynamic().

   Not Collective

   Level: advanced

.keywords: TSGLAdapt, register, destroy
.seealso: TSGLAdaptRegister(), TSGLAdaptRegisterAll(), TSGLAdaptRegisterDynamic()
@*/
PetscErrorCode PETSCTS_DLLEXPORT TSGLAdaptRegisterDestroy(void)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFListDestroy(&TSGLAdaptList);CHKERRQ(ierr);
  TSGLAdaptRegisterAllCalled = PETSC_FALSE;
  PetscFunctionReturn(0);
}


#undef __FUNCT__  
#define __FUNCT__ "TSGLAdaptSetType"
PetscErrorCode PETSCTS_DLLEXPORT TSGLAdaptSetType(TSGLAdapt adapt,const TSGLAdaptType type)
{
  PetscErrorCode ierr,(*r)(TSGLAdapt);

  PetscFunctionBegin;
  ierr = PetscFListFind(TSGLAdaptList,((PetscObject)adapt)->comm,type,(void(**)(void))&r);CHKERRQ(ierr);
  if (!r) SETERRQ1(PETSC_ERR_ARG_UNKNOWN_TYPE,"Unknown TSGLAdapt type \"%s\" given",type);
  if (((PetscObject)adapt)->type_name) {ierr = (*adapt->ops->destroy)(adapt);CHKERRQ(ierr);}
  ierr = (*r)(adapt);CHKERRQ(ierr);
  ierr = PetscObjectChangeTypeName((PetscObject)adapt,type);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "TSGLAdaptSetOptionsPrefix"
PetscErrorCode PETSCTS_DLLEXPORT TSGLAdaptSetOptionsPrefix(TSGLAdapt adapt,const char prefix[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscObjectSetOptionsPrefix((PetscObject)adapt,prefix);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "TSGLAdaptView"
PetscErrorCode PETSCTS_DLLEXPORT TSGLAdaptView(TSGLAdapt adapt,PetscViewer viewer)
{
  PetscErrorCode ierr;
  PetscTruth iascii;

  PetscFunctionBegin;
  ierr = PetscTypeCompare((PetscObject)viewer,PETSC_VIEWER_ASCII,&iascii);CHKERRQ(ierr);
  if (iascii) {
    if (((PetscObject)adapt)->prefix) {
      ierr = PetscViewerASCIIPrintf(viewer,"TSGLAdapt object: (%s)\n",((PetscObject)adapt)->prefix);CHKERRQ(ierr);
    } else {
      ierr = PetscViewerASCIIPrintf(viewer,"TSGLAdapt object:\n");CHKERRQ(ierr);
    }
    ierr = PetscViewerASCIIPrintf(viewer,"  type: %s\n",((PetscObject)adapt)->type_name?((PetscObject)adapt)->type_name:"(not set yet)");CHKERRQ(ierr);
    if (adapt->ops->view) {
      ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);
      ierr = (*adapt->ops->view)(adapt,viewer);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPopTab(viewer);CHKERRQ(ierr);
    }
  } else {
    SETERRQ1(PETSC_ERR_SUP,"Viewer type %s not supported for TS_GL",((PetscObject)viewer)->type_name);
  }

  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "TSGLAdaptDestroy"
PetscErrorCode PETSCTS_DLLEXPORT TSGLAdaptDestroy(TSGLAdapt adapt)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(adapt,TSGLADAPT_COOKIE,1);
  if (--((PetscObject)adapt)->refct > 0) PetscFunctionReturn(0);
  if (adapt->ops->destroy) {ierr = (*adapt->ops->destroy)(adapt);CHKERRQ(ierr);}
  ierr = PetscHeaderDestroy(adapt);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "TSGLAdaptSetFromOptions"
PetscErrorCode PETSCTS_DLLEXPORT TSGLAdaptSetFromOptions(TSGLAdapt adapt)
{
  PetscErrorCode ierr;
  char           type[256] = TSGLADAPT_BOTH;
  PetscTruth     flg;

  PetscFunctionBegin;
  /* This should use PetscOptionsBegin() if/when this becomes an object used outside of TSGL, but currently this
  * function can only be called from inside TSSetFromOptions_GL()  */
  ierr = PetscOptionsHead("TSGL Adaptivity options");CHKERRQ(ierr);
  ierr = PetscOptionsList("-ts_adapt_type","Algorithm to use for adaptivity","TSGLAdaptSetType",TSGLAdaptList,
                          ((PetscObject)adapt)->type_name?((PetscObject)adapt)->type_name:type,type,sizeof type,&flg);CHKERRQ(ierr);
  if (flg || !((PetscObject)adapt)->type_name) {
    ierr = TSGLAdaptSetType(adapt,type);CHKERRQ(ierr);
  }
  if (adapt->ops->setfromoptions) {ierr = (*adapt->ops->setfromoptions)(adapt);CHKERRQ(ierr);}
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "TSGLAdaptChoose"
PetscErrorCode PETSCTS_DLLEXPORT TSGLAdaptChoose(TSGLAdapt adapt,PetscInt n,const PetscInt orders[],const PetscReal errors[],const PetscReal cost[],PetscInt cur,PetscReal h,PetscReal tleft,PetscInt *next_sc,PetscReal *next_h,PetscTruth *finish)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(adapt,TSGLADAPT_COOKIE,1);
  PetscValidIntPointer(orders,3);
  PetscValidPointer(errors,4);
  PetscValidPointer(cost,5);
  PetscValidIntPointer(next_sc,9);
  PetscValidPointer(next_h,10);
  PetscValidIntPointer(finish,11);
  ierr = (*adapt->ops->choose)(adapt,n,orders,errors,cost,cur,h,tleft,next_sc,next_h,finish);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "TSGLAdaptCreate"
PetscErrorCode PETSCTS_DLLEXPORT TSGLAdaptCreate(MPI_Comm comm,TSGLAdapt *inadapt)
{
  PetscErrorCode ierr;
  TSGLAdapt adapt;

  PetscFunctionBegin;
  *inadapt = 0;
  ierr = PetscHeaderCreate(adapt,_p_TSGLAdapt,struct _TSGLAdaptOps,TSGLADAPT_COOKIE,0,"TSGLAdapt",comm,TSGLAdaptDestroy,TSGLAdaptView);CHKERRQ(ierr);
  *inadapt = adapt;
  PetscFunctionReturn(0);
}


/*
*  Implementations
*/

#undef __FUNCT__  
#define __FUNCT__ "TSGLAdaptDestroy_JustFree"
static PetscErrorCode TSGLAdaptDestroy_JustFree(TSGLAdapt adapt)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFree(adapt->data);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* -------------------------------- None ----------------------------------- */
typedef struct {
  PetscInt scheme;
  PetscReal h;
} TSGLAdapt_None;

#undef __FUNCT__  
#define __FUNCT__ "TSGLAdaptChoose_None"
static PetscErrorCode TSGLAdaptChoose_None(TSGLAdapt adapt,PetscInt n,const PetscInt orders[],const PetscReal errors[],const PetscReal cost[],PetscInt cur,PetscReal h,PetscReal tleft,PetscInt *next_sc,PetscReal *next_h,PetscTruth *finish)
{

  PetscFunctionBegin;
  *next_sc = cur;
  *next_h = h;
  if (*next_h > tleft) {
    *finish = PETSC_TRUE;
    *next_h = tleft;
  } else {
    *finish = PETSC_FALSE;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "TSGLAdaptCreate_None"
static PetscErrorCode TSGLAdaptCreate_None(TSGLAdapt adapt)
{
  PetscErrorCode ierr;
  TSGLAdapt_None *a;

  PetscFunctionBegin;
  ierr = PetscNewLog(adapt,TSGLAdapt_None,&a);CHKERRQ(ierr);
  adapt->data = (void*)a;
  adapt->ops->choose = TSGLAdaptChoose_None;
  adapt->ops->destroy = TSGLAdaptDestroy_JustFree;
  PetscFunctionReturn(0);
}


/* -------------------------------- Size ----------------------------------- */
typedef struct {
  PetscReal desired_h;
} TSGLAdapt_Size;


#undef __FUNCT__  
#define __FUNCT__ "TSGLAdaptChoose_Size"
static PetscErrorCode TSGLAdaptChoose_Size(TSGLAdapt adapt,PetscInt n,const PetscInt orders[],const PetscReal errors[],const PetscReal cost[],PetscInt cur,PetscReal h,PetscReal tleft,PetscInt *next_sc,PetscReal *next_h,PetscTruth *finish)
{
  TSGLAdapt_Size *sz = (TSGLAdapt_Size*)adapt->data;
  PetscReal dec = 0.2,inc = 5.0,safe = 0.9,optimal,last_desired_h;

  PetscFunctionBegin;
  *next_sc = cur;
  optimal = pow(errors[cur],-1./(safe*orders[cur]));
  /* Step sizes oscillate when there is no smoothing.  Here we use a geometric mean of the the current step size and the
  * one that would have been taken (without smoothing) on the last step. */
  last_desired_h = sz->desired_h;
  sz->desired_h = h*PetscMax(dec,PetscMin(inc,optimal)); /* Trim to [dec,inc] */
  if (last_desired_h > 1e-14) {                          /* Normally only happens on the first step */
    *next_h = sqrt(last_desired_h * sz->desired_h);
  } else {
    *next_h = sz->desired_h;
  }
  if (*next_h > tleft) {
    *finish = PETSC_TRUE;
    *next_h = tleft;
  } else {
    *finish = PETSC_FALSE;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "TSGLAdaptCreate_Size"
static PetscErrorCode TSGLAdaptCreate_Size(TSGLAdapt adapt)
{
  PetscErrorCode ierr;
  TSGLAdapt_Size *a;

  PetscFunctionBegin;
  ierr = PetscNewLog(adapt,TSGLAdapt_Size,&a);CHKERRQ(ierr);
  adapt->data = (void*)a;
  adapt->ops->choose = TSGLAdaptChoose_Size;
  adapt->ops->destroy = TSGLAdaptDestroy_JustFree;
  PetscFunctionReturn(0);
}

/* -------------------------------- Both ----------------------------------- */
typedef struct {
  PetscInt count_at_order;
  PetscReal desired_h;
} TSGLAdapt_Both;


#undef __FUNCT__  
#define __FUNCT__ "TSGLAdaptChoose_Both"
static PetscErrorCode TSGLAdaptChoose_Both(TSGLAdapt adapt,PetscInt n,const PetscInt orders[],const PetscReal errors[],const PetscReal cost[],PetscInt cur,PetscReal h,PetscReal tleft,PetscInt *next_sc,PetscReal *next_h,PetscTruth *finish)
{
  TSGLAdapt_Both *both = (TSGLAdapt_Both*)adapt->data;
  PetscErrorCode ierr;
  PetscReal dec = 0.2,inc = 5.0,safe = 0.9;
  struct {PetscInt id; PetscReal h,eff;} best={-1,0,0},trial={-1,0,0},current={-1,0,0};
  PetscInt i;

  PetscFunctionBegin;
  for (i=0; i<n; i++) {
    PetscReal optimal;
    trial.id = i;
    optimal = pow(errors[i],-1./(safe*orders[i]));
    trial.h = h*optimal;
    trial.eff = trial.h/cost[i];
    if (trial.eff > best.eff) {ierr = PetscMemcpy(&best,&trial,sizeof(trial));CHKERRQ(ierr);}
    if (i == cur) {ierr = PetscMemcpy(&current,&trial,sizeof(trial));CHKERRQ(ierr);}
  }
  /* Only switch orders if the scheme offers significant benefits over the current one.
  When the scheme is not changing, only change step size if it offers significant benefits. */
  if (best.eff < 1.2*current.eff || both->count_at_order < orders[cur]+2) {
    PetscReal last_desired_h;
    *next_sc = current.id;
    last_desired_h = both->desired_h;
    both->desired_h = PetscMax(h*dec,PetscMin(h*inc,current.h));
    *next_h = (both->count_at_order > 0)
      ? sqrt(last_desired_h * both->desired_h)
      : both->desired_h;
    both->count_at_order++;
  } else {
    PetscReal rat = cost[best.id]/cost[cur];
    *next_sc = best.id;
    *next_h  = PetscMax(h*rat*dec,PetscMin(h*rat*inc,best.h));
    both->count_at_order = 0;
    both->desired_h = best.h;
  }

  if (*next_h > tleft) {
    *finish = PETSC_TRUE;
    *next_h = tleft;
  } else {
    *finish = PETSC_FALSE;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "TSGLAdaptCreate_Both"
static PetscErrorCode TSGLAdaptCreate_Both(TSGLAdapt adapt)
{
  PetscErrorCode ierr;
  TSGLAdapt_Both *a;

  PetscFunctionBegin;
  ierr = PetscNewLog(adapt,TSGLAdapt_Both,&a);CHKERRQ(ierr);
  adapt->data = (void*)a;
  adapt->ops->choose = TSGLAdaptChoose_Both;
  adapt->ops->destroy = TSGLAdaptDestroy_JustFree;
  PetscFunctionReturn(0);
}
