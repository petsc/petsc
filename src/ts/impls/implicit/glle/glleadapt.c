
#include <../src/ts/impls/implicit/glle/glle.h> /*I  "petscts.h" I*/

static PetscFunctionList TSGLLEAdaptList;
static PetscBool         TSGLLEAdaptPackageInitialized;
static PetscBool         TSGLLEAdaptRegisterAllCalled;
static PetscClassId      TSGLLEADAPT_CLASSID;

struct _TSGLLEAdaptOps {
  PetscErrorCode (*choose)(TSGLLEAdapt,PetscInt,const PetscInt[],const PetscReal[],const PetscReal[],PetscInt,PetscReal,PetscReal,PetscInt*,PetscReal*,PetscBool*);
  PetscErrorCode (*destroy)(TSGLLEAdapt);
  PetscErrorCode (*view)(TSGLLEAdapt,PetscViewer);
  PetscErrorCode (*setfromoptions)(PetscOptionItems*,TSGLLEAdapt);
};

struct _p_TSGLLEAdapt {
  PETSCHEADER(struct _TSGLLEAdaptOps);
  void *data;
};

PETSC_EXTERN PetscErrorCode TSGLLEAdaptCreate_None(TSGLLEAdapt);
PETSC_EXTERN PetscErrorCode TSGLLEAdaptCreate_Size(TSGLLEAdapt);
PETSC_EXTERN PetscErrorCode TSGLLEAdaptCreate_Both(TSGLLEAdapt);

/*@C
   TSGLLEAdaptRegister -  adds a TSGLLEAdapt implementation

   Not Collective

   Input Parameters:
+  name_scheme - name of user-defined adaptivity scheme
-  routine_create - routine to create method context

   Notes:
   TSGLLEAdaptRegister() may be called multiple times to add several user-defined families.

   Sample usage:
.vb
   TSGLLEAdaptRegister("my_scheme",MySchemeCreate);
.ve

   Then, your scheme can be chosen with the procedural interface via
$     TSGLLEAdaptSetType(ts,"my_scheme")
   or at runtime via the option
$     -ts_adapt_type my_scheme

   Level: advanced

.seealso: TSGLLEAdaptRegisterAll()
@*/
PetscErrorCode  TSGLLEAdaptRegister(const char sname[],PetscErrorCode (*function)(TSGLLEAdapt))
{
  PetscFunctionBegin;
  CHKERRQ(TSGLLEAdaptInitializePackage());
  CHKERRQ(PetscFunctionListAdd(&TSGLLEAdaptList,sname,function));
  PetscFunctionReturn(0);
}

/*@C
  TSGLLEAdaptRegisterAll - Registers all of the adaptivity schemes in TSGLLEAdapt

  Not Collective

  Level: advanced

.seealso: TSGLLEAdaptRegisterDestroy()
@*/
PetscErrorCode  TSGLLEAdaptRegisterAll(void)
{
  PetscFunctionBegin;
  if (TSGLLEAdaptRegisterAllCalled) PetscFunctionReturn(0);
  TSGLLEAdaptRegisterAllCalled = PETSC_TRUE;
  CHKERRQ(TSGLLEAdaptRegister(TSGLLEADAPT_NONE,TSGLLEAdaptCreate_None));
  CHKERRQ(TSGLLEAdaptRegister(TSGLLEADAPT_SIZE,TSGLLEAdaptCreate_Size));
  CHKERRQ(TSGLLEAdaptRegister(TSGLLEADAPT_BOTH,TSGLLEAdaptCreate_Both));
  PetscFunctionReturn(0);
}

/*@C
  TSGLLEFinalizePackage - This function destroys everything in the TSGLLE package. It is
  called from PetscFinalize().

  Level: developer

.seealso: PetscFinalize()
@*/
PetscErrorCode  TSGLLEAdaptFinalizePackage(void)
{
  PetscFunctionBegin;
  CHKERRQ(PetscFunctionListDestroy(&TSGLLEAdaptList));
  TSGLLEAdaptPackageInitialized = PETSC_FALSE;
  TSGLLEAdaptRegisterAllCalled  = PETSC_FALSE;
  PetscFunctionReturn(0);
}

/*@C
  TSGLLEAdaptInitializePackage - This function initializes everything in the TSGLLEAdapt package. It is
  called from TSInitializePackage().

  Level: developer

.seealso: PetscInitialize()
@*/
PetscErrorCode  TSGLLEAdaptInitializePackage(void)
{
  PetscFunctionBegin;
  if (TSGLLEAdaptPackageInitialized) PetscFunctionReturn(0);
  TSGLLEAdaptPackageInitialized = PETSC_TRUE;
  CHKERRQ(PetscClassIdRegister("TSGLLEAdapt",&TSGLLEADAPT_CLASSID));
  CHKERRQ(TSGLLEAdaptRegisterAll());
  CHKERRQ(PetscRegisterFinalize(TSGLLEAdaptFinalizePackage));
  PetscFunctionReturn(0);
}

PetscErrorCode  TSGLLEAdaptSetType(TSGLLEAdapt adapt,TSGLLEAdaptType type)
{
  PetscErrorCode (*r)(TSGLLEAdapt);

  PetscFunctionBegin;
  CHKERRQ(PetscFunctionListFind(TSGLLEAdaptList,type,&r));
  PetscCheck(r,PETSC_COMM_SELF,PETSC_ERR_ARG_UNKNOWN_TYPE,"Unknown TSGLLEAdapt type \"%s\" given",type);
  if (((PetscObject)adapt)->type_name) CHKERRQ((*adapt->ops->destroy)(adapt));
  CHKERRQ((*r)(adapt));
  CHKERRQ(PetscObjectChangeTypeName((PetscObject)adapt,type));
  PetscFunctionReturn(0);
}

PetscErrorCode  TSGLLEAdaptSetOptionsPrefix(TSGLLEAdapt adapt,const char prefix[])
{
  PetscFunctionBegin;
  CHKERRQ(PetscObjectSetOptionsPrefix((PetscObject)adapt,prefix));
  PetscFunctionReturn(0);
}

PetscErrorCode  TSGLLEAdaptView(TSGLLEAdapt adapt,PetscViewer viewer)
{
  PetscBool      iascii;

  PetscFunctionBegin;
  CHKERRQ(PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii));
  if (iascii) {
    CHKERRQ(PetscObjectPrintClassNamePrefixType((PetscObject)adapt,viewer));
    if (adapt->ops->view) {
      CHKERRQ(PetscViewerASCIIPushTab(viewer));
      CHKERRQ((*adapt->ops->view)(adapt,viewer));
      CHKERRQ(PetscViewerASCIIPopTab(viewer));
    }
  }
  PetscFunctionReturn(0);
}

PetscErrorCode  TSGLLEAdaptDestroy(TSGLLEAdapt *adapt)
{
  PetscFunctionBegin;
  if (!*adapt) PetscFunctionReturn(0);
  PetscValidHeaderSpecific(*adapt,TSGLLEADAPT_CLASSID,1);
  if (--((PetscObject)(*adapt))->refct > 0) {*adapt = NULL; PetscFunctionReturn(0);}
  if ((*adapt)->ops->destroy) CHKERRQ((*(*adapt)->ops->destroy)(*adapt));
  CHKERRQ(PetscHeaderDestroy(adapt));
  PetscFunctionReturn(0);
}

PetscErrorCode  TSGLLEAdaptSetFromOptions(PetscOptionItems *PetscOptionsObject,TSGLLEAdapt adapt)
{
  PetscErrorCode ierr;
  char           type[256] = TSGLLEADAPT_BOTH;
  PetscBool      flg;

  PetscFunctionBegin;
  /* This should use PetscOptionsBegin() if/when this becomes an object used outside of TSGLLE, but currently this
  * function can only be called from inside TSSetFromOptions_GLLE()  */
  CHKERRQ(PetscOptionsHead(PetscOptionsObject,"TSGLLE Adaptivity options"));
  ierr = PetscOptionsFList("-ts_adapt_type","Algorithm to use for adaptivity","TSGLLEAdaptSetType",TSGLLEAdaptList,
                          ((PetscObject)adapt)->type_name ? ((PetscObject)adapt)->type_name : type,type,sizeof(type),&flg);CHKERRQ(ierr);
  if (flg || !((PetscObject)adapt)->type_name) {
    CHKERRQ(TSGLLEAdaptSetType(adapt,type));
  }
  if (adapt->ops->setfromoptions) CHKERRQ((*adapt->ops->setfromoptions)(PetscOptionsObject,adapt));
  CHKERRQ(PetscOptionsTail());
  PetscFunctionReturn(0);
}

PetscErrorCode  TSGLLEAdaptChoose(TSGLLEAdapt adapt,PetscInt n,const PetscInt orders[],const PetscReal errors[],const PetscReal cost[],PetscInt cur,PetscReal h,PetscReal tleft,PetscInt *next_sc,PetscReal *next_h,PetscBool  *finish)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(adapt,TSGLLEADAPT_CLASSID,1);
  PetscValidIntPointer(orders,3);
  PetscValidRealPointer(errors,4);
  PetscValidRealPointer(cost,5);
  PetscValidIntPointer(next_sc,9);
  PetscValidRealPointer(next_h,10);
  PetscValidBoolPointer(finish,11);
  CHKERRQ((*adapt->ops->choose)(adapt,n,orders,errors,cost,cur,h,tleft,next_sc,next_h,finish));
  PetscFunctionReturn(0);
}

PetscErrorCode  TSGLLEAdaptCreate(MPI_Comm comm,TSGLLEAdapt *inadapt)
{
  TSGLLEAdapt      adapt;

  PetscFunctionBegin;
  *inadapt = NULL;
  CHKERRQ(PetscHeaderCreate(adapt,TSGLLEADAPT_CLASSID,"TSGLLEAdapt","General Linear adaptivity","TS",comm,TSGLLEAdaptDestroy,TSGLLEAdaptView));
  *inadapt = adapt;
  PetscFunctionReturn(0);
}

/*
   Implementations
*/

static PetscErrorCode TSGLLEAdaptDestroy_JustFree(TSGLLEAdapt adapt)
{
  PetscFunctionBegin;
  CHKERRQ(PetscFree(adapt->data));
  PetscFunctionReturn(0);
}

/* -------------------------------- None ----------------------------------- */
typedef struct {
  PetscInt  scheme;
  PetscReal h;
} TSGLLEAdapt_None;

static PetscErrorCode TSGLLEAdaptChoose_None(TSGLLEAdapt adapt,PetscInt n,const PetscInt orders[],const PetscReal errors[],const PetscReal cost[],PetscInt cur,PetscReal h,PetscReal tleft,PetscInt *next_sc,PetscReal *next_h,PetscBool  *finish)
{
  PetscFunctionBegin;
  *next_sc = cur;
  *next_h  = h;
  if (*next_h > tleft) {
    *finish = PETSC_TRUE;
    *next_h = tleft;
  } else *finish = PETSC_FALSE;
  PetscFunctionReturn(0);
}

PetscErrorCode  TSGLLEAdaptCreate_None(TSGLLEAdapt adapt)
{
  TSGLLEAdapt_None *a;

  PetscFunctionBegin;
  CHKERRQ(PetscNewLog(adapt,&a));
  adapt->data         = (void*)a;
  adapt->ops->choose  = TSGLLEAdaptChoose_None;
  adapt->ops->destroy = TSGLLEAdaptDestroy_JustFree;
  PetscFunctionReturn(0);
}

/* -------------------------------- Size ----------------------------------- */
typedef struct {
  PetscReal desired_h;
} TSGLLEAdapt_Size;

static PetscErrorCode TSGLLEAdaptChoose_Size(TSGLLEAdapt adapt,PetscInt n,const PetscInt orders[],const PetscReal errors[],const PetscReal cost[],PetscInt cur,PetscReal h,PetscReal tleft,PetscInt *next_sc,PetscReal *next_h,PetscBool  *finish)
{
  TSGLLEAdapt_Size *sz = (TSGLLEAdapt_Size*)adapt->data;
  PetscReal      dec = 0.2,inc = 5.0,safe = 0.9,optimal,last_desired_h;

  PetscFunctionBegin;
  *next_sc = cur;
  optimal  = PetscPowReal((PetscReal)errors[cur],(PetscReal)-1./(safe*orders[cur]));
  /* Step sizes oscillate when there is no smoothing.  Here we use a geometric mean of the current step size and the
  * one that would have been taken (without smoothing) on the last step. */
  last_desired_h = sz->desired_h;
  sz->desired_h  = h*PetscMax(dec,PetscMin(inc,optimal)); /* Trim to [dec,inc] */

  /* Normally only happens on the first step */
  if (last_desired_h > 1e-14) *next_h = PetscSqrtReal(last_desired_h * sz->desired_h);
  else *next_h = sz->desired_h;

  if (*next_h > tleft) {
    *finish = PETSC_TRUE;
    *next_h = tleft;
  } else *finish = PETSC_FALSE;
  PetscFunctionReturn(0);
}

PetscErrorCode  TSGLLEAdaptCreate_Size(TSGLLEAdapt adapt)
{
  TSGLLEAdapt_Size *a;

  PetscFunctionBegin;
  CHKERRQ(PetscNewLog(adapt,&a));
  adapt->data         = (void*)a;
  adapt->ops->choose  = TSGLLEAdaptChoose_Size;
  adapt->ops->destroy = TSGLLEAdaptDestroy_JustFree;
  PetscFunctionReturn(0);
}

/* -------------------------------- Both ----------------------------------- */
typedef struct {
  PetscInt  count_at_order;
  PetscReal desired_h;
} TSGLLEAdapt_Both;

static PetscErrorCode TSGLLEAdaptChoose_Both(TSGLLEAdapt adapt,PetscInt n,const PetscInt orders[],const PetscReal errors[],const PetscReal cost[],PetscInt cur,PetscReal h,PetscReal tleft,PetscInt *next_sc,PetscReal *next_h,PetscBool  *finish)
{
  TSGLLEAdapt_Both *both = (TSGLLEAdapt_Both*)adapt->data;
  PetscReal        dec = 0.2,inc = 5.0,safe = 0.9;
  struct {PetscInt id; PetscReal h,eff;} best={-1,0,0},trial={-1,0,0},current={-1,0,0};
  PetscInt        i;

  PetscFunctionBegin;
  for (i=0; i<n; i++) {
    PetscReal optimal;
    trial.id  = i;
    optimal   = PetscPowReal((PetscReal)errors[i],(PetscReal)-1./(safe*orders[i]));
    trial.h   = h*optimal;
    trial.eff = trial.h/cost[i];
    if (trial.eff > best.eff) CHKERRQ(PetscArraycpy(&best,&trial,1));
    if (i == cur) CHKERRQ(PetscArraycpy(&current,&trial,1));
  }
  /* Only switch orders if the scheme offers significant benefits over the current one.
  When the scheme is not changing, only change step size if it offers significant benefits. */
  if (best.eff < 1.2*current.eff || both->count_at_order < orders[cur]+2) {
    PetscReal last_desired_h;
    *next_sc        = current.id;
    last_desired_h  = both->desired_h;
    both->desired_h = PetscMax(h*dec,PetscMin(h*inc,current.h));
    *next_h         = (both->count_at_order > 0)
                      ? PetscSqrtReal(last_desired_h * both->desired_h)
                      : both->desired_h;
    both->count_at_order++;
  } else {
    PetscReal rat = cost[best.id]/cost[cur];
    *next_sc = best.id;
    *next_h  = PetscMax(h*rat*dec,PetscMin(h*rat*inc,best.h));
    both->count_at_order = 0;
    both->desired_h      = best.h;
  }

  if (*next_h > tleft) {
    *finish = PETSC_TRUE;
    *next_h = tleft;
  } else *finish = PETSC_FALSE;
  PetscFunctionReturn(0);
}

PetscErrorCode TSGLLEAdaptCreate_Both(TSGLLEAdapt adapt)
{
  TSGLLEAdapt_Both *a;

  PetscFunctionBegin;
  CHKERRQ(PetscNewLog(adapt,&a));
  adapt->data         = (void*)a;
  adapt->ops->choose  = TSGLLEAdaptChoose_Both;
  adapt->ops->destroy = TSGLLEAdaptDestroy_JustFree;
  PetscFunctionReturn(0);
}
