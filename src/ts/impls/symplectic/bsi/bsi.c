/*
  Code for Timestepping with basic symplectic integrators for separable Hamiltonian systems
*/
#include <petsc/private/tsimpl.h>                /*I   "petscts.h"   I*/
#include <petscdm.h>

static TSBSIType TSBSIDefault = TSSIEULER;
static PetscBool TSBSIRegisterAllCalled;
static PetscBool TSBSIPackageInitialized;

typedef struct _BSIScheme *BSIScheme;
typedef struct _BSISchemeLink *BSISchemeLink;

struct _BSIScheme {
  char        *name;
  PetscInt    order;
  PetscInt    s;       /* number of stages */
  PetscReal   *c,*d;
};
struct _BSISchemeLink {
  struct _BSIScheme sch;
  BSISchemeLink     next;
};
static BSISchemeLink BSISchemeList;
typedef struct {
  Vec       update;   /* a nest work vector for generalized coordinates */
  BSIScheme scheme;
} TS_BSI;

/*MC
  TSSIEULER - first order semi-implicit Euler method
  Level: intermediate
.seealso: TSBSI
M*/

/*MC
  TSVELVERLET - second order Velocity Verlet method (leapfrog method with starting process and determing velocity and position at the same time)
Level: intermediate
.seealso: TSBSI
M*/

/*@C
  TSBSIRegisterAll - Registers all of the basic symplectic integration methods in TSBSI

  Not Collective, but should be called by all processes which will need the schemes to be registered

  Level: advanced

.keywords: TS, TSBSI, register, all

.seealso:  TSBSIRegisterDestroy()
@*/
PetscErrorCode TSBSIRegisterAll(void)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (TSBSIRegisterAllCalled) PetscFunctionReturn(0);
  TSBSIRegisterAllCalled = PETSC_TRUE;
  {
    const PetscReal c[1] = {1.0},d[1] = {1.0};
    ierr = TSBSIRegister(TSSIEULER,1,1,c,d);CHKERRQ(ierr);
  }
  {
    const PetscReal c[2] = {0,1.0},d[2] = {0.5,0.5};
    ierr = TSBSIRegister(TSVELVERLET,2,2,c,d);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*@C
   TSBSIRegisterDestroy - Frees the list of schemes that were registered by TSBSIRegister().

   Not Collective

   Level: advanced

.keywords: TSBSI, register, destroy
.seealso: TSBSIRegister(), TSBSIRegisterAll()
@*/
PetscErrorCode TSBSIRegisterDestroy(void)
{
  PetscErrorCode ierr;
  BSISchemeLink  link;

  PetscFunctionBegin;
  while ((link = BSISchemeList)) {
    BSIScheme scheme = &link->sch;
    BSISchemeList = link->next;
    ierr = PetscFree2(scheme->c,scheme->d);CHKERRQ(ierr);
    ierr = PetscFree (scheme->name);CHKERRQ(ierr);
    ierr = PetscFree (link);CHKERRQ(ierr);
  }
  TSBSIRegisterAllCalled = PETSC_FALSE;
  PetscFunctionReturn(0);
}

/*@C
  TSBSIInitializePackage - This function initializes everything in the TSBSI package. It is called
  from PetscDLLibraryRegister() when using dynamic libraries, and on the first call to TSCreate_BSI()
  when using static libraries.

  Level: developer

.keywords: TS, TSBSI, initialize, package
.seealso: PetscInitialize()
@*/
PetscErrorCode TSBSIInitializePackage(void)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (TSBSIPackageInitialized) PetscFunctionReturn(0);
  TSBSIPackageInitialized = PETSC_TRUE;
  ierr = TSBSIRegisterAll();CHKERRQ(ierr);
  ierr = PetscRegisterFinalize(TSBSIFinalizePackage);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
  TSBSIFinalizePackage - This function destroys everything in the TSBSI package. It is
  called from PetscFinalize().

  Level: developer

.keywords: Petsc, destroy, package
.seealso: PetscFinalize()
@*/
PetscErrorCode TSBSIFinalizePackage(void)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  TSBSIPackageInitialized = PETSC_FALSE;
  ierr = TSBSIRegisterDestroy();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
   TSBSIRegister - register a basic symplectic integration scheme by providing the coefficients.

   Not Collective, but the same schemes should be registered on all processes on which they will be used

   Input Parameters:
+  name - identifier for method
.  order - approximation order of method
.  s - number of stages, this is the dimension of the matrices below
.  c - coefficients for updating generalized position (dimension s)
-  d - coefficients for updating generalized momentum (dimension s)

   Notes:
   Several BSI methods are provided, this function is only needed to create new methods.

   Level: advanced

.keywords: TS, register

.seealso: TSBSI
@*/
PetscErrorCode TSBSIRegister(TSRosWType name,PetscInt order,PetscInt s,const PetscReal c[],const PetscReal d[])
{
  BSISchemeLink  link;
  BSIScheme      scheme;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidCharPointer(name,1);
  PetscValidPointer(c,4);
  PetscValidPointer(d,4);

  ierr = PetscCalloc1(1,&link);CHKERRQ(ierr);
  scheme = &link->sch;
  ierr = PetscStrallocpy(name,&scheme->name);CHKERRQ(ierr);
  scheme->order = order;
  scheme->s = s;
  ierr = PetscMalloc2(s,&scheme->c,s,&scheme->d);CHKERRQ(ierr);
  ierr = PetscMemcpy(scheme->c,c,s*sizeof(c[0]));CHKERRQ(ierr);
  ierr = PetscMemcpy(scheme->d,d,s*sizeof(d[0]));CHKERRQ(ierr);
  link->next = BSISchemeList;
  BSISchemeList = link;
  PetscFunctionReturn(0);
}

static PetscErrorCode TSStep_BSI(TS ts)
{
  TS_BSI         *bsi = (TS_BSI*)ts->data;
  BSIScheme      scheme = bsi->scheme;
  Vec            solution = ts->vec_sol,update = bsi->update,q,p,q_update,p_update;
  PetscBool      stageok;
  PetscReal      next_time_step = ts->time_step;
  PetscInt       iter;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecNestGetSubVec(solution,0,&p);CHKERRQ(ierr);
  ierr = VecNestGetSubVec(solution,1,&q);CHKERRQ(ierr);

  ierr = VecNestGetSubVec(update,0,&p_update);CHKERRQ(ierr);
  ierr = VecNestGetSubVec(update,1,&q_update);CHKERRQ(ierr);

  for (iter = 0;iter<scheme->s;iter++) {
    /* wikipedia says update q first, then p. It is wrong. */
    ierr = TSPreStage(ts,ts->ptime);CHKERRQ(ierr);
    ierr = TSComputeRHSFunctionSplit2w(ts,ts->ptime,q,p_update,1);CHKERRQ(ierr);
    /* update p */
    ierr = VecAXPY(p,scheme->d[iter]*ts->time_step,p_update);CHKERRQ(ierr);
    if (scheme->c[iter]) {
      ierr = TSComputeRHSFunctionSplit2w(ts,ts->ptime,p,q_update,2);CHKERRQ(ierr);
      /* update q */
      ierr = VecAXPY(q,scheme->c[iter]*ts->time_step,q_update);CHKERRQ(ierr);
      /* for nonautonomous systems */
      ts->ptime = ts->ptime+scheme->c[iter]*ts->time_step;
    }
    ierr = TSPostStage(ts,ts->ptime,0,&solution);CHKERRQ(ierr);
    ierr = TSAdaptCheckStage(ts->adapt,ts,ts->ptime,solution,&stageok);CHKERRQ(ierr);
    if(!stageok) {ts->reason = TS_DIVERGED_STEP_REJECTED; PetscFunctionReturn(0);}
    ierr = TSFunctionDomainError(ts,ts->ptime+ts->time_step,update,&stageok);CHKERRQ(ierr);
    if(!stageok) {ts->reason = TS_DIVERGED_STEP_REJECTED; PetscFunctionReturn(0);}
  }

  ts->time_step = next_time_step;
  PetscFunctionReturn(0);
}

static PetscErrorCode DMCoarsenHook_BSI(DM fine,DM coarse,void *ctx)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

static PetscErrorCode DMRestrictHook_BSI(DM fine,Mat restrct,Vec rscale,Mat inject,DM coarse,void *ctx)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

static PetscErrorCode DMSubDomainHook_BSI(DM dm,DM subdm,void *ctx)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

static PetscErrorCode DMSubDomainRestrictHook_BSI(DM dm,VecScatter gscat,VecScatter lscat,DM subdm,void *ctx)
{

  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

static PetscErrorCode TSSetUp_BSI(TS ts)
{
  TS_BSI               *bsi = (TS_BSI*)ts->data;
  DM                   dm;
  VecType              vtype;
  PetscBool            isNestVec;
  TSRHSFunctionSplit2w rhsfunction1,rhsfunction2;
  PetscErrorCode       ierr;

  PetscFunctionBegin;
  ierr =  TSGetRHSFunctionSplit2w(ts,NULL,&rhsfunction1,&rhsfunction2,NULL);CHKERRQ(ierr);
  if (!rhsfunction1 || !rhsfunction2) SETERRQ(PetscObjectComm((PetscObject)ts),PETSC_ERR_USER,"Must call TSSetRHSFunctionSplit2w() in order to use -ts_type bsi");
  ierr = VecGetType(ts->vec_sol,&vtype);CHKERRQ(ierr);
  ierr = PetscStrcmp(vtype,VECNEST,&isNestVec);CHKERRQ(ierr);
  if (!isNestVec) SETERRQ(PetscObjectComm((PetscObject)ts),PETSC_ERR_USER,"Must use nest vector for the solution in order to use -ts_type bsi");
  ierr = VecDuplicate(ts->vec_sol,&bsi->update);CHKERRQ(ierr);

  ierr = TSGetAdapt(ts,&ts->adapt);CHKERRQ(ierr);
  ierr = TSAdaptCandidatesClear(ts->adapt);CHKERRQ(ierr); /* make sure to use fixed time stepping */
  ierr = TSGetDM(ts,&dm);CHKERRQ(ierr);
  if (dm) {
    ierr = DMCoarsenHookAdd(dm,DMCoarsenHook_BSI,DMRestrictHook_BSI,ts);CHKERRQ(ierr);
    ierr = DMSubDomainHookAdd(dm,DMSubDomainHook_BSI,DMSubDomainRestrictHook_BSI,ts);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode TSReset_BSI(TS ts)
{
  TS_BSI     *bsi = (TS_BSI*)ts->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecDestroy(&bsi->update);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode TSDestroy_BSI(TS ts)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = TSReset_BSI(ts);CHKERRQ(ierr);
  ierr = PetscFree(ts->data);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode TSSetFromOptions_BSI(PetscOptionItems *PetscOptionsObject,TS ts)
{
  TS_BSI         *bsi = (TS_BSI*)ts->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscOptionsHead(PetscOptionsObject,"Basic symplectic integrator options");CHKERRQ(ierr);
  {
    BSISchemeLink link;
    PetscInt      count,choice;
    PetscBool     flg;
    const char    **namelist;

    for (link=BSISchemeList,count=0; link; link=link->next,count++) ;
    ierr = PetscMalloc1(count,&namelist);CHKERRQ(ierr);
    for (link=BSISchemeList,count=0; link; link=link->next,count++) namelist[count] = link->sch.name;
    ierr = PetscOptionsEList("-ts_bsi_type","Family of basic symplectic integration method","TSBSISetType",(const char*const*)namelist,count,bsi->scheme->name,&choice,&flg);CHKERRQ(ierr);
    if (flg) {ierr = TSBSISetType(ts,namelist[choice]);CHKERRQ(ierr);}
    ierr = PetscFree(namelist);CHKERRQ(ierr);
  }
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode TSView_BSI(TS ts,PetscViewer viewer)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

static PetscErrorCode TSInterpolate_BSI(TS ts,PetscReal t,Vec X)
{
  TS_BSI         *bsi = (TS_BSI*)ts->data;
  Vec            update = bsi->update;
  PetscReal      alpha = (ts->ptime - t)/ts->time_step;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecWAXPY(X,-ts->time_step,update,ts->vec_sol);CHKERRQ(ierr);
  ierr = VecAXPBY(X,1.0-alpha,alpha,ts->vec_sol);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode TSComputeLinearStability_BSI(TS ts,PetscReal xr,PetscReal xi,PetscReal *yr,PetscReal *yi)
{
  PetscFunctionBegin;
  *yr = 1.0 + xr;
  *yi = xi;
  PetscFunctionReturn(0);
}

/*@
  TSBSISetType - Set the type of the BSI method

  Logically Collective on TS

  Input Parameter:
+  ts - timestepping context
-  bsitype - type of the BSI scheme

  Options Database:
.  -ts_bsi_type <scheme>

  Level: intermediate

@*/
PetscErrorCode TSBSISetType(TS ts,TSBSIType bsitype)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  ierr = PetscTryMethod(ts,"TSBSISetType_C",(TS,TSBSIType),(ts,bsitype));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode TSBSIGetType(TS ts,TSBSIType *bsitype)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  ierr = PetscUseMethod(ts,"TSBSIGetType_C",(TS,TSBSIType*),(ts,bsitype));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode TSBSISetType_BSI(TS ts,TSBSIType bsitype)
{
  TS_BSI         *bsi = (TS_BSI*)ts->data;
  BSISchemeLink  link;
  PetscBool      match;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (bsi->scheme) {
    ierr = PetscStrcmp(bsi->scheme->name,bsitype,&match);CHKERRQ(ierr);
    if (match) PetscFunctionReturn(0);
  }
  for (link = BSISchemeList; link; link=link->next) {
    ierr = PetscStrcmp(link->sch.name,bsitype,&match);CHKERRQ(ierr);
    if (match) {
      bsi->scheme = &link->sch;
      PetscFunctionReturn(0);
    }
  }
  SETERRQ1(PetscObjectComm((PetscObject)ts),PETSC_ERR_ARG_UNKNOWN_TYPE,"Could not find '%s'",bsitype);
  PetscFunctionReturn(0);
}

static PetscErrorCode  TSBSIGetType_BSI(TS ts,TSBSIType *bsitype)
{
  TS_BSI *bsi = (TS_BSI*)ts->data;

  PetscFunctionBegin;
  *bsitype = bsi->scheme->name;
  PetscFunctionReturn(0);
}

/*MC
  TSBSI - ODE solver using basic symplectic integration schemes

  These methods are intened for separable Hamiltonian systems

$  pdot = -dH(q,p,t)/dq
$  qdot = dH(q,p,t)/dp

  where the Hamiltonian can be split into the sum of kinetic energy and potential energy

$  H(q,p,t) = T(p,t) + V(q,t).

  As a result, the system can be genearlly represented by

$  pdot = f(q,t) = -dV(q,t)/dq
$  qdot = g(p,t) = dT(p,t)/dp

  and solved iteratively with

$  q_new = q_old + c_i*h*g(p_old,t_old)
$  t_new = t_old + c_i*h
$  p_new = p_old + d_i*h*f(q_new,t_new)
$  i=0,1,...,n.

  The solution is represented by a nest vec [p,q].
  f and g are provided by RHSFunction1 and RHSFunction2 respectively.

  Reference: wikipedia

  Level: beginner

.seealso:  TSCreate(), TS, TSSetType(), TSSIEULER, TSVELVERLET

M*/
PETSC_EXTERN PetscErrorCode TSCreate_BSI(TS ts)
{
  TS_BSI         *bsi;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = TSBSIInitializePackage();CHKERRQ(ierr);
  ierr = PetscNewLog(ts,&bsi);CHKERRQ(ierr);
  ts->data = (void*)bsi;

  ts->ops->setup           = TSSetUp_BSI;
  ts->ops->step            = TSStep_BSI;
  ts->ops->reset           = TSReset_BSI;
  ts->ops->destroy         = TSDestroy_BSI;
  ts->ops->setfromoptions  = TSSetFromOptions_BSI;
  ts->ops->view            = TSView_BSI;
  ts->ops->interpolate     = TSInterpolate_BSI;
  ts->ops->linearstability = TSComputeLinearStability_BSI;

  ierr = PetscObjectComposeFunction((PetscObject)ts,"TSBSISetType_C",TSBSISetType_BSI);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)ts,"TSBSIGetType_C",TSBSIGetType_BSI);CHKERRQ(ierr);

  ierr = TSBSISetType(ts,TSBSIDefault);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
