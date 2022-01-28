/*
  Code for Timestepping with basic symplectic integrators for separable Hamiltonian systems
*/
#include <petsc/private/tsimpl.h>                /*I   "petscts.h"   I*/
#include <petscdm.h>

static TSBasicSymplecticType TSBasicSymplecticDefault = TSBASICSYMPLECTICSIEULER;
static PetscBool TSBasicSymplecticRegisterAllCalled;
static PetscBool TSBasicSymplecticPackageInitialized;

typedef struct _BasicSymplecticScheme *BasicSymplecticScheme;
typedef struct _BasicSymplecticSchemeLink *BasicSymplecticSchemeLink;

struct _BasicSymplecticScheme {
  char      *name;
  PetscInt  order;
  PetscInt  s;       /* number of stages */
  PetscReal *c,*d;
};
struct _BasicSymplecticSchemeLink {
  struct _BasicSymplecticScheme sch;
  BasicSymplecticSchemeLink     next;
};
static BasicSymplecticSchemeLink BasicSymplecticSchemeList;
typedef struct {
  TS                    subts_p,subts_q; /* sub TS contexts that holds the RHSFunction pointers */
  IS                    is_p,is_q; /* IS sets for position and momentum respectively */
  Vec                   update;    /* a nest work vector for generalized coordinates */
  BasicSymplecticScheme scheme;
} TS_BasicSymplectic;

/*MC
  TSBASICSYMPLECTICSIEULER - first order semi-implicit Euler method
  Level: intermediate
.seealso: TSBASICSYMPLECTIC
M*/

/*MC
  TSBASICSYMPLECTICVELVERLET - second order Velocity Verlet method (leapfrog method with starting process and determing velocity and position at the same time)
Level: intermediate
.seealso: TSBASICSYMPLECTIC
M*/

/*@C
  TSBasicSymplecticRegisterAll - Registers all of the basic symplectic integration methods in TSBasicSymplectic

  Not Collective, but should be called by all processes which will need the schemes to be registered

  Level: advanced

.seealso:  TSBasicSymplecticRegisterDestroy()
@*/
PetscErrorCode TSBasicSymplecticRegisterAll(void)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (TSBasicSymplecticRegisterAllCalled) PetscFunctionReturn(0);
  TSBasicSymplecticRegisterAllCalled = PETSC_TRUE;
  {
    PetscReal c[1] = {1.0},d[1] = {1.0};
    ierr = TSBasicSymplecticRegister(TSBASICSYMPLECTICSIEULER,1,1,c,d);CHKERRQ(ierr);
  }
  {
    PetscReal c[2] = {0,1.0},d[2] = {0.5,0.5};
    ierr = TSBasicSymplecticRegister(TSBASICSYMPLECTICVELVERLET,2,2,c,d);CHKERRQ(ierr);
  }
  {
    PetscReal c[3] = {1,-2.0/3.0,2.0/3.0},d[3] = {-1.0/24.0,3.0/4.0,7.0/24.0};
    ierr = TSBasicSymplecticRegister(TSBASICSYMPLECTIC3,3,3,c,d);CHKERRQ(ierr);
  }
  {
#define CUBEROOTOFTWO 1.2599210498948731647672106
    PetscReal c[4] = {1.0/2.0/(2.0-CUBEROOTOFTWO),(1.0-CUBEROOTOFTWO)/2.0/(2.0-CUBEROOTOFTWO),(1.0-CUBEROOTOFTWO)/2.0/(2.0-CUBEROOTOFTWO),1.0/2.0/(2.0-CUBEROOTOFTWO)},d[4] = {1.0/(2.0-CUBEROOTOFTWO),-CUBEROOTOFTWO/(2.0-CUBEROOTOFTWO),1.0/(2.0-CUBEROOTOFTWO),0};
    ierr = TSBasicSymplecticRegister(TSBASICSYMPLECTIC4,4,4,c,d);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*@C
   TSBasicSymplecticRegisterDestroy - Frees the list of schemes that were registered by TSBasicSymplecticRegister().

   Not Collective

   Level: advanced

.seealso: TSBasicSymplecticRegister(), TSBasicSymplecticRegisterAll()
@*/
PetscErrorCode TSBasicSymplecticRegisterDestroy(void)
{
  PetscErrorCode            ierr;
  BasicSymplecticSchemeLink link;

  PetscFunctionBegin;
  while ((link = BasicSymplecticSchemeList)) {
    BasicSymplecticScheme scheme = &link->sch;
    BasicSymplecticSchemeList = link->next;
    ierr = PetscFree2(scheme->c,scheme->d);CHKERRQ(ierr);
    ierr = PetscFree(scheme->name);CHKERRQ(ierr);
    ierr = PetscFree(link);CHKERRQ(ierr);
  }
  TSBasicSymplecticRegisterAllCalled = PETSC_FALSE;
  PetscFunctionReturn(0);
}

/*@C
  TSBasicSymplecticInitializePackage - This function initializes everything in the TSBasicSymplectic package. It is called
  from TSInitializePackage().

  Level: developer

.seealso: PetscInitialize()
@*/
PetscErrorCode TSBasicSymplecticInitializePackage(void)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (TSBasicSymplecticPackageInitialized) PetscFunctionReturn(0);
  TSBasicSymplecticPackageInitialized = PETSC_TRUE;
  ierr = TSBasicSymplecticRegisterAll();CHKERRQ(ierr);
  ierr = PetscRegisterFinalize(TSBasicSymplecticFinalizePackage);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
  TSBasicSymplecticFinalizePackage - This function destroys everything in the TSBasicSymplectic package. It is
  called from PetscFinalize().

  Level: developer

.seealso: PetscFinalize()
@*/
PetscErrorCode TSBasicSymplecticFinalizePackage(void)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  TSBasicSymplecticPackageInitialized = PETSC_FALSE;
  ierr = TSBasicSymplecticRegisterDestroy();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
   TSBasicSymplecticRegister - register a basic symplectic integration scheme by providing the coefficients.

   Not Collective, but the same schemes should be registered on all processes on which they will be used

   Input Parameters:
+  name - identifier for method
.  order - approximation order of method
.  s - number of stages, this is the dimension of the matrices below
.  c - coefficients for updating generalized position (dimension s)
-  d - coefficients for updating generalized momentum (dimension s)

   Notes:
   Several symplectic methods are provided, this function is only needed to create new methods.

   Level: advanced

.seealso: TSBasicSymplectic
@*/
PetscErrorCode TSBasicSymplecticRegister(TSRosWType name,PetscInt order,PetscInt s,PetscReal c[],PetscReal d[])
{
  BasicSymplecticSchemeLink link;
  BasicSymplecticScheme     scheme;
  PetscErrorCode            ierr;

  PetscFunctionBegin;
  PetscValidCharPointer(name,1);
  PetscValidPointer(c,4);
  PetscValidPointer(d,5);

  ierr = TSBasicSymplecticInitializePackage();CHKERRQ(ierr);
  ierr = PetscNew(&link);CHKERRQ(ierr);
  scheme = &link->sch;
  ierr = PetscStrallocpy(name,&scheme->name);CHKERRQ(ierr);
  scheme->order = order;
  scheme->s = s;
  ierr = PetscMalloc2(s,&scheme->c,s,&scheme->d);CHKERRQ(ierr);
  ierr = PetscArraycpy(scheme->c,c,s);CHKERRQ(ierr);
  ierr = PetscArraycpy(scheme->d,d,s);CHKERRQ(ierr);
  link->next = BasicSymplecticSchemeList;
  BasicSymplecticSchemeList = link;
  PetscFunctionReturn(0);
}

/*
The simplified form of the equations are:

$ p_{i+1} = p_i + c_i*g(q_i)*h
$ q_{i+1} = q_i + d_i*f(p_{i+1},t_{i+1})*h

Several symplectic integrators are given below. An illustrative way to use them is to consider a particle with position q and velocity p.

To apply a timestep with values c_{1,2},d_{1,2} to the particle, carry out the following steps:

- Update the velocity of the particle by adding to it its acceleration multiplied by c_1
- Update the position of the particle by adding to it its (updated) velocity multiplied by d_1
- Update the velocity of the particle by adding to it its acceleration (at the updated position) multiplied by c_2
- Update the position of the particle by adding to it its (double-updated) velocity multiplied by d_2

*/
static PetscErrorCode TSStep_BasicSymplectic(TS ts)
{
  TS_BasicSymplectic    *bsymp = (TS_BasicSymplectic*)ts->data;
  BasicSymplecticScheme scheme = bsymp->scheme;
  Vec                   solution = ts->vec_sol,update = bsymp->update,q,p,q_update,p_update;
  IS                    is_q = bsymp->is_q,is_p = bsymp->is_p;
  TS                    subts_q = bsymp->subts_q,subts_p = bsymp->subts_p;
  PetscBool             stageok;
  PetscReal             next_time_step = ts->time_step;
  PetscInt              iter;
  PetscErrorCode        ierr;

  PetscFunctionBegin;
  ierr = VecGetSubVector(solution,is_q,&q);CHKERRQ(ierr);
  ierr = VecGetSubVector(solution,is_p,&p);CHKERRQ(ierr);
  ierr = VecGetSubVector(update,is_q,&q_update);CHKERRQ(ierr);
  ierr = VecGetSubVector(update,is_p,&p_update);CHKERRQ(ierr);

  for (iter = 0;iter<scheme->s;iter++) {
    ierr = TSPreStage(ts,ts->ptime);CHKERRQ(ierr);
    /* update velocity p */
    if (scheme->c[iter]) {
      ierr = TSComputeRHSFunction(subts_p,ts->ptime,q,p_update);CHKERRQ(ierr);
      ierr = VecAXPY(p,scheme->c[iter]*ts->time_step,p_update);CHKERRQ(ierr);
    }
    /* update position q */
    if (scheme->d[iter]) {
      ierr = TSComputeRHSFunction(subts_q,ts->ptime,p,q_update);CHKERRQ(ierr);
      ierr = VecAXPY(q,scheme->d[iter]*ts->time_step,q_update);CHKERRQ(ierr);
      ts->ptime = ts->ptime+scheme->d[iter]*ts->time_step;
    }
    ierr = TSPostStage(ts,ts->ptime,0,&solution);CHKERRQ(ierr);
    ierr = TSAdaptCheckStage(ts->adapt,ts,ts->ptime,solution,&stageok);CHKERRQ(ierr);
    if (!stageok) {ts->reason = TS_DIVERGED_STEP_REJECTED; PetscFunctionReturn(0);}
    ierr = TSFunctionDomainError(ts,ts->ptime+ts->time_step,update,&stageok);CHKERRQ(ierr);
    if (!stageok) {ts->reason = TS_DIVERGED_STEP_REJECTED; PetscFunctionReturn(0);}
  }

  ts->time_step = next_time_step;
  ierr = VecRestoreSubVector(solution,is_q,&q);CHKERRQ(ierr);
  ierr = VecRestoreSubVector(solution,is_p,&p);CHKERRQ(ierr);
  ierr = VecRestoreSubVector(update,is_q,&q_update);CHKERRQ(ierr);
  ierr = VecRestoreSubVector(update,is_p,&p_update);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode DMCoarsenHook_BasicSymplectic(DM fine,DM coarse,void *ctx)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

static PetscErrorCode DMRestrictHook_BasicSymplectic(DM fine,Mat restrct,Vec rscale,Mat inject,DM coarse,void *ctx)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

static PetscErrorCode DMSubDomainHook_BasicSymplectic(DM dm,DM subdm,void *ctx)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

static PetscErrorCode DMSubDomainRestrictHook_BasicSymplectic(DM dm,VecScatter gscat,VecScatter lscat,DM subdm,void *ctx)
{

  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

static PetscErrorCode TSSetUp_BasicSymplectic(TS ts)
{
  TS_BasicSymplectic *bsymp = (TS_BasicSymplectic*)ts->data;
  DM                 dm;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  ierr = TSRHSSplitGetIS(ts,"position",&bsymp->is_q);CHKERRQ(ierr);
  ierr = TSRHSSplitGetIS(ts,"momentum",&bsymp->is_p);CHKERRQ(ierr);
  PetscAssertFalse(!bsymp->is_q || !bsymp->is_p,PetscObjectComm((PetscObject)ts),PETSC_ERR_USER,"Must set up RHSSplits with TSRHSSplitSetIS() using split names positon and momentum respectively in order to use -ts_type basicsymplectic");
  ierr = TSRHSSplitGetSubTS(ts,"position",&bsymp->subts_q);CHKERRQ(ierr);
  ierr = TSRHSSplitGetSubTS(ts,"momentum",&bsymp->subts_p);CHKERRQ(ierr);
  PetscAssertFalse(!bsymp->subts_q || !bsymp->subts_p,PetscObjectComm((PetscObject)ts),PETSC_ERR_USER,"Must set up the RHSFunctions for position and momentum using TSRHSSplitSetRHSFunction() or calling TSSetRHSFunction() for each sub-TS");

  ierr = VecDuplicate(ts->vec_sol,&bsymp->update);CHKERRQ(ierr);

  ierr = TSGetAdapt(ts,&ts->adapt);CHKERRQ(ierr);
  ierr = TSAdaptCandidatesClear(ts->adapt);CHKERRQ(ierr); /* make sure to use fixed time stepping */
  ierr = TSGetDM(ts,&dm);CHKERRQ(ierr);
  if (dm) {
    ierr = DMCoarsenHookAdd(dm,DMCoarsenHook_BasicSymplectic,DMRestrictHook_BasicSymplectic,ts);CHKERRQ(ierr);
    ierr = DMSubDomainHookAdd(dm,DMSubDomainHook_BasicSymplectic,DMSubDomainRestrictHook_BasicSymplectic,ts);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode TSReset_BasicSymplectic(TS ts)
{
  TS_BasicSymplectic *bsymp = (TS_BasicSymplectic*)ts->data;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  ierr = VecDestroy(&bsymp->update);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode TSDestroy_BasicSymplectic(TS ts)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = TSReset_BasicSymplectic(ts);CHKERRQ(ierr);
  ierr = PetscFree(ts->data);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode TSSetFromOptions_BasicSymplectic(PetscOptionItems *PetscOptionsObject,TS ts)
{
  TS_BasicSymplectic *bsymp = (TS_BasicSymplectic*)ts->data;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  ierr = PetscOptionsHead(PetscOptionsObject,"Basic symplectic integrator options");CHKERRQ(ierr);
  {
    BasicSymplecticSchemeLink link;
    PetscInt                  count,choice;
    PetscBool                 flg;
    const char                **namelist;

    for (link=BasicSymplecticSchemeList,count=0; link; link=link->next,count++) ;
    ierr = PetscMalloc1(count,(char***)&namelist);CHKERRQ(ierr);
    for (link=BasicSymplecticSchemeList,count=0; link; link=link->next,count++) namelist[count] = link->sch.name;
    ierr = PetscOptionsEList("-ts_basicsymplectic_type","Family of basic symplectic integration method","TSBasicSymplecticSetType",(const char*const*)namelist,count,bsymp->scheme->name,&choice,&flg);CHKERRQ(ierr);
    if (flg) {ierr = TSBasicSymplecticSetType(ts,namelist[choice]);CHKERRQ(ierr);}
    ierr = PetscFree(namelist);CHKERRQ(ierr);
  }
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode TSView_BasicSymplectic(TS ts,PetscViewer viewer)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

static PetscErrorCode TSInterpolate_BasicSymplectic(TS ts,PetscReal t,Vec X)
{
  TS_BasicSymplectic *bsymp = (TS_BasicSymplectic*)ts->data;
  Vec                update = bsymp->update;
  PetscReal          alpha = (ts->ptime - t)/ts->time_step;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  ierr = VecWAXPY(X,-ts->time_step,update,ts->vec_sol);CHKERRQ(ierr);
  ierr = VecAXPBY(X,1.0-alpha,alpha,ts->vec_sol);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode TSComputeLinearStability_BasicSymplectic(TS ts,PetscReal xr,PetscReal xi,PetscReal *yr,PetscReal *yi)
{
  PetscFunctionBegin;
  *yr = 1.0 + xr;
  *yi = xi;
  PetscFunctionReturn(0);
}

/*@C
  TSBasicSymplecticSetType - Set the type of the basic symplectic method

  Logically Collective on TS

  Input Parameters:
+  ts - timestepping context
-  bsymptype - type of the symplectic scheme

  Options Database:
.  -ts_basicsymplectic_type <scheme>

  Notes:
  The symplectic solver always expects a two-way splitting with the split names being "position" and "momentum". Each split is associated with an IS object and a sub-TS that is intended to store the user-provided RHS function.

  Level: intermediate
@*/
PetscErrorCode TSBasicSymplecticSetType(TS ts,TSBasicSymplecticType bsymptype)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  ierr = PetscTryMethod(ts,"TSBasicSymplecticSetType_C",(TS,TSBasicSymplecticType),(ts,bsymptype));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
  TSBasicSymplecticGetType - Get the type of the basic symplectic method

  Logically Collective on TS

  Input Parameters:
+  ts - timestepping context
-  bsymptype - type of the basic symplectic scheme

  Level: intermediate
@*/
PetscErrorCode TSBasicSymplecticGetType(TS ts,TSBasicSymplecticType *bsymptype)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  ierr = PetscUseMethod(ts,"TSBasicSymplecticGetType_C",(TS,TSBasicSymplecticType*),(ts,bsymptype));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode TSBasicSymplecticSetType_BasicSymplectic(TS ts,TSBasicSymplecticType bsymptype)
{
  TS_BasicSymplectic        *bsymp = (TS_BasicSymplectic*)ts->data;
  BasicSymplecticSchemeLink link;
  PetscBool                 match;
  PetscErrorCode            ierr;

  PetscFunctionBegin;
  if (bsymp->scheme) {
    ierr = PetscStrcmp(bsymp->scheme->name,bsymptype,&match);CHKERRQ(ierr);
    if (match) PetscFunctionReturn(0);
  }
  for (link = BasicSymplecticSchemeList; link; link=link->next) {
    ierr = PetscStrcmp(link->sch.name,bsymptype,&match);CHKERRQ(ierr);
    if (match) {
      bsymp->scheme = &link->sch;
      PetscFunctionReturn(0);
    }
  }
  SETERRQ(PetscObjectComm((PetscObject)ts),PETSC_ERR_ARG_UNKNOWN_TYPE,"Could not find '%s'",bsymptype);
}

static PetscErrorCode  TSBasicSymplecticGetType_BasicSymplectic(TS ts,TSBasicSymplecticType *bsymptype)
{
  TS_BasicSymplectic *bsymp = (TS_BasicSymplectic*)ts->data;

  PetscFunctionBegin;
  *bsymptype = bsymp->scheme->name;
  PetscFunctionReturn(0);
}

/*MC
  TSBasicSymplectic - ODE solver using basic symplectic integration schemes

  These methods are intened for separable Hamiltonian systems

$  qdot = dH(q,p,t)/dp
$  pdot = -dH(q,p,t)/dq

  where the Hamiltonian can be split into the sum of kinetic energy and potential energy

$  H(q,p,t) = T(p,t) + V(q,t).

  As a result, the system can be genearlly represented by

$  qdot = f(p,t) = dT(p,t)/dp
$  pdot = g(q,t) = -dV(q,t)/dq

  and solved iteratively with

$  q_new = q_old + d_i*h*f(p_old,t_old)
$  t_new = t_old + d_i*h
$  p_new = p_old + c_i*h*g(p_new,t_new)
$  i=0,1,...,n.

  The solution vector should contain both q and p, which correspond to (generalized) position and momentum respectively. Note that the momentum component could simply be velocity in some representations.
  The symplectic solver always expects a two-way splitting with the split names being "position" and "momentum". Each split is associated with an IS object and a sub-TS that is intended to store the user-provided RHS function.

  Reference: wikipedia (https://en.wikipedia.org/wiki/Symplectic_integrator)

  Level: beginner

.seealso:  TSCreate(), TSSetType(), TSRHSSplitSetIS(), TSRHSSplitSetRHSFunction()

M*/
PETSC_EXTERN PetscErrorCode TSCreate_BasicSymplectic(TS ts)
{
  TS_BasicSymplectic *bsymp;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  ierr = TSBasicSymplecticInitializePackage();CHKERRQ(ierr);
  ierr = PetscNewLog(ts,&bsymp);CHKERRQ(ierr);
  ts->data = (void*)bsymp;

  ts->ops->setup           = TSSetUp_BasicSymplectic;
  ts->ops->step            = TSStep_BasicSymplectic;
  ts->ops->reset           = TSReset_BasicSymplectic;
  ts->ops->destroy         = TSDestroy_BasicSymplectic;
  ts->ops->setfromoptions  = TSSetFromOptions_BasicSymplectic;
  ts->ops->view            = TSView_BasicSymplectic;
  ts->ops->interpolate     = TSInterpolate_BasicSymplectic;
  ts->ops->linearstability = TSComputeLinearStability_BasicSymplectic;

  ierr = PetscObjectComposeFunction((PetscObject)ts,"TSBasicSymplecticSetType_C",TSBasicSymplecticSetType_BasicSymplectic);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)ts,"TSBasicSymplecticGetType_C",TSBasicSymplecticGetType_BasicSymplectic);CHKERRQ(ierr);

  ierr = TSBasicSymplecticSetType(ts,TSBasicSymplecticDefault);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
