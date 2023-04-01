/*
  Code for Timestepping with basic symplectic integrators for separable Hamiltonian systems
*/
#include <petsc/private/tsimpl.h> /*I   "petscts.h"   I*/
#include <petscdm.h>

static TSBasicSymplecticType TSBasicSymplecticDefault = TSBASICSYMPLECTICSIEULER;
static PetscBool             TSBasicSymplecticRegisterAllCalled;
static PetscBool             TSBasicSymplecticPackageInitialized;

typedef struct _BasicSymplecticScheme     *BasicSymplecticScheme;
typedef struct _BasicSymplecticSchemeLink *BasicSymplecticSchemeLink;

struct _BasicSymplecticScheme {
  char      *name;
  PetscInt   order;
  PetscInt   s; /* number of stages */
  PetscReal *c, *d;
};
struct _BasicSymplecticSchemeLink {
  struct _BasicSymplecticScheme sch;
  BasicSymplecticSchemeLink     next;
};
static BasicSymplecticSchemeLink BasicSymplecticSchemeList;
typedef struct {
  TS                    subts_p, subts_q; /* sub TS contexts that holds the RHSFunction pointers */
  IS                    is_p, is_q;       /* IS sets for position and momentum respectively */
  Vec                   update;           /* a nest work vector for generalized coordinates */
  BasicSymplecticScheme scheme;
} TS_BasicSymplectic;

/*MC
  TSBASICSYMPLECTICSIEULER - first order semi-implicit Euler method

  Level: intermediate

.seealso: [](chapter_ts), `TSBASICSYMPLECTIC`
M*/

/*MC
  TSBASICSYMPLECTICVELVERLET - second order Velocity Verlet method (leapfrog method with starting process and determining velocity and position at the same time)

Level: intermediate

.seealso: [](chapter_ts), `TSBASICSYMPLECTIC`
M*/

/*@C
  TSBasicSymplecticRegisterAll - Registers all of the basic symplectic integration methods in `TSBASICSYMPLECTIC`

  Not Collective, but should be called by all processes which will need the schemes to be registered

  Level: advanced

.seealso: [](chapter_ts), `TSBASICSYMPLECTIC`, `TSBasicSymplecticRegisterDestroy()`
@*/
PetscErrorCode TSBasicSymplecticRegisterAll(void)
{
  PetscFunctionBegin;
  if (TSBasicSymplecticRegisterAllCalled) PetscFunctionReturn(PETSC_SUCCESS);
  TSBasicSymplecticRegisterAllCalled = PETSC_TRUE;
  {
    PetscReal c[1] = {1.0}, d[1] = {1.0};
    PetscCall(TSBasicSymplecticRegister(TSBASICSYMPLECTICSIEULER, 1, 1, c, d));
  }
  {
    PetscReal c[2] = {0, 1.0}, d[2] = {0.5, 0.5};
    PetscCall(TSBasicSymplecticRegister(TSBASICSYMPLECTICVELVERLET, 2, 2, c, d));
  }
  {
    PetscReal c[3] = {1, -2.0 / 3.0, 2.0 / 3.0}, d[3] = {-1.0 / 24.0, 3.0 / 4.0, 7.0 / 24.0};
    PetscCall(TSBasicSymplecticRegister(TSBASICSYMPLECTIC3, 3, 3, c, d));
  }
  {
#define CUBEROOTOFTWO 1.2599210498948731647672106
    PetscReal c[4] = {1.0 / 2.0 / (2.0 - CUBEROOTOFTWO), (1.0 - CUBEROOTOFTWO) / 2.0 / (2.0 - CUBEROOTOFTWO), (1.0 - CUBEROOTOFTWO) / 2.0 / (2.0 - CUBEROOTOFTWO), 1.0 / 2.0 / (2.0 - CUBEROOTOFTWO)}, d[4] = {1.0 / (2.0 - CUBEROOTOFTWO), -CUBEROOTOFTWO / (2.0 - CUBEROOTOFTWO), 1.0 / (2.0 - CUBEROOTOFTWO), 0};
    PetscCall(TSBasicSymplecticRegister(TSBASICSYMPLECTIC4, 4, 4, c, d));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
   TSBasicSymplecticRegisterDestroy - Frees the list of schemes that were registered by `TSBasicSymplecticRegister()`.

   Not Collective

   Level: advanced

.seealso: [](chapter_ts), `TSBasicSymplecticRegister()`, `TSBasicSymplecticRegisterAll()`, `TSBASICSYMPLECTIC`
@*/
PetscErrorCode TSBasicSymplecticRegisterDestroy(void)
{
  BasicSymplecticSchemeLink link;

  PetscFunctionBegin;
  while ((link = BasicSymplecticSchemeList)) {
    BasicSymplecticScheme scheme = &link->sch;
    BasicSymplecticSchemeList    = link->next;
    PetscCall(PetscFree2(scheme->c, scheme->d));
    PetscCall(PetscFree(scheme->name));
    PetscCall(PetscFree(link));
  }
  TSBasicSymplecticRegisterAllCalled = PETSC_FALSE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  TSBasicSymplecticInitializePackage - This function initializes everything in the `TSBASICSYMPLECTIC` package. It is called
  from `TSInitializePackage()`.

  Level: developer

.seealso: [](chapter_ts), `PetscInitialize()`, `TSBASICSYMPLECTIC`
@*/
PetscErrorCode TSBasicSymplecticInitializePackage(void)
{
  PetscFunctionBegin;
  if (TSBasicSymplecticPackageInitialized) PetscFunctionReturn(PETSC_SUCCESS);
  TSBasicSymplecticPackageInitialized = PETSC_TRUE;
  PetscCall(TSBasicSymplecticRegisterAll());
  PetscCall(PetscRegisterFinalize(TSBasicSymplecticFinalizePackage));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  TSBasicSymplecticFinalizePackage - This function destroys everything in the `TSBASICSYMPLECTIC` package. It is
  called from `PetscFinalize()`.

  Level: developer

.seealso: [](chapter_ts), `PetscFinalize()`, `TSBASICSYMPLECTIC`
@*/
PetscErrorCode TSBasicSymplecticFinalizePackage(void)
{
  PetscFunctionBegin;
  TSBasicSymplecticPackageInitialized = PETSC_FALSE;
  PetscCall(TSBasicSymplecticRegisterDestroy());
  PetscFunctionReturn(PETSC_SUCCESS);
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

   Level: advanced

   Notes:
   Several symplectic methods are provided, this function is only needed to create new methods.

.seealso: [](chapter_ts), `TSBASICSYMPLECTIC`
@*/
PetscErrorCode TSBasicSymplecticRegister(TSRosWType name, PetscInt order, PetscInt s, PetscReal c[], PetscReal d[])
{
  BasicSymplecticSchemeLink link;
  BasicSymplecticScheme     scheme;

  PetscFunctionBegin;
  PetscValidCharPointer(name, 1);
  PetscValidRealPointer(c, 4);
  PetscValidRealPointer(d, 5);

  PetscCall(TSBasicSymplecticInitializePackage());
  PetscCall(PetscNew(&link));
  scheme = &link->sch;
  PetscCall(PetscStrallocpy(name, &scheme->name));
  scheme->order = order;
  scheme->s     = s;
  PetscCall(PetscMalloc2(s, &scheme->c, s, &scheme->d));
  PetscCall(PetscArraycpy(scheme->c, c, s));
  PetscCall(PetscArraycpy(scheme->d, d, s));
  link->next                = BasicSymplecticSchemeList;
  BasicSymplecticSchemeList = link;
  PetscFunctionReturn(PETSC_SUCCESS);
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
  TS_BasicSymplectic   *bsymp    = (TS_BasicSymplectic *)ts->data;
  BasicSymplecticScheme scheme   = bsymp->scheme;
  Vec                   solution = ts->vec_sol, update = bsymp->update, q, p, q_update, p_update;
  IS                    is_q = bsymp->is_q, is_p = bsymp->is_p;
  TS                    subts_q = bsymp->subts_q, subts_p = bsymp->subts_p;
  PetscBool             stageok;
  PetscReal             next_time_step = ts->time_step;
  PetscInt              iter;

  PetscFunctionBegin;
  PetscCall(VecGetSubVector(solution, is_q, &q));
  PetscCall(VecGetSubVector(solution, is_p, &p));
  PetscCall(VecGetSubVector(update, is_q, &q_update));
  PetscCall(VecGetSubVector(update, is_p, &p_update));

  for (iter = 0; iter < scheme->s; iter++) {
    PetscCall(TSPreStage(ts, ts->ptime));
    /* update velocity p */
    if (scheme->c[iter]) {
      PetscCall(TSComputeRHSFunction(subts_p, ts->ptime, q, p_update));
      PetscCall(VecAXPY(p, scheme->c[iter] * ts->time_step, p_update));
    }
    /* update position q */
    if (scheme->d[iter]) {
      PetscCall(TSComputeRHSFunction(subts_q, ts->ptime, p, q_update));
      PetscCall(VecAXPY(q, scheme->d[iter] * ts->time_step, q_update));
      ts->ptime = ts->ptime + scheme->d[iter] * ts->time_step;
    }
    PetscCall(TSPostStage(ts, ts->ptime, 0, &solution));
    PetscCall(TSAdaptCheckStage(ts->adapt, ts, ts->ptime, solution, &stageok));
    if (!stageok) {
      ts->reason = TS_DIVERGED_STEP_REJECTED;
      PetscFunctionReturn(PETSC_SUCCESS);
    }
    PetscCall(TSFunctionDomainError(ts, ts->ptime + ts->time_step, update, &stageok));
    if (!stageok) {
      ts->reason = TS_DIVERGED_STEP_REJECTED;
      PetscFunctionReturn(PETSC_SUCCESS);
    }
  }

  ts->time_step = next_time_step;
  PetscCall(VecRestoreSubVector(solution, is_q, &q));
  PetscCall(VecRestoreSubVector(solution, is_p, &p));
  PetscCall(VecRestoreSubVector(update, is_q, &q_update));
  PetscCall(VecRestoreSubVector(update, is_p, &p_update));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DMCoarsenHook_BasicSymplectic(DM fine, DM coarse, void *ctx)
{
  PetscFunctionBegin;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DMRestrictHook_BasicSymplectic(DM fine, Mat restrct, Vec rscale, Mat inject, DM coarse, void *ctx)
{
  PetscFunctionBegin;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DMSubDomainHook_BasicSymplectic(DM dm, DM subdm, void *ctx)
{
  PetscFunctionBegin;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DMSubDomainRestrictHook_BasicSymplectic(DM dm, VecScatter gscat, VecScatter lscat, DM subdm, void *ctx)
{
  PetscFunctionBegin;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TSSetUp_BasicSymplectic(TS ts)
{
  TS_BasicSymplectic *bsymp = (TS_BasicSymplectic *)ts->data;
  DM                  dm;

  PetscFunctionBegin;
  PetscCall(TSRHSSplitGetIS(ts, "position", &bsymp->is_q));
  PetscCall(TSRHSSplitGetIS(ts, "momentum", &bsymp->is_p));
  PetscCheck(bsymp->is_q && bsymp->is_p, PetscObjectComm((PetscObject)ts), PETSC_ERR_USER, "Must set up RHSSplits with TSRHSSplitSetIS() using split names position and momentum respectively in order to use -ts_type basicsymplectic");
  PetscCall(TSRHSSplitGetSubTS(ts, "position", &bsymp->subts_q));
  PetscCall(TSRHSSplitGetSubTS(ts, "momentum", &bsymp->subts_p));
  PetscCheck(bsymp->subts_q && bsymp->subts_p, PetscObjectComm((PetscObject)ts), PETSC_ERR_USER, "Must set up the RHSFunctions for position and momentum using TSRHSSplitSetRHSFunction() or calling TSSetRHSFunction() for each sub-TS");

  PetscCall(VecDuplicate(ts->vec_sol, &bsymp->update));

  PetscCall(TSGetAdapt(ts, &ts->adapt));
  PetscCall(TSAdaptCandidatesClear(ts->adapt)); /* make sure to use fixed time stepping */
  PetscCall(TSGetDM(ts, &dm));
  if (dm) {
    PetscCall(DMCoarsenHookAdd(dm, DMCoarsenHook_BasicSymplectic, DMRestrictHook_BasicSymplectic, ts));
    PetscCall(DMSubDomainHookAdd(dm, DMSubDomainHook_BasicSymplectic, DMSubDomainRestrictHook_BasicSymplectic, ts));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TSReset_BasicSymplectic(TS ts)
{
  TS_BasicSymplectic *bsymp = (TS_BasicSymplectic *)ts->data;

  PetscFunctionBegin;
  PetscCall(VecDestroy(&bsymp->update));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TSDestroy_BasicSymplectic(TS ts)
{
  PetscFunctionBegin;
  PetscCall(TSReset_BasicSymplectic(ts));
  PetscCall(PetscObjectComposeFunction((PetscObject)ts, "TSBasicSymplecticSetType_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)ts, "TSBasicSymplecticGetType_C", NULL));
  PetscCall(PetscFree(ts->data));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TSSetFromOptions_BasicSymplectic(TS ts, PetscOptionItems *PetscOptionsObject)
{
  TS_BasicSymplectic *bsymp = (TS_BasicSymplectic *)ts->data;

  PetscFunctionBegin;
  PetscOptionsHeadBegin(PetscOptionsObject, "Basic symplectic integrator options");
  {
    BasicSymplecticSchemeLink link;
    PetscInt                  count, choice;
    PetscBool                 flg;
    const char              **namelist;

    for (link = BasicSymplecticSchemeList, count = 0; link; link = link->next, count++)
      ;
    PetscCall(PetscMalloc1(count, (char ***)&namelist));
    for (link = BasicSymplecticSchemeList, count = 0; link; link = link->next, count++) namelist[count] = link->sch.name;
    PetscCall(PetscOptionsEList("-ts_basicsymplectic_type", "Family of basic symplectic integration method", "TSBasicSymplecticSetType", (const char *const *)namelist, count, bsymp->scheme->name, &choice, &flg));
    if (flg) PetscCall(TSBasicSymplecticSetType(ts, namelist[choice]));
    PetscCall(PetscFree(namelist));
  }
  PetscOptionsHeadEnd();
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TSView_BasicSymplectic(TS ts, PetscViewer viewer)
{
  PetscFunctionBegin;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TSInterpolate_BasicSymplectic(TS ts, PetscReal t, Vec X)
{
  TS_BasicSymplectic *bsymp  = (TS_BasicSymplectic *)ts->data;
  Vec                 update = bsymp->update;
  PetscReal           alpha  = (ts->ptime - t) / ts->time_step;

  PetscFunctionBegin;
  PetscCall(VecWAXPY(X, -ts->time_step, update, ts->vec_sol));
  PetscCall(VecAXPBY(X, 1.0 - alpha, alpha, ts->vec_sol));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TSComputeLinearStability_BasicSymplectic(TS ts, PetscReal xr, PetscReal xi, PetscReal *yr, PetscReal *yi)
{
  PetscFunctionBegin;
  *yr = 1.0 + xr;
  *yi = xi;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  TSBasicSymplecticSetType - Set the type of the basic symplectic method

  Logically Collective

  Input Parameters:
+  ts - timestepping context
-  bsymptype - type of the symplectic scheme

  Options Database Key:
.  -ts_basicsymplectic_type <scheme> - select the scheme

  Level: intermediate

  Note:
    The symplectic solver always expects a two-way splitting with the split names being "position" and "momentum". Each split is associated with an `IS` object and a sub-`TS`
    that is intended to store the user-provided RHS function.

.seealso: [](chapter_ts), `TSBASICSYMPLECTIC`, `TSBasicSymplecticType`, `TSBasicSymplecticSetType()`
@*/
PetscErrorCode TSBasicSymplecticSetType(TS ts, TSBasicSymplecticType bsymptype)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts, TS_CLASSID, 1);
  PetscTryMethod(ts, "TSBasicSymplecticSetType_C", (TS, TSBasicSymplecticType), (ts, bsymptype));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  TSBasicSymplecticGetType - Get the type of the basic symplectic method

  Logically Collective

  Input Parameters:
+  ts - timestepping context
-  bsymptype - type of the basic symplectic scheme

  Level: intermediate

.seealso: [](chapter_ts), `TSBASICSYMPLECTIC`, `TSBasicSymplecticType`, `TSBasicSymplecticSetType()`
@*/
PetscErrorCode TSBasicSymplecticGetType(TS ts, TSBasicSymplecticType *bsymptype)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts, TS_CLASSID, 1);
  PetscUseMethod(ts, "TSBasicSymplecticGetType_C", (TS, TSBasicSymplecticType *), (ts, bsymptype));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TSBasicSymplecticSetType_BasicSymplectic(TS ts, TSBasicSymplecticType bsymptype)
{
  TS_BasicSymplectic       *bsymp = (TS_BasicSymplectic *)ts->data;
  BasicSymplecticSchemeLink link;
  PetscBool                 match;

  PetscFunctionBegin;
  if (bsymp->scheme) {
    PetscCall(PetscStrcmp(bsymp->scheme->name, bsymptype, &match));
    if (match) PetscFunctionReturn(PETSC_SUCCESS);
  }
  for (link = BasicSymplecticSchemeList; link; link = link->next) {
    PetscCall(PetscStrcmp(link->sch.name, bsymptype, &match));
    if (match) {
      bsymp->scheme = &link->sch;
      PetscFunctionReturn(PETSC_SUCCESS);
    }
  }
  SETERRQ(PetscObjectComm((PetscObject)ts), PETSC_ERR_ARG_UNKNOWN_TYPE, "Could not find '%s'", bsymptype);
}

static PetscErrorCode TSBasicSymplecticGetType_BasicSymplectic(TS ts, TSBasicSymplecticType *bsymptype)
{
  TS_BasicSymplectic *bsymp = (TS_BasicSymplectic *)ts->data;

  PetscFunctionBegin;
  *bsymptype = bsymp->scheme->name;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*MC
  TSBASICSYMPLECTIC - ODE solver using basic symplectic integration schemes

  These methods are intended for separable Hamiltonian systems
.vb
  qdot = dH(q,p,t)/dp
  pdot = -dH(q,p,t)/dq
.ve

  where the Hamiltonian can be split into the sum of kinetic energy and potential energy
.vb
  H(q,p,t) = T(p,t) + V(q,t).
.ve

  As a result, the system can be genearlly represented by
.vb
  qdot = f(p,t) = dT(p,t)/dp
  pdot = g(q,t) = -dV(q,t)/dq
.ve

  and solved iteratively with
.vb
  q_new = q_old + d_i*h*f(p_old,t_old)
  t_new = t_old + d_i*h
  p_new = p_old + c_i*h*g(p_new,t_new)
  i=0,1,...,n.
.ve

  The solution vector should contain both q and p, which correspond to (generalized) position and momentum respectively. Note that the momentum component
  could simply be velocity in some representations. The symplectic solver always expects a two-way splitting with the split names being "position" and "momentum".
  Each split is associated with an `IS` object and a sub-`TS` that is intended to store the user-provided RHS function.

  Level: beginner

  Reference:
. * -  wikipedia (https://en.wikipedia.org/wiki/Symplectic_integrator)

.seealso: [](chapter_ts), `TSCreate()`, `TSSetType()`, `TSRHSSplitSetIS()`, `TSRHSSplitSetRHSFunction()`, `TSType`
M*/
PETSC_EXTERN PetscErrorCode TSCreate_BasicSymplectic(TS ts)
{
  TS_BasicSymplectic *bsymp;

  PetscFunctionBegin;
  PetscCall(TSBasicSymplecticInitializePackage());
  PetscCall(PetscNew(&bsymp));
  ts->data = (void *)bsymp;

  ts->ops->setup           = TSSetUp_BasicSymplectic;
  ts->ops->step            = TSStep_BasicSymplectic;
  ts->ops->reset           = TSReset_BasicSymplectic;
  ts->ops->destroy         = TSDestroy_BasicSymplectic;
  ts->ops->setfromoptions  = TSSetFromOptions_BasicSymplectic;
  ts->ops->view            = TSView_BasicSymplectic;
  ts->ops->interpolate     = TSInterpolate_BasicSymplectic;
  ts->ops->linearstability = TSComputeLinearStability_BasicSymplectic;

  PetscCall(PetscObjectComposeFunction((PetscObject)ts, "TSBasicSymplecticSetType_C", TSBasicSymplecticSetType_BasicSymplectic));
  PetscCall(PetscObjectComposeFunction((PetscObject)ts, "TSBasicSymplecticGetType_C", TSBasicSymplecticGetType_BasicSymplectic));

  PetscCall(TSBasicSymplecticSetType(ts, TSBasicSymplecticDefault));
  PetscFunctionReturn(PETSC_SUCCESS);
}
