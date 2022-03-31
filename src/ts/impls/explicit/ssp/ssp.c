/*
       Code for Timestepping with explicit SSP.
*/
#include <petsc/private/tsimpl.h>                /*I   "petscts.h"   I*/

PetscFunctionList TSSSPList = NULL;
static PetscBool TSSSPPackageInitialized;

typedef struct {
  PetscErrorCode (*onestep)(TS,PetscReal,PetscReal,Vec);
  char           *type_name;
  PetscInt       nstages;
  Vec            *work;
  PetscInt       nwork;
  PetscBool      workout;
} TS_SSP;

static PetscErrorCode TSSSPGetWorkVectors(TS ts,PetscInt n,Vec **work)
{
  TS_SSP         *ssp = (TS_SSP*)ts->data;

  PetscFunctionBegin;
  PetscCheck(!ssp->workout,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Work vectors already gotten");
  if (ssp->nwork < n) {
    if (ssp->nwork > 0) {
      PetscCall(VecDestroyVecs(ssp->nwork,&ssp->work));
    }
    PetscCall(VecDuplicateVecs(ts->vec_sol,n,&ssp->work));
    ssp->nwork = n;
  }
  *work = ssp->work;
  ssp->workout = PETSC_TRUE;
  PetscFunctionReturn(0);
}

static PetscErrorCode TSSSPRestoreWorkVectors(TS ts,PetscInt n,Vec **work)
{
  TS_SSP *ssp = (TS_SSP*)ts->data;

  PetscFunctionBegin;
  PetscCheck(ssp->workout,PETSC_COMM_SELF,PETSC_ERR_ORDER,"Work vectors have not been gotten");
  PetscCheckFalse(*work != ssp->work,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Wrong work vectors checked out");
  ssp->workout = PETSC_FALSE;
  *work = NULL;
  PetscFunctionReturn(0);
}

/*MC
   TSSSPRKS2 - Optimal second order SSP Runge-Kutta method, low-storage, c_eff=(s-1)/s

   Pseudocode 2 of Ketcheson 2008

   Level: beginner

.seealso: TSSSP, TSSSPSetType(), TSSSPSetNumStages()
M*/
static PetscErrorCode TSSSPStep_RK_2(TS ts,PetscReal t0,PetscReal dt,Vec sol)
{
  TS_SSP         *ssp = (TS_SSP*)ts->data;
  Vec            *work,F;
  PetscInt       i,s;

  PetscFunctionBegin;
  s    = ssp->nstages;
  PetscCall(TSSSPGetWorkVectors(ts,2,&work));
  F    = work[1];
  PetscCall(VecCopy(sol,work[0]));
  for (i=0; i<s-1; i++) {
    PetscReal stage_time = t0+dt*(i/(s-1.));
    PetscCall(TSPreStage(ts,stage_time));
    PetscCall(TSComputeRHSFunction(ts,stage_time,work[0],F));
    PetscCall(VecAXPY(work[0],dt/(s-1.),F));
  }
  PetscCall(TSComputeRHSFunction(ts,t0+dt,work[0],F));
  PetscCall(VecAXPBYPCZ(sol,(s-1.)/s,dt/s,1./s,work[0],F));
  PetscCall(TSSSPRestoreWorkVectors(ts,2,&work));
  PetscFunctionReturn(0);
}

/*MC
   TSSSPRKS3 - Optimal third order SSP Runge-Kutta, low-storage, c_eff=(PetscSqrtReal(s)-1)/PetscSqrtReal(s), where PetscSqrtReal(s) is an integer

   Pseudocode 2 of Ketcheson 2008

   Level: beginner

.seealso: TSSSP, TSSSPSetType(), TSSSPSetNumStages()
M*/
static PetscErrorCode TSSSPStep_RK_3(TS ts,PetscReal t0,PetscReal dt,Vec sol)
{
  TS_SSP         *ssp = (TS_SSP*)ts->data;
  Vec            *work,F;
  PetscInt       i,s,n,r;
  PetscReal      c,stage_time;

  PetscFunctionBegin;
  s = ssp->nstages;
  n = (PetscInt)(PetscSqrtReal((PetscReal)s)+0.001);
  r = s-n;
  PetscCheck(n*n == s,PETSC_COMM_SELF,PETSC_ERR_SUP,"No support for optimal third order schemes with %d stages, must be a square number at least 4",s);
  PetscCall(TSSSPGetWorkVectors(ts,3,&work));
  F    = work[2];
  PetscCall(VecCopy(sol,work[0]));
  for (i=0; i<(n-1)*(n-2)/2; i++) {
    c          = (i<n*(n+1)/2) ? 1.*i/(s-n) : (1.*i-n)/(s-n);
    stage_time = t0+c*dt;
    PetscCall(TSPreStage(ts,stage_time));
    PetscCall(TSComputeRHSFunction(ts,stage_time,work[0],F));
    PetscCall(VecAXPY(work[0],dt/r,F));
  }
  PetscCall(VecCopy(work[0],work[1]));
  for (; i<n*(n+1)/2-1; i++) {
    c          = (i<n*(n+1)/2) ? 1.*i/(s-n) : (1.*i-n)/(s-n);
    stage_time = t0+c*dt;
    PetscCall(TSPreStage(ts,stage_time));
    PetscCall(TSComputeRHSFunction(ts,stage_time,work[0],F));
    PetscCall(VecAXPY(work[0],dt/r,F));
  }
  {
    c          = (i<n*(n+1)/2) ? 1.*i/(s-n) : (1.*i-n)/(s-n);
    stage_time = t0+c*dt;
    PetscCall(TSPreStage(ts,stage_time));
    PetscCall(TSComputeRHSFunction(ts,stage_time,work[0],F));
    PetscCall(VecAXPBYPCZ(work[0],1.*n/(2*n-1.),(n-1.)*dt/(r*(2*n-1)),(n-1.)/(2*n-1.),work[1],F));
    i++;
  }
  for (; i<s; i++) {
    c          = (i<n*(n+1)/2) ? 1.*i/(s-n) : (1.*i-n)/(s-n);
    stage_time = t0+c*dt;
    PetscCall(TSPreStage(ts,stage_time));
    PetscCall(TSComputeRHSFunction(ts,stage_time,work[0],F));
    PetscCall(VecAXPY(work[0],dt/r,F));
  }
  PetscCall(VecCopy(work[0],sol));
  PetscCall(TSSSPRestoreWorkVectors(ts,3,&work));
  PetscFunctionReturn(0);
}

/*MC
   TSSSPRKS104 - Optimal fourth order SSP Runge-Kutta, low-storage (2N), c_eff=0.6

   SSPRK(10,4), Pseudocode 3 of Ketcheson 2008

   Level: beginner

.seealso: TSSSP, TSSSPSetType()
M*/
static PetscErrorCode TSSSPStep_RK_10_4(TS ts,PetscReal t0,PetscReal dt,Vec sol)
{
  const PetscReal c[10] = {0, 1./6, 2./6, 3./6, 4./6, 2./6, 3./6, 4./6, 5./6, 1};
  Vec             *work,F;
  PetscInt        i;
  PetscReal       stage_time;

  PetscFunctionBegin;
  PetscCall(TSSSPGetWorkVectors(ts,3,&work));
  F    = work[2];
  PetscCall(VecCopy(sol,work[0]));
  for (i=0; i<5; i++) {
    stage_time = t0+c[i]*dt;
    PetscCall(TSPreStage(ts,stage_time));
    PetscCall(TSComputeRHSFunction(ts,stage_time,work[0],F));
    PetscCall(VecAXPY(work[0],dt/6,F));
  }
  PetscCall(VecAXPBYPCZ(work[1],1./25,9./25,0,sol,work[0]));
  PetscCall(VecAXPBY(work[0],15,-5,work[1]));
  for (; i<9; i++) {
    stage_time = t0+c[i]*dt;
    PetscCall(TSPreStage(ts,stage_time));
    PetscCall(TSComputeRHSFunction(ts,stage_time,work[0],F));
    PetscCall(VecAXPY(work[0],dt/6,F));
  }
  stage_time = t0+dt;
  PetscCall(TSPreStage(ts,stage_time));
  PetscCall(TSComputeRHSFunction(ts,stage_time,work[0],F));
  PetscCall(VecAXPBYPCZ(work[1],3./5,dt/10,1,work[0],F));
  PetscCall(VecCopy(work[1],sol));
  PetscCall(TSSSPRestoreWorkVectors(ts,3,&work));
  PetscFunctionReturn(0);
}

static PetscErrorCode TSSetUp_SSP(TS ts)
{
  PetscFunctionBegin;
  PetscCall(TSCheckImplicitTerm(ts));
  PetscCall(TSGetAdapt(ts,&ts->adapt));
  PetscCall(TSAdaptCandidatesClear(ts->adapt));
  PetscFunctionReturn(0);
}

static PetscErrorCode TSStep_SSP(TS ts)
{
  TS_SSP         *ssp = (TS_SSP*)ts->data;
  Vec            sol  = ts->vec_sol;
  PetscBool      stageok,accept = PETSC_TRUE;
  PetscReal      next_time_step = ts->time_step;

  PetscFunctionBegin;
  PetscCall((*ssp->onestep)(ts,ts->ptime,ts->time_step,sol));
  PetscCall(TSPostStage(ts,ts->ptime,0,&sol));
  PetscCall(TSAdaptCheckStage(ts->adapt,ts,ts->ptime+ts->time_step,sol,&stageok));
  if (!stageok) {ts->reason = TS_DIVERGED_STEP_REJECTED; PetscFunctionReturn(0);}

  PetscCall(TSAdaptChoose(ts->adapt,ts,ts->time_step,NULL,&next_time_step,&accept));
  if (!accept) {ts->reason = TS_DIVERGED_STEP_REJECTED; PetscFunctionReturn(0);}

  ts->ptime += ts->time_step;
  ts->time_step = next_time_step;
  PetscFunctionReturn(0);
}
/*------------------------------------------------------------*/

static PetscErrorCode TSReset_SSP(TS ts)
{
  TS_SSP         *ssp = (TS_SSP*)ts->data;

  PetscFunctionBegin;
  if (ssp->work) PetscCall(VecDestroyVecs(ssp->nwork,&ssp->work));
  ssp->nwork   = 0;
  ssp->workout = PETSC_FALSE;
  PetscFunctionReturn(0);
}

static PetscErrorCode TSDestroy_SSP(TS ts)
{
  TS_SSP         *ssp = (TS_SSP*)ts->data;

  PetscFunctionBegin;
  PetscCall(TSReset_SSP(ts));
  PetscCall(PetscFree(ssp->type_name));
  PetscCall(PetscFree(ts->data));
  PetscCall(PetscObjectComposeFunction((PetscObject)ts,"TSSSPGetType_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)ts,"TSSSPSetType_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)ts,"TSSSPGetNumStages_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)ts,"TSSSPSetNumStages_C",NULL));
  PetscFunctionReturn(0);
}
/*------------------------------------------------------------*/

/*@C
   TSSSPSetType - set the SSP time integration scheme to use

   Logically Collective

   Input Parameters:
+  ts - time stepping object
-  ssptype - type of scheme to use

   Options Database Keys:
   -ts_ssp_type <rks2>: Type of SSP method (one of) rks2 rks3 rk104
   -ts_ssp_nstages <5>: Number of stages

   Level: beginner

.seealso: TSSSP, TSSSPGetType(), TSSSPSetNumStages(), TSSSPRKS2, TSSSPRKS3, TSSSPRK104
@*/
PetscErrorCode TSSSPSetType(TS ts,TSSSPType ssptype)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  PetscValidCharPointer(ssptype,2);
  PetscTryMethod(ts,"TSSSPSetType_C",(TS,TSSSPType),(ts,ssptype));
  PetscFunctionReturn(0);
}

/*@C
   TSSSPGetType - get the SSP time integration scheme

   Logically Collective

   Input Parameter:
.  ts - time stepping object

   Output Parameter:
.  type - type of scheme being used

   Level: beginner

.seealso: TSSSP, TSSSPSettype(), TSSSPSetNumStages(), TSSSPRKS2, TSSSPRKS3, TSSSPRK104
@*/
PetscErrorCode TSSSPGetType(TS ts,TSSSPType *type)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  PetscUseMethod(ts,"TSSSPGetType_C",(TS,TSSSPType*),(ts,type));
  PetscFunctionReturn(0);
}

/*@
   TSSSPSetNumStages - set the number of stages to use with the SSP method

   Logically Collective

   Input Parameters:
+  ts - time stepping object
-  nstages - number of stages

   Options Database Keys:
   -ts_ssp_type <rks2>: NumStages of SSP method (one of) rks2 rks3 rk104
   -ts_ssp_nstages <5>: Number of stages

   Level: beginner

.seealso: TSSSP, TSSSPGetNumStages(), TSSSPSetNumStages(), TSSSPRKS2, TSSSPRKS3, TSSSPRK104
@*/
PetscErrorCode TSSSPSetNumStages(TS ts,PetscInt nstages)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  PetscTryMethod(ts,"TSSSPSetNumStages_C",(TS,PetscInt),(ts,nstages));
  PetscFunctionReturn(0);
}

/*@
   TSSSPGetNumStages - get the number of stages in the SSP time integration scheme

   Logically Collective

   Input Parameter:
.  ts - time stepping object

   Output Parameter:
.  nstages - number of stages

   Level: beginner

.seealso: TSSSP, TSSSPGetType(), TSSSPSetNumStages(), TSSSPRKS2, TSSSPRKS3, TSSSPRK104
@*/
PetscErrorCode TSSSPGetNumStages(TS ts,PetscInt *nstages)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  PetscUseMethod(ts,"TSSSPGetNumStages_C",(TS,PetscInt*),(ts,nstages));
  PetscFunctionReturn(0);
}

static PetscErrorCode TSSSPSetType_SSP(TS ts,TSSSPType type)
{
  TS_SSP         *ssp = (TS_SSP*)ts->data;
  PetscErrorCode (*r)(TS,PetscReal,PetscReal,Vec);

  PetscFunctionBegin;
  PetscCall(PetscFunctionListFind(TSSSPList,type,&r));
  PetscCheck(r,PETSC_COMM_SELF,PETSC_ERR_ARG_UNKNOWN_TYPE,"Unknown TS_SSP type %s given",type);
  ssp->onestep = r;
  PetscCall(PetscFree(ssp->type_name));
  PetscCall(PetscStrallocpy(type,&ssp->type_name));
  ts->default_adapt_type = TSADAPTNONE;
  PetscFunctionReturn(0);
}
static PetscErrorCode TSSSPGetType_SSP(TS ts,TSSSPType *type)
{
  TS_SSP *ssp = (TS_SSP*)ts->data;

  PetscFunctionBegin;
  *type = ssp->type_name;
  PetscFunctionReturn(0);
}
static PetscErrorCode TSSSPSetNumStages_SSP(TS ts,PetscInt nstages)
{
  TS_SSP *ssp = (TS_SSP*)ts->data;

  PetscFunctionBegin;
  ssp->nstages = nstages;
  PetscFunctionReturn(0);
}
static PetscErrorCode TSSSPGetNumStages_SSP(TS ts,PetscInt *nstages)
{
  TS_SSP *ssp = (TS_SSP*)ts->data;

  PetscFunctionBegin;
  *nstages = ssp->nstages;
  PetscFunctionReturn(0);
}

static PetscErrorCode TSSetFromOptions_SSP(PetscOptionItems *PetscOptionsObject,TS ts)
{
  char           tname[256] = TSSSPRKS2;
  TS_SSP         *ssp       = (TS_SSP*)ts->data;
  PetscBool      flg;

  PetscFunctionBegin;
  PetscCall(PetscOptionsHead(PetscOptionsObject,"SSP ODE solver options"));
  {
    PetscCall(PetscOptionsFList("-ts_ssp_type","Type of SSP method","TSSSPSetType",TSSSPList,tname,tname,sizeof(tname),&flg));
    if (flg) {
      PetscCall(TSSSPSetType(ts,tname));
    }
    PetscCall(PetscOptionsInt("-ts_ssp_nstages","Number of stages","TSSSPSetNumStages",ssp->nstages,&ssp->nstages,NULL));
  }
  PetscCall(PetscOptionsTail());
  PetscFunctionReturn(0);
}

static PetscErrorCode TSView_SSP(TS ts,PetscViewer viewer)
{
  TS_SSP         *ssp = (TS_SSP*)ts->data;
  PetscBool      ascii;

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&ascii));
  if (ascii) PetscCall(PetscViewerASCIIPrintf(viewer,"  Scheme: %s\n",ssp->type_name));
  PetscFunctionReturn(0);
}

/* ------------------------------------------------------------ */

/*MC
      TSSSP - Explicit strong stability preserving ODE solver

  Most hyperbolic conservation laws have exact solutions that are total variation diminishing (TVD) or total variation
  bounded (TVB) although these solutions often contain discontinuities.  Spatial discretizations such as Godunov's
  scheme and high-resolution finite volume methods (TVD limiters, ENO/WENO) are designed to preserve these properties,
  but they are usually formulated using a forward Euler time discretization or by coupling the space and time
  discretization as in the classical Lax-Wendroff scheme.  When the space and time discretization is coupled, it is very
  difficult to produce schemes with high temporal accuracy while preserving TVD properties.  An alternative is the
  semidiscrete formulation where we choose a spatial discretization that is TVD with forward Euler and then choose a
  time discretization that preserves the TVD property.  Such integrators are called strong stability preserving (SSP).

  Let c_eff be the minimum number of function evaluations required to step as far as one step of forward Euler while
  still being SSP.  Some theoretical bounds

  1. There are no explicit methods with c_eff > 1.

  2. There are no explicit methods beyond order 4 (for nonlinear problems) and c_eff > 0.

  3. There are no implicit methods with order greater than 1 and c_eff > 2.

  This integrator provides Runge-Kutta methods of order 2, 3, and 4 with maximal values of c_eff.  More stages allows
  for larger values of c_eff which improves efficiency.  These implementations are low-memory and only use 2 or 3 work
  vectors regardless of the total number of stages, so e.g. 25-stage 3rd order methods may be an excellent choice.

  Methods can be chosen with -ts_ssp_type {rks2,rks3,rk104}

  rks2: Second order methods with any number s>1 of stages.  c_eff = (s-1)/s

  rks3: Third order methods with s=n^2 stages, n>1.  c_eff = (s-n)/s

  rk104: A 10-stage fourth order method.  c_eff = 0.6

  Level: beginner

  References:
+  * - Ketcheson, Highly efficient strong stability preserving Runge Kutta methods with low storage implementations, SISC, 2008.
-  * - Gottlieb, Ketcheson, and Shu, High order strong stability preserving time discretizations, J Scientific Computing, 2009.

.seealso:  TSCreate(), TS, TSSetType()

M*/
PETSC_EXTERN PetscErrorCode TSCreate_SSP(TS ts)
{
  TS_SSP         *ssp;

  PetscFunctionBegin;
  PetscCall(TSSSPInitializePackage());

  ts->ops->setup          = TSSetUp_SSP;
  ts->ops->step           = TSStep_SSP;
  ts->ops->reset          = TSReset_SSP;
  ts->ops->destroy        = TSDestroy_SSP;
  ts->ops->setfromoptions = TSSetFromOptions_SSP;
  ts->ops->view           = TSView_SSP;

  PetscCall(PetscNewLog(ts,&ssp));
  ts->data = (void*)ssp;

  PetscCall(PetscObjectComposeFunction((PetscObject)ts,"TSSSPGetType_C",TSSSPGetType_SSP));
  PetscCall(PetscObjectComposeFunction((PetscObject)ts,"TSSSPSetType_C",TSSSPSetType_SSP));
  PetscCall(PetscObjectComposeFunction((PetscObject)ts,"TSSSPGetNumStages_C",TSSSPGetNumStages_SSP));
  PetscCall(PetscObjectComposeFunction((PetscObject)ts,"TSSSPSetNumStages_C",TSSSPSetNumStages_SSP));

  PetscCall(TSSSPSetType(ts,TSSSPRKS2));
  ssp->nstages = 5;
  PetscFunctionReturn(0);
}

/*@C
  TSSSPInitializePackage - This function initializes everything in the TSSSP package. It is called
  from TSInitializePackage().

  Level: developer

.seealso: PetscInitialize()
@*/
PetscErrorCode TSSSPInitializePackage(void)
{
  PetscFunctionBegin;
  if (TSSSPPackageInitialized) PetscFunctionReturn(0);
  TSSSPPackageInitialized = PETSC_TRUE;
  PetscCall(PetscFunctionListAdd(&TSSSPList,TSSSPRKS2, TSSSPStep_RK_2));
  PetscCall(PetscFunctionListAdd(&TSSSPList,TSSSPRKS3, TSSSPStep_RK_3));
  PetscCall(PetscFunctionListAdd(&TSSSPList,TSSSPRK104,TSSSPStep_RK_10_4));
  PetscCall(PetscRegisterFinalize(TSSSPFinalizePackage));
  PetscFunctionReturn(0);
}

/*@C
  TSSSPFinalizePackage - This function destroys everything in the TSSSP package. It is
  called from PetscFinalize().

  Level: developer

.seealso: PetscFinalize()
@*/
PetscErrorCode TSSSPFinalizePackage(void)
{
  PetscFunctionBegin;
  TSSSPPackageInitialized = PETSC_FALSE;
  PetscCall(PetscFunctionListDestroy(&TSSSPList));
  PetscFunctionReturn(0);
}
