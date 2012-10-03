/*
       Code for Timestepping with explicit SSP.
*/
#include <petsc-private/tsimpl.h>                /*I   "petscts.h"   I*/

PetscFList TSSSPList = 0;

typedef struct {
  PetscErrorCode (*onestep)(TS,PetscReal,PetscReal,Vec);
  char *type_name;
  PetscInt nstages;
  Vec *work;
  PetscInt nwork;
  PetscBool  workout;
} TS_SSP;


#undef __FUNCT__
#define __FUNCT__ "TSSSPGetWorkVectors"
static PetscErrorCode TSSSPGetWorkVectors(TS ts,PetscInt n,Vec **work)
{
  TS_SSP *ssp = (TS_SSP*)ts->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (ssp->workout) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Work vectors already gotten");
  if (ssp->nwork < n) {
    if (ssp->nwork > 0) {
      ierr = VecDestroyVecs(ssp->nwork,&ssp->work);CHKERRQ(ierr);
    }
    ierr = VecDuplicateVecs(ts->vec_sol,n,&ssp->work);CHKERRQ(ierr);
    ssp->nwork = n;
  }
  *work = ssp->work;
  ssp->workout = PETSC_TRUE;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSSSPRestoreWorkVectors"
static PetscErrorCode TSSSPRestoreWorkVectors(TS ts,PetscInt n,Vec **work)
{
  TS_SSP *ssp = (TS_SSP*)ts->data;

  PetscFunctionBegin;
  if (!ssp->workout) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ORDER,"Work vectors have not been gotten");
  if (*work != ssp->work) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Wrong work vectors checked out");
  ssp->workout = PETSC_FALSE;
  *work = PETSC_NULL;
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "TSSSPStep_RK_2"
/*MC
   TSSSPRKS2 - Optimal second order SSP Runge-Kutta method, low-storage, c_eff=(s-1)/s

   Pseudocode 2 of Ketcheson 2008

   Level: beginner

.seealso: TSSSP, TSSSPSetType(), TSSSPSetNumStages()
M*/
static PetscErrorCode TSSSPStep_RK_2(TS ts,PetscReal t0,PetscReal dt,Vec sol)
{
  TS_SSP *ssp = (TS_SSP*)ts->data;
  Vec *work,F;
  PetscInt i,s;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  s = ssp->nstages;
  ierr = TSSSPGetWorkVectors(ts,2,&work);CHKERRQ(ierr);
  F = work[1];
  ierr = VecCopy(sol,work[0]);CHKERRQ(ierr);
  for (i=0; i<s-1; i++) {
    PetscReal stage_time = t0+dt*(i/(s-1.));
    ierr = TSPreStage(ts,stage_time);CHKERRQ(ierr);
    ierr = TSComputeRHSFunction(ts,stage_time,work[0],F);CHKERRQ(ierr);
    ierr = VecAXPY(work[0],dt/(s-1.),F);CHKERRQ(ierr);
  }
  ierr = TSComputeRHSFunction(ts,t0+dt,work[0],F);CHKERRQ(ierr);
  ierr = VecAXPBYPCZ(sol,(s-1.)/s,dt/s,1./s,work[0],F);CHKERRQ(ierr);
  ierr = TSSSPRestoreWorkVectors(ts,2,&work);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSSSPStep_RK_3"
/*MC
   TSSSPRKS3 - Optimal third order SSP Runge-Kutta, low-storage, c_eff=(PetscSqrtReal(s)-1)/PetscSqrtReal(s), where PetscSqrtReal(s) is an integer

   Pseudocode 2 of Ketcheson 2008

   Level: beginner

.seealso: TSSSP, TSSSPSetType(), TSSSPSetNumStages()
M*/
static PetscErrorCode TSSSPStep_RK_3(TS ts,PetscReal t0,PetscReal dt,Vec sol)
{
  TS_SSP *ssp = (TS_SSP*)ts->data;
  Vec *work,F;
  PetscInt i,s,n,r;
  PetscReal c,stage_time;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  s = ssp->nstages;
  n = (PetscInt)(PetscSqrtReal((PetscReal)s)+0.001);
  r = s-n;
  if (n*n != s) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_SUP,"No support for optimal third order schemes with %d stages, must be a square number at least 4",s);
  ierr = TSSSPGetWorkVectors(ts,3,&work);CHKERRQ(ierr);
  F = work[2];
  ierr = VecCopy(sol,work[0]);CHKERRQ(ierr);
  for (i=0; i<(n-1)*(n-2)/2; i++) {
    c = (i<n*(n+1)/2) ? 1.*i/(s-n) : (1.*i-n)/(s-n);
    stage_time = t0+c*dt;
    ierr = TSPreStage(ts,stage_time);CHKERRQ(ierr);
    ierr = TSComputeRHSFunction(ts,stage_time,work[0],F);CHKERRQ(ierr);
    ierr = VecAXPY(work[0],dt/r,F);CHKERRQ(ierr);
  }
  ierr = VecCopy(work[0],work[1]);CHKERRQ(ierr);
  for ( ; i<n*(n+1)/2-1; i++) {
    c = (i<n*(n+1)/2) ? 1.*i/(s-n) : (1.*i-n)/(s-n);
    stage_time = t0+c*dt;
    ierr = TSPreStage(ts,stage_time);CHKERRQ(ierr);
    ierr = TSComputeRHSFunction(ts,stage_time,work[0],F);CHKERRQ(ierr);
    ierr = VecAXPY(work[0],dt/r,F);CHKERRQ(ierr);
  }
  {
    c = (i<n*(n+1)/2) ? 1.*i/(s-n) : (1.*i-n)/(s-n);
    stage_time = t0+c*dt;
    ierr = TSPreStage(ts,stage_time);CHKERRQ(ierr);
    ierr = TSComputeRHSFunction(ts,stage_time,work[0],F);CHKERRQ(ierr);
    ierr = VecAXPBYPCZ(work[0],1.*n/(2*n-1.),(n-1.)*dt/(r*(2*n-1)),(n-1.)/(2*n-1.),work[1],F);CHKERRQ(ierr);
    i++;
  }
  for ( ; i<s; i++) {
    c = (i<n*(n+1)/2) ? 1.*i/(s-n) : (1.*i-n)/(s-n);
    stage_time = t0+c*dt;
    ierr = TSPreStage(ts,stage_time);CHKERRQ(ierr);
    ierr = TSComputeRHSFunction(ts,stage_time,work[0],F);CHKERRQ(ierr);
    ierr = VecAXPY(work[0],dt/r,F);CHKERRQ(ierr);
  }
  ierr = VecCopy(work[0],sol);CHKERRQ(ierr);
  ierr = TSSSPRestoreWorkVectors(ts,3,&work);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSSSPStep_RK_10_4"
/*MC
   TSSSPRKS104 - Optimal fourth order SSP Runge-Kutta, low-storage (2N), c_eff=0.6

   SSPRK(10,4), Pseudocode 3 of Ketcheson 2008

   Level: beginner

.seealso: TSSSP, TSSSPSetType()
M*/
static PetscErrorCode TSSSPStep_RK_10_4(TS ts,PetscReal t0,PetscReal dt,Vec sol)
{
  const PetscReal c[10] = {0, 1./6, 2./6, 3./6, 4./6, 2./6, 3./6, 4./6, 5./6, 1};
  Vec *work,F;
  PetscInt i;
  PetscReal stage_time;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = TSSSPGetWorkVectors(ts,3,&work);CHKERRQ(ierr);
  F = work[2];
  ierr = VecCopy(sol,work[0]);CHKERRQ(ierr);
  for (i=0; i<5; i++) {
    stage_time = t0+c[i]*dt;
    ierr = TSPreStage(ts,stage_time);CHKERRQ(ierr);
    ierr = TSComputeRHSFunction(ts,stage_time,work[0],F);CHKERRQ(ierr);
    ierr = VecAXPY(work[0],dt/6,F);CHKERRQ(ierr);
  }
  ierr = VecAXPBYPCZ(work[1],1./25,9./25,0,sol,work[0]);CHKERRQ(ierr);
  ierr = VecAXPBY(work[0],15,-5,work[1]);CHKERRQ(ierr);
  for ( ; i<9; i++) {
    stage_time = t0+c[i]*dt;
    ierr = TSPreStage(ts,stage_time);CHKERRQ(ierr);
    ierr = TSComputeRHSFunction(ts,stage_time,work[0],F);CHKERRQ(ierr);
    ierr = VecAXPY(work[0],dt/6,F);CHKERRQ(ierr);
  }
  stage_time = t0+dt;
  ierr = TSPreStage(ts,stage_time);CHKERRQ(ierr);
  ierr = TSComputeRHSFunction(ts,stage_time,work[0],F);CHKERRQ(ierr);
  ierr = VecAXPBYPCZ(work[1],3./5,dt/10,1,work[0],F);CHKERRQ(ierr);
  ierr = VecCopy(work[1],sol);CHKERRQ(ierr);
  ierr = TSSSPRestoreWorkVectors(ts,3,&work);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "TSSetUp_SSP"
static PetscErrorCode TSSetUp_SSP(TS ts)
{

  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSStep_SSP"
static PetscErrorCode TSStep_SSP(TS ts)
{
  TS_SSP        *ssp = (TS_SSP*)ts->data;
  Vec            sol = ts->vec_sol;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = TSPreStep(ts);CHKERRQ(ierr);
  ierr = (*ssp->onestep)(ts,ts->ptime,ts->time_step,sol);CHKERRQ(ierr);
  ts->ptime += ts->time_step;
  ts->steps++;
  PetscFunctionReturn(0);
}
/*------------------------------------------------------------*/
#undef __FUNCT__
#define __FUNCT__ "TSReset_SSP"
static PetscErrorCode TSReset_SSP(TS ts)
{
  TS_SSP         *ssp = (TS_SSP*)ts->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (ssp->work) {ierr = VecDestroyVecs(ssp->nwork,&ssp->work);CHKERRQ(ierr);}
  ssp->nwork = 0;
  ssp->workout = PETSC_FALSE;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSDestroy_SSP"
static PetscErrorCode TSDestroy_SSP(TS ts)
{
  TS_SSP         *ssp = (TS_SSP*)ts->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = TSReset_SSP(ts);CHKERRQ(ierr);
  ierr = PetscFree(ssp->type_name);CHKERRQ(ierr);
  ierr = PetscFree(ts->data);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)ts,"TSSSPGetType_C","",PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)ts,"TSSSPSetType_C","",PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)ts,"TSSSPGetNumStages_C","",PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)ts,"TSSSPSetNumStages_C","",PETSC_NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
/*------------------------------------------------------------*/

#undef __FUNCT__
#define __FUNCT__ "TSSSPSetType"
/*@C
   TSSSPSetType - set the SSP time integration scheme to use

   Logically Collective

   Input Arguments:
   ts - time stepping object
   type - type of scheme to use

   Options Database Keys:
   -ts_ssp_type <rks2>: Type of SSP method (one of) rks2 rks3 rk104
   -ts_ssp_nstages <5>: Number of stages

   Level: beginner

.seealso: TSSSP, TSSSPGetType(), TSSSPSetNumStages(), TSSSPRKS2, TSSSPRKS3, TSSSPRK104
@*/
PetscErrorCode TSSSPSetType(TS ts,TSSSPType type)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  ierr = PetscTryMethod(ts,"TSSSPSetType_C",(TS,TSSSPType),(ts,type));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSSSPGetType"
/*@C
   TSSSPGetType - get the SSP time integration scheme

   Logically Collective

   Input Argument:
   ts - time stepping object

   Output Argument:
   type - type of scheme being used

   Level: beginner

.seealso: TSSSP, TSSSPSettype(), TSSSPSetNumStages(), TSSSPRKS2, TSSSPRKS3, TSSSPRK104
@*/
PetscErrorCode TSSSPGetType(TS ts,TSSSPType *type)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  ierr = PetscTryMethod(ts,"TSSSPGetType_C",(TS,TSSSPType*),(ts,type));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSSSPSetNumStages"
/*@
   TSSSPSetNumStages - set the number of stages to use with the SSP method

   Logically Collective

   Input Arguments:
   ts - time stepping object
   nstages - number of stages

   Options Database Keys:
   -ts_ssp_type <rks2>: NumStages of SSP method (one of) rks2 rks3 rk104
   -ts_ssp_nstages <5>: Number of stages

   Level: beginner

.seealso: TSSSP, TSSSPGetNumStages(), TSSSPSetNumStages(), TSSSPRKS2, TSSSPRKS3, TSSSPRK104
@*/
PetscErrorCode TSSSPSetNumStages(TS ts,PetscInt nstages)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  ierr = PetscTryMethod(ts,"TSSSPSetNumStages_C",(TS,PetscInt),(ts,nstages));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSSSPGetNumStages"
/*@
   TSSSPGetNumStages - get the number of stages in the SSP time integration scheme

   Logically Collective

   Input Argument:
   ts - time stepping object

   Output Argument:
   nstages - number of stages

   Level: beginner

.seealso: TSSSP, TSSSPGetType(), TSSSPSetNumStages(), TSSSPRKS2, TSSSPRKS3, TSSSPRK104
@*/
PetscErrorCode TSSSPGetNumStages(TS ts,PetscInt *nstages)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  ierr = PetscTryMethod(ts,"TSSSPGetNumStages_C",(TS,PetscInt*),(ts,nstages));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
#undef __FUNCT__
#define __FUNCT__ "TSSSPSetType_SSP"
PetscErrorCode TSSSPSetType_SSP(TS ts,TSSSPType type)
{
  PetscErrorCode ierr,(*r)(TS,PetscReal,PetscReal,Vec);
  TS_SSP *ssp = (TS_SSP*)ts->data;

  PetscFunctionBegin;
  ierr = PetscFListFind(TSSSPList,((PetscObject)ts)->comm,type,PETSC_TRUE,(PetscVoidStarFunction)&r);CHKERRQ(ierr);
  if (!r) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_UNKNOWN_TYPE,"Unknown TS_SSP type %s given",type);
  ssp->onestep = r;
  ierr = PetscFree(ssp->type_name);CHKERRQ(ierr);
  ierr = PetscStrallocpy(type,&ssp->type_name);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
#undef __FUNCT__
#define __FUNCT__ "TSSSPGetType_SSP"
PetscErrorCode TSSSPGetType_SSP(TS ts,TSSSPType *type)
{
  TS_SSP *ssp = (TS_SSP*)ts->data;

  PetscFunctionBegin;
  *type = ssp->type_name;
  PetscFunctionReturn(0);
}
#undef __FUNCT__
#define __FUNCT__ "TSSSPSetNumStages_SSP"
PetscErrorCode TSSSPSetNumStages_SSP(TS ts,PetscInt nstages)
{
  TS_SSP *ssp = (TS_SSP*)ts->data;

  PetscFunctionBegin;
  ssp->nstages = nstages;
  PetscFunctionReturn(0);
}
#undef __FUNCT__
#define __FUNCT__ "TSSSPGetNumStages_SSP"
PetscErrorCode TSSSPGetNumStages_SSP(TS ts,PetscInt *nstages)
{
  TS_SSP *ssp = (TS_SSP*)ts->data;

  PetscFunctionBegin;
  *nstages = ssp->nstages;
  PetscFunctionReturn(0);
}
EXTERN_C_END

#undef __FUNCT__
#define __FUNCT__ "TSSetFromOptions_SSP"
static PetscErrorCode TSSetFromOptions_SSP(TS ts)
{
  char tname[256] = TSSSPRKS2;
  TS_SSP *ssp = (TS_SSP*)ts->data;
  PetscErrorCode ierr;
  PetscBool  flg;

  PetscFunctionBegin;
  ierr = PetscOptionsHead("SSP ODE solver options");CHKERRQ(ierr);
  {
    ierr = PetscOptionsList("-ts_ssp_type","Type of SSP method","TSSSPSetType",TSSSPList,tname,tname,sizeof(tname),&flg);CHKERRQ(ierr);
    if (flg) {
      ierr = TSSSPSetType(ts,tname);CHKERRQ(ierr);
    }
    ierr = PetscOptionsInt("-ts_ssp_nstages","Number of stages","TSSSPSetNumStages",ssp->nstages,&ssp->nstages,PETSC_NULL);CHKERRQ(ierr);
  }
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSView_SSP"
static PetscErrorCode TSView_SSP(TS ts,PetscViewer viewer)
{
  PetscFunctionBegin;
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
  Ketcheson, Highly efficient strong stability preserving Runge-Kutta methods with low-storage implementations, SISC, 2008.

  Gottlieb, Ketcheson, and Shu, High order strong stability preserving time discretizations, J Scientific Computing, 2009.

.seealso:  TSCreate(), TS, TSSetType()

M*/
EXTERN_C_BEGIN
#undef __FUNCT__
#define __FUNCT__ "TSCreate_SSP"
PetscErrorCode  TSCreate_SSP(TS ts)
{
  TS_SSP       *ssp;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!TSSSPList) {
    ierr = PetscFListAdd(&TSSSPList,TSSSPRKS2,  "TSSSPStep_RK_2",   (void(*)(void))TSSSPStep_RK_2);CHKERRQ(ierr);
    ierr = PetscFListAdd(&TSSSPList,TSSSPRKS3,  "TSSSPStep_RK_3",   (void(*)(void))TSSSPStep_RK_3);CHKERRQ(ierr);
    ierr = PetscFListAdd(&TSSSPList,TSSSPRK104, "TSSSPStep_RK_10_4",(void(*)(void))TSSSPStep_RK_10_4);CHKERRQ(ierr);
  }

  ts->ops->setup           = TSSetUp_SSP;
  ts->ops->step            = TSStep_SSP;
  ts->ops->reset           = TSReset_SSP;
  ts->ops->destroy         = TSDestroy_SSP;
  ts->ops->setfromoptions  = TSSetFromOptions_SSP;
  ts->ops->view            = TSView_SSP;

  ierr = PetscNewLog(ts,TS_SSP,&ssp);CHKERRQ(ierr);
  ts->data = (void*)ssp;

  ierr = PetscObjectComposeFunctionDynamic((PetscObject)ts,"TSSSPGetType_C","TSSSPGetType_SSP",TSSSPGetType_SSP);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)ts,"TSSSPSetType_C","TSSSPSetType_SSP",TSSSPSetType_SSP);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)ts,"TSSSPGetNumStages_C","TSSSPGetNumStages_SSP",TSSSPGetNumStages_SSP);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)ts,"TSSSPSetNumStages_C","TSSSPSetNumStages_SSP",TSSSPSetNumStages_SSP);CHKERRQ(ierr);

  ierr = TSSSPSetType(ts,TSSSPRKS2);CHKERRQ(ierr);
  ssp->nstages = 5;
  PetscFunctionReturn(0);
}
EXTERN_C_END
