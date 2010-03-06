#define PETSCTS_DLL

/*
       Code for Timestepping with explicit SSP.
*/
#include "private/tsimpl.h"                /*I   "petscts.h"   I*/

PetscFList TSSSPList = 0;
#define TSSSPType char*

#define TSSSPRKS2  "rks2"
#define TSSSPRKS3  "rks3"
#define TSSSPRK104 "rk104"

typedef struct {
  PetscErrorCode (*onestep)(TS,PetscReal,PetscReal,Vec);
  PetscInt nstages;
  Vec xdot;
  Vec *work;
  PetscInt nwork;
  PetscTruth workout;
} TS_SSP;


#undef __FUNCT__  
#define __FUNCT__ "SSPGetWorkVectors"
static PetscErrorCode SSPGetWorkVectors(TS ts,PetscInt n,Vec **work)
{
  TS_SSP *ssp = (TS_SSP*)ts->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (ssp->workout) SETERRQ(PETSC_ERR_PLIB,"Work vectors already gotten");
  if (ssp->nwork < n) {
    if (ssp->nwork > 0) {
      ierr = VecDestroyVecs(ssp->work,ssp->nwork);CHKERRQ(ierr);
    }
    ierr = VecDuplicateVecs(ts->vec_sol,n,&ssp->work);CHKERRQ(ierr);
    ssp->nwork = n;
  }
  *work = ssp->work;
  ssp->workout = PETSC_TRUE;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "SSPRestoreWorkVectors"
static PetscErrorCode SSPRestoreWorkVectors(TS ts,PetscInt n,Vec **work)
{
  TS_SSP *ssp = (TS_SSP*)ts->data;

  PetscFunctionBegin;
  if (!ssp->workout) SETERRQ(PETSC_ERR_ORDER,"Work vectors have not been gotten");
  if (*work != ssp->work) SETERRQ(PETSC_ERR_PLIB,"Wrong work vectors checked out");
  ssp->workout = PETSC_FALSE;
  *work = PETSC_NULL;
  PetscFunctionReturn(0);
}


#undef __FUNCT__  
#define __FUNCT__ "SSPStep_RK_2"
/* Optimal second order SSP Runge-Kutta, low-storage, c_eff=(s-1)/s */
/* Pseudocode 2 of Ketcheson 2008 */
static PetscErrorCode SSPStep_RK_2(TS ts,PetscReal t0,PetscReal dt,Vec sol)
{
  TS_SSP *ssp = (TS_SSP*)ts->data;
  Vec *work,F;
  PetscInt i,s;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  s = ssp->nstages;
  ierr = SSPGetWorkVectors(ts,2,&work);CHKERRQ(ierr);
  F = work[1];
  ierr = VecCopy(sol,work[0]);CHKERRQ(ierr);
  for (i=0; i<s-1; i++) {
    ierr = TSComputeRHSFunction(ts,t0+dt*(i/(s-1.)),work[0],F);CHKERRQ(ierr);
    ierr = VecAXPY(work[0],dt/(s-1.),F);CHKERRQ(ierr);
  }
  ierr = TSComputeRHSFunction(ts,t0+dt,work[0],F);CHKERRQ(ierr);
  ierr = VecAXPBYPCZ(sol,(s-1.)/s,dt/s,1./s,work[0],F);CHKERRQ(ierr);
  ierr = SSPRestoreWorkVectors(ts,2,&work);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "SSPStep_RK_3"
/* Optimal third order SSP Runge-Kutta, low-storage, c_eff=(sqrt(s)-1)/sqrt(s), where sqrt(s) is an integer */
/* Pseudocode 2 of Ketcheson 2008 */
static PetscErrorCode SSPStep_RK_3(TS ts,PetscReal t0,PetscReal dt,Vec sol)
{
  TS_SSP *ssp = (TS_SSP*)ts->data;
  Vec *work,F;
  PetscInt i,s,n,r;
  PetscReal c;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  s = ssp->nstages;
  n = (PetscInt)(sqrt((PetscReal)s)+0.001);
  r = s-n;
  if (n*n != s) SETERRQ1(PETSC_ERR_SUP,"No support for optimal third order schemes with %d stages, must be a square number at least 4",s);
  ierr = SSPGetWorkVectors(ts,3,&work);CHKERRQ(ierr);
  F = work[2];
  ierr = VecCopy(sol,work[0]);CHKERRQ(ierr);
  for (i=0; i<(n-1)*(n-2)/2; i++) {
    c = (i<n*(n+1)/2) ? 1.*i/(s-n) : (1.*i-n)/(s-n);
    ierr = TSComputeRHSFunction(ts,t0+c*dt,work[0],F);CHKERRQ(ierr);
    ierr = VecAXPY(work[0],dt/r,F);CHKERRQ(ierr);
  }
  ierr = VecCopy(work[0],work[1]);CHKERRQ(ierr);
  for ( ; i<n*(n+1)/2-1; i++) {
    c = (i<n*(n+1)/2) ? 1.*i/(s-n) : (1.*i-n)/(s-n);
    ierr = TSComputeRHSFunction(ts,t0+c*dt,work[0],F);CHKERRQ(ierr);
    ierr = VecAXPY(work[0],dt/r,F);CHKERRQ(ierr);
  }
  {
    c = (i<n*(n+1)/2) ? 1.*i/(s-n) : (1.*i-n)/(s-n);
    ierr = TSComputeRHSFunction(ts,t0+c*dt,work[0],F);CHKERRQ(ierr);
    ierr = VecAXPBYPCZ(work[0],1.*n/(2*n-1.),(n-1.)*dt/(r*(2*n-1)),(n-1.)/(2*n-1.),work[1],F);CHKERRQ(ierr);
    i++;
  }
  for ( ; i<s; i++) {
    c = (i<n*(n+1)/2) ? 1.*i/(s-n) : (1.*i-n)/(s-n);
    ierr = TSComputeRHSFunction(ts,t0+c*dt,work[0],F);CHKERRQ(ierr);
    ierr = VecAXPY(work[0],dt/r,F);CHKERRQ(ierr);
  }
  ierr = VecCopy(work[0],sol);CHKERRQ(ierr);
  ierr = SSPRestoreWorkVectors(ts,3,&work);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "SSPStep_RK_10_4"
/* Optimal fourth order SSP Runge-Kutta, low-storage (2N), c_eff=0.6 */
/* SSPRK(10,4), Pseudocode 3 of Ketcheson 2008 */
static PetscErrorCode SSPStep_RK_10_4(TS ts,PetscReal t0,PetscReal dt,Vec sol)
{
  TS_SSP *ssp = (TS_SSP*)ts->data;
  const PetscReal c[10] = {0, 1./6, 2./6, 3./6, 4./6, 2./6, 3./6, 4./6, 5./6, 1};
  Vec *work,F;
  PetscInt i,s;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  s = ssp->nstages;
  ierr = SSPGetWorkVectors(ts,3,&work);CHKERRQ(ierr);
  F = work[2];
  ierr = VecCopy(sol,work[0]);CHKERRQ(ierr);
  for (i=0; i<5; i++) {
    ierr = TSComputeRHSFunction(ts,t0+c[i],work[0],F);CHKERRQ(ierr);
    ierr = VecAXPY(work[0],dt/6,F);CHKERRQ(ierr);
  }
  ierr = VecAXPBYPCZ(work[1],1./25,9./25,0,sol,work[0]);CHKERRQ(ierr);
  ierr = VecAXPBY(work[0],15,-5,work[1]);CHKERRQ(ierr);
  for ( ; i<9; i++) {
    ierr = TSComputeRHSFunction(ts,t0+c[i],work[0],F);CHKERRQ(ierr);
    ierr = VecAXPY(work[0],dt/6,F);CHKERRQ(ierr);
  }
  ierr = TSComputeRHSFunction(ts,t0+dt,work[0],F);CHKERRQ(ierr);
  ierr = VecAXPBYPCZ(work[1],3./5,dt/10,1,work[0],F);CHKERRQ(ierr);
  ierr = VecCopy(work[1],sol);CHKERRQ(ierr);
  ierr = SSPRestoreWorkVectors(ts,3,&work);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


#undef __FUNCT__  
#define __FUNCT__ "TSSetUp_SSP"
static PetscErrorCode TSSetUp_SSP(TS ts)
{
  /* TS_SSP       *ssp = (TS_SSP*)ts->data; */
  /* PetscErrorCode ierr; */

  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "TSStep_SSP"
static PetscErrorCode TSStep_SSP(TS ts,PetscInt *steps,PetscReal *ptime)
{
  TS_SSP        *ssp = (TS_SSP*)ts->data;
  Vec            sol = ts->vec_sol;
  PetscErrorCode ierr;
  PetscInt       i,max_steps = ts->max_steps;

  PetscFunctionBegin;
  *steps = -ts->steps;
  ierr = TSMonitor(ts,ts->steps,ts->ptime,sol);CHKERRQ(ierr);

  for (i=0; i<max_steps; i++) {
    PetscReal dt = ts->time_step;

    ierr = TSPreStep(ts);CHKERRQ(ierr);
    ts->ptime += dt;
    ierr = (*ssp->onestep)(ts,ts->ptime-dt,dt,sol);CHKERRQ(ierr);
    ts->steps++;
    ierr = TSPostStep(ts);CHKERRQ(ierr);
    ierr = TSMonitor(ts,ts->steps,ts->ptime,sol);CHKERRQ(ierr);
    if (ts->ptime > ts->max_time) break;
  }

  *steps += ts->steps;
  *ptime  = ts->ptime;
  PetscFunctionReturn(0);
}
/*------------------------------------------------------------*/
#undef __FUNCT__  
#define __FUNCT__ "TSDestroy_SSP"
static PetscErrorCode TSDestroy_SSP(TS ts)
{
  TS_SSP       *ssp = (TS_SSP*)ts->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (ssp->work) {ierr = VecDestroyVecs(ssp->work,ssp->nwork);CHKERRQ(ierr);}
  ierr = PetscFree(ssp);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
/*------------------------------------------------------------*/

#undef __FUNCT__  
#define __FUNCT__ "TSSSPSetType"
static PetscErrorCode TSSSPSetType(TS ts,const TSSSPType type)
{
  PetscErrorCode ierr,(*r)(TS,PetscReal,PetscReal,Vec);
  TS_SSP *ssp = (TS_SSP*)ts->data;

  PetscFunctionBegin;
  ierr = PetscFListFind(TSSSPList,((PetscObject)ts)->comm,type,(void(**)(void))&r);CHKERRQ(ierr);
  if (!r) SETERRQ1(PETSC_ERR_ARG_UNKNOWN_TYPE,"Unknown TS_SSP type %s given",type);
  ssp->onestep = r;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "TSSetFromOptions_SSP"
static PetscErrorCode TSSetFromOptions_SSP(TS ts)
{
  char tname[256] = TSSSPRKS2;
  TS_SSP *ssp = (TS_SSP*)ts->data;
  PetscErrorCode ierr;
  PetscTruth flg;

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

.seealso:  TSCreate(), TS, TSSetType()

M*/
EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "TSCreate_SSP"
PetscErrorCode PETSCTS_DLLEXPORT TSCreate_SSP(TS ts)
{
  TS_SSP       *ssp;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!TSSSPList) {
    ierr = PetscFListAdd(&TSSSPList,TSSSPRKS2,  "SSPStep_RK_2",   (void(*)(void))SSPStep_RK_2);CHKERRQ(ierr);
    ierr = PetscFListAdd(&TSSSPList,TSSSPRKS3,  "SSPStep_RK_3",   (void(*)(void))SSPStep_RK_3);CHKERRQ(ierr);
    ierr = PetscFListAdd(&TSSSPList,TSSSPRK104, "SSPStep_RK_10_4",(void(*)(void))SSPStep_RK_10_4);CHKERRQ(ierr);
  }

  ts->ops->setup           = TSSetUp_SSP;
  ts->ops->step            = TSStep_SSP;
  ts->ops->destroy         = TSDestroy_SSP;
  ts->ops->setfromoptions  = TSSetFromOptions_SSP;
  ts->ops->view            = TSView_SSP;

  ierr = PetscNewLog(ts,TS_SSP,&ssp);CHKERRQ(ierr);
  ts->data = (void*)ssp;

  ierr = TSSSPSetType(ts,TSSSPRKS2);CHKERRQ(ierr);
  ssp->nstages = 5;
  PetscFunctionReturn(0);
}
EXTERN_C_END




