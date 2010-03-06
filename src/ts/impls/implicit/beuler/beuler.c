#define PETSCTS_DLL

/*
       Code for Timestepping with implicit backwards Euler.
*/
#include "private/tsimpl.h"                /*I   "petscts.h"   I*/

typedef struct {
  Vec  update;      /* work vector where new solution is formed */
  Vec  func;        /* work vector where F(t[i],u[i]) is stored */
  Vec  rhs;         /* work vector for RHS; vec_sol/dt */
} TS_BEuler;

/*------------------------------------------------------------------------------*/
/* 
   Set ts->A = ts->Arhs = 1/dt*Alhs - Arhs, used in KSPSolve() 
*/
#undef __FUNCT__  
#define __FUNCT__ "TSSetKSPOperators_BEuler"
PetscErrorCode TSSetKSPOperators_BEuler(TS ts)
{
  PetscErrorCode ierr;
  PetscScalar    mdt = 1.0/ts->time_step;

  PetscFunctionBegin;
  if (!ts->A){
    ierr  = PetscObjectReference((PetscObject)ts->Arhs);CHKERRQ(ierr);
    ts->A = ts->Arhs;
  }

  ierr = MatScale(ts->A,-1.0);CHKERRQ(ierr);
  if (ts->Alhs){
    ierr = MatAXPY(ts->A,mdt,ts->Alhs,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);
  } else {
    ierr = MatShift(ts->A,mdt);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*
    Version for linear PDE where RHS does not depend on time. Has built a
  single matrix that is to be used for all timesteps.
*/
#undef __FUNCT__  
#define __FUNCT__ "TSStep_BEuler_Linear_Constant_Matrix"
static PetscErrorCode TSStep_BEuler_Linear_Constant_Matrix(TS ts,PetscInt *steps,PetscReal *ptime)
{
  TS_BEuler      *beuler = (TS_BEuler*)ts->data;
  Vec            sol = ts->vec_sol,update = beuler->update;
  Vec            rhs = beuler->rhs;
  PetscErrorCode ierr;
  PetscInt       i,max_steps = ts->max_steps,its;
  PetscScalar    mdt = 1.0/ts->time_step;
  KSP            ksp;

  PetscFunctionBegin;
  ierr = TSGetKSP(ts,&ksp);CHKERRQ(ierr);
  *steps = -ts->steps;
  ierr = TSMonitor(ts,ts->steps,ts->ptime,sol);CHKERRQ(ierr);

  /* set initial guess to be previous solution */
  ierr = VecCopy(sol,update);CHKERRQ(ierr);

  for (i=0; i<max_steps; i++) {
    if (ts->ptime + ts->time_step > ts->max_time) break;

    ierr = TSPreStep(ts);CHKERRQ(ierr);
    /* set rhs = 1/dt*Alhs*sol */
    if (ts->Alhs){
      ierr = MatMult(ts->Alhs,sol,rhs);CHKERRQ(ierr);
    } else {
      ierr = VecCopy(sol,rhs);CHKERRQ(ierr);
    }
    ierr = VecScale(rhs,mdt);CHKERRQ(ierr);

    ts->ptime += ts->time_step;

    /* solve (1/dt*Alhs - A)*update = rhs */
    ierr = KSPSolve(ts->ksp,rhs,update);CHKERRQ(ierr);
    ierr = KSPGetIterationNumber(ksp,&its);CHKERRQ(ierr);
    ts->linear_its += its;
    ierr = VecCopy(update,sol);CHKERRQ(ierr);
    ts->steps++;
    ierr = TSPostStep(ts);CHKERRQ(ierr);
    ierr = TSMonitor(ts,ts->steps,ts->ptime,sol);CHKERRQ(ierr);
  }

  *steps += ts->steps;
  *ptime  = ts->ptime;
  PetscFunctionReturn(0);
}

/*
      Version where matrix depends on time 
*/
#undef __FUNCT__  
#define __FUNCT__ "TSStep_BEuler_Linear_Variable_Matrix"
static PetscErrorCode TSStep_BEuler_Linear_Variable_Matrix(TS ts,PetscInt *steps,PetscReal *ptime)
{
  TS_BEuler      *beuler = (TS_BEuler*)ts->data;
  Vec            sol = ts->vec_sol,update = beuler->update,rhs = beuler->rhs;
  PetscErrorCode ierr;
  PetscInt       i,max_steps = ts->max_steps,its;
  PetscScalar    mdt = 1.0/ts->time_step;
  PetscReal      t_mid;
  MatStructure   str;
  KSP            ksp;

  PetscFunctionBegin;
  ierr = TSGetKSP(ts,&ksp);CHKERRQ(ierr);
  *steps = -ts->steps;
  ierr = TSMonitor(ts,ts->steps,ts->ptime,sol);CHKERRQ(ierr);

  /* set initial guess to be previous solution */
  ierr = VecCopy(sol,update);CHKERRQ(ierr);

  for (i=0; i<max_steps; i++) {
    if (ts->ptime +ts->time_step > ts->max_time) break;

    ierr = TSPreStep(ts);CHKERRQ(ierr);
    /* set rhs = 1/dt*Alhs(t_mid)*sol */
    if (ts->Alhs){
      /* evaluate lhs matrix function at t_mid */
      t_mid = ts->ptime+ts->time_step/2.0;
      ierr = (*ts->ops->lhsmatrix)(ts,t_mid,&ts->Alhs,PETSC_NULL,&str,ts->jacPlhs);CHKERRQ(ierr);
      ierr = MatMult(ts->Alhs,sol,rhs);CHKERRQ(ierr);
    } else {
      ierr = VecCopy(sol,rhs);CHKERRQ(ierr);
    }
    ierr = VecScale(rhs,mdt);CHKERRQ(ierr);

    ts->ptime += ts->time_step;

    /* evaluate rhs matrix function at current ptime */
    ierr = (*ts->ops->rhsmatrix)(ts,ts->ptime,&ts->Arhs,&ts->B,&str,ts->jacP);CHKERRQ(ierr);

    /* set ts->A = ts->Arhs = 1/dt*Alhs - Arhs, used in KSPSolve() */
    ierr = TSSetKSPOperators_BEuler(ts);CHKERRQ(ierr);
    ierr = KSPSetOperators(ts->ksp,ts->A,ts->A,SAME_NONZERO_PATTERN);CHKERRQ(ierr);

    /* solve (1/dt*Alhs(t_mid) - A(t_n+1))*update = rhs */
    ierr = KSPSolve(ts->ksp,rhs,update);CHKERRQ(ierr);
    ierr = KSPGetIterationNumber(ksp,&its);CHKERRQ(ierr);
    ts->linear_its += its;
    ierr = VecCopy(update,sol);CHKERRQ(ierr);
    ts->steps++;
    ierr = TSPostStep(ts);CHKERRQ(ierr);
    ierr = TSMonitor(ts,ts->steps,ts->ptime,sol);CHKERRQ(ierr);
  }

  *steps += ts->steps;
  *ptime  = ts->ptime;
  PetscFunctionReturn(0);
}
/*
    Version for nonlinear PDE.
*/
#undef __FUNCT__  
#define __FUNCT__ "TSStep_BEuler_Nonlinear"
static PetscErrorCode TSStep_BEuler_Nonlinear(TS ts,PetscInt *steps,PetscReal *ptime)
{
  Vec            sol = ts->vec_sol;
  PetscErrorCode ierr;
  PetscInt       i,max_steps = ts->max_steps,its,lits;
  TS_BEuler      *beuler = (TS_BEuler*)ts->data;
  
  PetscFunctionBegin;
  *steps = -ts->steps;
  ierr = TSMonitor(ts,ts->steps,ts->ptime,sol);CHKERRQ(ierr);

  for (i=0; i<max_steps; i++) {
    if (ts->ptime + ts->time_step > ts->max_time) break;
    ierr = TSPreStep(ts);CHKERRQ(ierr);
    ts->ptime += ts->time_step;
    ierr = VecCopy(sol,beuler->update);CHKERRQ(ierr);
    ierr = SNESSolve(ts->snes,PETSC_NULL,beuler->update);CHKERRQ(ierr);
    ierr = SNESGetLinearSolveIterations(ts->snes,&lits);CHKERRQ(ierr);
    ierr = SNESGetIterationNumber(ts->snes,&its);CHKERRQ(ierr);
    ts->nonlinear_its += its; ts->linear_its += lits;
    ierr = VecCopy(beuler->update,sol);CHKERRQ(ierr);
    ts->steps++;
    ierr = TSPostStep(ts);CHKERRQ(ierr);
    ierr = TSMonitor(ts,ts->steps,ts->ptime,sol);CHKERRQ(ierr);
  }

  *steps += ts->steps;
  *ptime  = ts->ptime;
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/
#undef __FUNCT__  
#define __FUNCT__ "TSDestroy_BEuler"
static PetscErrorCode TSDestroy_BEuler(TS ts)
{
  TS_BEuler      *beuler = (TS_BEuler*)ts->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (beuler->update) {ierr = VecDestroy(beuler->update);CHKERRQ(ierr);}
  if (beuler->func) {ierr = VecDestroy(beuler->func);CHKERRQ(ierr);}
  if (beuler->rhs) {ierr = VecDestroy(beuler->rhs);CHKERRQ(ierr);}
  ierr = PetscFree(beuler);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* 
    This defines the nonlinear equation that is to be solved with SNES
      1/dt* (U^{n+1} - U^{n}) - F(U^{n+1}) 
*/
#undef __FUNCT__  
#define __FUNCT__ "TSBEulerFunction"
PetscErrorCode TSBEulerFunction(SNES snes,Vec x,Vec y,void *ctx)
{
  TS             ts = (TS) ctx;
  PetscScalar    mdt = 1.0/ts->time_step,*unp1,*un,*Funp1;
  PetscErrorCode ierr;
  PetscInt       i,n;

  PetscFunctionBegin;
  /* apply user-provided function */
  ierr = TSComputeRHSFunction(ts,ts->ptime,x,y);CHKERRQ(ierr);
  ierr = VecGetArray(ts->vec_sol,&un);CHKERRQ(ierr);
  ierr = VecGetArray(x,&unp1);CHKERRQ(ierr);
  ierr = VecGetArray(y,&Funp1);CHKERRQ(ierr);
  ierr = VecGetLocalSize(x,&n);CHKERRQ(ierr);
  for (i=0; i<n; i++) {
    Funp1[i] = mdt*(unp1[i] - un[i]) - Funp1[i];
  }
  ierr = VecRestoreArray(ts->vec_sol,&un);CHKERRQ(ierr);
  ierr = VecRestoreArray(x,&unp1);CHKERRQ(ierr);
  ierr = VecRestoreArray(y,&Funp1);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
   This constructs the Jacobian needed for SNES 
     J = I/dt - J_{F}   where J_{F} is the given Jacobian of F at t_{n+1}.
     x  - input vector
     AA - Jacobian matrix 
     BB - preconditioner matrix, usually the same as AA
*/
#undef __FUNCT__  
#define __FUNCT__ "TSBEulerJacobian"
PetscErrorCode TSBEulerJacobian(SNES snes,Vec x,Mat *AA,Mat *BB,MatStructure *str,void *ctx)
{
  TS             ts = (TS) ctx;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /* construct user's Jacobian */
  ierr = TSComputeRHSJacobian(ts,ts->ptime,x,AA,BB,str);CHKERRQ(ierr);

  /* shift and scale Jacobian */
  /* this test is a undesirable hack, we assume that if it is MATMFFD then it is
     obtained from -snes_mf_operator and there is computed directly from the 
     FormFunction() SNES is given and therefor does not need to be shifted/scaled
     BUT maybe it could be MATMFFD and does require shift in some other case??? */
  ierr = TSSetKSPOperators_BEuler(ts);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* ------------------------------------------------------------*/
#undef __FUNCT__  
#define __FUNCT__ "TSSetUp_BEuler_Linear_Constant_Matrix"
static PetscErrorCode TSSetUp_BEuler_Linear_Constant_Matrix(TS ts)
{
  TS_BEuler      *beuler = (TS_BEuler*)ts->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecDuplicate(ts->vec_sol,&beuler->update);CHKERRQ(ierr);  
  ierr = VecDuplicate(ts->vec_sol,&beuler->rhs);CHKERRQ(ierr);  
    
  /* build linear system to be solved - should move into TSStep() if dt changes! */
  /* Set ts->A = ts->Arhs = 1/dt*Alhs - Arhs, used in KSPSolve() */
  ierr = TSSetKSPOperators_BEuler(ts);CHKERRQ(ierr);
  ierr = KSPSetOperators(ts->ksp,ts->A,ts->A,SAME_NONZERO_PATTERN);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "TSSetUp_BEuler_Linear_Variable_Matrix"
static PetscErrorCode TSSetUp_BEuler_Linear_Variable_Matrix(TS ts)
{
  TS_BEuler      *beuler = (TS_BEuler*)ts->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecDuplicate(ts->vec_sol,&beuler->update);CHKERRQ(ierr);  
  ierr = VecDuplicate(ts->vec_sol,&beuler->rhs);CHKERRQ(ierr);  
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "TSSetUp_BEuler_Nonlinear"
static PetscErrorCode TSSetUp_BEuler_Nonlinear(TS ts)
{
  TS_BEuler      *beuler = (TS_BEuler*)ts->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecDuplicate(ts->vec_sol,&beuler->update);CHKERRQ(ierr);  
  ierr = VecDuplicate(ts->vec_sol,&beuler->func);CHKERRQ(ierr);  
  ierr = SNESSetFunction(ts->snes,beuler->func,TSBEulerFunction,ts);CHKERRQ(ierr);
  ierr = SNESSetJacobian(ts->snes,ts->Arhs,ts->B,TSBEulerJacobian,ts);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
/*------------------------------------------------------------*/

#undef __FUNCT__  
#define __FUNCT__ "TSSetFromOptions_BEuler_Linear"
static PetscErrorCode TSSetFromOptions_BEuler_Linear(TS ts)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "TSSetFromOptions_BEuler_Nonlinear"
static PetscErrorCode TSSetFromOptions_BEuler_Nonlinear(TS ts)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "TSView_BEuler"
static PetscErrorCode TSView_BEuler(TS ts,PetscViewer viewer)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

/* ------------------------------------------------------------ */
/*MC
      TSBEULER - ODE solver using the implicit backward Euler method

  Level: beginner

.seealso:  TSCreate(), TS, TSSetType(), TSEULER

M*/
EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "TSCreate_BEuler"
PetscErrorCode PETSCTS_DLLEXPORT TSCreate_BEuler(TS ts)
{
  TS_BEuler      *beuler;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ts->ops->destroy = TSDestroy_BEuler;
  ts->ops->view    = TSView_BEuler;

  if (ts->problem_type == TS_LINEAR) {
    if (!ts->Arhs) {
      SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Must set rhs matrix for linear problem");
    }
    if (!ts->ops->rhsmatrix) {
      ts->ops->setup  = TSSetUp_BEuler_Linear_Constant_Matrix;
      ts->ops->step   = TSStep_BEuler_Linear_Constant_Matrix;
    } else {
      ts->ops->setup  = TSSetUp_BEuler_Linear_Variable_Matrix;  
      ts->ops->step   = TSStep_BEuler_Linear_Variable_Matrix;
    }
    ts->ops->setfromoptions  = TSSetFromOptions_BEuler_Linear;
    ierr = KSPCreate(((PetscObject)ts)->comm,&ts->ksp);CHKERRQ(ierr);
    ierr = PetscObjectIncrementTabLevel((PetscObject)ts->ksp,(PetscObject)ts,1);CHKERRQ(ierr);
    ierr = KSPSetInitialGuessNonzero(ts->ksp,PETSC_TRUE);CHKERRQ(ierr);
  } else if (ts->problem_type == TS_NONLINEAR) {
    ts->ops->setup           = TSSetUp_BEuler_Nonlinear;  
    ts->ops->step            = TSStep_BEuler_Nonlinear;
    ts->ops->setfromoptions  = TSSetFromOptions_BEuler_Nonlinear;
    ierr = SNESCreate(((PetscObject)ts)->comm,&ts->snes);CHKERRQ(ierr);
    ierr = PetscObjectIncrementTabLevel((PetscObject)ts->snes,(PetscObject)ts,1);CHKERRQ(ierr);
  } else SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,"No such problem");

  ierr = PetscNewLog(ts,TS_BEuler,&beuler);CHKERRQ(ierr);
  ts->data = (void*)beuler;

  PetscFunctionReturn(0);
}
EXTERN_C_END





