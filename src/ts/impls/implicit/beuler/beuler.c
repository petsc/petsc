/*$Id: beuler.c,v 1.61 2001/09/11 16:34:19 bsmith Exp $*/
/*
       Code for Timestepping with implicit backwards Euler.
*/
#include "src/ts/tsimpl.h"                /*I   "petscts.h"   I*/

typedef struct {
  Vec  update;      /* work vector where new solution is formed */
  Vec  func;        /* work vector where F(t[i],u[i]) is stored */
  Vec  rhs;         /* work vector for RHS; vec_sol/dt */
} TS_BEuler;

/*------------------------------------------------------------------------------*/

/*
    Version for linear PDE where RHS does not depend on time. Has built a
  single matrix that is to be used for all timesteps.
*/
#undef __FUNCT__  
#define __FUNCT__ "TSStep_BEuler_Linear_Constant_Matrix"
static int TSStep_BEuler_Linear_Constant_Matrix(TS ts,int *steps,PetscReal *ptime)
{
  TS_BEuler   *beuler = (TS_BEuler*)ts->data;
  Vec         sol = ts->vec_sol,update = beuler->update;
  Vec         rhs = beuler->rhs;
  int         ierr,i,max_steps = ts->max_steps,its;
  PetscScalar mdt = 1.0/ts->time_step;
  KSP         ksp;

  PetscFunctionBegin;
  ierr = TSGetKSP(ts,&ksp);CHKERRQ(ierr);
  *steps = -ts->steps;
  ierr = TSMonitor(ts,ts->steps,ts->ptime,sol);CHKERRQ(ierr);

  /* set initial guess to be previous solution */
  ierr = VecCopy(sol,update);CHKERRQ(ierr);

  for (i=0; i<max_steps; i++) {
    ierr = VecCopy(sol,rhs);CHKERRQ(ierr);
    ierr = VecScale(&mdt,rhs);CHKERRQ(ierr);
    /* apply user-provided boundary conditions (only needed if they are time dependent) */
    ierr = TSComputeRHSBoundaryConditions(ts,ts->ptime,rhs);CHKERRQ(ierr);

    ts->ptime += ts->time_step;
    if (ts->ptime > ts->max_time) break;
    ierr = KSPSolve(ts->ksp,rhs,update);CHKERRQ(ierr);
    ierr = KSPGetIterationNumber(ksp,&its);CHKERRQ(ierr);
    ts->linear_its += its;
    ierr = VecCopy(update,sol);CHKERRQ(ierr);
    ts->steps++;
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
static int TSStep_BEuler_Linear_Variable_Matrix(TS ts,int *steps,PetscReal *ptime)
{
  TS_BEuler    *beuler = (TS_BEuler*)ts->data;
  Vec          sol = ts->vec_sol,update = beuler->update,rhs = beuler->rhs;
  int          ierr,i,max_steps = ts->max_steps,its;
  PetscScalar  mdt = 1.0/ts->time_step,mone = -1.0;
  MatStructure str;
  KSP          ksp;

  PetscFunctionBegin;
  ierr = TSGetKSP(ts,&ksp);CHKERRQ(ierr);
  *steps = -ts->steps;
  ierr = TSMonitor(ts,ts->steps,ts->ptime,sol);CHKERRQ(ierr);

  /* set initial guess to be previous solution */
  ierr = VecCopy(sol,update);CHKERRQ(ierr);

  for (i=0; i<max_steps; i++) {
    ierr = VecCopy(sol,rhs);CHKERRQ(ierr);
    ierr = VecScale(&mdt,rhs);CHKERRQ(ierr);
    /* apply user-provided boundary conditions (only needed if they are time dependent) */
    ierr = TSComputeRHSBoundaryConditions(ts,ts->ptime,rhs);CHKERRQ(ierr);

    ts->ptime += ts->time_step;
    if (ts->ptime > ts->max_time) break;
    /*
        evaluate matrix function 
    */
    ierr = (*ts->ops->rhsmatrix)(ts,ts->ptime,&ts->A,&ts->B,&str,ts->jacP);CHKERRQ(ierr);
    ierr = MatScale(&mone,ts->A);CHKERRQ(ierr);
    ierr = MatShift(&mdt,ts->A);CHKERRQ(ierr);
    if (ts->B != ts->A && str != SAME_PRECONDITIONER) {
      ierr = MatScale(&mone,ts->B);CHKERRQ(ierr);
      ierr = MatShift(&mdt,ts->B);CHKERRQ(ierr);
    }
    ierr = KSPSetOperators(ts->ksp,ts->A,ts->B,str);CHKERRQ(ierr);
    ierr = KSPSolve(ts->ksp,rhs,update);CHKERRQ(ierr);
    ierr = KSPGetIterationNumber(ksp,&its);CHKERRQ(ierr);
    ts->linear_its += its;
    ierr = VecCopy(update,sol);CHKERRQ(ierr);
    ts->steps++;
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
static int TSStep_BEuler_Nonlinear(TS ts,int *steps,PetscReal *ptime)
{
  Vec       sol = ts->vec_sol;
  int       ierr,i,max_steps = ts->max_steps,its,lits;
  TS_BEuler *beuler = (TS_BEuler*)ts->data;
  
  PetscFunctionBegin;
  *steps = -ts->steps;
  ierr = TSMonitor(ts,ts->steps,ts->ptime,sol);CHKERRQ(ierr);

  for (i=0; i<max_steps; i++) {
    ts->ptime += ts->time_step;
    if (ts->ptime > ts->max_time) break;
    ierr = VecCopy(sol,beuler->update);CHKERRQ(ierr);
    ierr = SNESSolve(ts->snes,beuler->update);CHKERRQ(ierr);
    ierr = SNESGetNumberLinearIterations(ts->snes,&lits);CHKERRQ(ierr);
    ierr = SNESGetIterationNumber(ts->snes,&its);CHKERRQ(ierr);
    ts->nonlinear_its += its; ts->linear_its += lits;
    ierr = VecCopy(beuler->update,sol);CHKERRQ(ierr);
    ts->steps++;
    ierr = TSMonitor(ts,ts->steps,ts->ptime,sol);CHKERRQ(ierr);
  }

  *steps += ts->steps;
  *ptime  = ts->ptime;
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/
#undef __FUNCT__  
#define __FUNCT__ "TSDestroy_BEuler"
static int TSDestroy_BEuler(TS ts)
{
  TS_BEuler *beuler = (TS_BEuler*)ts->data;
  int       ierr;

  PetscFunctionBegin;
  if (beuler->update) {ierr = VecDestroy(beuler->update);CHKERRQ(ierr);}
  if (beuler->func) {ierr = VecDestroy(beuler->func);CHKERRQ(ierr);}
  if (beuler->rhs) {ierr = VecDestroy(beuler->rhs);CHKERRQ(ierr);}
  ierr = PetscFree(beuler);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}



/* 
    This defines the nonlinear equation that is to be solved with SNES

              U^{n+1} - dt*F(U^{n+1}) - U^{n}
*/
#undef __FUNCT__  
#define __FUNCT__ "TSBEulerFunction"
int TSBEulerFunction(SNES snes,Vec x,Vec y,void *ctx)
{
  TS     ts = (TS) ctx;
  PetscScalar mdt = 1.0/ts->time_step,*unp1,*un,*Funp1;
  int    ierr,i,n;

  PetscFunctionBegin;
  /* apply user-provided function */
  ierr = TSComputeRHSFunction(ts,ts->ptime,x,y);CHKERRQ(ierr);
  /* (u^{n+1} - U^{n})/dt - F(u^{n+1}) */
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

             J = I/dt - J_{F}   where J_{F} is the given Jacobian of F.
*/
#undef __FUNCT__  
#define __FUNCT__ "TSBEulerJacobian"
int TSBEulerJacobian(SNES snes,Vec x,Mat *AA,Mat *BB,MatStructure *str,void *ctx)
{
  TS           ts = (TS) ctx;
  int          ierr;
  PetscScalar  mone = -1.0,mdt = 1.0/ts->time_step;

  PetscFunctionBegin;
  /* construct user's Jacobian */
  ierr = TSComputeRHSJacobian(ts,ts->ptime,x,AA,BB,str);CHKERRQ(ierr);

  /* shift and scale Jacobian */
  ierr = MatScale(&mone,*AA);CHKERRQ(ierr);
  ierr = MatShift(&mdt,*AA);CHKERRQ(ierr);
  if (*BB != *AA && *str != SAME_PRECONDITIONER) {
    ierr = MatScale(&mone,*BB);CHKERRQ(ierr);
    ierr = MatShift(&mdt,*BB);CHKERRQ(ierr);
  }

  PetscFunctionReturn(0);
}

/* ------------------------------------------------------------*/
#undef __FUNCT__  
#define __FUNCT__ "TSSetUp_BEuler_Linear_Constant_Matrix"
static int TSSetUp_BEuler_Linear_Constant_Matrix(TS ts)
{
  TS_BEuler *beuler = (TS_BEuler*)ts->data;
  int       ierr;
  PetscScalar    mdt = 1.0/ts->time_step,mone = -1.0;

  PetscFunctionBegin;
  ierr = KSPSetFromOptions(ts->ksp);CHKERRQ(ierr);
  ierr = VecDuplicate(ts->vec_sol,&beuler->update);CHKERRQ(ierr);  
  ierr = VecDuplicate(ts->vec_sol,&beuler->rhs);CHKERRQ(ierr);  
    
  /* build linear system to be solved */
  ierr = MatScale(&mone,ts->A);CHKERRQ(ierr);
  ierr = MatShift(&mdt,ts->A);CHKERRQ(ierr);
  if (ts->A != ts->B) {
    ierr = MatScale(&mone,ts->B);CHKERRQ(ierr);
    ierr = MatShift(&mdt,ts->B);CHKERRQ(ierr);
  }
  ierr = KSPSetOperators(ts->ksp,ts->A,ts->B,SAME_NONZERO_PATTERN);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "TSSetUp_BEuler_Linear_Variable_Matrix"
static int TSSetUp_BEuler_Linear_Variable_Matrix(TS ts)
{
  TS_BEuler *beuler = (TS_BEuler*)ts->data;
  int       ierr;

  PetscFunctionBegin;
  ierr = KSPSetFromOptions(ts->ksp);CHKERRQ(ierr);
  ierr = VecDuplicate(ts->vec_sol,&beuler->update);CHKERRQ(ierr);  
  ierr = VecDuplicate(ts->vec_sol,&beuler->rhs);CHKERRQ(ierr);  
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "TSSetUp_BEuler_Nonlinear"
static int TSSetUp_BEuler_Nonlinear(TS ts)
{
  TS_BEuler *beuler = (TS_BEuler*)ts->data;
  int       ierr;

  PetscFunctionBegin;
  ierr = VecDuplicate(ts->vec_sol,&beuler->update);CHKERRQ(ierr);  
  ierr = VecDuplicate(ts->vec_sol,&beuler->func);CHKERRQ(ierr);  
  ierr = SNESSetFunction(ts->snes,beuler->func,TSBEulerFunction,ts);CHKERRQ(ierr);
  ierr = SNESSetJacobian(ts->snes,ts->A,ts->B,TSBEulerJacobian,ts);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
/*------------------------------------------------------------*/

#undef __FUNCT__  
#define __FUNCT__ "TSSetFromOptions_BEuler_Linear"
static int TSSetFromOptions_BEuler_Linear(TS ts)
{
  PetscFunctionBegin;
  
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "TSSetFromOptions_BEuler_Nonlinear"
static int TSSetFromOptions_BEuler_Nonlinear(TS ts)
{
  PetscFunctionBegin;
  
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "TSView_BEuler"
static int TSView_BEuler(TS ts,PetscViewer viewer)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

/* ------------------------------------------------------------ */
/*MC
      TS_BEULER - ODE solver using the implicit backward Euler method

.seealso:  TSCreate(), TS, TSSetType(), TS_EULER

M*/
EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "TSCreate_BEuler"
int TSCreate_BEuler(TS ts)
{
  TS_BEuler  *beuler;
  int        ierr;

  PetscFunctionBegin;
  ts->ops->destroy = TSDestroy_BEuler;
  ts->ops->view    = TSView_BEuler;

  if (ts->problem_type == TS_LINEAR) {
    if (!ts->A) {
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
    ierr = KSPCreate(ts->comm,&ts->ksp);CHKERRQ(ierr);
    ierr = KSPSetInitialGuessNonzero(ts->ksp,PETSC_TRUE);CHKERRQ(ierr);
  } else if (ts->problem_type == TS_NONLINEAR) {
    ts->ops->setup           = TSSetUp_BEuler_Nonlinear;  
    ts->ops->step            = TSStep_BEuler_Nonlinear;
    ts->ops->setfromoptions  = TSSetFromOptions_BEuler_Nonlinear;
    ierr = SNESCreate(ts->comm,&ts->snes);CHKERRQ(ierr);
  } else SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,"No such problem");

  ierr = PetscNew(TS_BEuler,&beuler);CHKERRQ(ierr);
  PetscLogObjectMemory(ts,sizeof(TS_BEuler));
  ierr     = PetscMemzero(beuler,sizeof(TS_BEuler));CHKERRQ(ierr);
  ts->data = (void*)beuler;

  PetscFunctionReturn(0);
}
EXTERN_C_END





