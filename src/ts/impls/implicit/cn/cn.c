/*$Id: cn.c,v 1.35 2001/09/11 16:34:19 bsmith Exp $*/
/*
       Code for Timestepping with implicit Crank-Nicholson method.
    THIS IS NOT YET COMPLETE -- DO NOT USE!!
*/
#include "src/ts/tsimpl.h"                /*I   "petscts.h"   I*/

typedef struct {
  Vec  update;      /* work vector where new solution is formed */
  Vec  func;        /* work vector where F(t[i],u[i]) is stored */
  Vec  rhs;         /* work vector for RHS; vec_sol/dt */
} TS_CN;

/*------------------------------------------------------------------------------*/
#undef __FUNCT__  
#define __FUNCT__ "TSComputeRHSFunctionEuler"
/*
   TSComputeRHSFunctionEuler - Evaluates the right-hand-side function. 

   Note: If the user did not provide a function but merely a matrix,
   this routine applies the matrix.
*/
int TSComputeRHSFunctionEuler(TS ts,PetscReal t,Vec x,Vec y)
{
  int         ierr;
  PetscScalar neg_two = -2.0,neg_mdt = -1.0/ts->time_step;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_COOKIE,1);
  PetscValidHeaderSpecific(x,VEC_COOKIE,2);  
  PetscValidHeaderSpecific(y,VEC_COOKIE,3);

  if (ts->ops->rhsfunction) {
    PetscStackPush("TS user right-hand-side function");
    ierr = (*ts->ops->rhsfunction)(ts,t,x,y,ts->funP);CHKERRQ(ierr);
    PetscStackPop;
    PetscFunctionReturn(0);
  }

  if (ts->ops->rhsmatrix) { /* assemble matrix for this timestep */
    MatStructure flg;
    PetscStackPush("TS user right-hand-side matrix function");
    ierr = (*ts->ops->rhsmatrix)(ts,t,&ts->A,&ts->B,&flg,ts->jacP);CHKERRQ(ierr);
    PetscStackPop;
  }
  ierr = MatMult(ts->A,x,y);CHKERRQ(ierr);
  /* shift: y = y -2*x */
  ierr = VecAXPY(&neg_two,x,y);CHKERRQ(ierr);
  /* scale: y = y -2*x */
  ierr = VecScale(&neg_mdt,y);CHKERRQ(ierr);

  /* apply user-provided boundary conditions (only needed if these are time dependent) */
  ierr = TSComputeRHSBoundaryConditions(ts,t,y);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

/*
    Version for linear PDE where RHS does not depend on time. Has built a
  single matrix that is to be used for all timesteps.
*/
#undef __FUNCT__  
#define __FUNCT__ "TSStep_CN_Linear_Constant_Matrix"
static int TSStep_CN_Linear_Constant_Matrix(TS ts,int *steps,PetscReal *ptime)
{
  TS_CN       *cn = (TS_CN*)ts->data;
  Vec         sol = ts->vec_sol,update = cn->update;
  Vec         rhs = cn->rhs;
  int         ierr,i,max_steps = ts->max_steps,its;
  PetscScalar dt = ts->time_step,two = 2.0;
  KSP         ksp;

  PetscFunctionBegin;
  ierr   = TSGetKSP(ts,&ksp);CHKERRQ(ierr);
  *steps = -ts->steps;
  ierr   = TSMonitor(ts,ts->steps,ts->ptime,sol);CHKERRQ(ierr);

  /* set initial guess to be previous solution */
  ierr = VecCopy(sol,update);CHKERRQ(ierr);

  for (i=0; i<max_steps; i++) {
    ts->ptime += ts->time_step;
    if (ts->ptime > ts->max_time) break;

    /* phase 1 - explicit step */
    ierr = TSComputeRHSFunctionEuler(ts,ts->ptime,sol,update);CHKERRQ(ierr);
    ierr = VecAXPBY(&dt,&two,update,sol);CHKERRQ(ierr);

    /* phase 2 - implicit step */
    ierr = VecCopy(sol,rhs);CHKERRQ(ierr);
    /* apply user-provided boundary conditions (only needed if they are time dependent) */
    ierr = TSComputeRHSBoundaryConditions(ts,ts->ptime,rhs);CHKERRQ(ierr);

    ierr = KSPSolve(ts->ksp,rhs,update);CHKERRQ(ierr);
    ierr = KSPGetIterationNumber(ksp,&its);CHKERRQ(ierr);
    ts->linear_its += PetscAbsInt(its);
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
#define __FUNCT__ "TSStep_CN_Linear_Variable_Matrix"
static int TSStep_CN_Linear_Variable_Matrix(TS ts,int *steps,PetscReal *ptime)
{
  TS_CN        *cn = (TS_CN*)ts->data;
  Vec          sol = ts->vec_sol,update = cn->update,rhs = cn->rhs;
  int          ierr,i,max_steps = ts->max_steps,its;
  PetscScalar  dt = ts->time_step,two = 2.0,neg_dt = -1.0*ts->time_step;
  MatStructure str;
  KSP          ksp;

  PetscFunctionBegin;
  ierr   = TSGetKSP(ts,&ksp);CHKERRQ(ierr);
  *steps = -ts->steps;
  ierr   = TSMonitor(ts,ts->steps,ts->ptime,sol);CHKERRQ(ierr);

  /* set initial guess to be previous solution */
  ierr = VecCopy(sol,update);CHKERRQ(ierr);

  for (i=0; i<max_steps; i++) {
    ts->ptime += ts->time_step;
    if (ts->ptime > ts->max_time) break;
    /*
        evaluate matrix function 
    */
    ierr = (*ts->ops->rhsmatrix)(ts,ts->ptime,&ts->A,&ts->B,&str,ts->jacP);CHKERRQ(ierr);
    ierr = MatScale(&neg_dt,ts->A);CHKERRQ(ierr);
    ierr = MatShift(&two,ts->A);CHKERRQ(ierr);
    if (ts->B != ts->A && str != SAME_PRECONDITIONER) {
      ierr = MatScale(&neg_dt,ts->B);CHKERRQ(ierr);
      ierr = MatShift(&two,ts->B);CHKERRQ(ierr);
    }

    /* phase 1 - explicit step */
    ierr = TSComputeRHSFunctionEuler(ts,ts->ptime,sol,update);CHKERRQ(ierr);
    ierr = VecAXPBY(&dt,&two,update,sol);CHKERRQ(ierr);

    /* phase 2 - implicit step */
    ierr = VecCopy(sol,rhs);CHKERRQ(ierr);

    /* apply user-provided boundary conditions (only needed if they are time dependent) */
    ierr = TSComputeRHSBoundaryConditions(ts,ts->ptime,rhs);CHKERRQ(ierr);

    ierr = KSPSetOperators(ts->ksp,ts->A,ts->B,str);CHKERRQ(ierr);
    ierr = KSPSolve(ts->ksp,rhs,update);CHKERRQ(ierr);
    ierr = KSPGetIterationNumber(ksp,&its);CHKERRQ(ierr);
    ts->linear_its += PetscAbsInt(its);
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
#define __FUNCT__ "TSStep_CN_Nonlinear"
static int TSStep_CN_Nonlinear(TS ts,int *steps,PetscReal *ptime)
{
  Vec   sol = ts->vec_sol;
  int   ierr,i,max_steps = ts->max_steps,its,lits;
  TS_CN *cn = (TS_CN*)ts->data;
  
  PetscFunctionBegin;
  *steps = -ts->steps;
  ierr = TSMonitor(ts,ts->steps,ts->ptime,sol);CHKERRQ(ierr);

  for (i=0; i<max_steps; i++) {
    ts->ptime += ts->time_step;
    if (ts->ptime > ts->max_time) break;
    ierr = VecCopy(sol,cn->update);CHKERRQ(ierr);
    ierr = SNESSolve(ts->snes,cn->update);CHKERRQ(ierr);
    ierr = SNESGetIterationNumber(ts->snes,&its);CHKERRQ(ierr);
    ierr = SNESGetNumberLinearIterations(ts->snes,&lits);CHKERRQ(ierr);
    ts->nonlinear_its += its; ts->linear_its += lits;
    ierr = VecCopy(cn->update,sol);CHKERRQ(ierr);
    ts->steps++;
    ierr = TSMonitor(ts,ts->steps,ts->ptime,sol);CHKERRQ(ierr);
  }

  *steps += ts->steps;
  *ptime  = ts->ptime;
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/
#undef __FUNCT__  
#define __FUNCT__ "TSDestroy_CN"
static int TSDestroy_CN(TS ts)
{
  TS_CN *cn = (TS_CN*)ts->data;
  int   ierr;

  PetscFunctionBegin;
  if (cn->update) {ierr = VecDestroy(cn->update);CHKERRQ(ierr);}
  if (cn->func) {ierr = VecDestroy(cn->func);CHKERRQ(ierr);}
  if (cn->rhs) {ierr = VecDestroy(cn->rhs);CHKERRQ(ierr);}
  ierr = PetscFree(cn);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* 
    This defines the nonlinear equation that is to be solved with SNES

              U^{n+1} - dt*F(U^{n+1}) - U^{n}
*/
#undef __FUNCT__  
#define __FUNCT__ "TSCnFunction"
int TSCnFunction(SNES snes,Vec x,Vec y,void *ctx)
{
  TS          ts = (TS) ctx;
  PetscScalar mdt = 1.0/ts->time_step,*unp1,*un,*Funp1;
  int         ierr,i,n;

  PetscFunctionBegin;
  /* apply user provided function */
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
#define __FUNCT__ "TSCnJacobian"
int TSCnJacobian(SNES snes,Vec x,Mat *AA,Mat *BB,MatStructure *str,void *ctx)
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
#define __FUNCT__ "TSSetUp_CN_Linear_Constant_Matrix"
static int TSSetUp_CN_Linear_Constant_Matrix(TS ts)
{
  TS_CN        *cn = (TS_CN*)ts->data;
  int          ierr;
  PetscScalar  two = 2.0,neg_dt = -1.0*ts->time_step;

  PetscFunctionBegin;
  ierr = VecDuplicate(ts->vec_sol,&cn->update);CHKERRQ(ierr);  
  ierr = VecDuplicate(ts->vec_sol,&cn->rhs);CHKERRQ(ierr);  
    
  /* build linear system to be solved */
  ierr = MatScale(&neg_dt,ts->A);CHKERRQ(ierr);
  ierr = MatShift(&two,ts->A);CHKERRQ(ierr);
  if (ts->A != ts->B) {
    ierr = MatScale(&neg_dt,ts->B);CHKERRQ(ierr);
    ierr = MatShift(&two,ts->B);CHKERRQ(ierr);
  }
  ierr = KSPSetOperators(ts->ksp,ts->A,ts->B,SAME_NONZERO_PATTERN);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "TSSetUp_CN_Linear_Variable_Matrix"
static int TSSetUp_CN_Linear_Variable_Matrix(TS ts)
{
  TS_CN *cn = (TS_CN*)ts->data;
  int   ierr;

  PetscFunctionBegin;
  ierr = VecDuplicate(ts->vec_sol,&cn->update);CHKERRQ(ierr);  
  ierr = VecDuplicate(ts->vec_sol,&cn->rhs);CHKERRQ(ierr);  
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "TSSetUp_CN_Nonlinear"
static int TSSetUp_CN_Nonlinear(TS ts)
{
  TS_CN *cn = (TS_CN*)ts->data;
  int   ierr;

  PetscFunctionBegin;
  ierr = VecDuplicate(ts->vec_sol,&cn->update);CHKERRQ(ierr);  
  ierr = VecDuplicate(ts->vec_sol,&cn->func);CHKERRQ(ierr);  
  ierr = SNESSetFunction(ts->snes,cn->func,TSCnFunction,ts);CHKERRQ(ierr);
  ierr = SNESSetJacobian(ts->snes,ts->A,ts->B,TSCnJacobian,ts);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
/*------------------------------------------------------------*/

#undef __FUNCT__  
#define __FUNCT__ "TSSetFromOptions_CN_Linear"
static int TSSetFromOptions_CN_Linear(TS ts)
{
  int ierr;

  PetscFunctionBegin;
  ierr = KSPSetFromOptions(ts->ksp);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "TSSetFromOptions_CN_Nonlinear"
static int TSSetFromOptions_CN_Nonlinear(TS ts)
{
  int ierr;

  PetscFunctionBegin;
  ierr = SNESSetFromOptions(ts->snes);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "TSView_CN"
static int TSView_CN(TS ts,PetscViewer viewer)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

/* ------------------------------------------------------------ */
/*MC
      TS_CN - ODE solver using the implicit Crank-Nicholson method

.seealso:  TSCreate(), TS, TSSetType()

M*/
EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "TSCreate_CN"
int TSCreate_CN(TS ts)
{
  TS_CN      *cn;
  int        ierr;
  KSP        ksp;

  PetscFunctionBegin;
  ts->ops->destroy         = TSDestroy_CN;
  ts->ops->view            = TSView_CN;

  if (ts->problem_type == TS_LINEAR) {
    if (!ts->A) {
      SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Must set rhs matrix for linear problem");
    }
    if (!ts->ops->rhsmatrix) {
      ts->ops->setup  = TSSetUp_CN_Linear_Constant_Matrix;
      ts->ops->step   = TSStep_CN_Linear_Constant_Matrix;
    } else {
      ts->ops->setup  = TSSetUp_CN_Linear_Variable_Matrix;  
      ts->ops->step   = TSStep_CN_Linear_Variable_Matrix;
    }
    ts->ops->setfromoptions  = TSSetFromOptions_CN_Linear;
    ierr = KSPCreate(ts->comm,&ts->ksp);CHKERRQ(ierr);
    ierr = TSGetKSP(ts,&ksp);CHKERRQ(ierr);
    ierr = KSPSetInitialGuessNonzero(ksp,PETSC_TRUE);CHKERRQ(ierr);
  } else if (ts->problem_type == TS_NONLINEAR) {
    ts->ops->setup           = TSSetUp_CN_Nonlinear;  
    ts->ops->step            = TSStep_CN_Nonlinear;
    ts->ops->setfromoptions  = TSSetFromOptions_CN_Nonlinear;
    ierr = SNESCreate(ts->comm,&ts->snes);CHKERRQ(ierr);
  } else SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,"No such problem");

  ierr = PetscNew(TS_CN,&cn);CHKERRQ(ierr);
  PetscLogObjectMemory(ts,sizeof(TS_CN));
  ierr     = PetscMemzero(cn,sizeof(TS_CN));CHKERRQ(ierr);
  ts->data = (void*)cn;

  PetscFunctionReturn(0);
}
EXTERN_C_END





