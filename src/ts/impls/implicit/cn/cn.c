#define PETSCTS_DLL

/*
       Code for Timestepping with implicit Crank-Nicholson method.
*/
#include "private/tsimpl.h"                /*I   "petscts.h"   I*/

typedef struct {
  Vec  update;         /* work vector where new solution is formed */
  Vec  func;           /* work vector where F(t[i],u[i]) is stored */
  Vec  rhsfunc, rhsfunc_old; /* work vectors to hold rhs function provided by user */
  Vec  rhs;            /* work vector for RHS; vec_sol/dt */
  TS   ts;             /* used by ShellMult_private() */
  PetscScalar mdt;     /* 1/dt, used by ShellMult_private() */
  PetscReal rhsfunc_time,rhsfunc_old_time; /* time at which rhsfunc holds the value */
} TS_CN;

/*------------------------------------------------------------------------------*/
/* 
   Scale ts->Alhs = 1/dt*Alhs, ts->Arhs = 0.5*Arhs
   Set   ts->A    = Alhs - Arhs, used in KSPSolve()
*/
#undef __FUNCT__  
#define __FUNCT__ "TSSetKSPOperators_CN_Matrix"
PetscErrorCode TSSetKSPOperators_CN_Matrix(TS ts)
{
  PetscErrorCode ierr;
  PetscScalar    mdt = 1.0/ts->time_step;

  PetscFunctionBegin;
  /* scale Arhs = 0.5*Arhs, Alhs = 1/dt*Alhs - assume dt is constant! */
  ierr = MatScale(ts->Arhs,0.5);CHKERRQ(ierr);
  if (ts->Alhs){
    ierr = MatScale(ts->Alhs,mdt);CHKERRQ(ierr);
  }
  if (ts->A){
    ierr = MatDestroy(ts->A);CHKERRQ(ierr);
  }
  ierr = MatDuplicate(ts->Arhs,MAT_COPY_VALUES,&ts->A);CHKERRQ(ierr);
 
  if (ts->Alhs){
    /* ts->A = - Arhs + Alhs */
    ierr = MatAYPX(ts->A,-1.0,ts->Alhs,ts->matflg);CHKERRQ(ierr);
  } else { 
    /* ts->A = 1/dt - Arhs */
    ierr = MatScale(ts->A,-1.0);CHKERRQ(ierr);
    ierr = MatShift(ts->A,mdt);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/* 
   Scale ts->Alhs = 1/dt*Alhs, ts->Arhs = 0.5*Arhs
   Set   ts->A    = Alhs - Arhs, used in KSPSolve()
*/
#undef __FUNCT__
#define __FUNCT__ "ShellMult_private"
PetscErrorCode ShellMult_private(Mat mat,Vec x,Vec y)
{
  PetscErrorCode  ierr;
  void            *ctx;
  TS_CN           *cn;

  PetscFunctionBegin;
  ierr = MatShellGetContext(mat,(void **)&ctx);CHKERRQ(ierr);
  cn   = (TS_CN*)ctx;
  ierr = MatMult(cn->ts->Arhs,x,y);CHKERRQ(ierr); /* y = 0.5*Arhs*x */
  ierr = VecScale(y,-1.0);CHKERRQ(ierr);          /* y = -0.5*Arhs*x */
  if (cn->ts->Alhs){
    ierr = MatMultAdd(cn->ts->Alhs,x,y,y);CHKERRQ(ierr); /* y = 1/dt*Alhs*x + y */
  } else {
    ierr = VecAXPY(y,cn->mdt,x);CHKERRQ(ierr); /* y = 1/dt*x + y */
  }
  PetscFunctionReturn(0);
}
#undef __FUNCT__  
#define __FUNCT__ "TSSetKSPOperators_CN_No_Matrix"
PetscErrorCode TSSetKSPOperators_CN_No_Matrix(TS ts)
{
  PetscErrorCode ierr;
  PetscScalar    mdt = 1.0/ts->time_step;
  Mat            Arhs = ts->Arhs;
  MPI_Comm       comm;
  PetscInt       m,n,M,N;
  TS_CN          *cn = (TS_CN*)ts->data;

  PetscFunctionBegin;
  /* scale Arhs = 0.5*Arhs, Alhs = 1/dt*Alhs - assume dt is constant! */
  ierr = MatScale(ts->Arhs,0.5);CHKERRQ(ierr);
  if (ts->Alhs){
    ierr = MatScale(ts->Alhs,mdt);CHKERRQ(ierr);
  }
 
  cn->ts  = ts;
  cn->mdt = mdt;
  if (ts->A) {
    ierr = MatDestroy(ts->A);CHKERRQ(ierr);
  }
  ierr = MatGetSize(Arhs,&M,&N);CHKERRQ(ierr);
  ierr = MatGetLocalSize(Arhs,&m,&n);CHKERRQ(ierr);
  ierr = PetscObjectGetComm((PetscObject)Arhs,&comm);CHKERRQ(ierr);
  ierr = MatCreateShell(comm,m,n,M,N,cn,&ts->A);CHKERRQ(ierr);
  ierr = MatShellSetOperation(ts->A,MATOP_MULT,(void(*)(void))ShellMult_private);CHKERRQ(ierr); 
  PetscFunctionReturn(0);
}

/*
    Version for linear PDE where RHS does not depend on time. Has built a
  single matrix that is to be used for all timesteps.
*/
#undef __FUNCT__  
#define __FUNCT__ "TSStep_CN_Linear_Constant_Matrix"
static PetscErrorCode TSStep_CN_Linear_Constant_Matrix(TS ts,PetscInt *steps,PetscReal *ptime)
{
  TS_CN          *cn = (TS_CN*)ts->data;
  Vec            sol = ts->vec_sol,update = cn->update,rhs = cn->rhs;
  PetscErrorCode ierr;
  PetscInt       i,max_steps = ts->max_steps,its;
  PetscScalar    mdt = 1.0/ts->time_step;

  PetscFunctionBegin;
  *steps = -ts->steps;
  ierr   = TSMonitor(ts,ts->steps,ts->ptime,sol);CHKERRQ(ierr);

  /* set initial guess to be previous solution */
  ierr = VecCopy(sol,update);CHKERRQ(ierr);

  for (i=0; i<max_steps; i++) {
    if (ts->ptime + ts->time_step > ts->max_time) break;
    ierr = TSPreStep(ts);CHKERRQ(ierr);
    /* set rhs = (1/dt*Alhs + 0.5*Arhs)*sol */
    ierr = MatMult(ts->Arhs,sol,rhs);CHKERRQ(ierr); /* rhs = 0.5*Arhs*sol */
    if (ts->Alhs){
      ierr = MatMultAdd(ts->Alhs,sol,rhs,rhs);      /* rhs = rhs + 1/dt*Alhs*sol */
    } else {
      ierr = VecAXPY(rhs,mdt,sol);CHKERRQ(ierr);    /* rhs = rhs + 1/dt*sol */
    }  

    ts->ptime += ts->time_step;

    /* solve (1/dt*Alhs - 0.5*Arhs)*update = rhs */
    ierr = KSPSolve(ts->ksp,rhs,update);CHKERRQ(ierr);
    ierr = KSPGetIterationNumber(ts->ksp,&its);CHKERRQ(ierr);
    ts->linear_its += PetscAbsInt(its);
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
#define __FUNCT__ "TSStep_CN_Linear_Variable_Matrix"
static PetscErrorCode TSStep_CN_Linear_Variable_Matrix(TS ts,PetscInt *steps,PetscReal *ptime)
{
  TS_CN          *cn = (TS_CN*)ts->data;
  Vec            sol = ts->vec_sol,update = cn->update,rhs = cn->rhs;
  PetscErrorCode ierr;
  PetscInt       i,max_steps = ts->max_steps,its;
  PetscScalar    mdt = 1.0/ts->time_step;
  PetscReal      t_mid;
  MatStructure   str;

  PetscFunctionBegin;
  *steps = -ts->steps;
  ierr   = TSMonitor(ts,ts->steps,ts->ptime,sol);CHKERRQ(ierr);

  /* set initial guess to be previous solution */
  ierr = VecCopy(sol,update);CHKERRQ(ierr);

  for (i=0; i<max_steps; i++) {
    if (ts->ptime + ts->time_step > ts->max_time) break;
    ierr = TSPreStep(ts);CHKERRQ(ierr);

    /* set rhs = (1/dt*Alhs(t_mid) + 0.5*Arhs(t_n)) * sol */
    if (i==0){
      /* evaluate 0.5*Arhs(t_0) */
      ierr = (*ts->ops->rhsmatrix)(ts,ts->ptime,&ts->Arhs,PETSC_NULL,&str,ts->jacP);CHKERRQ(ierr);
      ierr = MatScale(ts->Arhs,0.5);CHKERRQ(ierr);
    }
    if (ts->Alhs){
      /* evaluate Alhs(t_mid) */
      t_mid = ts->ptime+ts->time_step/2.0;
      ierr = (*ts->ops->lhsmatrix)(ts,t_mid,&ts->Alhs,PETSC_NULL,&str,ts->jacPlhs);CHKERRQ(ierr);
      ierr = MatMult(ts->Alhs,sol,rhs);CHKERRQ(ierr); /* rhs = Alhs_mid*sol */
      ierr = VecScale(rhs,mdt);CHKERRQ(ierr);         /* rhs = 1/dt*Alhs_mid*sol */
      ierr = MatMultAdd(ts->Arhs,sol,rhs,rhs);        /* rhs = rhs + 0.5*Arhs_mid*sol */
    } else {
      ierr = MatMult(ts->Arhs,sol,rhs);CHKERRQ(ierr); /* rhs = 0.5*Arhs_n*sol */
      ierr = VecAXPY(rhs,mdt,sol);CHKERRQ(ierr);      /* rhs = rhs + 1/dt*sol */
    }  

    ts->ptime += ts->time_step;

    /* evaluate Arhs at current ptime t_{n+1} */
    ierr = (*ts->ops->rhsmatrix)(ts,ts->ptime,&ts->Arhs,PETSC_NULL,&str,ts->jacP);CHKERRQ(ierr);
    ierr = TSSetKSPOperators_CN_Matrix(ts);CHKERRQ(ierr);

    ierr = KSPSetOperators(ts->ksp,ts->A,ts->A,SAME_NONZERO_PATTERN);CHKERRQ(ierr);
    ierr = KSPSolve(ts->ksp,rhs,update);CHKERRQ(ierr);
    ierr = KSPGetIterationNumber(ts->ksp,&its);CHKERRQ(ierr);
    ts->linear_its += PetscAbsInt(its);
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
#define __FUNCT__ "TSStep_CN_Nonlinear"
static PetscErrorCode TSStep_CN_Nonlinear(TS ts,PetscInt *steps,PetscReal *ptime)
{
  Vec            sol = ts->vec_sol;
  PetscErrorCode ierr;
  PetscInt       i,max_steps = ts->max_steps,its,lits;
  TS_CN          *cn = (TS_CN*)ts->data;
  
  PetscFunctionBegin;
  *steps = -ts->steps;
  ierr = TSMonitor(ts,ts->steps,ts->ptime,sol);CHKERRQ(ierr);

  for (i=0; i<max_steps; i++) {
    if (ts->ptime + ts->time_step > ts->max_time) break;
    ierr = TSPreStep(ts);CHKERRQ(ierr);
    ts->ptime += ts->time_step;
   
    ierr = VecCopy(sol,cn->update);CHKERRQ(ierr);
    ierr = SNESSolve(ts->snes,PETSC_NULL,cn->update);CHKERRQ(ierr);
    ierr = SNESGetIterationNumber(ts->snes,&its);CHKERRQ(ierr);
    ierr = SNESGetLinearSolveIterations(ts->snes,&lits);CHKERRQ(ierr);
    ts->nonlinear_its += its; ts->linear_its += lits;
    ierr = VecCopy(cn->update,sol);CHKERRQ(ierr);
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
#define __FUNCT__ "TSDestroy_CN"
static PetscErrorCode TSDestroy_CN(TS ts)
{
  TS_CN          *cn = (TS_CN*)ts->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (cn->update) {ierr = VecDestroy(cn->update);CHKERRQ(ierr);}
  if (cn->func) {ierr = VecDestroy(cn->func);CHKERRQ(ierr);}
  if (cn->rhsfunc) {ierr = VecDestroy(cn->rhsfunc);CHKERRQ(ierr);}
  if (cn->rhsfunc_old) {ierr = VecDestroy(cn->rhsfunc_old);CHKERRQ(ierr);}
  if (cn->rhs) {ierr = VecDestroy(cn->rhs);CHKERRQ(ierr);}
  ierr = PetscFree(cn);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* 
    This defines the nonlinear equation that is to be solved with SNES
       1/dt*Alhs*(U^{n+1} - U^{n}) - 0.5*(F(U^{n+1}) + F(U^{n}))
*/
#undef __FUNCT__  
#define __FUNCT__ "TSCnFunction"
PetscErrorCode TSCnFunction(SNES snes,Vec x,Vec y,void *ctx)
{
  TS             ts = (TS) ctx;
  PetscScalar    mdt = 1.0/ts->time_step,*unp1,*un,*Funp1,*Fun,*yarray;
  PetscErrorCode ierr;
  PetscInt       i,n;
  TS_CN          *cn = (TS_CN*)ts->data;

  PetscFunctionBegin;
  /* apply user provided function */
  if (cn->rhsfunc_time == (ts->ptime - ts->time_step)){
    /* printf("   copy rhsfunc to rhsfunc_old, then eval rhsfunc\n"); */
    ierr = VecCopy(cn->rhsfunc,cn->rhsfunc_old);CHKERRQ(ierr);
    cn->rhsfunc_old_time = cn->rhsfunc_time;
  } else if (cn->rhsfunc_time != ts->ptime && cn->rhsfunc_old_time != ts->ptime-ts->time_step){
    /* printf("   eval both rhsfunc_old and rhsfunc\n"); */
    ierr = TSComputeRHSFunction(ts,ts->ptime-ts->time_step,ts->vec_sol,cn->rhsfunc_old);CHKERRQ(ierr); /* rhsfunc_old=F(U^{n}) */
    cn->rhsfunc_old_time = ts->ptime - ts->time_step;
  } 
  
  if (ts->Alhs){
    /* compute y=Alhs*(U^{n+1} - U^{n}) with cn->rhsfunc as workspce */
    ierr = VecWAXPY(cn->rhsfunc,-1.0,ts->vec_sol,x);CHKERRQ(ierr);
    ierr = MatMult(ts->Alhs,cn->rhsfunc,y);CHKERRQ(ierr);
  }

  ierr = TSComputeRHSFunction(ts,ts->ptime,x,cn->rhsfunc);CHKERRQ(ierr); /* rhsfunc = F(U^{n+1}) */
  cn->rhsfunc_time = ts->ptime;
    
  ierr = VecGetArray(ts->vec_sol,&un);CHKERRQ(ierr); /* U^{n} */
  ierr = VecGetArray(x,&unp1);CHKERRQ(ierr);         /* U^{n+1} */  
  ierr = VecGetArray(cn->rhsfunc,&Funp1);CHKERRQ(ierr);
  ierr = VecGetArray(cn->rhsfunc_old,&Fun);CHKERRQ(ierr);  
  ierr = VecGetArray(y,&yarray);CHKERRQ(ierr);  
  ierr = VecGetLocalSize(x,&n);CHKERRQ(ierr);
  if (ts->Alhs){ 
    for (i=0; i<n; i++) {
      yarray[i] = mdt*yarray[i] - 0.5*(Funp1[i] + Fun[i]);
    }
  } else {
    for (i=0; i<n; i++) {
      yarray[i] = mdt*(unp1[i] - un[i]) - 0.5*(Funp1[i] + Fun[i]);
    }
  }
  ierr = VecRestoreArray(ts->vec_sol,&un);CHKERRQ(ierr);
  ierr = VecRestoreArray(x,&unp1);CHKERRQ(ierr);
  ierr = VecRestoreArray(cn->rhsfunc,&Funp1);CHKERRQ(ierr);
  ierr = VecRestoreArray(cn->rhsfunc_old,&Fun);CHKERRQ(ierr);
  ierr = VecRestoreArray(y,&yarray);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* Set B = 1/dt*Alh - 0.5*B */
#undef __FUNCT__  
#define __FUNCT__ "TSScaleShiftMatrices_CN"
PetscErrorCode TSScaleShiftMatrices_CN(TS ts,Mat A,Mat B,MatStructure str)
{
  PetscTruth     flg;
  PetscErrorCode ierr;
  PetscScalar    mdt = 1.0/ts->time_step;

  PetscFunctionBegin;
  ierr = PetscTypeCompare((PetscObject)B,MATMFFD,&flg);CHKERRQ(ierr);
  if (!flg) {
    ierr = MatScale(B,-0.5);CHKERRQ(ierr);
    if (ts->Alhs){
      ierr = MatAXPY(B,mdt,ts->Alhs,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr); /* DIFFERENT_NONZERO_PATTERN? */
    } else {
      ierr = MatShift(B,mdt);CHKERRQ(ierr);
    }
  } else {
    SETERRQ(PETSC_ERR_SUP,"Matrix type MATMFFD is not supported yet"); /* ref TSScaleShiftMatrices() */
  }
  PetscFunctionReturn(0);
}

/*
   This constructs the Jacobian needed for SNES 
     J = I/dt - 0.5*J_{F}   where J_{F} is the given Jacobian of F.
     x  - input vector
     AA - Jacobian matrix 
     BB - preconditioner matrix, usually the same as AA
*/
#undef __FUNCT__  
#define __FUNCT__ "TSCnJacobian"
PetscErrorCode TSCnJacobian(SNES snes,Vec x,Mat *AA,Mat *BB,MatStructure *str,void *ctx)
{
  TS             ts = (TS) ctx;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /* construct user's Jacobian */
  ierr = TSComputeRHSJacobian(ts,ts->ptime,x,AA,BB,str);CHKERRQ(ierr); /* BB = J_{F} */

  /* shift and scale Jacobian */
  ierr = TSScaleShiftMatrices_CN(ts,*AA,*BB,*str);CHKERRQ(ierr); /* Set BB = 1/dt*Alhs - 0.5*BB */
  PetscFunctionReturn(0);
}

/* ------------------------------------------------------------*/
#undef __FUNCT__  
#define __FUNCT__ "TSSetUp_CN_Linear_Constant_Matrix"
static PetscErrorCode TSSetUp_CN_Linear_Constant_Matrix(TS ts)
{
  TS_CN          *cn = (TS_CN*)ts->data;
  PetscErrorCode ierr;
  PetscTruth shelltype;

  PetscFunctionBegin;
  ierr = VecDuplicate(ts->vec_sol,&cn->update);CHKERRQ(ierr);  
  ierr = VecDuplicate(ts->vec_sol,&cn->rhs);CHKERRQ(ierr);  
    
  /* build linear system to be solved */
  /* scale ts->Alhs = 1/dt*Alhs, ts->Arhs = 0.5*Arhs; set ts->A = Alhs - Arhs */
  ierr = PetscTypeCompare((PetscObject)ts->Arhs,MATSHELL,&shelltype);CHKERRQ(ierr);
  if (shelltype){
    ierr = TSSetKSPOperators_CN_No_Matrix(ts);CHKERRQ(ierr);
  } else {
    ierr = TSSetKSPOperators_CN_Matrix(ts);CHKERRQ(ierr);  
  } 
  ierr = KSPSetOperators(ts->ksp,ts->A,ts->A,SAME_NONZERO_PATTERN);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "TSSetUp_CN_Linear_Variable_Matrix"
static PetscErrorCode TSSetUp_CN_Linear_Variable_Matrix(TS ts)
{
  TS_CN          *cn = (TS_CN*)ts->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecDuplicate(ts->vec_sol,&cn->update);CHKERRQ(ierr);  
  ierr = VecDuplicate(ts->vec_sol,&cn->rhs);CHKERRQ(ierr);  
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "TSSetUp_CN_Nonlinear"
static PetscErrorCode TSSetUp_CN_Nonlinear(TS ts)
{
  TS_CN          *cn = (TS_CN*)ts->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecDuplicate(ts->vec_sol,&cn->update);CHKERRQ(ierr);  
  ierr = VecDuplicate(ts->vec_sol,&cn->func);CHKERRQ(ierr);  
  ierr = VecDuplicate(ts->vec_sol,&cn->rhsfunc);CHKERRQ(ierr); 
  ierr = VecDuplicate(ts->vec_sol,&cn->rhsfunc_old);CHKERRQ(ierr); 
  ierr = SNESSetFunction(ts->snes,cn->func,TSCnFunction,ts);CHKERRQ(ierr);
  ierr = SNESSetJacobian(ts->snes,ts->A,ts->B,TSCnJacobian,ts);CHKERRQ(ierr);
  cn->rhsfunc_time     = -100.0; /* cn->rhsfunc is not evaluated yet */
  cn->rhsfunc_old_time = -100.0;
  PetscFunctionReturn(0);
}
/*------------------------------------------------------------*/

#undef __FUNCT__  
#define __FUNCT__ "TSSetFromOptions_CN_Linear"
static PetscErrorCode TSSetFromOptions_CN_Linear(TS ts)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "TSSetFromOptions_CN_Nonlinear"
static PetscErrorCode TSSetFromOptions_CN_Nonlinear(TS ts)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "TSView_CN"
static PetscErrorCode TSView_CN(TS ts,PetscViewer viewer)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

/* ------------------------------------------------------------ */
/*MC
      TSCN - ODE solver using the implicit Crank-Nicholson method

  Level: beginner

.seealso:  TSCreate(), TS, TSSetType()

M*/
EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "TSCreate_CN"
PetscErrorCode PETSCTS_DLLEXPORT TSCreate_CN(TS ts)
{
  TS_CN          *cn;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ts->ops->destroy = TSDestroy_CN;
  ts->ops->view    = TSView_CN;

  if (ts->problem_type == TS_LINEAR) {
    if (!ts->Arhs) {
      SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Must set rhs matrix for linear problem");
    }    
    if (!ts->ops->rhsmatrix) {     
      ts->ops->setup = TSSetUp_CN_Linear_Constant_Matrix;
      ts->ops->step  = TSStep_CN_Linear_Constant_Matrix;
    } else {
      ts->ops->setup = TSSetUp_CN_Linear_Variable_Matrix;  
      ts->ops->step  = TSStep_CN_Linear_Variable_Matrix;
    }
    ts->ops->setfromoptions = TSSetFromOptions_CN_Linear;
    ierr = KSPCreate(((PetscObject)ts)->comm,&ts->ksp);CHKERRQ(ierr);
    ierr = PetscObjectIncrementTabLevel((PetscObject)ts->ksp,(PetscObject)ts,1);CHKERRQ(ierr);
    ierr = KSPSetInitialGuessNonzero(ts->ksp,PETSC_TRUE);CHKERRQ(ierr);
  } else if (ts->problem_type == TS_NONLINEAR) {
    ts->ops->setup          = TSSetUp_CN_Nonlinear;  
    ts->ops->step           = TSStep_CN_Nonlinear;
    ts->ops->setfromoptions = TSSetFromOptions_CN_Nonlinear;
    ierr = SNESCreate(((PetscObject)ts)->comm,&ts->snes);CHKERRQ(ierr);
    ierr = PetscObjectIncrementTabLevel((PetscObject)ts->snes,(PetscObject)ts,1);CHKERRQ(ierr);
  } else SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,"No such problem");

  ierr = PetscNewLog(ts,TS_CN,&cn);CHKERRQ(ierr);
  ts->data = (void*)cn;
  PetscFunctionReturn(0);
}
EXTERN_C_END





