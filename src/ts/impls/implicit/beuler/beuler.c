#ifndef lint
static char vcid[] = "$Id: beuler.c,v 1.4 1996/03/23 18:34:51 bsmith Exp bsmith $";
#endif
/*
       Code for Time Stepping with implicit backwards Euler.
*/
#include <math.h>
#include "tsimpl.h"                /*I   "ts.h"   I*/
#include "pinclude/pviewer.h"


typedef struct {
  Vec update;      /* work vector where new solution is formed */
  Vec func   ;     /* work vector where F(t[i],u[i]) is stored */
  Vec rhs;         /* work vector for RHS; vec_sol/dt */
} TS_BEuler;

/*
    Version for linear PDE where RHS does not depend on time. Has built a
  single matrix that is to be used for all time steps.
*/
static int TSStep_BEuler_Linear_Constant_Matrix(TS ts,int *steps,Scalar *time)
{
  TS_BEuler *beuler = (TS_BEuler*) ts->data;
  Vec       sol = ts->vec_sol,update = beuler->update;
  Vec       rhs = beuler->rhs;
  int       ierr,i,max_steps = ts->max_steps,its;
  Scalar    mdt = 1.0/ts->time_step;
  
  *steps = -ts->steps;
  ierr = TSMonitor(ts,ts->steps,ts->ptime,sol); CHKERRQ(ierr);

  /* set initial guess to be previous solution */
  ierr = VecCopy(sol,update); CHKERRQ(ierr);

  for ( i=0; i<max_steps; i++ ) {
    ts->ptime += ts->time_step;
    if (ts->ptime > ts->max_time) break;
    ierr = VecCopy(sol,rhs); CHKERRQ(ierr);
    ierr = VecScale(&mdt,rhs); CHKERRQ(ierr);
    ierr = SLESSolve(ts->sles,rhs,update,&its); CHKERRQ(ierr);
    ierr = VecCopy(update,sol); CHKERRQ(ierr);
    ts->steps++;
    ierr = TSMonitor(ts,ts->steps,ts->ptime,sol);CHKERRQ(ierr);
  }

  *steps += ts->steps;
  *time  = ts->ptime;
  return 0;
}
/*
      Version where matrix depends on time 
*/
static int TSStep_BEuler_Linear_Variable_Matrix(TS ts,int *steps,Scalar *time)
{
  TS_BEuler    *beuler = (TS_BEuler*) ts->data;
  Vec          sol = ts->vec_sol,update = beuler->update, rhs = beuler->rhs;
  int          ierr,i,max_steps = ts->max_steps,its;
  Scalar       mdt = 1.0/ts->time_step, mone = -1.0;
  MatStructure str;

  *steps = -ts->steps;
  ierr = TSMonitor(ts,ts->steps,ts->ptime,sol);CHKERRQ(ierr);

  /* set initial guess to be previous solution */
  ierr = VecCopy(sol,update); CHKERRQ(ierr);

  for ( i=0; i<max_steps; i++ ) {
    ts->ptime += ts->time_step;
    if (ts->ptime > ts->max_time) break;
    /*
        evaluate matrix function 
    */
    ierr = (*ts->rhsmatrix)(ts,ts->ptime,&ts->A,&ts->B,&str,ts->funP);CHKERRQ(ierr);
    if (!ts->Ashell) {
      ierr = MatScale(&mone,ts->A); CHKERRQ(ierr);
      ierr = MatShift(&mdt,ts->A); CHKERRQ(ierr);
    }
    if (ts->B != ts->A && ts->Ashell != ts->B && str != SAME_PRECONDITIONER) {
      ierr = MatScale(&mone,ts->B); CHKERRQ(ierr);
      ierr = MatShift(&mdt,ts->B); CHKERRQ(ierr);
    }
    ierr = VecCopy(sol,rhs); CHKERRQ(ierr);
    ierr = VecScale(&mdt,rhs); CHKERRQ(ierr);
    ierr = SLESSetOperators(ts->sles,ts->A,ts->B,str); CHKERRQ(ierr);
    ierr = SLESSolve(ts->sles,rhs,update,&its); CHKERRQ(ierr);
    ierr = VecCopy(update,sol); CHKERRQ(ierr);
    ts->steps++;
    ierr = TSMonitor(ts,ts->steps,ts->ptime,sol); CHKERRQ(ierr);
  }

  *steps += ts->steps;
  *time  = ts->ptime;
  return 0;
}
/*
    Version for nonlinear PDE.
*/
static int TSStep_BEuler_Nonlinear(TS ts,int *steps,Scalar *time)
{
  Vec       sol = ts->vec_sol;
  int       ierr,i,max_steps = ts->max_steps,its;
  TS_BEuler *beuler = (TS_BEuler*) ts->data;
  
  *steps = -ts->steps;
  ierr = TSMonitor(ts,ts->steps,ts->ptime,sol); CHKERRQ(ierr);

  for ( i=0; i<max_steps; i++ ) {
    ts->ptime += ts->time_step;
    if (ts->ptime > ts->max_time) break;
    ierr = VecCopy(sol,beuler->update); CHKERRQ(ierr);
    ierr = SNESSolve(ts->snes,beuler->update,&its); CHKERRQ(ierr);
    ierr = VecCopy(beuler->update,sol); CHKERRQ(ierr);
    ts->steps++;
    ierr = TSMonitor(ts,ts->steps,ts->ptime,sol);CHKERRQ(ierr);
  }

  *steps += ts->steps;
  *time  = ts->ptime;
  return 0;
}

/*------------------------------------------------------------*/
static int TSDestroy_BEuler(PetscObject obj )
{
  TS        ts = (TS) obj;
  TS_BEuler *beuler = (TS_BEuler*) ts->data;
  int       ierr;

  ierr = VecDestroy(beuler->update); CHKERRQ(ierr);
  if (beuler->func) {ierr = VecDestroy(beuler->func);CHKERRQ(ierr);}
  if (beuler->rhs) {ierr = VecDestroy(beuler->rhs);CHKERRQ(ierr);}
  if (ts->Ashell) {ierr = MatDestroy(ts->A); CHKERRQ(ierr);}
  PetscFree(beuler);
  return 0;
}


/*------------------------------------------------------------*/
/*
    This matrix shell multiply where user provided Shell matrix
*/

int TSBEulerMatMult(Mat mat,Vec x,Vec y)
{
  TS     ts;
  Scalar mdt,mone = -1.0;
  int    ierr;

  MatShellGetContext(mat,(void **)&ts);
  mdt = 1.0/ts->time_step;

  /* apply user provided function */
  ierr = MatMult(ts->Ashell,x,y); CHKERRQ(ierr);
  /* shift and scale by 1/dt - F */
  ierr = VecAXPBY(&mdt,&mone,x,y); CHKERRQ(ierr);
  return 0;
}

/* 
    This defines the nonlinear equation that is to be solved with SNES

              U^{n+1} - dt*F(U^{n+1}) - U^{n}
*/
int TSBEulerFunction(SNES snes,Vec x,Vec y,void *ctx)
{
  TS     ts = (TS) ctx;
  Scalar mdt = 1.0/ts->time_step,*unp1,*un,*Funp1;
  int    ierr,i,n;

  /* apply user provided function */
  ierr = TSComputeRHSFunction(ts,ts->ptime,x,y); CHKERRQ(ierr);
  /* (u^{n+1) - U^{n})/dt - F(u^{n+1}) */
  ierr = VecGetArray(ts->vec_sol,&un); CHKERRQ(ierr);
  ierr = VecGetArray(x,&unp1); CHKERRQ(ierr);
  ierr = VecGetArray(y,&Funp1); CHKERRQ(ierr);
  ierr = VecGetLocalSize(x,&n); CHKERRQ(ierr);

  for ( i=0; i<n; i++ ) {
    Funp1[i] = mdt*(unp1[i] - un[i]) - Funp1[i];
  }
  ierr = VecRestoreArray(ts->vec_sol,&un);
  ierr = VecRestoreArray(x,&unp1);
  ierr = VecRestoreArray(y,&Funp1);
  return 0;
}

/*
   This constructs the Jacobian needed for SNES 

             J = I/dt - J_{F}   where J_{F} is the given Jacobian of F.
*/
int TSBEulerJacobian(SNES snes,Vec x,Mat *AA,Mat *BB,MatStructure *str,void *ctx)
{
  TS     ts = (TS) ctx;
  Mat    A,B;
  int    ierr;
  Scalar mone = -1.0, mdt = 1.0/ts->time_step;

  /* construct users Jacobian */
  if (ts->rhsjacobian) {
    ierr = (*ts->rhsjacobian)(ts,ts->ptime,x,AA,BB,str,ts->jacP);CHKERRQ(ierr);
  }
  A = *AA; B = *BB;
  /* shift and scale Jacobian */
  if (!ts->Ashell) {
    ierr = MatScale(&mone,A); CHKERRQ(ierr);
    ierr = MatShift(&mdt,A); CHKERRQ(ierr);
  }
  if (ts->B != ts->A && ts->Ashell != ts->B && *str != SAME_PRECONDITIONER) {
    ierr = MatScale(&mone,B); CHKERRQ(ierr);
    ierr = MatShift(&mdt,B); CHKERRQ(ierr);
  }

  return 0;
}

/* ------------------------------------------------------------*/
static int TSSetUp_BEuler_Linear_Constant_Matrix(TS ts)
{
  TS_BEuler *beuler = (TS_BEuler*) ts->data;
  int       ierr,M;
  Scalar    mdt = 1.0/ts->time_step, mone = -1.0;

  ierr = VecDuplicate(ts->vec_sol,&beuler->update); CHKERRQ(ierr);  
  ierr = VecDuplicate(ts->vec_sol,&beuler->rhs); CHKERRQ(ierr);  
    
  /* build linear system to be solved */
  if (!ts->Ashell) {
    ierr = MatScale(&mone,ts->A); CHKERRQ(ierr);
    ierr = MatShift(&mdt,ts->A); CHKERRQ(ierr);
  } else {
    /* construct new shell matrix */
    ierr = VecGetSize(ts->vec_sol,&M); CHKERRQ(ierr);
    ierr = MatCreateShell(ts->comm,M,M,ts,&ts->A); CHKERRQ(ierr);
    ierr = MatShellSetOperation(ts->A,MAT_MULT,TSBEulerMatMult); CHKERRQ(ierr);
  }
  if (ts->A != ts->B && ts->Ashell != ts->B) {
    ierr = MatScale(&mone,ts->B); CHKERRQ(ierr);
    ierr = MatShift(&mdt,ts->B); CHKERRQ(ierr);
  }
  ierr = SLESSetOperators(ts->sles,ts->A,ts->B,SAME_NONZERO_PATTERN);CHKERRQ(ierr);
  return 0;
}

static int TSSetUp_BEuler_Linear_Variable_Matrix(TS ts)
{
  TS_BEuler *beuler = (TS_BEuler*) ts->data;
  int       ierr, M;

  ierr = VecDuplicate(ts->vec_sol,&beuler->update); CHKERRQ(ierr);  
  ierr = VecDuplicate(ts->vec_sol,&beuler->rhs); CHKERRQ(ierr);  
  if (ts->Ashell) { /* construct new shell matrix */
    ierr = VecGetSize(ts->vec_sol,&M); CHKERRQ(ierr);
    ierr = MatCreateShell(ts->comm,M,M,ts,&ts->A); CHKERRQ(ierr);
    ierr = MatShellSetOperation(ts->A,MAT_MULT,TSBEulerMatMult); CHKERRQ(ierr);
  }
  return 0;
}


static int TSSetUp_BEuler_Nonlinear(TS ts)
{
  TS_BEuler *beuler = (TS_BEuler*) ts->data;
  int       ierr,M;

  ierr = VecDuplicate(ts->vec_sol,&beuler->update); CHKERRQ(ierr);  
  ierr = VecDuplicate(ts->vec_sol,&beuler->func); CHKERRQ(ierr);  
  ierr = SNESSetFunction(ts->snes,beuler->func,TSBEulerFunction,ts);CHKERRQ(ierr);
  if (ts->Ashell) { /* construct new shell matrix */
    ierr = VecGetSize(ts->vec_sol,&M); CHKERRQ(ierr);
    ierr = MatCreateShell(ts->comm,M,M,ts,&ts->A); CHKERRQ(ierr);
    ierr = MatShellSetOperation(ts->A,MAT_MULT,TSBEulerMatMult); CHKERRQ(ierr);
  }
  ierr = SNESSetJacobian(ts->snes,ts->A,ts->B,TSBEulerJacobian,ts);CHKERRQ(ierr);
  return 0;
}
/*------------------------------------------------------------*/

static int TSSetFromOptions_BEuler_Linear(TS ts)
{
  int ierr;
  ierr = SLESSetFromOptions(ts->sles); CHKERRQ(ierr);
  
  return 0;
}

static int TSSetFromOptions_BEuler_Nonlinear(TS ts)
{
  int ierr;
  ierr = SNESSetFromOptions(ts->snes); CHKERRQ(ierr);
  
  return 0;
}

static int TSPrintHelp_BEuler(TS ts)
{

  return 0;
}

static int TSView_BEuler(PetscObject obj,Viewer viewer)
{
  return 0;
}

/* ------------------------------------------------------------ */
int TSCreate_BEuler(TS ts )
{
  TS_BEuler *beuler;
  int       ierr;
  KSP       ksp;
  MatType   mtype;

  ts->type 	      = TS_BEULER;
  ts->destroy         = TSDestroy_BEuler;
  ts->printhelp       = TSPrintHelp_BEuler;
  ts->view            = TSView_BEuler;

  if (ts->problem_type == TS_LINEAR) {
    if (!ts->A) {
      SETERRQ(1,"TSCreate_BEuler:Must set rhs matrix for linear problem");
    }
    ierr = MatGetType(ts->A,&mtype,PETSC_NULL);
    if (!ts->rhsmatrix) {
      if (mtype == MATSHELL) {
        ts->Ashell = ts->A;
      }
      ts->setup  = TSSetUp_BEuler_Linear_Constant_Matrix;
      ts->step   = TSStep_BEuler_Linear_Constant_Matrix;
    }
    else {
      if (mtype == MATSHELL) {
        ts->Ashell = ts->A;
      }
      ts->setup  = TSSetUp_BEuler_Linear_Variable_Matrix;  
      ts->step   = TSStep_BEuler_Linear_Variable_Matrix;
    }
    ts->setfromoptions  = TSSetFromOptions_BEuler_Linear;
    ierr = SLESCreate(ts->comm,&ts->sles); CHKERRQ(ierr);
    ierr = SLESGetKSP(ts->sles,&ksp); CHKERRQ(ierr);
    ierr = KSPSetInitialGuessNonzero(ksp); CHKERRQ(ierr);
  } else if (ts->problem_type == TS_NONLINEAR) {
    if (!ts->A) {
      SETERRQ(1,"TSCreate_BEuler:Must set Jacobian for nonlinear problem");
    }
    ierr = MatGetType(ts->A,&mtype,PETSC_NULL);
    if (mtype == MATSHELL) {
      ts->Ashell = ts->A;
    }
    ts->setup           = TSSetUp_BEuler_Nonlinear;  
    ts->step            = TSStep_BEuler_Nonlinear;
    ts->setfromoptions  = TSSetFromOptions_BEuler_Nonlinear;
    ierr = SNESCreate(ts->comm,SNES_NONLINEAR_EQUATIONS,&ts->snes);CHKERRQ(ierr);
  } else SETERRQ(1,"TSCreate_BEuler:No such problem");

  beuler   = PetscNew(TS_BEuler); CHKPTRQ(beuler);
  PetscMemzero(beuler,sizeof(TS_BEuler));
  ts->data = (void *) beuler;

  return 0;
}





