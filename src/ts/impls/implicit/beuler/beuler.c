#ifndef lint
static char vcid[] = "$Id: beuler.c,v 1.1 1996/01/17 04:45:47 bsmith Exp bsmith $";
#endif
/*
       Code for Time Stepping with implicit backwards Euler.
*/
#include <math.h>
#include "tsimpl.h"                /*I   "ts.h"   I*/
#include "pinclude/pviewer.h"


typedef struct {
  Vec update;     /* work vector where F(t[i],u[i]) is stored */
} TS_BEuler;

/*
    Version for linear PDE where RHS does not depend on time. Builds a
  single matrix that is to be used for all time steps.
*/
static int TSStep_BEuler_Linear_Constant_Matrix(TS ts,int *steps,Scalar *time)
{
  TS_BEuler *beuler = (TS_BEuler*) ts->data;
  Vec       sol = ts->vec_sol,update = beuler->update;
  int       ierr,i,max_steps = ts->max_steps,its;
  
  *steps = -ts->steps;
  ierr = (*ts->monitor)(ts,ts->steps,ts->ptime,sol,ts->monP); CHKERRQ(ierr);

  for ( i=0; i<max_steps; i++ ) {
    ierr = SLESSolve(ts->sles,sol,update,&its); CHKERRQ(ierr);
    ierr = VecCopy(update,sol); CHKERRQ(ierr);
    ts->ptime += ts->time_step;
    ts->steps++;
    ierr = (*ts->monitor)(ts,ts->steps,ts->ptime,sol,ts->monP); CHKERRQ(ierr);
    if (ts->ptime > ts->max_time) break;
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
  Vec          sol = ts->vec_sol,update = beuler->update;
  int          ierr,i,max_steps = ts->max_steps,its;
  Scalar       mdt = -ts->time_step,one = 1.0;
  MatStructure str;

  *steps = -ts->steps;
  ierr = (*ts->monitor)(ts,ts->steps,ts->ptime,sol,ts->monP); CHKERRQ(ierr);

  for ( i=0; i<max_steps; i++ ) {
    /* 
        evaluate matrix function 
    */
    ierr = (*ts->rhsmatrix)(ts,ts->ptime,&ts->A,&ts->B,&str,ts->funP); CHKERRQ(ierr);
    ierr = MatScale(&mdt,ts->A); CHKERRQ(ierr);
    ierr = MatShift(&one,ts->A); CHKERRQ(ierr);
    if (ts->B != ts->A) {
      ierr = MatScale(&mdt,ts->B); CHKERRQ(ierr);
      ierr = MatShift(&one,ts->B); CHKERRQ(ierr);
    }

    ierr = SLESSetOperators(ts->sles,ts->A,ts->B,str); CHKERRQ(ierr);
    ierr = SLESSolve(ts->sles,sol,update,&its); CHKERRQ(ierr);
    ierr = VecCopy(update,sol); CHKERRQ(ierr);
    ierr = (*ts->monitor)(ts,ts->steps,ts->ptime,sol,ts->monP); CHKERRQ(ierr);
    ts->ptime += ts->time_step;
    ts->steps++;
    if (ts->ptime > ts->max_time) break;
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
  ierr = (*ts->monitor)(ts,ts->steps,ts->ptime,sol,ts->monP); CHKERRQ(ierr);

  for ( i=0; i<max_steps; i++ ) {
    ierr = VecCopy(sol,beuler->update); CHKERRQ(ierr);
    ierr = SNESSolve(ts->snes,beuler->update,&its); CHKERRQ(ierr);
    ierr = VecCopy(beuler->update,sol); CHKERRQ(ierr);
    ierr = (*ts->monitor)(ts,ts->steps,ts->ptime,sol,ts->monP); CHKERRQ(ierr);
    ts->ptime += ts->time_step;
    ts->steps++;
    if (ts->ptime > ts->max_time) break;
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
  PetscFree(beuler);
  return 0;
}
/*------------------------------------------------------------*/
/*
    This matrix shell multiply is for linear problems where the
  user does not supply a matrix.
*/

int TSBEulerMatMult_Linear(void *ctx,Vec x,Vec y)
{
  TS     ts = (TS) ctx;
  Scalar mdt = -ts->time_step;
  int    ierr;
 
  /* apply user provided function */
  ierr = TSComputeRHSFunction(ts,ts->ptime,x,y); CHKERRQ(ierr);
  /* shift and scale by 1 - dt*F */
  ierr = VecAYPX(&mdt,x,y); CHKERRQ(ierr);
  return 0;
}

/* 
    This defines the nonlinear equation that is to be solved with SNES

              U^{n+1} - dt*F(U^{n+1}) - U^{n}
*/
int TSBEulerFunction_Nonlinear(SNES snes,Vec x,Vec y,void *ctx)
{
  TS     ts = (TS) ctx;
  Scalar mdt = -ts->time_step,mone = -1.0;
  int    ierr;

  /* apply user provided function */
  ierr = TSComputeRHSFunction(ts,ts->ptime,x,y); CHKERRQ(ierr);
  /* shift and scale by x - dt*F(x) */
  ierr = VecAYPX(&mdt,x,y); CHKERRQ(ierr);
  /* subtract off old value */
  ierr = VecAXPY(&mone,ts->vec_sol,y); CHKERRQ(ierr);
  return 0;
}

/*
   This constructs the Jacobian needed for SNES 

             J = I - dt*J_{F}   where J_{F} is the given Jacobian of F.
*/
int TSBEulerJacobian_Nonlinear(SNES snes,Vec x,Mat *AA,Mat *BB,MatStructure *str,void *ctx)
{
  TS     ts = (TS) ctx;
  Mat    A,B;
  int    ierr;
  Scalar one = 1.0, mdt = -ts->time_step;

  /* construct users Jacobian */
  ierr = (*ts->rhsjacobian)(ts,ts->ptime,x,AA,BB,str,ts->jacP); CHKERRQ(ierr);
  A = *AA; B = *BB;
  /* shift and scale Jacobian */
  ierr = MatScale(&mdt,A); CHKERRQ(ierr);
  ierr = MatShift(&one,A); CHKERRQ(ierr);
  if (ts->B != ts->A) {
    ierr = MatScale(&mdt,B); CHKERRQ(ierr);
    ierr = MatShift(&one,B); CHKERRQ(ierr);
  }

  return 0;
}

/* ------------------------------------------------------------*/
static int TSSetUp_BEuler_Linear_Constant_Matrix(TS ts)
{
  TS_BEuler *beuler = (TS_BEuler*) ts->data;
  int       ierr;
  Scalar    mdt = -ts->time_step, one = 1.0;
  KSP       ksp;

  ierr = VecDuplicate(ts->vec_sol,&beuler->update); CHKERRQ(ierr);  
    
  /* build linear system to be solved */
  ierr = MatScale(&mdt,ts->A); CHKERRQ(ierr);
  ierr = MatShift(&one,ts->A); CHKERRQ(ierr);
  if (ts->A != ts->B) {
    ierr = MatScale(&mdt,ts->B); CHKERRQ(ierr);
    ierr = MatShift(&one,ts->B); CHKERRQ(ierr);
  }
  ierr = SLESSetOperators(ts->sles,ts->A,ts->B,MAT_SAME_NONZERO_PATTERN);CHKERRQ(ierr);
  ierr = SLESGetKSP(ts->sles,&ksp); CHKERRQ(ierr);
  ierr = KSPSetInitialGuessNonzero(ksp); CHKERRQ(ierr);
  return 0;
}

static int TSSetUp_BEuler_Linear_Variable_Matrix(TS ts)
{
  TS_BEuler *beuler = (TS_BEuler*) ts->data;
  int      ierr;

  ierr = VecDuplicate(ts->vec_sol,&beuler->update); CHKERRQ(ierr);  

  return 0;
}

static int TSSetUp_BEuler_Linear_No_Matrix(TS ts)
{
  TS_BEuler *beuler = (TS_BEuler*) ts->data;
  int      ierr,M;

  ierr = VecDuplicate(ts->vec_sol,&beuler->update); CHKERRQ(ierr);  

  ierr = VecGetSize(ts->vec_sol,&M); CHKERRQ(ierr);
  /* create shell matrix */
  ierr = MatCreateShell(ts->comm,M,M,ts,&ts->A); CHKERRQ(ierr);
  ierr = MatShellSetMult(ts->A,TSBEulerMatMult_Linear); CHKERRQ(ierr);
  /* no support yet for different B from A */
  ts->B = ts->A;
  ierr = SLESSetOperators(ts->sles,ts->A,ts->B,MAT_SAME_NONZERO_PATTERN);CHKERRQ(ierr);
  return 0;
}

static int TSSetUp_BEuler_Nonlinear_No_Jacobian(TS ts)
{
  TS_BEuler *beuler = (TS_BEuler*) ts->data;
  int       ierr;
  Mat       J;

  /*
      Create Jacobian (to be computed matrix free )
  */
  ierr = SNESDefaultMatrixFreeMatCreate(ts->snes,ts->vec_sol,&J); CHKERRQ(ierr);
  ierr = VecDuplicate(ts->vec_sol,&beuler->update); CHKERRQ(ierr);  
  ierr = SNESSetFunction(ts->snes,beuler->update,TSBEulerFunction_Nonlinear,ts);
         CHKERRQ(ierr);

  ts->B = ts->A = J;
  ierr = SNESSetJacobian(ts->snes,ts->A,ts->B,PETSC_NULL,ts);CHKERRQ(ierr);
  return 0;
}

static int TSSetUp_BEuler_Nonlinear_Jacobian(TS ts)
{
  TS_BEuler *beuler = (TS_BEuler*) ts->data;
  int       ierr;

  ierr = VecDuplicate(ts->vec_sol,&beuler->update); CHKERRQ(ierr);  
  ierr = SNESSetFunction(ts->snes,beuler->update,TSBEulerFunction_Nonlinear,ts);CHKERRQ(ierr);

  ierr = SNESSetJacobian(ts->snes,ts->A,ts->B,TSBEulerJacobian_Nonlinear,ts);CHKERRQ(ierr);
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

  ts->type 	      = 0;
  ts->destroy         = TSDestroy_BEuler;
  ts->printhelp       = TSPrintHelp_BEuler;
  ts->view            = TSView_BEuler;

  if (ts->problem_type == TS_LINEAR_CONSTANT_MATRIX) {
    ts->setup	        = TSSetUp_BEuler_Linear_Constant_Matrix;
    ts->step            = TSStep_BEuler_Linear_Constant_Matrix;
    ts->setfromoptions  = TSSetFromOptions_BEuler_Linear;
    ierr = SLESCreate(ts->comm,&ts->sles); CHKERRQ(ierr);
  } else if (ts->problem_type == TS_LINEAR_VARIABLE_MATRIX) {
    ts->setup	        = TSSetUp_BEuler_Linear_Variable_Matrix;
    ts->step            = TSStep_BEuler_Linear_Variable_Matrix;
    ts->setfromoptions  = TSSetFromOptions_BEuler_Linear;
    ierr = SLESCreate(ts->comm,&ts->sles); CHKERRQ(ierr);
  } else if (ts->problem_type == TS_LINEAR_NO_MATRIX) {
    ts->setup	        = TSSetUp_BEuler_Linear_No_Matrix;
    ts->step            = TSStep_BEuler_Linear_Constant_Matrix;
    ts->setfromoptions  = TSSetFromOptions_BEuler_Linear;
    ierr = SLESCreate(ts->comm,&ts->sles); CHKERRQ(ierr);
  } else if (ts->problem_type == TS_NONLINEAR_NO_JACOBIAN) {
    ts->setup	        = TSSetUp_BEuler_Nonlinear_No_Jacobian;
    ts->step            = TSStep_BEuler_Nonlinear;
    ts->setfromoptions  = TSSetFromOptions_BEuler_Nonlinear;
    ierr = SNESCreate(ts->comm,SNES_NONLINEAR_EQUATIONS,&ts->snes); CHKERRQ(ierr);
  } else if (ts->problem_type == TS_NONLINEAR_JACOBIAN) {
    ts->setup	        = TSSetUp_BEuler_Nonlinear_Jacobian;
    ts->step            = TSStep_BEuler_Nonlinear;
    ts->setfromoptions  = TSSetFromOptions_BEuler_Nonlinear;
    ierr = SNESCreate(ts->comm,SNES_NONLINEAR_EQUATIONS,&ts->snes); CHKERRQ(ierr);
  } else SETERRQ(1,"TSCreate_BEuler:No such problem");

  beuler   = PetscNew(TS_BEuler); CHKPTRQ(beuler);
  ts->data = (void *) beuler;

  return 0;
}
