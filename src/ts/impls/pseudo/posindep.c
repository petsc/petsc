#ifndef lint
static char vcid[] = "$Id: posindep.c,v 1.5 1996/09/14 12:37:20 bsmith Exp bsmith $";
#endif
/*
       Code for Time Stepping with implicit backwards Euler.
*/
#include <math.h>
#include "src/ts/tsimpl.h"                /*I   "ts.h"   I*/
#include "pinclude/pviewer.h"


typedef struct {
  Vec  update;      /* work vector where new solution is formed */
  Vec  func;        /* work vector where F(t[i],u[i]) is stored */
  Vec  rhs;         /* work vector for RHS; vec_sol/dt */

  /* information used for Pseudo-timestepping */

  int    (*dt)(TS,double*,void*);
  void   *dtctx;
  double initial_fnorm,fnorm;  /* original and current norm of F(u) */

} TS_PosIndep;

/* ------------------------------------------------------------------------------*/
/*@C
    TSPseudoDefaultPosIndTimeStep - Default code to compute 
      pseudo timestepping. Use with TSPseudoSetPosIndTimeStep().

  Input Parameters:
.   ts - the time step context
.   dtctx - 

  Output Parameter:
.   newdt - the time step to use for the next step

@*/
int TSPseudoDefaultPosIndTimeStep(TS ts,double* newdt,void* dtctx)
{
  TS_PosIndep *posindep = (TS_PosIndep*) ts->data;
  int         ierr;

  ierr = TSComputeRHSFunction(ts,ts->ptime,ts->vec_sol,posindep->func);CHKERRQ(ierr);  
  ierr = VecNorm(posindep->func,NORM_2,&posindep->fnorm); CHKERRQ(ierr); 
  if (posindep->initial_fnorm == 0.0) {
    /* first time through so compute initial function norm */
    posindep->initial_fnorm = posindep->fnorm;
  }
  if (posindep->fnorm == 0.0) *newdt = 1.e12*1.1*ts->time_step; 
  else *newdt = 1.1*ts->time_step*posindep->initial_fnorm/posindep->fnorm;
  return 0;
}

/*@
    TSPseudoSetPosIndTimeStep - Sets the user routine to be
        called at each pseudo-time-step to update the time-step.

  Input Parameters:
.  ts - time step context
.  dt - function to compute timestep
.  ctx - [optional] context required by function

@*/
int TSPseudoSetPosIndTimeStep(TS ts,int (*dt)(TS,double*,void*),void* ctx)
{
  TS_PosIndep *posindep;

  PetscValidHeaderSpecific(ts,TS_COOKIE);
  if (ts->type != TS_PSEUDO_POSIND) return 0;
  posindep          = (TS_PosIndep*) ts->data;
  posindep->dt      = dt;
  posindep->dtctx   = ctx;
  return 0;
}

/*
    
*/
static int TSStep_PosIndep(TS ts,int *steps,double *time)
{
  Vec         sol = ts->vec_sol;
  int         ierr,i,max_steps = ts->max_steps,its;
  TS_PosIndep *posindep = (TS_PosIndep*) ts->data;
  
  *steps = -ts->steps;

  ierr = VecCopy(sol,posindep->update); CHKERRQ(ierr);
  for ( i=0; i<max_steps && ts->ptime < ts->max_time; i++ ) {
    ierr = (*posindep->dt)(ts,&ts->time_step, posindep->dtctx); CHKERRQ(ierr);
    ts->ptime += ts->time_step;
    ierr = SNESSolve(ts->snes,posindep->update,&its); CHKERRQ(ierr);
    ierr = VecCopy(posindep->update,sol); CHKERRQ(ierr);
    ts->steps++;
    ierr = TSMonitor(ts,ts->steps,ts->ptime,sol);CHKERRQ(ierr);
  }

  *steps += ts->steps;
  *time  = ts->ptime;
  return 0;
}

/*------------------------------------------------------------*/
static int TSDestroy_PosIndep(PetscObject obj )
{
  TS          ts = (TS) obj;
  TS_PosIndep *posindep = (TS_PosIndep*) ts->data;
  int         ierr;

  ierr = VecDestroy(posindep->update); CHKERRQ(ierr);
  if (posindep->func) {ierr = VecDestroy(posindep->func);CHKERRQ(ierr);}
  if (posindep->rhs) {ierr = VecDestroy(posindep->rhs);CHKERRQ(ierr);}
  if (ts->Ashell) {ierr = MatDestroy(ts->A); CHKERRQ(ierr);}
  PetscFree(posindep);
  return 0;
}


/*------------------------------------------------------------*/
/*
    This matrix shell multiply where user provided Shell matrix
*/

int TSPosIndepMatMult(Mat mat,Vec x,Vec y)
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

              (U^{n+1} - U^{n})/dt - F(U^{n+1})
*/
int TSPosIndepFunction(SNES snes,Vec x,Vec y,void *ctx)
{
  TS     ts = (TS) ctx;
  Scalar mdt = 1.0/ts->time_step,*unp1,*un,*Funp1;
  int    ierr,i,n;

  /* apply user provided function */
  ierr = TSComputeRHSFunction(ts,ts->ptime,x,y); CHKERRQ(ierr);
  /* (u^{n+1) - u^{n})/dt - F(u^{n+1}) */
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
int TSPosIndepJacobian(SNES snes,Vec x,Mat *AA,Mat *BB,MatStructure *str,void *ctx)
{
  TS      ts = (TS) ctx;
  int     ierr;
  Scalar  mone = -1.0, mdt = 1.0/ts->time_step;
  MatType mtype;

  /* construct users Jacobian */
  if (ts->rhsjacobian) {
    ierr = (*ts->rhsjacobian)(ts,ts->ptime,x,AA,BB,str,ts->jacP);CHKERRQ(ierr);
  }

  /* shift and scale Jacobian, if not a shell matrix */
  ierr = MatGetType(*AA,&mtype,PETSC_NULL);
  if (mtype != MATSHELL) {
    ierr = MatScale(&mone,*AA); CHKERRQ(ierr);
    ierr = MatShift(&mdt,*AA); CHKERRQ(ierr);
  }
  ierr = MatGetType(*BB,&mtype,PETSC_NULL);
  if (*BB != *AA && *str != SAME_PRECONDITIONER && mtype != MATSHELL) {
    ierr = MatScale(&mone,*BB); CHKERRQ(ierr);
    ierr = MatShift(&mdt,*BB); CHKERRQ(ierr);
  }

  return 0;
}


static int TSSetUp_PosIndep(TS ts)
{
  TS_PosIndep *posindep = (TS_PosIndep*) ts->data;
  int         ierr, M, m;

  ierr = VecDuplicate(ts->vec_sol,&posindep->update); CHKERRQ(ierr);  
  ierr = VecDuplicate(ts->vec_sol,&posindep->func); CHKERRQ(ierr);  
  ierr = SNESSetFunction(ts->snes,posindep->func,TSPosIndepFunction,ts);CHKERRQ(ierr);
  if (ts->Ashell) { /* construct new shell matrix */
    ierr = VecGetSize(ts->vec_sol,&M); CHKERRQ(ierr);
    ierr = VecGetLocalSize(ts->vec_sol,&m); CHKERRQ(ierr);
    ierr = MatCreateShell(ts->comm,m,M,M,M,ts,&ts->A); CHKERRQ(ierr);
    ierr = MatShellSetOperation(ts->A,MAT_MULT,(void*)TSPosIndepMatMult);CHKERRQ(ierr);
  }
  ierr = SNESSetJacobian(ts->snes,ts->A,ts->B,TSPosIndepJacobian,ts);CHKERRQ(ierr);
  return 0;
}
/*------------------------------------------------------------*/

int TSPseudoDefaultMonitor(TS ts, int step, double time,Vec v, void *ctx)
{
  TS_PosIndep *posindep = (TS_PosIndep*) ts->data;

  PetscPrintf(ts->comm,"TS %d dt %g time %g fnorm %g\n",step,ts->time_step,time,
              posindep->fnorm);
  return 0;
}


static int TSSetFromOptions_PosIndep(TS ts)
{
  int ierr,flg;

  ierr = SNESSetFromOptions(ts->snes); CHKERRQ(ierr);

  ierr = OptionsHasName(ts->prefix,"-ts_monitor",&flg); CHKERRQ(ierr);
  if (flg) {
    TSSetMonitor(ts,TSPseudoDefaultMonitor,0);
  }
  return 0;
}

static int TSPrintHelp_PosIndep(TS ts)
{

  return 0;
}

static int TSView_PosIndep(PetscObject obj,Viewer viewer)
{
  return 0;
}

/* ------------------------------------------------------------ */
int TSCreate_PosIndep(TS ts )
{
  TS_PosIndep *posindep;
  int         ierr;
  MatType     mtype;

  ts->type 	      = TS_PSEUDO_POSIND;
  ts->destroy         = TSDestroy_PosIndep;
  ts->printhelp       = TSPrintHelp_PosIndep;
  ts->view            = TSView_PosIndep;

  if (ts->problem_type == TS_LINEAR) {
    SETERRQ(1,"TSCreate_PosIndep:Only for nonlinear problems");
  }
  if (!ts->A) {
    SETERRQ(1,"TSCreate_PosIndep:Must set Jacobian");
  }
  ierr = MatGetType(ts->A,&mtype,PETSC_NULL);
  if (mtype == MATSHELL) {
    ts->Ashell = ts->A;
  }
  ts->setup           = TSSetUp_PosIndep;  
  ts->step            = TSStep_PosIndep;
  ts->setfromoptions  = TSSetFromOptions_PosIndep;
  ierr = SNESCreate(ts->comm,SNES_NONLINEAR_EQUATIONS,&ts->snes);CHKERRQ(ierr);

  posindep   = PetscNew(TS_PosIndep); CHKPTRQ(posindep);
  PetscMemzero(posindep,sizeof(TS_PosIndep));
  ts->data = (void *) posindep;

  return 0;
}





