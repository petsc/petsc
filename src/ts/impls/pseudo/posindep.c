#ifndef lint
static char vcid[] = "$Id: posindep.c,v 1.3 1996/09/30 20:23:41 curfman Exp bsmith $";
#endif
/*
       Code for Timestepping with implicit backwards Euler.
*/
#include <math.h>
#include "src/ts/tsimpl.h"                /*I   "ts.h"   I*/
#include "pinclude/pviewer.h"


typedef struct {
  Vec  update;      /* work vector where new solution is formed */
  Vec  func;        /* work vector where F(t[i],u[i]) is stored */
  Vec  rhs;         /* work vector for RHS; vec_sol/dt */

  /* information used for Pseudo-timestepping */

  int    (*dt)(TS,double*,void*);              /* compute next timestep, and related context */
  void   *dtctx;              
  int    (*verify)(TS,Vec,void*,double*,int*); /* verify previous timestep and related context */
  void   *verifyctx;     

  double initial_fnorm,fnorm;                  /* original and current norm of F(u) */

  double dt_increment;        /* scaling that dt is incremented each time-step */
} TS_Pseudo;

/* ------------------------------------------------------------------------------*/
/*@C
   TSPseudoDefaultTimeStep - Default code to compute pseudo-timestepping.
   Use with TSPseudoSetTimeStep().

   Input Parameters:
.  ts - the timestep context
.  dtctx - unused timestep context

   Output Parameter:
.  newdt - the timestep to use for the next step

.keywords: timestep, pseudo, default

.seealso: TSPseudoSetTimeStep()
@*/
int TSPseudoDefaultTimeStep(TS ts,double* newdt,void* dtctx)
{
  TS_Pseudo *pseudo = (TS_Pseudo*) ts->data;
  double    inc = pseudo->dt_increment;
  int       ierr;

  ierr = TSComputeRHSFunction(ts,ts->ptime,ts->vec_sol,pseudo->func);CHKERRQ(ierr);  
  ierr = VecNorm(pseudo->func,NORM_2,&pseudo->fnorm); CHKERRQ(ierr); 
  if (pseudo->initial_fnorm == 0.0) {
    /* first time through so compute initial function norm */
    pseudo->initial_fnorm = pseudo->fnorm;
  }
  if (pseudo->fnorm == 0.0) *newdt = 1.e12*inc*ts->time_step; 
  else                      *newdt = inc*ts->time_step*pseudo->initial_fnorm/pseudo->fnorm;
  return 0;
}

/*@
   TSPseudoSetTimeStep - Sets the user-defined routine to be
   called at each pseudo-timestep to update the timestep.

   Input Parameters:
.  ts - timestep context
.  dt - function to compute timestep
.  ctx - [optional] context required by function

.keywords: timestep, pseudo, set

.seealso: TSPseudoDefaultTimeStep()
@*/
int TSPseudoSetTimeStep(TS ts,int (*dt)(TS,double*,void*),void* ctx)
{
  TS_Pseudo *pseudo;

  PetscValidHeaderSpecific(ts,TS_COOKIE);
  if (ts->type != TS_PSEUDO) return 0;

  pseudo          = (TS_Pseudo*) ts->data;
  pseudo->dt      = dt;
  pseudo->dtctx   = ctx;
  return 0;
}

/*@
    TSPseudoComputeTimeStep - Computes the next timestep for a currently running
    pseudo-timestepping.

    Input Parameter:
.   ts - timestep context

    Output Parameter:
.   dt - newly computed timestep

.keywords: timestep, pseudo, compute
@*/
int TSPseudoComputeTimeStep(TS ts,double *dt)
{
  TS_Pseudo *pseudo = (TS_Pseudo*) ts->data;
  int       ierr;

  PLogEventBegin(TS_PseudoComputeTimeStep,ts,0,0,0);
  ierr = (*pseudo->dt)(ts,dt, pseudo->dtctx); CHKERRQ(ierr);
  PLogEventEnd(TS_PseudoComputeTimeStep,ts,0,0,0);
  return 0;
}


/* ------------------------------------------------------------------------------*/
/*@C
   TSPseudoDefaultVerifyTimeStep - Default code to verify last timestep.

   Input Parameters:
.  ts - the timestep context
.  dtctx - unused timestep context

   Output Parameter:
.  newdt - the timestep to use for the next step

@*/
int TSPseudoDefaultVerifyTimeStep(TS ts,Vec update,void* dtctx,double* newdt,int *flag)
{
  *flag = 1;
  return 0;
}

/*@
   TSPseudoSetVerifyTimeStep - Sets the user routine to verify quality of last timestep.

   Input Parameters:
.  ts - timestep context
.  dt - function to verify
.  ctx - [optional] context required by function

@*/
int TSPseudoSetVerifyTimeStep(TS ts,int (*dt)(TS,Vec,void*,double*,int*),void* ctx)
{
  TS_Pseudo *pseudo;

  PetscValidHeaderSpecific(ts,TS_COOKIE);
  if (ts->type != TS_PSEUDO) return 0;

  pseudo              = (TS_Pseudo*) ts->data;
  pseudo->verify      = dt;
  pseudo->verifyctx   = ctx;
  return 0;
}

/*@
    TSPseudoVerifyTimeStep - Verifies that the last timestep was OK.

    Input Parameters:
.   ts - timestep context
.   update - latest solution

    Output Parameters:
.   dt - newly computed timestep (if it had to shrink)
.   flag - indicates if current timestep was ok

@*/
int TSPseudoVerifyTimeStep(TS ts,Vec update,double *dt,int *flag)
{
  TS_Pseudo *pseudo = (TS_Pseudo*) ts->data;
  int       ierr;

  if (!pseudo->verify) {*flag = 1; return 0;}

  ierr = (*pseudo->verify)(ts,update,pseudo->verifyctx,dt,flag ); CHKERRQ(ierr);

  return 0;
}

/* --------------------------------------------------------------------------------*/

static int TSStep_Pseudo(TS ts,int *steps,double *time)
{
  Vec       sol = ts->vec_sol;
  int       ierr,i,max_steps = ts->max_steps,its,ok;
  TS_Pseudo *pseudo = (TS_Pseudo*) ts->data;
  double    current_time_step;
  
  *steps = -ts->steps;

  ierr = VecCopy(sol,pseudo->update); CHKERRQ(ierr);
  for ( i=0; i<max_steps && ts->ptime < ts->max_time; i++ ) {
    ierr = TSPseudoComputeTimeStep(ts,&ts->time_step); CHKERRQ(ierr);
    current_time_step = ts->time_step;
    while (1) {
      ts->ptime  += current_time_step;
      ierr = SNESSolve(ts->snes,pseudo->update,&its); CHKERRQ(ierr);
      ierr = TSPseudoVerifyTimeStep(ts,pseudo->update,&ts->time_step,&ok); CHKERRQ(ierr);
      if (ok) break;
      ts->ptime        -= current_time_step;
      current_time_step = ts->time_step;
    }
    ierr = VecCopy(pseudo->update,sol); CHKERRQ(ierr);
    ts->steps++;
    ierr = TSMonitor(ts,ts->steps,ts->ptime,sol);CHKERRQ(ierr);
  }

  *steps += ts->steps;
  *time  = ts->ptime;
  return 0;
}

/*------------------------------------------------------------*/
static int TSDestroy_Pseudo(PetscObject obj )
{
  TS        ts = (TS) obj;
  TS_Pseudo *pseudo = (TS_Pseudo*) ts->data;
  int       ierr;

  ierr = VecDestroy(pseudo->update); CHKERRQ(ierr);
  if (pseudo->func) {ierr = VecDestroy(pseudo->func);CHKERRQ(ierr);}
  if (pseudo->rhs)  {ierr = VecDestroy(pseudo->rhs);CHKERRQ(ierr);}
  if (ts->Ashell)   {ierr = MatDestroy(ts->A); CHKERRQ(ierr);}
  PetscFree(pseudo);
  return 0;
}


/*------------------------------------------------------------*/
/*
    This matrix shell multiply where user provided Shell matrix
*/

int TSPseudoMatMult(Mat mat,Vec x,Vec y)
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
int TSPseudoFunction(SNES snes,Vec x,Vec y,void *ctx)
{
  TS     ts = (TS) ctx;
  Scalar mdt = 1.0/ts->time_step,*unp1,*un,*Funp1;
  int    ierr,i,n;

  /* apply user provided function */
  ierr = TSComputeRHSFunction(ts,ts->ptime,x,y); CHKERRQ(ierr);
  /* compute (u^{n+1) - u^{n})/dt - F(u^{n+1}) */
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
int TSPseudoJacobian(SNES snes,Vec x,Mat *AA,Mat *BB,MatStructure *str,void *ctx)
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


static int TSSetUp_Pseudo(TS ts)
{
  TS_Pseudo *pseudo = (TS_Pseudo*) ts->data;
  int         ierr, M, m;

  ierr = VecDuplicate(ts->vec_sol,&pseudo->update); CHKERRQ(ierr);  
  ierr = VecDuplicate(ts->vec_sol,&pseudo->func); CHKERRQ(ierr);  
  ierr = SNESSetFunction(ts->snes,pseudo->func,TSPseudoFunction,ts);CHKERRQ(ierr);
  if (ts->Ashell) { /* construct new shell matrix */
    ierr = VecGetSize(ts->vec_sol,&M); CHKERRQ(ierr);
    ierr = VecGetLocalSize(ts->vec_sol,&m); CHKERRQ(ierr);
    ierr = MatCreateShell(ts->comm,m,M,M,M,ts,&ts->A); CHKERRQ(ierr);
    ierr = MatShellSetOperation(ts->A,MATOP_MULT,(void*)TSPseudoMatMult);CHKERRQ(ierr);
  }
  ierr = SNESSetJacobian(ts->snes,ts->A,ts->B,TSPseudoJacobian,ts);CHKERRQ(ierr);
  return 0;
}
/*------------------------------------------------------------*/

int TSPseudoDefaultMonitor(TS ts, int step, double time,Vec v, void *ctx)
{
  TS_Pseudo *pseudo = (TS_Pseudo*) ts->data;

  PetscPrintf(ts->comm,"TS %d dt %g time %g fnorm %g\n",step,ts->time_step,time,pseudo->fnorm);
  return 0;
}

static int TSSetFromOptions_Pseudo(TS ts)
{
  int    ierr,flg;
  double inc;

  ierr = SNESSetFromOptions(ts->snes); CHKERRQ(ierr);

  ierr = OptionsHasName(ts->prefix,"-ts_monitor",&flg); CHKERRQ(ierr);
  if (flg) {
    ierr = TSSetMonitor(ts,TSPseudoDefaultMonitor,0); CHKERRQ(ierr);
  }
  ierr = OptionsGetDouble(ts->prefix,"-ts_pseudo_increment",&inc,&flg);  CHKERRQ(ierr);
  if (flg) {
    ierr = TSPseudoSetTimeStepIncrement(ts,inc);  CHKERRQ(ierr);
  }
  return 0;
}

static int TSPrintHelp_Pseudo(TS ts)
{

  return 0;
}

static int TSView_Pseudo(PetscObject obj,Viewer viewer)
{
  return 0;
}

/* ------------------------------------------------------------ */
int TSCreate_Pseudo(TS ts )
{
  TS_Pseudo *pseudo;
  int       ierr;
  MatType   mtype;

  ts->type 	      = TS_PSEUDO;
  ts->destroy         = TSDestroy_Pseudo;
  ts->printhelp       = TSPrintHelp_Pseudo;
  ts->view            = TSView_Pseudo;

  if (ts->problem_type == TS_LINEAR) {
    SETERRQ(1,"TSCreate_Pseudo:Only for nonlinear problems");
  }
  if (!ts->A) {
    SETERRQ(1,"TSCreate_Pseudo:Must set Jacobian");
  }
  ierr = MatGetType(ts->A,&mtype,PETSC_NULL);
  if (mtype == MATSHELL) {
    ts->Ashell = ts->A;
  }
  ts->setup           = TSSetUp_Pseudo;  
  ts->step            = TSStep_Pseudo;
  ts->setfromoptions  = TSSetFromOptions_Pseudo;

  /* create the required nonlinear solver context */
  ierr = SNESCreate(ts->comm,SNES_NONLINEAR_EQUATIONS,&ts->snes);CHKERRQ(ierr);

  pseudo   = PetscNew(TS_Pseudo); CHKPTRQ(pseudo);
  PetscMemzero(pseudo,sizeof(TS_Pseudo));
  ts->data = (void *) pseudo;

  pseudo->dt_increment = 1.1;
  pseudo->dt           = TSPseudoDefaultTimeStep;
  return 0;
}


/*@
      TSPseudoSetTimeStepIncrement - Sets the scaling increment applied to 
         dt when using the TSPseudoDefaultTimeStep() routine.

  Input Parameters:
.   ts - the timestep context
.   inc - the scaling factor >= 1.0

   Options Database Key:
$  -ts_pseudo_increment <increment>

.seealso: TSPseudoSetTimeStep(), TSPseudoDefaultTimeStep()
@*/
int TSPseudoSetTimeStepIncrement(TS ts,double inc)
{
  TS_Pseudo *pseudo;

  PetscValidHeaderSpecific(ts,TS_COOKIE);
  if (ts->type != TS_PSEUDO) return 0;

  pseudo               = (TS_Pseudo*) ts->data;
  pseudo->dt_increment = inc;
  return 0;
}



