
#ifndef lint
static char vcid[] = "$Id: euler.c,v 1.4 1996/08/08 14:45:37 bsmith Exp curfman $";
#endif
/*
       Code for Timestepping with explicit Euler.
*/
#include <math.h>
#include "src/ts/tsimpl.h"                /*I   "ts.h"   I*/
#include "pinclude/pviewer.h"


typedef struct {
  Vec update;     /* work vector where F(t[i],u[i]) is stored */
} TS_Euler;

static int TSSetUp_Euler(TS ts)
{
  TS_Euler *euler = (TS_Euler*) ts->data;
  int      ierr;

  ierr = VecDuplicate(ts->vec_sol,&euler->update); CHKERRQ(ierr);  
  return 0;
}

static int TSStep_Euler(TS ts,int *steps,double *time)
{
  TS_Euler *euler = (TS_Euler*) ts->data;
  Vec      sol = ts->vec_sol,update = euler->update;
  int      ierr,i,max_steps = ts->max_steps;
  Scalar   dt = ts->time_step;
  
  *steps = -ts->steps;
  ierr = TSMonitor(ts,ts->steps,ts->ptime,sol); CHKERRQ(ierr);

  for ( i=0; i<max_steps; i++ ) {
    ts->ptime += ts->time_step;
    ierr = TSComputeRHSFunction(ts,ts->ptime,sol,update); CHKERRQ(ierr);
    ierr = VecAXPY(&dt,update,sol); CHKERRQ(ierr);
    ts->steps++;
    ierr = TSMonitor(ts,ts->steps,ts->ptime,sol); CHKERRQ(ierr);
    if (ts->ptime > ts->max_time) break;
  }

  *steps += ts->steps;
  *time  = ts->ptime;
  return 0;
}
/*------------------------------------------------------------*/
static int TSDestroy_Euler(PetscObject obj )
{
  TS       ts = (TS) obj;
  TS_Euler *euler = (TS_Euler*) ts->data;

  VecDestroy(euler->update);
  PetscFree(euler);
  return 0;
}
/*------------------------------------------------------------*/

static int TSSetFromOptions_Euler(TS ts)
{

  return 0;
}

static int TSPrintHelp_Euler(TS ts)
{

  return 0;
}

static int TSView_Euler(PetscObject obj,Viewer viewer)
{
  return 0;
}

/* ------------------------------------------------------------ */
int TSCreate_Euler(TS ts )
{
  TS_Euler *euler;

  ts->type 	      = TS_EULER;
  ts->setup	      = TSSetUp_Euler;
  ts->step            = TSStep_Euler;
  ts->destroy         = TSDestroy_Euler;
  ts->printhelp       = TSPrintHelp_Euler;
  ts->setfromoptions  = TSSetFromOptions_Euler;
  ts->view            = TSView_Euler;

  euler    = PetscNew(TS_Euler); CHKPTRQ(euler);
  ts->data = (void *) euler;

  return 0;
}





