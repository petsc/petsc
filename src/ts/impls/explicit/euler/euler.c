
#ifndef lint
static char vcid[] = "$Id: euler.c,v 1.1 1996/01/06 16:31:06 bsmith Exp bsmith $";
#endif
/*
       Code for Time Stepping with explicit Euler.
*/
#include <math.h>
#include "tsimpl.h"                /*I   "ts.h"   I*/
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

static int TSStep_Euler(TS ts,int *steps,Scalar *time)
{
  TS_Euler *euler = (TS_Euler*) ts->data;
  Vec      sol = ts->vec_sol,update = euler->update;
  int      ierr,i,max_steps = ts->max_steps;
  
  *steps = -ts->steps;
  ierr = (*ts->monitor)(ts,ts->steps,ts->ptime,sol,ts->monP); CHKERRQ(ierr);

  for ( i=0; i<max_steps; i++ ) {
    ierr = TSComputeRHSFunction(ts,ts->ptime,sol,update); CHKERRQ(ierr);
    ierr = VecAXPY(&ts->time_step,update,sol); CHKERRQ(ierr);
    ts->ptime += ts->time_step;
    ts->steps++;
    ierr = (*ts->monitor)(ts,ts->steps,ts->ptime,sol,ts->monP); CHKERRQ(ierr);
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

  ts->type 	      = 0;
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
