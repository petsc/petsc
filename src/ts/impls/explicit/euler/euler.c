#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: euler.c,v 1.16 1999/05/04 20:36:43 balay Exp balay $";
#endif
/*
       Code for Timestepping with explicit Euler.
*/
#include "src/ts/tsimpl.h"                /*I   "ts.h"   I*/

typedef struct {
  Vec update;     /* work vector where F(t[i],u[i]) is stored */
} TS_Euler;

#undef __FUNC__  
#define __FUNC__ "TSSetUp_Euler"
static int TSSetUp_Euler(TS ts)
{
  TS_Euler *euler = (TS_Euler*) ts->data;
  int      ierr;

  PetscFunctionBegin;
  ierr = VecDuplicate(ts->vec_sol,&euler->update);CHKERRQ(ierr);  
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "TSStep_Euler"
static int TSStep_Euler(TS ts,int *steps,double *time)
{
  TS_Euler *euler = (TS_Euler*) ts->data;
  Vec      sol = ts->vec_sol,update = euler->update;
  int      ierr,i,max_steps = ts->max_steps;
  Scalar   dt = ts->time_step;
  
  PetscFunctionBegin;
  *steps = -ts->steps;
  ierr = TSMonitor(ts,ts->steps,ts->ptime,sol);CHKERRQ(ierr);

  for ( i=0; i<max_steps; i++ ) {
    ts->ptime += ts->time_step;
    ierr = TSComputeRHSFunction(ts,ts->ptime,sol,update);CHKERRQ(ierr);
    ierr = VecAXPY(&dt,update,sol);CHKERRQ(ierr);
    ts->steps++;
    ierr = TSMonitor(ts,ts->steps,ts->ptime,sol);CHKERRQ(ierr);
    if (ts->ptime > ts->max_time) break;
  }

  *steps += ts->steps;
  *time  = ts->ptime;
  PetscFunctionReturn(0);
}
/*------------------------------------------------------------*/
#undef __FUNC__  
#define __FUNC__ "TSDestroy_Euler"
static int TSDestroy_Euler(TS ts)
{
  TS_Euler *euler = (TS_Euler*) ts->data;
  int      ierr;

  PetscFunctionBegin;
  VecDestroy(euler->update);
  ierr = PetscFree(euler);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
/*------------------------------------------------------------*/

#undef __FUNC__  
#define __FUNC__ "TSSetFromOptions_Euler"
static int TSSetFromOptions_Euler(TS ts)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "TSPrintHelp_Euler"
static int TSPrintHelp_Euler(TS ts,char *p)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "TSView_Euler"
static int TSView_Euler(TS ts,Viewer viewer)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

/* ------------------------------------------------------------ */
EXTERN_C_BEGIN
#undef __FUNC__  
#define __FUNC__ "TSCreate_Euler"
int TSCreate_Euler(TS ts )
{
  TS_Euler *euler;

  PetscFunctionBegin;
  ts->setup	      = TSSetUp_Euler;
  ts->step            = TSStep_Euler;
  ts->destroy         = TSDestroy_Euler;
  ts->printhelp       = TSPrintHelp_Euler;
  ts->setfromoptions  = TSSetFromOptions_Euler;
  ts->view            = TSView_Euler;

  euler    = PetscNew(TS_Euler);CHKPTRQ(euler);
  PLogObjectMemory(ts,sizeof(TS_Euler));
  ts->data = (void *) euler;

  PetscFunctionReturn(0);
}
EXTERN_C_END




