#ifndef lint
static char vcid[] = "$Id: exnonlin.c,v 1.1 1996/01/05 23:27:00 bsmith Exp bsmith $";
#endif
/*
       Code for Time Stepping for explict non-linear problems.
*/
#include <math.h>
#include "tsimpl.h"                /*I   "ts.h"   I*/
#include "pinclude/pviewer.h"


static int TSSetUp_ExNonLin(TS ts)
{
  Vec sol = ts->vec_sol;
  int ierr;

  /* form initial functions */
  ierr = (*ts->computeinitial)(ts,sol,ts->gusP); CHKERRQ(ierr);

  


  return 0;
}

static int TSStep_ExNonLin(TS ts,int *steps)
{
  Vec sol = ts->vec_sol;
  int ierr;

  


  return 0;
}
/*------------------------------------------------------------*/
static int TSDestroy_ExNonLin(PetscObject obj )
{
  TS ts = (TS) ts;

  return 0;
}
/*------------------------------------------------------------*/

static int TSSetFromOptions_ExNonLin(TS ts)
{

  return 0;
}

static int TSPrintHelp_ExNonLin(TS ts)
{

  return 0;
}

static int TSView_ExNonLin(PetscObject obj,Viewer viewer)
{
  TS    ts = (TS)obj;

  return 0;
}

/* ------------------------------------------------------------ */
int TSCreate_ExNonLin(TS ts )
{
  ts->type 	      = 0;
  ts->setup	      = TSSetUp_ExNonLin;
  ts->step            = TSStep_ExNonLin;
  ts->destroy         = TSDestroy_ExNonLin;
  ts->printhelp       = TSPrintHelp_ExNonLin;
  ts->setfromoptions  = TSSetFromOptions_ExNonLin;
  ts->view            = TSView_ExNonLin;

  return 0;
}
