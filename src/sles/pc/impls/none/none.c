#ifndef lint
static char vcid[] = "$Id: none.c,v 1.10 1996/03/04 05:15:23 bsmith Exp bsmith $";
#endif
/*
    Identity preconditioner, simply copies vector x to y.
*/
#include "pcimpl.h"          /*I "pc.h" I*/

int PCApply_None(PC pc,Vec x,Vec y)
{
  int ierr;
  PLogEventBegin(PC_Apply,pc,x,y,0);
  ierr = VecCopy(x,y); CHKERRQ(ierr);
  PLogEventEnd(PC_Apply,pc,x,y,0);
  return 0;
}

int PCCreate_None(PC pc)
{
  pc->type           = PCNONE;
  pc->apply          = PCApply_None;
  pc->destroy        = 0;
  pc->setup          = 0;
  pc->data           = 0;
  pc->view           = 0;
  pc->applysymmetricleft  = PCApply_None;
  pc->applysymmetricright = PCApply_None;
  return 0;
}
