#ifndef lint
static char vcid[] = "$Id: none.c,v 1.13 1996/12/17 17:12:03 balay Exp balay $";
#endif
/*
    Identity preconditioner, simply copies vector x to y.
*/
#include "src/pc/pcimpl.h"          /*I "pc.h" I*/

#undef __FUNC__  
#define __FUNC__ "PCApply_None"
int PCApply_None(PC pc,Vec x,Vec y)
{
  int ierr;
  PLogEventBegin(PC_Apply,pc,x,y,0);
  ierr = VecCopy(x,y); CHKERRQ(ierr);
  PLogEventEnd(PC_Apply,pc,x,y,0);
  return 0;
}

#undef __FUNC__  
#define __FUNC__ "PCCreate_None"
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
