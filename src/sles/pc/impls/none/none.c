#ifndef lint
static char vcid[] = "$Id: none.c,v 1.8 1995/12/12 22:47:18 curfman Exp curfman $";
#endif
/*
    Identity preconditioner, simply copies vector x to y.
*/
#include "pcimpl.h"          /*I "pc.h" I*/

int PCApply_None(PC ptr,Vec x,Vec y)
{
  return VecCopy(x,y);
}

int PCCreate_None(PC pc)
{
  pc->type           = PCNONE;
  pc->apply          = PCApply_None;
  pc->destroy        = 0;
  pc->setup          = 0;
  pc->data           = 0;
  pc->view           = 0;
  pc->applysymmleft  = PCApply_None;
  pc->applysymmright = PCApply_None;
  return 0;
}
