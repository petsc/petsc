#ifndef lint
static char vcid[] = "$Id: none.c,v 1.3 1995/03/06 04:13:21 bsmith Exp bsmith $";
#endif
/*

*/
#include "pcimpl.h"

int PCApply_None(PC ptr,Vec x,Vec y)
{
  return VecCopy(x,y);
}

int PCCreate_None(PC pc)
{
  pc->apply   = PCApply_None;
  pc->destroy = 0;
  pc->setup   = 0;
  pc->data    = 0;
  return 0;
}
