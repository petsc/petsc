#ifndef lint
static char vcid[] = "$Id: none.c,v 1.4 1995/03/25 01:26:20 bsmith Exp curfman $";
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
  pc->view    = 0;
  return 0;
}
