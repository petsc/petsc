#ifndef lint
static char vcid[] = "$Id: none.c,v 1.5 1995/07/26 02:26:00 curfman Exp bsmith $";
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
  pc->type    = PCNONE;
  pc->apply   = PCApply_None;
  pc->destroy = 0;
  pc->setup   = 0;
  pc->data    = 0;
  pc->view    = 0;
  return 0;
}
