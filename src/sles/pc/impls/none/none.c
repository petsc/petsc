#ifndef lint
static char vcid[] = "$Id: $";
#endif
/*

*/
#include "pcimpl.h"

int PCiNoneApply(PC ptr,Vec x,Vec y)
{
  return VecCopy(x,y);
}

int PCiNoneCreate(PC pc)
{
  pc->apply   = PCiNoneApply;
  pc->destroy = 0;
  pc->setup   = 0;
  pc->data    = 0;
  return 0;
}
