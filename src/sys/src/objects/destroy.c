#ifndef lint
static char vcid[] = "$Id: pbvec.c,v 1.7 1995/03/06 03:56:21 bsmith Exp bsmith $";
#endif
#include "ptscimpl.h"

int PetscDestroy(PetscObject obj)
{
  if (obj->destroy) return (*obj->destroy)(obj);
  return 0;
}
