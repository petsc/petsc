
#include "ptscimpl.h"

int PetscDestroy(PetscObject obj)
{
  if (obj->destroy) return (*obj->destroy)(obj);
  return 0;
}
