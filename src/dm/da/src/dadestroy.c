#ifndef lint
static char vcid[] = "$Id: dadestroy.c,v 1.1 1996/01/30 04:27:59 bsmith Exp bsmith $";
#endif
 
/*
  Code for manipulating distributed regular arrays in parallel.
*/

#include "daimpl.h"    /*I   "da.h"   I*/

/*@C
   DADestroy - Destroy a distributed array.

   Input Parameter:
.  da - the distributed array to destroy 

.keywords: distributed array, destroy

.seealso: DACreate2d()
@*/
int DADestroy(DA da)
{
  PetscValidHeaderSpecific(da,DA_COOKIE);
  PLogObjectDestroy(da);
  PetscFree(da->idx);
  VecScatterDestroy(da->ltog);
  VecScatterDestroy(da->gtol);
  VecScatterDestroy(da->ltol);
  PetscHeaderDestroy(da);
  return 0;
}

