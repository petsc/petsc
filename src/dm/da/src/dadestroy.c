#ifndef lint
static char vcid[] = "$Id: da2.c,v 1.32 1996/01/27 04:56:41 bsmith Exp $";
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
  PETSCVALIDHEADERSPECIFIC(da,DA_COOKIE);
  PLogObjectDestroy(da);
  PetscFree(da->idx);
  VecScatterDestroy(da->ltog);
  VecScatterDestroy(da->gtol);
  VecScatterDestroy(da->ltol);
  PetscHeaderDestroy(da);
  return 0;
}

