#ifndef lint
static char vcid[] = "$Id: dadestroy.c,v 1.3 1996/04/14 00:49:26 curfman Exp curfman $";
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
  int ierr;
  PetscValidHeaderSpecific(da,DA_COOKIE);
  PLogObjectDestroy(da);
  PetscFree(da->idx);
  VecScatterDestroy(da->ltog);
  VecScatterDestroy(da->gtol);
  VecScatterDestroy(da->ltol);
  if (da->gtog1) PetscFree(da->gtog1);
  if (da->dfshell) {ierr = DFShellDestroy(da->dfshell); CHKERRQ(ierr);}
  PetscHeaderDestroy(da);
  return 0;
}

