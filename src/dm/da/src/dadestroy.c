#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: dadestroy.c,v 1.10 1997/02/22 02:29:24 bsmith Exp balay $";
#endif
 
/*
  Code for manipulating distributed regular arrays in parallel.
*/

#include "src/da/daimpl.h"    /*I   "da.h"   I*/

#undef __FUNC__  
#define __FUNC__ "DADestroy" /* ADIC Ignore */
/*@C
   DADestroy - Destroys a distributed array.

   Input Parameter:
.  da - the distributed array to destroy 

.keywords: distributed array, destroy

.seealso: DACreate1d(), DACreate2d(), DACreate3d()
@*/
int DADestroy(DA da)
{
  int ierr;

  PetscValidHeaderSpecific(da,DA_COOKIE);
  PLogObjectDestroy(da);
  PetscFree(da->idx);
  ierr = VecScatterDestroy(da->ltog);CHKERRQ(ierr);
  ierr = VecScatterDestroy(da->gtol);CHKERRQ(ierr);
  ierr = VecScatterDestroy(da->ltol);CHKERRQ(ierr);
  ierr = AODestroy(da->ao); CHKERRQ(ierr);
  if (da->gtog1) PetscFree(da->gtog1);
  if (da->dfshell) {ierr = DFShellDestroy(da->dfshell); CHKERRQ(ierr);}
  PetscHeaderDestroy(da);
  return 0;
}

