#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: dadestroy.c,v 1.11 1997/07/09 21:00:44 balay Exp bsmith $";
#endif
 
/*
  Code for manipulating distributed regular arrays in parallel.
*/

#include "src/da/daimpl.h"    /*I   "da.h"   I*/

#undef __FUNC__  
#define __FUNC__ "DADestroy"
/*@C
   DADestroy - Destroys a distributed array.

   Input Parameter:
.  da - the distributed array to destroy 

.keywords: distributed array, destroy

.seealso: DACreate1d(), DACreate2d(), DACreate3d()
@*/int DADestroy(DA da)
{
  int ierr;

  PetscValidHeaderSpecific(da,DA_COOKIE);
  if (--da->refct > 0) return 0;

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

