#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: dadestroy.c,v 1.23 1999/03/01 04:58:26 bsmith Exp bsmith $";
#endif
 
/*
  Code for manipulating distributed regular arrays in parallel.
*/

#include "src/dm/da/daimpl.h"    /*I   "da.h"   I*/

#undef __FUNC__  
#define __FUNC__ "DADestroy"
/*@C
   DADestroy - Destroys a distributed array.

   Collective on DA

   Input Parameter:
.  da - the distributed array to destroy 

.keywords: distributed array, destroy

.seealso: DACreate1d(), DACreate2d(), DACreate3d()
@*/
int DADestroy(DA da)
{
  int ierr,i;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(da,DA_COOKIE);
  if (--da->refct > 0) PetscFunctionReturn(0);
  /*
         Need this test because the da references the vectors that 
     reference the da, so destroying the da calls destroy on the 
     vectors that cause another destroy on the da
  */
  if (da->refct < 0) PetscFunctionReturn(0);

  PLogObjectDestroy(da);
  PetscFree(da->idx);
  ierr = VecScatterDestroy(da->ltog);CHKERRQ(ierr);
  ierr = VecScatterDestroy(da->gtol);CHKERRQ(ierr);
  ierr = VecScatterDestroy(da->ltol);CHKERRQ(ierr);
  if (!da->globalused) {
    ierr = VecDestroy(da->global);CHKERRQ(ierr);
  }
  if (!da->localused) {
    ierr = VecDestroy(da->local);CHKERRQ(ierr);
  }
  if (da->natural){
    ierr = VecDestroy(da->natural);CHKERRQ(ierr);
  }
  ierr = AODestroy(da->ao); CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingDestroy(da->ltogmap); CHKERRQ(ierr);

  if (da->lx) PetscFree(da->lx);  
  if (da->ly) PetscFree(da->ly);  
  if (da->lz) PetscFree(da->lz);  

  for ( i=0; i<da->w; i++ ) {
    if (da->fieldname[i]) PetscFree(da->fieldname[i]);
  }
  PetscFree(da->fieldname);

  if (da->coordinates) ierr = VecDestroy(da->coordinates);CHKERRQ(ierr);
  if (da->gtog1) PetscFree(da->gtog1);
  if (da->dfshell) {ierr = DFShellDestroy(da->dfshell);CHKERRQ(ierr);}
  PetscHeaderDestroy(da);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "DAGetISLocalToGlobalMapping"
/*@C
   DAGetISLocalToGlobalMapping - Accesses the local to global mapping in a DA.

   Not Collective

   Input Parameter:
.  da - the distributed array to destroy 

   Output Parameter:
.  ltog - the mapping

.keywords: distributed array, destroy

.seealso: DACreate1d(), DACreate2d(), DACreate3d()
@*/
int DAGetISLocalToGlobalMapping(DA da,ISLocalToGlobalMapping *map)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(da,DA_COOKIE);
  *map = da->ltogmap;
  PetscFunctionReturn(0);
}
