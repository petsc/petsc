/*$Id: dadestroy.c,v 1.40 2000/09/13 03:13:00 bsmith Exp bsmith $*/
 
/*
  Code for manipulating distributed regular arrays in parallel.
*/

#include "src/dm/da/daimpl.h"    /*I   "petscda.h"   I*/

#undef __FUNC__  
#define __FUNC__ "DADestroy"
/*@C
   DADestroy - Destroys a distributed array.

   Collective on DA

   Input Parameter:
.  da - the distributed array to destroy 

   Level: beginner

.keywords: distributed array, destroy

.seealso: DACreate1d(), DACreate2d(), DACreate3d()
@*/
int DADestroy(DA da)
{
  int ierr,i,cnt = 0;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(da,DA_COOKIE);

  for (i=0; i<10; i++) {
    if (da->localin[i])  {cnt++;}
    if (da->globalin[i]) {cnt++;}
  }

  if (--da->refct - cnt > 0) PetscFunctionReturn(0);
  /*
         Need this test because the da references the vectors that 
     reference the da, so destroying the da calls destroy on the 
     vectors that cause another destroy on the da
  */
  if (da->refct < 0) PetscFunctionReturn(0);
  da->refct = 0;

  for (i=0; i<10; i++) {
    if (da->localout[i]) SETERRQ(1,"Destroying a DA that has a local vector obtained with DAGetLocalVector()");
    if (da->localin[i]) {ierr = VecDestroy(da->localin[i]);CHKERRQ(ierr);}
    if (da->globalout[i]) SETERRQ(1,"Destroying a DA that has a global vector obtained with DAGetGlobalVector()");
    if (da->globalin[i]) {ierr = VecDestroy(da->globalin[i]);CHKERRQ(ierr);}
  }

  /* if memory was published with AMS then destroy it */
  ierr = PetscObjectDepublish(da);CHKERRQ(ierr);

  PetscLogObjectDestroy(da);
  ierr = PetscFree(da->idx);CHKERRQ(ierr);
  ierr = VecScatterDestroy(da->ltog);CHKERRQ(ierr);
  ierr = VecScatterDestroy(da->gtol);CHKERRQ(ierr);
  ierr = VecScatterDestroy(da->ltol);CHKERRQ(ierr);
  ierr = VecDestroy(da->global);CHKERRQ(ierr);
  ierr = VecDestroy(da->local);CHKERRQ(ierr);
  if (da->natural){
    ierr = VecDestroy(da->natural);CHKERRQ(ierr);
  }
  if (da->gton) {
    ierr = VecScatterDestroy(da->gton);CHKERRQ(ierr);
  }

  if (da->ao) {
    ierr = AODestroy(da->ao);CHKERRQ(ierr);
  }
  ierr = ISLocalToGlobalMappingDestroy(da->ltogmap);CHKERRQ(ierr);

  if (da->lx) {ierr = PetscFree(da->lx);CHKERRQ(ierr);}
  if (da->ly) {ierr = PetscFree(da->ly);CHKERRQ(ierr);}
  if (da->lz) {ierr = PetscFree(da->lz);CHKERRQ(ierr);}

  for (i=0; i<da->w; i++) {
    ierr = PetscStrfree(da->fieldname[i]);CHKERRQ(ierr);
  }
  ierr = PetscFree(da->fieldname);CHKERRQ(ierr);

  if (da->coordinates) {ierr = VecDestroy(da->coordinates);CHKERRQ(ierr);}
  if (da->gtog1) {ierr = PetscFree(da->gtog1);CHKERRQ(ierr);}
  PetscHeaderDestroy(da);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "DAGetISLocalToGlobalMapping"
/*@C
   DAGetISLocalToGlobalMapping - Accesses the local-to-global mapping in a DA.

   Not Collective

   Input Parameter:
.  da - the distributed array that provides the mapping 

   Output Parameter:
.  ltog - the mapping

   Level: intermediate

   Notes:
   This mapping can them be used by VecSetLocalToGlobalMapping() or 
   MatSetLocalToGlobalMapping().

   Essentially the same data is returned in the form of an integer array
   with the routine DAGetGlobalIndices().

.keywords: distributed array, destroy

.seealso: DACreate1d(), DACreate2d(), DACreate3d(), VecSetLocalToGlobalMapping(),
          MatSetLocalToGlobalMapping(), DAGetGlobalIndices()
@*/
int DAGetISLocalToGlobalMapping(DA da,ISLocalToGlobalMapping *map)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(da,DA_COOKIE);
  *map = da->ltogmap;
  PetscFunctionReturn(0);
}
