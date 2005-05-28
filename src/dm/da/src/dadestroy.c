#define PETSCDM_DLL

/*
  Code for manipulating distributed regular arrays in parallel.
*/

#include "src/dm/da/daimpl.h"    /*I   "petscda.h"   I*/

/* Logging support */
PetscCookie PETSCDM_DLLEXPORT DA_COOKIE = 0;
PetscEvent  DA_GlobalToLocal = 0, DA_LocalToGlobal = 0, DA_LocalADFunction = 0;

#undef __FUNCT__  
#define __FUNCT__ "DADestroy"
/*@
   DADestroy - Destroys a distributed array.

   Collective on DA

   Input Parameter:
.  da - the distributed array to destroy 

   Level: beginner

.keywords: distributed array, destroy

.seealso: DACreate1d(), DACreate2d(), DACreate3d()
@*/
PetscErrorCode PETSCDM_DLLEXPORT DADestroy(DA da)
{
  PetscErrorCode ierr;
  PetscErrorCode i,cnt = 0;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(da,DA_COOKIE,1);

  for (i=0; i<DA_MAX_WORK_VECTORS; i++) {
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

  for (i=0; i<DA_MAX_WORK_VECTORS; i++) {
    if (da->localout[i]) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Destroying a DA that has a local vector obtained with DAGetLocalVector()");
    if (da->localin[i]) {ierr = VecDestroy(da->localin[i]);CHKERRQ(ierr);}
    if (da->globalout[i]) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Destroying a DA that has a global vector obtained with DAGetGlobalVector()");
    if (da->globalin[i]) {ierr = VecDestroy(da->globalin[i]);CHKERRQ(ierr);}
  }

  for (i=0; i<DA_MAX_AD_ARRAYS; i++) {
    if (da->adstartghostedout[i]){
      ierr = PetscFree(da->adstartghostedout[i]);CHKERRQ(ierr);
    }
    if (da->adstartghostedin[i]){
      ierr = PetscFree(da->adstartghostedin[i]);CHKERRQ(ierr);
    }
    if (da->adstartout[i]){
      ierr = PetscFree(da->adstartout[i]);CHKERRQ(ierr);
    }
    if (da->adstartin[i]){
      ierr = PetscFree(da->adstartin[i]);CHKERRQ(ierr);
    }
  }
  for (i=0; i<DA_MAX_AD_ARRAYS; i++) {
    if (da->admfstartghostedout[i]){
      ierr = PetscFree(da->admfstartghostedout[i]);CHKERRQ(ierr);
    }
    if (da->admfstartghostedin[i]){
      ierr = PetscFree(da->admfstartghostedin[i]);CHKERRQ(ierr);
    }
    if (da->admfstartout[i]){
      ierr = PetscFree(da->admfstartout[i]);CHKERRQ(ierr);
    }
    if (da->admfstartin[i]){
      ierr = PetscFree(da->admfstartin[i]);CHKERRQ(ierr);
    }
  }
  for (i=0; i<DA_MAX_WORK_ARRAYS; i++) {
    if (da->startghostedout[i]){
      ierr = PetscFree(da->startghostedout[i]);CHKERRQ(ierr);
    }
    if (da->startghostedin[i]){
      ierr = PetscFree(da->startghostedin[i]);CHKERRQ(ierr);
    }
    if (da->startout[i]){
      ierr = PetscFree(da->startout[i]);CHKERRQ(ierr);
    }
    if (da->startin[i]){
      ierr = PetscFree(da->startin[i]);CHKERRQ(ierr);
    }
  }

  /* if memory was published with AMS then destroy it */
  ierr = PetscObjectDepublish(da);CHKERRQ(ierr);

  if (da->ltog)   {ierr = VecScatterDestroy(da->ltog);CHKERRQ(ierr);}
  if (da->gtol)   {ierr = VecScatterDestroy(da->gtol);CHKERRQ(ierr);}
  if (da->ltol)   {ierr = VecScatterDestroy(da->ltol);CHKERRQ(ierr);}
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
  ierr = ISLocalToGlobalMappingDestroy(da->ltogmapb);CHKERRQ(ierr);

  if (da->lx) {ierr = PetscFree(da->lx);CHKERRQ(ierr);}
  if (da->ly) {ierr = PetscFree(da->ly);CHKERRQ(ierr);}
  if (da->lz) {ierr = PetscFree(da->lz);CHKERRQ(ierr);}

  for (i=0; i<da->w; i++) {
    ierr = PetscStrfree(da->fieldname[i]);CHKERRQ(ierr);
  }
  ierr = PetscFree(da->fieldname);CHKERRQ(ierr);

  if (da->localcoloring) {
    ierr = ISColoringDestroy(da->localcoloring);CHKERRQ(ierr);
  }
  if (da->ghostedcoloring) {
    ierr = ISColoringDestroy(da->ghostedcoloring);CHKERRQ(ierr);
  }

  if (da->coordinates) {ierr = VecDestroy(da->coordinates);CHKERRQ(ierr);}
  if (da->ghosted_coordinates) {ierr = VecDestroy(da->ghosted_coordinates);CHKERRQ(ierr);}
  if (da->da_coordinates && da != da->da_coordinates) {ierr = DADestroy(da->da_coordinates);CHKERRQ(ierr);}

  if (da->dfill) {ierr = PetscFree(da->dfill);CHKERRQ(ierr);}
  if (da->ofill) {ierr = PetscFree(da->ofill);CHKERRQ(ierr);}
  if (da->e) {ierr = PetscFree(da->e);CHKERRQ(ierr);}

  ierr = PetscHeaderDestroy(da);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "DAGetISLocalToGlobalMapping"
/*@
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
          MatSetLocalToGlobalMapping(), DAGetGlobalIndices(), DAGetISLocalToGlobalMappingBlck()
@*/
PetscErrorCode PETSCDM_DLLEXPORT DAGetISLocalToGlobalMapping(DA da,ISLocalToGlobalMapping *map)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(da,DA_COOKIE,1);
  PetscValidPointer(map,2);
  *map = da->ltogmap;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "DAGetISLocalToGlobalMappingBlck"
/*@
   DAGetISLocalToGlobalMappingBlck - Accesses the local-to-global mapping in a DA.

   Not Collective

   Input Parameter:
.  da - the distributed array that provides the mapping 

   Output Parameter:
.  ltog - the mapping

   Level: intermediate

   Notes:
   This mapping can them be used by VecSetLocalToGlobalMappingBlock() or 
   MatSetLocalToGlobalMappingBlock().

   Essentially the same data is returned in the form of an integer array
   with the routine DAGetGlobalIndices().

.keywords: distributed array, destroy

.seealso: DACreate1d(), DACreate2d(), DACreate3d(), VecSetLocalToGlobalMapping(),
          MatSetLocalToGlobalMapping(), DAGetGlobalIndices(), DAGetISLocalToGlobalMapping()
@*/
PetscErrorCode PETSCDM_DLLEXPORT DAGetISLocalToGlobalMappingBlck(DA da,ISLocalToGlobalMapping *map)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(da,DA_COOKIE,1);
  PetscValidPointer(map,2);
  *map = da->ltogmapb;
  PetscFunctionReturn(0);
}


