#define PETSCDM_DLL

/*
  Code for manipulating distributed regular arrays in parallel.
*/

#include "private/daimpl.h"    /*I   "petscda.h"   I*/

/* Logging support */
PetscCookie PETSCDM_DLLEXPORT DM_COOKIE;
PetscCookie PETSCDM_DLLEXPORT ADDA_COOKIE;
PetscLogEvent  DA_GlobalToLocal, DA_LocalToGlobal, DA_LocalADFunction;

#undef __FUNCT__  
#define __FUNCT__ "DMDestroy_Private"
/*
   DMDestroy_Private - handles the work vectors created by DMGetGlobalVector() and DMGetLocalVector()

*/
PetscErrorCode PETSCDM_DLLEXPORT DMDestroy_Private(DM dm,PetscTruth *done)
{
  PetscErrorCode ierr;
  PetscErrorCode i,cnt = 0;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_COOKIE,1);
  *done = PETSC_FALSE;

  for (i=0; i<DM_MAX_WORK_VECTORS; i++) {
    if (dm->localin[i])  {cnt++;}
    if (dm->globalin[i]) {cnt++;}
  }

  if (--((PetscObject)dm)->refct - cnt > 0) PetscFunctionReturn(0);

  /*
         Need this test because the dm references the vectors that 
     reference the dm, so destroying the dm calls destroy on the 
     vectors that cause another destroy on the dm
  */
  if (((PetscObject)dm)->refct < 0) PetscFunctionReturn(0);
  ((PetscObject)dm)->refct = 0;

  for (i=0; i<DM_MAX_WORK_VECTORS; i++) {
    if (dm->localout[i]) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Destroying a DM that has a local vector obtained with DMGetLocalVector()");
    if (dm->localin[i]) {ierr = VecDestroy(dm->localin[i]);CHKERRQ(ierr);}
    if (dm->globalout[i]) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Destroying a DM that has a global vector obtained with DMGetGlobalVector()");
    if (dm->globalin[i]) {ierr = VecDestroy(dm->globalin[i]);CHKERRQ(ierr);}
  }
  *done = PETSC_TRUE;
  PetscFunctionReturn(0);
}

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
  PetscErrorCode i;
  PetscTruth     done;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(da,DM_COOKIE,1);

  ierr = DMDestroy_Private((DM)da,&done);CHKERRQ(ierr);
  if (!done) PetscFunctionReturn(0);
  /* destroy the internal part */
  if (da->ops->destroy) {
    ierr = (*da->ops->destroy)(da);CHKERRQ(ierr);
  }
  /* destroy the external/common part */
  for (i=0; i<DA_MAX_AD_ARRAYS; i++) {
    ierr = PetscFree(da->adstartghostedout[i]);CHKERRQ(ierr);
    ierr = PetscFree(da->adstartghostedin[i]);CHKERRQ(ierr);
    ierr = PetscFree(da->adstartout[i]);CHKERRQ(ierr);
    ierr = PetscFree(da->adstartin[i]);CHKERRQ(ierr);
  }
  for (i=0; i<DA_MAX_AD_ARRAYS; i++) {
    ierr = PetscFree(da->admfstartghostedout[i]);CHKERRQ(ierr);
    ierr = PetscFree(da->admfstartghostedin[i]);CHKERRQ(ierr);
    ierr = PetscFree(da->admfstartout[i]);CHKERRQ(ierr);
    ierr = PetscFree(da->admfstartin[i]);CHKERRQ(ierr);
  }
  for (i=0; i<DA_MAX_WORK_ARRAYS; i++) {
    ierr = PetscFree(da->startghostedout[i]);CHKERRQ(ierr);
    ierr = PetscFree(da->startghostedin[i]);CHKERRQ(ierr);
    ierr = PetscFree(da->startout[i]);CHKERRQ(ierr);
    ierr = PetscFree(da->startin[i]);CHKERRQ(ierr);
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
  if (da->ltogmap) {
    ierr = ISLocalToGlobalMappingDestroy(da->ltogmap);CHKERRQ(ierr);
  }
  if (da->ltogmapb) {
    ierr = ISLocalToGlobalMappingDestroy(da->ltogmapb);CHKERRQ(ierr);
  }

  ierr = PetscFree(da->lx);CHKERRQ(ierr);
  ierr = PetscFree(da->ly);CHKERRQ(ierr);
  ierr = PetscFree(da->lz);CHKERRQ(ierr);

  if (da->fieldname) {
    for (i=0; i<da->w; i++) {
      ierr = PetscStrfree(da->fieldname[i]);CHKERRQ(ierr);
    }
    ierr = PetscFree(da->fieldname);CHKERRQ(ierr);
  }

  if (da->localcoloring) {
    ierr = ISColoringDestroy(da->localcoloring);CHKERRQ(ierr);
  }
  if (da->ghostedcoloring) {
    ierr = ISColoringDestroy(da->ghostedcoloring);CHKERRQ(ierr);
  }

  if (da->coordinates) {ierr = VecDestroy(da->coordinates);CHKERRQ(ierr);}
  if (da->ghosted_coordinates) {ierr = VecDestroy(da->ghosted_coordinates);CHKERRQ(ierr);}
  if (da->da_coordinates && da != da->da_coordinates) {ierr = DADestroy(da->da_coordinates);CHKERRQ(ierr);}

  ierr = PetscFree(da->neighbors);CHKERRQ(ierr);
  ierr = PetscFree(da->dfill);CHKERRQ(ierr);
  ierr = PetscFree(da->ofill);CHKERRQ(ierr);
  ierr = PetscFree(da->e);CHKERRQ(ierr);

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
  PetscValidHeaderSpecific(da,DM_COOKIE,1);
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
  PetscValidHeaderSpecific(da,DM_COOKIE,1);
  PetscValidPointer(map,2);
  *map = da->ltogmapb;
  PetscFunctionReturn(0);
}


