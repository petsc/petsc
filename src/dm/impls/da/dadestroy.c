
/*
  Code for manipulating distributed regular arrays in parallel.
*/

#include <petsc-private/daimpl.h>    /*I   "petscdmda.h"   I*/

/* Logging support */
PetscClassId  ADDA_CLASSID;
PetscLogEvent DMDA_LocalADFunction;

#undef __FUNCT__  
#define __FUNCT__ "DMDestroy_Private"
/*
   DMDestroy_Private - handles the work vectors created by DMGetGlobalVector() and DMGetLocalVector()

*/
PetscErrorCode  DMDestroy_Private(DM dm,PetscBool  *done)
{
  PetscErrorCode ierr;
  PetscErrorCode i,cnt = 0;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
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
    if (dm->localout[i]) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Destroying a DM that has a local vector obtained with DMGetLocalVector()");
    ierr = VecDestroy(&dm->localin[i]);CHKERRQ(ierr);
    if (dm->globalout[i]) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Destroying a DM that has a global vector obtained with DMGetGlobalVector()");
    ierr = VecDestroy(&dm->globalin[i]);CHKERRQ(ierr);
  }
  ierr = ISLocalToGlobalMappingDestroy(&dm->ltogmap);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingDestroy(&dm->ltogmapb);CHKERRQ(ierr);

  *done = PETSC_TRUE;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "DMDestroy_DA"
PetscErrorCode  DMDestroy_DA(DM da)
{
  PetscErrorCode ierr;
  PetscErrorCode i;
  DM_DA          *dd = (DM_DA*)da->data;

  PetscFunctionBegin;
  /* destroy the external/common part */
  for (i=0; i<DMDA_MAX_AD_ARRAYS; i++) {
    ierr = PetscFree(dd->adstartghostedout[i]);CHKERRQ(ierr);
    ierr = PetscFree(dd->adstartghostedin[i]);CHKERRQ(ierr);
    ierr = PetscFree(dd->adstartout[i]);CHKERRQ(ierr);
    ierr = PetscFree(dd->adstartin[i]);CHKERRQ(ierr);
  }
  for (i=0; i<DMDA_MAX_AD_ARRAYS; i++) {
    ierr = PetscFree(dd->admfstartghostedout[i]);CHKERRQ(ierr);
    ierr = PetscFree(dd->admfstartghostedin[i]);CHKERRQ(ierr);
    ierr = PetscFree(dd->admfstartout[i]);CHKERRQ(ierr);
    ierr = PetscFree(dd->admfstartin[i]);CHKERRQ(ierr);
  }
  for (i=0; i<DMDA_MAX_WORK_ARRAYS; i++) {
    ierr = PetscFree(dd->startghostedout[i]);CHKERRQ(ierr);
    ierr = PetscFree(dd->startghostedin[i]);CHKERRQ(ierr);
    ierr = PetscFree(dd->startout[i]);CHKERRQ(ierr);
    ierr = PetscFree(dd->startin[i]);CHKERRQ(ierr);
  }

  /* if memory was published with AMS then destroy it */
  ierr = PetscObjectDepublish(da);CHKERRQ(ierr);

  ierr = VecScatterDestroy(&dd->ltog);CHKERRQ(ierr);
  ierr = VecScatterDestroy(&dd->gtol);CHKERRQ(ierr);
  ierr = VecScatterDestroy(&dd->ltol);CHKERRQ(ierr);
  ierr = VecDestroy(&dd->natural);CHKERRQ(ierr);
  ierr = VecScatterDestroy(&dd->gton);CHKERRQ(ierr);
  ierr = AODestroy(&dd->ao);CHKERRQ(ierr);

  ierr = PetscFree(dd->idx);CHKERRQ(ierr);
  ierr = PetscFree(dd->lx);CHKERRQ(ierr);
  ierr = PetscFree(dd->ly);CHKERRQ(ierr);
  ierr = PetscFree(dd->lz);CHKERRQ(ierr);

  if (dd->fieldname) {
    for (i=0; i<dd->w; i++) {
      ierr = PetscFree(dd->fieldname[i]);CHKERRQ(ierr);
    }
    ierr = PetscFree(dd->fieldname);CHKERRQ(ierr);
  }
  ierr = ISColoringDestroy(&dd->localcoloring);CHKERRQ(ierr);
  ierr = ISColoringDestroy(&dd->ghostedcoloring);CHKERRQ(ierr);

  ierr = VecDestroy(&dd->coordinates);CHKERRQ(ierr);
  ierr = VecDestroy(&dd->ghosted_coordinates);CHKERRQ(ierr);
  ierr = DMDestroy(&dd->da_coordinates);CHKERRQ(ierr);

  ierr = PetscFree(dd->neighbors);CHKERRQ(ierr);
  ierr = PetscFree(dd->dfill);CHKERRQ(ierr);
  ierr = PetscFree(dd->ofill);CHKERRQ(ierr);
  ierr = PetscFree(dd->e);CHKERRQ(ierr);

  ierr = PetscSectionDestroy(&dd->defaultSection);CHKERRQ(ierr);
  /* ierr = PetscSectionDestroy(&dd->defaultGlobalSection);CHKERRQ(ierr); */
  PetscFunctionReturn(0);
}
