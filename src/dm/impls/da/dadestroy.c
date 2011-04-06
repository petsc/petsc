
/*
  Code for manipulating distributed regular arrays in parallel.
*/

#include <private/daimpl.h>    /*I   "petscdmda.h"   I*/

/* Logging support */
PetscClassId  ADDA_CLASSID;
PetscLogEvent DMDA_LocalADFunction;

#undef __FUNCT__  
#define __FUNCT__ "DMDestroy_DA"
PetscErrorCode  DMDestroy_DA(DM dm)
{
  PetscErrorCode ierr;
  PetscErrorCode i;
  DM_DA         *dd = (DM_DA *) dm->data;

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

  if (dd->ltog)   {ierr = VecScatterDestroy(dd->ltog);CHKERRQ(ierr);}
  if (dd->gtol)   {ierr = VecScatterDestroy(dd->gtol);CHKERRQ(ierr);}
  if (dd->ltol)   {ierr = VecScatterDestroy(dd->ltol);CHKERRQ(ierr);}
  if (dd->natural){
    ierr = VecDestroy(dd->natural);CHKERRQ(ierr);
  }
  if (dd->gton) {
    ierr = VecScatterDestroy(dd->gton);CHKERRQ(ierr);
  }

  if (dd->ao) {
    ierr = AODestroy(dd->ao);CHKERRQ(ierr);
  }

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

  if (dd->localcoloring) {
    ierr = ISColoringDestroy(dd->localcoloring);CHKERRQ(ierr);
  }
  if (dd->ghostedcoloring) {
    ierr = ISColoringDestroy(dd->ghostedcoloring);CHKERRQ(ierr);
  }

  if (dd->coordinates) {ierr = VecDestroy(dd->coordinates);CHKERRQ(ierr);}
  if (dd->ghosted_coordinates) {ierr = VecDestroy(dd->ghosted_coordinates);CHKERRQ(ierr);}
  if (dd->da_coordinates && dm != dd->da_coordinates) {ierr = DMDestroy(dd->da_coordinates);CHKERRQ(ierr);}

  ierr = PetscFree(dd->neighbors);CHKERRQ(ierr);
  ierr = PetscFree(dd->dfill);CHKERRQ(ierr);
  ierr = PetscFree(dd->ofill);CHKERRQ(ierr);
  ierr = PetscFree(dd->e);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
