
/*
  Code for manipulating distributed regular arrays in parallel.
*/

#include <petsc/private/dmdaimpl.h>    /*I   "petscdmda.h"   I*/

/* Logging support */
PetscLogEvent DMDA_LocalADFunction;

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
    if (dm->localin[i])  cnt++;
    if (dm->globalin[i]) cnt++;
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

  *done = PETSC_TRUE;
  PetscFunctionReturn(0);
}

PetscErrorCode  DMDestroy_DA(DM da)
{
  PetscErrorCode ierr;
  PetscErrorCode i;
  DM_DA          *dd = (DM_DA*)da->data;

  PetscFunctionBegin;
  /* destroy the external/common part */
  for (i=0; i<DMDA_MAX_WORK_ARRAYS; i++) {
    ierr = PetscFree(dd->startghostedout[i]);CHKERRQ(ierr);
    ierr = PetscFree(dd->startghostedin[i]);CHKERRQ(ierr);
    ierr = PetscFree(dd->startout[i]);CHKERRQ(ierr);
    ierr = PetscFree(dd->startin[i]);CHKERRQ(ierr);
  }

  ierr = VecScatterDestroy(&dd->gtol);CHKERRQ(ierr);
  ierr = VecScatterDestroy(&dd->ltol);CHKERRQ(ierr);
  ierr = VecDestroy(&dd->natural);CHKERRQ(ierr);
  ierr = VecScatterDestroy(&dd->gton);CHKERRQ(ierr);
  ierr = AODestroy(&dd->ao);CHKERRQ(ierr);
  ierr = PetscFree(dd->aotype);CHKERRQ(ierr);

  ierr = PetscFree(dd->lx);CHKERRQ(ierr);
  ierr = PetscFree(dd->ly);CHKERRQ(ierr);
  ierr = PetscFree(dd->lz);CHKERRQ(ierr);

  ierr = PetscFree(dd->refine_x_hier);CHKERRQ(ierr);
  ierr = PetscFree(dd->refine_y_hier);CHKERRQ(ierr);
  ierr = PetscFree(dd->refine_z_hier);CHKERRQ(ierr);

  if (dd->fieldname) {
    for (i=0; i<dd->w; i++) {
      ierr = PetscFree(dd->fieldname[i]);CHKERRQ(ierr);
    }
    ierr = PetscFree(dd->fieldname);CHKERRQ(ierr);
  }
  if (dd->coordinatename) {
    for (i=0; i<da->dim; i++) {
      ierr = PetscFree(dd->coordinatename[i]);CHKERRQ(ierr);
    }
    ierr = PetscFree(dd->coordinatename);CHKERRQ(ierr);
  }
  ierr = ISColoringDestroy(&dd->localcoloring);CHKERRQ(ierr);
  ierr = ISColoringDestroy(&dd->ghostedcoloring);CHKERRQ(ierr);

  ierr = PetscFree(dd->neighbors);CHKERRQ(ierr);
  ierr = PetscFree(dd->dfill);CHKERRQ(ierr);
  ierr = PetscFree(dd->ofill);CHKERRQ(ierr);
  ierr = PetscFree(dd->ofillcols);CHKERRQ(ierr);
  ierr = PetscFree(dd->e);CHKERRQ(ierr);

  ierr = PetscObjectComposeFunction((PetscObject)da,"DMSetUpGLVisViewer_C",NULL);CHKERRQ(ierr);

  /* ierr = PetscSectionDestroy(&dd->defaultGlobalSection);CHKERRQ(ierr); */
  /* This was originally freed in DMDestroy(), but that prevents reference counting of backend objects */
  ierr = PetscFree(dd);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
