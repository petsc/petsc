
/*
  Code for manipulating distributed regular arrays in parallel.
*/

#include <petsc/private/dmdaimpl.h>    /*I   "petscdmda.h"   I*/

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
  ierr = ISDestroy(&dd->ecorners);CHKERRQ(ierr);

  ierr = PetscObjectComposeFunction((PetscObject)da,"DMSetUpGLVisViewer_C",NULL);CHKERRQ(ierr);

  ierr = PetscFree(dd);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
