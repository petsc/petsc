
/*
  Code for manipulating distributed regular arrays in parallel.
*/

#include <petsc/private/dmdaimpl.h>    /*I   "petscdmda.h"   I*/

PetscErrorCode  DMDestroy_DA(DM da)
{
  PetscErrorCode i;
  DM_DA          *dd = (DM_DA*)da->data;

  PetscFunctionBegin;
  /* destroy the external/common part */
  for (i=0; i<DMDA_MAX_WORK_ARRAYS; i++) {
    CHKERRQ(PetscFree(dd->startghostedout[i]));
    CHKERRQ(PetscFree(dd->startghostedin[i]));
    CHKERRQ(PetscFree(dd->startout[i]));
    CHKERRQ(PetscFree(dd->startin[i]));
  }

  CHKERRQ(VecScatterDestroy(&dd->gtol));
  CHKERRQ(VecScatterDestroy(&dd->ltol));
  CHKERRQ(VecDestroy(&dd->natural));
  CHKERRQ(VecScatterDestroy(&dd->gton));
  CHKERRQ(AODestroy(&dd->ao));
  CHKERRQ(PetscFree(dd->aotype));

  CHKERRQ(PetscFree(dd->lx));
  CHKERRQ(PetscFree(dd->ly));
  CHKERRQ(PetscFree(dd->lz));

  CHKERRQ(PetscFree(dd->refine_x_hier));
  CHKERRQ(PetscFree(dd->refine_y_hier));
  CHKERRQ(PetscFree(dd->refine_z_hier));

  if (dd->fieldname) {
    for (i=0; i<dd->w; i++) {
      CHKERRQ(PetscFree(dd->fieldname[i]));
    }
    CHKERRQ(PetscFree(dd->fieldname));
  }
  if (dd->coordinatename) {
    for (i=0; i<da->dim; i++) {
      CHKERRQ(PetscFree(dd->coordinatename[i]));
    }
    CHKERRQ(PetscFree(dd->coordinatename));
  }
  CHKERRQ(ISColoringDestroy(&dd->localcoloring));
  CHKERRQ(ISColoringDestroy(&dd->ghostedcoloring));

  CHKERRQ(PetscFree(dd->neighbors));
  CHKERRQ(PetscFree(dd->dfill));
  CHKERRQ(PetscFree(dd->ofill));
  CHKERRQ(PetscFree(dd->ofillcols));
  CHKERRQ(PetscFree(dd->e));
  CHKERRQ(ISDestroy(&dd->ecorners));

  CHKERRQ(PetscObjectComposeFunction((PetscObject)da,"DMSetUpGLVisViewer_C",NULL));

  CHKERRQ(PetscFree(dd));
  PetscFunctionReturn(0);
}
