
/*
     Additive Multigrid V Cycle routine
*/
#include <petsc/private/pcmgimpl.h>

PetscErrorCode PCMGACycle_Private(PC pc,PC_MG_Levels **mglevels,PetscBool transpose,PetscBool matapp)
{
  PetscErrorCode ierr;
  PetscInt       i,l = mglevels[0]->levels;

  PetscFunctionBegin;
  /* compute RHS on each level */
  for (i=l-1; i>0; i--) {
    if (mglevels[i]->eventinterprestrict) {ierr = PetscLogEventBegin(mglevels[i]->eventinterprestrict,0,0,0,0);CHKERRQ(ierr);}
    if (!transpose) {
      if (matapp) { ierr = MatMatRestrict(mglevels[i]->restrct,mglevels[i]->B,&mglevels[i-1]->B);CHKERRQ(ierr); }
      else { ierr = MatRestrict(mglevels[i]->restrct,mglevels[i]->b,mglevels[i-1]->b);CHKERRQ(ierr); }
    } else {
      if (matapp) { ierr = MatMatRestrict(mglevels[i]->interpolate,mglevels[i]->B,&mglevels[i-1]->B);CHKERRQ(ierr); }
      else { ierr = MatRestrict(mglevels[i]->interpolate,mglevels[i]->b,mglevels[i-1]->b);CHKERRQ(ierr); }
    }
    if (mglevels[i]->eventinterprestrict) {ierr = PetscLogEventEnd(mglevels[i]->eventinterprestrict,0,0,0,0);CHKERRQ(ierr);}
  }
  /* solve separately on each level */
  for (i=0; i<l; i++) {
    if (matapp) {
      if (!mglevels[i]->X) {
        ierr = MatDuplicate(mglevels[i]->B,MAT_DO_NOT_COPY_VALUES,&mglevels[i]->X);CHKERRQ(ierr);
      } else {
        ierr = MatZeroEntries(mglevels[i]->X);CHKERRQ(ierr);
      }
    } else {
      ierr = VecZeroEntries(mglevels[i]->x);CHKERRQ(ierr);
    }
    if (mglevels[i]->eventsmoothsolve) {ierr = PetscLogEventBegin(mglevels[i]->eventsmoothsolve,0,0,0,0);CHKERRQ(ierr);}
    if (!transpose) {
      if (matapp) {
        ierr = KSPMatSolve(mglevels[i]->smoothd,mglevels[i]->B,mglevels[i]->X);CHKERRQ(ierr);
        ierr = KSPCheckSolve(mglevels[i]->smoothd,pc,NULL);CHKERRQ(ierr);
      } else {
        ierr = KSPSolve(mglevels[i]->smoothd,mglevels[i]->b,mglevels[i]->x);CHKERRQ(ierr);
        ierr = KSPCheckSolve(mglevels[i]->smoothd,pc,mglevels[i]->x);CHKERRQ(ierr);
      }
    } else {
      PetscCheckFalse(matapp,PetscObjectComm((PetscObject)pc),PETSC_ERR_SUP,"Not supported");
      ierr = KSPSolveTranspose(mglevels[i]->smoothu,mglevels[i]->b,mglevels[i]->x);CHKERRQ(ierr);
      ierr = KSPCheckSolve(mglevels[i]->smoothu,pc,mglevels[i]->x);CHKERRQ(ierr);
    }
    if (mglevels[i]->eventsmoothsolve) {ierr = PetscLogEventEnd(mglevels[i]->eventsmoothsolve,0,0,0,0);CHKERRQ(ierr);}
  }
  for (i=1; i<l; i++) {
    if (mglevels[i]->eventinterprestrict) {ierr = PetscLogEventBegin(mglevels[i]->eventinterprestrict,0,0,0,0);CHKERRQ(ierr);}
    if (!transpose) {
      if (matapp) { ierr = MatMatInterpolateAdd(mglevels[i]->interpolate,mglevels[i-1]->X,mglevels[i]->X,&mglevels[i]->X);CHKERRQ(ierr); }
      else { ierr = MatInterpolateAdd(mglevels[i]->interpolate,mglevels[i-1]->x,mglevels[i]->x,mglevels[i]->x);CHKERRQ(ierr); }
    } else {
      if (matapp) { ierr = MatMatInterpolateAdd(mglevels[i]->restrct,mglevels[i-1]->X,mglevels[i]->X,&mglevels[i]->X);CHKERRQ(ierr); }
      else { ierr = MatInterpolateAdd(mglevels[i]->restrct,mglevels[i-1]->x,mglevels[i]->x,mglevels[i]->x);CHKERRQ(ierr); }
    }
    if (mglevels[i]->eventinterprestrict) {ierr = PetscLogEventEnd(mglevels[i]->eventinterprestrict,0,0,0,0);CHKERRQ(ierr);}
  }
  PetscFunctionReturn(0);
}
