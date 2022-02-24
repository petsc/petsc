
/*
     Additive Multigrid V Cycle routine
*/
#include <petsc/private/pcmgimpl.h>

PetscErrorCode PCMGACycle_Private(PC pc,PC_MG_Levels **mglevels,PetscBool transpose,PetscBool matapp)
{
  PetscInt       i,l = mglevels[0]->levels;

  PetscFunctionBegin;
  /* compute RHS on each level */
  for (i=l-1; i>0; i--) {
    if (mglevels[i]->eventinterprestrict) CHKERRQ(PetscLogEventBegin(mglevels[i]->eventinterprestrict,0,0,0,0));
    if (!transpose) {
      if (matapp) CHKERRQ(MatMatRestrict(mglevels[i]->restrct,mglevels[i]->B,&mglevels[i-1]->B));
      else CHKERRQ(MatRestrict(mglevels[i]->restrct,mglevels[i]->b,mglevels[i-1]->b));
    } else {
      if (matapp) CHKERRQ(MatMatRestrict(mglevels[i]->interpolate,mglevels[i]->B,&mglevels[i-1]->B));
      else CHKERRQ(MatRestrict(mglevels[i]->interpolate,mglevels[i]->b,mglevels[i-1]->b));
    }
    if (mglevels[i]->eventinterprestrict) CHKERRQ(PetscLogEventEnd(mglevels[i]->eventinterprestrict,0,0,0,0));
  }
  /* solve separately on each level */
  for (i=0; i<l; i++) {
    if (matapp) {
      if (!mglevels[i]->X) {
        CHKERRQ(MatDuplicate(mglevels[i]->B,MAT_DO_NOT_COPY_VALUES,&mglevels[i]->X));
      } else {
        CHKERRQ(MatZeroEntries(mglevels[i]->X));
      }
    } else {
      CHKERRQ(VecZeroEntries(mglevels[i]->x));
    }
    if (mglevels[i]->eventsmoothsolve) CHKERRQ(PetscLogEventBegin(mglevels[i]->eventsmoothsolve,0,0,0,0));
    if (!transpose) {
      if (matapp) {
        CHKERRQ(KSPMatSolve(mglevels[i]->smoothd,mglevels[i]->B,mglevels[i]->X));
        CHKERRQ(KSPCheckSolve(mglevels[i]->smoothd,pc,NULL));
      } else {
        CHKERRQ(KSPSolve(mglevels[i]->smoothd,mglevels[i]->b,mglevels[i]->x));
        CHKERRQ(KSPCheckSolve(mglevels[i]->smoothd,pc,mglevels[i]->x));
      }
    } else {
      PetscCheckFalse(matapp,PetscObjectComm((PetscObject)pc),PETSC_ERR_SUP,"Not supported");
      CHKERRQ(KSPSolveTranspose(mglevels[i]->smoothu,mglevels[i]->b,mglevels[i]->x));
      CHKERRQ(KSPCheckSolve(mglevels[i]->smoothu,pc,mglevels[i]->x));
    }
    if (mglevels[i]->eventsmoothsolve) CHKERRQ(PetscLogEventEnd(mglevels[i]->eventsmoothsolve,0,0,0,0));
  }
  for (i=1; i<l; i++) {
    if (mglevels[i]->eventinterprestrict) CHKERRQ(PetscLogEventBegin(mglevels[i]->eventinterprestrict,0,0,0,0));
    if (!transpose) {
      if (matapp) CHKERRQ(MatMatInterpolateAdd(mglevels[i]->interpolate,mglevels[i-1]->X,mglevels[i]->X,&mglevels[i]->X));
      else CHKERRQ(MatInterpolateAdd(mglevels[i]->interpolate,mglevels[i-1]->x,mglevels[i]->x,mglevels[i]->x));
    } else {
      if (matapp) CHKERRQ(MatMatInterpolateAdd(mglevels[i]->restrct,mglevels[i-1]->X,mglevels[i]->X,&mglevels[i]->X));
      else CHKERRQ(MatInterpolateAdd(mglevels[i]->restrct,mglevels[i-1]->x,mglevels[i]->x,mglevels[i]->x));
    }
    if (mglevels[i]->eventinterprestrict) CHKERRQ(PetscLogEventEnd(mglevels[i]->eventinterprestrict,0,0,0,0));
  }
  PetscFunctionReturn(0);
}
