
/*
     Additive Multigrid V Cycle routine
*/
#include <petsc/private/pcmgimpl.h>

PetscErrorCode PCMGACycle_Private(PC pc, PC_MG_Levels **mglevels, PetscBool transpose, PetscBool matapp)
{
  PetscInt i, l = mglevels[0]->levels;

  PetscFunctionBegin;
  /* compute RHS on each level */
  for (i = l - 1; i > 0; i--) {
    if (mglevels[i]->eventinterprestrict) PetscCall(PetscLogEventBegin(mglevels[i]->eventinterprestrict, 0, 0, 0, 0));
    if (!transpose) {
      if (matapp) PetscCall(MatMatRestrict(mglevels[i]->restrct, mglevels[i]->B, &mglevels[i - 1]->B));
      else PetscCall(MatRestrict(mglevels[i]->restrct, mglevels[i]->b, mglevels[i - 1]->b));
    } else {
      if (matapp) PetscCall(MatMatRestrict(mglevels[i]->interpolate, mglevels[i]->B, &mglevels[i - 1]->B));
      else PetscCall(MatRestrict(mglevels[i]->interpolate, mglevels[i]->b, mglevels[i - 1]->b));
    }
    if (mglevels[i]->eventinterprestrict) PetscCall(PetscLogEventEnd(mglevels[i]->eventinterprestrict, 0, 0, 0, 0));
  }
  /* solve separately on each level */
  for (i = 0; i < l; i++) {
    if (matapp) {
      if (!mglevels[i]->X) {
        PetscCall(MatDuplicate(mglevels[i]->B, MAT_DO_NOT_COPY_VALUES, &mglevels[i]->X));
      } else {
        PetscCall(MatZeroEntries(mglevels[i]->X));
      }
    } else {
      PetscCall(VecZeroEntries(mglevels[i]->x));
    }
    if (mglevels[i]->eventsmoothsolve) PetscCall(PetscLogEventBegin(mglevels[i]->eventsmoothsolve, 0, 0, 0, 0));
    if (!transpose) {
      if (matapp) {
        PetscCall(KSPMatSolve(mglevels[i]->smoothd, mglevels[i]->B, mglevels[i]->X));
        PetscCall(KSPCheckSolve(mglevels[i]->smoothd, pc, NULL));
      } else {
        PetscCall(KSPSolve(mglevels[i]->smoothd, mglevels[i]->b, mglevels[i]->x));
        PetscCall(KSPCheckSolve(mglevels[i]->smoothd, pc, mglevels[i]->x));
      }
    } else {
      PetscCheck(!matapp, PetscObjectComm((PetscObject)pc), PETSC_ERR_SUP, "Not supported");
      PetscCall(KSPSolveTranspose(mglevels[i]->smoothu, mglevels[i]->b, mglevels[i]->x));
      PetscCall(KSPCheckSolve(mglevels[i]->smoothu, pc, mglevels[i]->x));
    }
    if (mglevels[i]->eventsmoothsolve) PetscCall(PetscLogEventEnd(mglevels[i]->eventsmoothsolve, 0, 0, 0, 0));
  }
  for (i = 1; i < l; i++) {
    if (mglevels[i]->eventinterprestrict) PetscCall(PetscLogEventBegin(mglevels[i]->eventinterprestrict, 0, 0, 0, 0));
    if (!transpose) {
      if (matapp) PetscCall(MatMatInterpolateAdd(mglevels[i]->interpolate, mglevels[i - 1]->X, mglevels[i]->X, &mglevels[i]->X));
      else PetscCall(MatInterpolateAdd(mglevels[i]->interpolate, mglevels[i - 1]->x, mglevels[i]->x, mglevels[i]->x));
    } else {
      if (matapp) PetscCall(MatMatInterpolateAdd(mglevels[i]->restrct, mglevels[i - 1]->X, mglevels[i]->X, &mglevels[i]->X));
      else PetscCall(MatInterpolateAdd(mglevels[i]->restrct, mglevels[i - 1]->x, mglevels[i]->x, mglevels[i]->x));
    }
    if (mglevels[i]->eventinterprestrict) PetscCall(PetscLogEventEnd(mglevels[i]->eventinterprestrict, 0, 0, 0, 0));
  }
  PetscFunctionReturn(0);
}
