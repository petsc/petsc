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
      if (matapp == PETSC_TRUE) PetscCall(MatMatRestrict(mglevels[i]->restrct, PCMGMatB(pc, i), &PCMGMatB(pc, i - 1)));
      else PetscCall(MatRestrict(mglevels[i]->restrct, mglevels[i]->b, mglevels[i - 1]->b));
    } else {
      if (matapp == PETSC_TRUE) PetscCall(MatMatRestrict(mglevels[i]->interpolate, PCMGMatB(pc, i), &PCMGMatB(pc, i - 1)));
      else PetscCall(MatRestrict(mglevels[i]->interpolate, mglevels[i]->b, mglevels[i - 1]->b));
    }
    if (mglevels[i]->eventinterprestrict) PetscCall(PetscLogEventEnd(mglevels[i]->eventinterprestrict, 0, 0, 0, 0));
  }
  /* solve separately on each level */
  for (i = 0; i < l; i++) {
    if (matapp == PETSC_TRUE) {
      if (PCMGMatX(pc, i) == NULL) PetscCall(MatDuplicate(PCMGMatB(pc, i), MAT_DO_NOT_COPY_VALUES, &PCMGMatX(pc, i)));
      else PetscCall(MatZeroEntries(PCMGMatX(pc, i)));
    } else PetscCall(VecZeroEntries(mglevels[i]->x));
    if (mglevels[i]->eventsmoothsolve) PetscCall(PetscLogEventBegin(mglevels[i]->eventsmoothsolve, 0, 0, 0, 0));
    if (!transpose) {
      if (matapp == PETSC_TRUE) {
        PetscCall(KSPMatSolve(mglevels[i]->smoothd, PCMGMatB(pc, i), PCMGMatX(pc, i)));
        PetscCall(KSPCheckMatSolve(mglevels[i]->smoothd, pc, PCMGMatX(pc, i)));
      } else {
        PetscCall(KSPSolve(mglevels[i]->smoothd, mglevels[i]->b, mglevels[i]->x));
        PetscCall(KSPCheckSolve(mglevels[i]->smoothd, pc, mglevels[i]->x));
      }
    } else {
      PetscCheck(matapp == PETSC_FALSE, PetscObjectComm((PetscObject)pc), PETSC_ERR_SUP, "Not supported");
      PetscCall(KSPSolveTranspose(mglevels[i]->smoothd, mglevels[i]->b, mglevels[i]->x));
      PetscCall(KSPCheckSolve(mglevels[i]->smoothd, pc, mglevels[i]->x));
    }
    if (mglevels[i]->eventsmoothsolve) PetscCall(PetscLogEventEnd(mglevels[i]->eventsmoothsolve, 0, 0, 0, 0));
  }
  for (i = 1; i < l; i++) {
    if (mglevels[i]->eventinterprestrict) PetscCall(PetscLogEventBegin(mglevels[i]->eventinterprestrict, 0, 0, 0, 0));
    if (!transpose) {
      if (matapp == PETSC_TRUE) PetscCall(PCMGMatInterpolate_Private(pc, i, mglevels[i]->interpolate, PETSC_TRUE));
      else PetscCall(MatInterpolateAdd(mglevels[i]->interpolate, mglevels[i - 1]->x, mglevels[i]->x, mglevels[i]->x));
    } else {
      if (matapp == PETSC_TRUE) PetscCall(PCMGMatInterpolate_Private(pc, i, mglevels[i]->restrct, PETSC_TRUE));
      else PetscCall(MatInterpolateAdd(mglevels[i]->restrct, mglevels[i - 1]->x, mglevels[i]->x, mglevels[i]->x));
    }
    if (mglevels[i]->eventinterprestrict) PetscCall(PetscLogEventEnd(mglevels[i]->eventinterprestrict, 0, 0, 0, 0));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}
