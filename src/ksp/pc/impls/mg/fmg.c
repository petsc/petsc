/*
     Full multigrid using either additive or multiplicative V or W cycle
*/
#include <petsc/private/pcmgimpl.h>

PetscErrorCode PCMGFCycle_Private(PC pc, PC_MG_Levels **mglevels, PetscBool transpose, PetscBool matapp)
{
  PetscInt i, l = mglevels[0]->levels;

  PetscFunctionBegin;
  if (!transpose) {
    /* restrict the RHS through all levels to coarsest. */
    for (i = l - 1; i > 0; i--) {
      if (mglevels[i]->eventinterprestrict) PetscCall(PetscLogEventBegin(mglevels[i]->eventinterprestrict, 0, 0, 0, 0));
      if (matapp) PetscCall(MatMatRestrict(mglevels[i]->restrct, mglevels[i]->B, &mglevels[i - 1]->B));
      else PetscCall(MatRestrict(mglevels[i]->restrct, mglevels[i]->b, mglevels[i - 1]->b));
      if (mglevels[i]->eventinterprestrict) PetscCall(PetscLogEventEnd(mglevels[i]->eventinterprestrict, 0, 0, 0, 0));
    }

    /* work our way up through the levels */
    if (matapp) {
      if (!mglevels[0]->X) {
        PetscCall(MatDuplicate(mglevels[0]->B, MAT_DO_NOT_COPY_VALUES, &mglevels[0]->X));
      } else {
        PetscCall(MatZeroEntries(mglevels[0]->X));
      }
    } else {
      PetscCall(VecZeroEntries(mglevels[0]->x));
    }
    for (i = 0; i < l - 1; i++) {
      PetscCall(PCMGMCycle_Private(pc, &mglevels[i], transpose, matapp, NULL));
      if (mglevels[i + 1]->eventinterprestrict) PetscCall(PetscLogEventBegin(mglevels[i + 1]->eventinterprestrict, 0, 0, 0, 0));
      if (matapp) PetscCall(MatMatInterpolate(mglevels[i + 1]->interpolate, mglevels[i]->X, &mglevels[i + 1]->X));
      else PetscCall(MatInterpolate(mglevels[i + 1]->interpolate, mglevels[i]->x, mglevels[i + 1]->x));
      if (mglevels[i + 1]->eventinterprestrict) PetscCall(PetscLogEventEnd(mglevels[i + 1]->eventinterprestrict, 0, 0, 0, 0));
    }
    PetscCall(PCMGMCycle_Private(pc, &mglevels[l - 1], transpose, matapp, NULL));
  } else {
    PetscCall(PCMGMCycle_Private(pc, &mglevels[l - 1], transpose, matapp, NULL));
    for (i = l - 2; i >= 0; i--) {
      if (mglevels[i + 1]->eventinterprestrict) PetscCall(PetscLogEventBegin(mglevels[i + 1]->eventinterprestrict, 0, 0, 0, 0));
      if (matapp) PetscCall(MatMatRestrict(mglevels[i + 1]->interpolate, mglevels[i + 1]->X, &mglevels[i]->X));
      else PetscCall(MatRestrict(mglevels[i + 1]->interpolate, mglevels[i + 1]->x, mglevels[i]->x));
      if (mglevels[i + 1]->eventinterprestrict) PetscCall(PetscLogEventEnd(mglevels[i + 1]->eventinterprestrict, 0, 0, 0, 0));
      PetscCall(PCMGMCycle_Private(pc, &mglevels[i], transpose, matapp, NULL));
    }
    for (i = 1; i < l; i++) {
      if (mglevels[i]->eventinterprestrict) PetscCall(PetscLogEventBegin(mglevels[i]->eventinterprestrict, 0, 0, 0, 0));
      if (matapp) PetscCall(MatMatInterpolate(mglevels[i]->restrct, mglevels[i - 1]->B, &mglevels[i]->B));
      else PetscCall(MatInterpolate(mglevels[i]->restrct, mglevels[i - 1]->b, mglevels[i]->b));
      if (mglevels[i]->eventinterprestrict) PetscCall(PetscLogEventEnd(mglevels[i]->eventinterprestrict, 0, 0, 0, 0));
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PCMGKCycle_Private(PC pc, PC_MG_Levels **mglevels, PetscBool transpose, PetscBool matapp)
{
  PetscInt i, l = mglevels[0]->levels;

  PetscFunctionBegin;
  /* restrict the RHS through all levels to coarsest. */
  for (i = l - 1; i > 0; i--) {
    if (mglevels[i]->eventinterprestrict) PetscCall(PetscLogEventBegin(mglevels[i]->eventinterprestrict, 0, 0, 0, 0));
    if (matapp) PetscCall(MatMatRestrict(mglevels[i]->restrct, mglevels[i]->B, &mglevels[i - 1]->B));
    else PetscCall(MatRestrict(mglevels[i]->restrct, mglevels[i]->b, mglevels[i - 1]->b));
    if (mglevels[i]->eventinterprestrict) PetscCall(PetscLogEventEnd(mglevels[i]->eventinterprestrict, 0, 0, 0, 0));
  }

  /* work our way up through the levels */
  if (matapp) {
    if (!mglevels[0]->X) {
      PetscCall(MatDuplicate(mglevels[0]->B, MAT_DO_NOT_COPY_VALUES, &mglevels[0]->X));
    } else {
      PetscCall(MatZeroEntries(mglevels[0]->X));
    }
  } else {
    PetscCall(VecZeroEntries(mglevels[0]->x));
  }
  for (i = 0; i < l - 1; i++) {
    if (mglevels[i]->eventsmoothsolve) PetscCall(PetscLogEventBegin(mglevels[i]->eventsmoothsolve, 0, 0, 0, 0));
    if (matapp) {
      PetscCall(KSPMatSolve(mglevels[i]->smoothd, mglevels[i]->B, mglevels[i]->X));
      PetscCall(KSPCheckSolve(mglevels[i]->smoothd, pc, NULL));
    } else {
      PetscCall(KSPSolve(mglevels[i]->smoothd, mglevels[i]->b, mglevels[i]->x));
      PetscCall(KSPCheckSolve(mglevels[i]->smoothd, pc, mglevels[i]->x));
    }
    if (mglevels[i]->eventsmoothsolve) PetscCall(PetscLogEventEnd(mglevels[i]->eventsmoothsolve, 0, 0, 0, 0));
    if (mglevels[i + 1]->eventinterprestrict) PetscCall(PetscLogEventBegin(mglevels[i + 1]->eventinterprestrict, 0, 0, 0, 0));
    if (matapp) PetscCall(MatMatInterpolate(mglevels[i + 1]->interpolate, mglevels[i]->X, &mglevels[i + 1]->X));
    else PetscCall(MatInterpolate(mglevels[i + 1]->interpolate, mglevels[i]->x, mglevels[i + 1]->x));
    if (mglevels[i + 1]->eventinterprestrict) PetscCall(PetscLogEventEnd(mglevels[i + 1]->eventinterprestrict, 0, 0, 0, 0));
  }
  if (mglevels[l - 1]->eventsmoothsolve) PetscCall(PetscLogEventBegin(mglevels[l - 1]->eventsmoothsolve, 0, 0, 0, 0));
  if (matapp) {
    PetscCall(KSPMatSolve(mglevels[l - 1]->smoothd, mglevels[l - 1]->B, mglevels[l - 1]->X));
    PetscCall(KSPCheckSolve(mglevels[l - 1]->smoothd, pc, NULL));
  } else {
    PetscCall(KSPSolve(mglevels[l - 1]->smoothd, mglevels[l - 1]->b, mglevels[l - 1]->x));
    PetscCall(KSPCheckSolve(mglevels[l - 1]->smoothd, pc, mglevels[l - 1]->x));
  }
  if (mglevels[l - 1]->eventsmoothsolve) PetscCall(PetscLogEventEnd(mglevels[l - 1]->eventsmoothsolve, 0, 0, 0, 0));
  PetscFunctionReturn(PETSC_SUCCESS);
}
