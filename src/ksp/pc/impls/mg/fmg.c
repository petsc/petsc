/*
     Full multigrid using either additive or multiplicative V or W cycle
*/
#include <petsc/private/pcmgimpl.h>

/*
  Transpose of the kaskade cycle, and of the full cycle when full is PETSC_TRUE.

  The kaskade cycle applies M_i = E_i P_i M_{i-1} R_i + N_i on level i, where N_i is the level solve and
  E_i = I - N_i A_i, hence M_i^T = R_i^T M_{i-1}^T P_i^T E_i^T + N_i^T with M_0^T = N_0^T. N_i^T is applied by
  KSPSolveTranspose() from a zero initial guess and E_i^T by the transposed residual that follows it.

  The full cycle is the same recursion with the level multiplicative cycle in place of the single smoother solve,
  so the transposed multiplicative cycle plays the smoother's role there.
*/
static PetscErrorCode PCMGKCycleTranspose_Private(PC pc, PC_MG_Levels **mglevels, PetscBool full)
{
  PetscInt l = mglevels[0]->levels;

  PetscFunctionBegin;
  /* apply the transposed level solves from the finest level down, restricting the transposed residual with P^T */
  for (PetscInt i = l - 1; i >= 0; i--) {
    PetscCall(VecZeroEntries(mglevels[i]->x));
    if (full) PetscCall(PCMGMCycle_Private(pc, &mglevels[i], PETSC_TRUE, PETSC_FALSE, NULL));
    else {
      if (mglevels[i]->eventsmoothsolve) PetscCall(PetscLogEventBegin(mglevels[i]->eventsmoothsolve, 0, 0, 0, 0));
      PetscCall(KSPSolveTranspose(mglevels[i]->smoothd, mglevels[i]->b, mglevels[i]->x));
      PetscCall(KSPCheckSolve(mglevels[i]->smoothd, pc, mglevels[i]->x));
      if (mglevels[i]->eventsmoothsolve) PetscCall(PetscLogEventEnd(mglevels[i]->eventsmoothsolve, 0, 0, 0, 0));
    }
    if (i) {
      if (mglevels[i]->eventresidual) PetscCall(PetscLogEventBegin(mglevels[i]->eventresidual, 0, 0, 0, 0));
      PetscCall((*mglevels[i]->residualtranspose)(mglevels[i]->A, mglevels[i]->b, mglevels[i]->x, mglevels[i]->r));
      if (mglevels[i]->eventresidual) PetscCall(PetscLogEventEnd(mglevels[i]->eventresidual, 0, 0, 0, 0));
      if (mglevels[i]->eventinterprestrict) PetscCall(PetscLogEventBegin(mglevels[i]->eventinterprestrict, 0, 0, 0, 0));
      PetscCall(MatRestrict(mglevels[i]->interpolate, mglevels[i]->r, mglevels[i - 1]->b));
      if (mglevels[i]->eventinterprestrict) PetscCall(PetscLogEventEnd(mglevels[i]->eventinterprestrict, 0, 0, 0, 0));
    }
  }

  /* accumulate the coarse contributions back up through the levels with R^T */
  for (PetscInt i = 1; i < l; i++) {
    if (mglevels[i]->eventinterprestrict) PetscCall(PetscLogEventBegin(mglevels[i]->eventinterprestrict, 0, 0, 0, 0));
    PetscCall(MatInterpolateAdd(mglevels[i]->restrct, mglevels[i - 1]->x, mglevels[i]->x, mglevels[i]->x));
    if (mglevels[i]->eventinterprestrict) PetscCall(PetscLogEventEnd(mglevels[i]->eventinterprestrict, 0, 0, 0, 0));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PCMGFCycle_Private(PC pc, PC_MG_Levels **mglevels, PetscBool transpose, PetscBool matapp)
{
  PetscInt l = mglevels[0]->levels;

  PetscFunctionBegin;
  if (!transpose) {
    /* restrict the RHS through all levels to coarsest. */
    for (PetscInt i = l - 1; i > 0; i--) {
      if (mglevels[i]->eventinterprestrict) PetscCall(PetscLogEventBegin(mglevels[i]->eventinterprestrict, 0, 0, 0, 0));
      if (matapp) PetscCall(MatMatRestrict(mglevels[i]->restrct, mglevels[i]->B, &mglevels[i - 1]->B));
      else PetscCall(MatRestrict(mglevels[i]->restrct, mglevels[i]->b, mglevels[i - 1]->b));
      if (mglevels[i]->eventinterprestrict) PetscCall(PetscLogEventEnd(mglevels[i]->eventinterprestrict, 0, 0, 0, 0));
    }

    /* work our way up through the levels */
    if (matapp) {
      if (!mglevels[0]->X) PetscCall(MatDuplicate(mglevels[0]->B, MAT_DO_NOT_COPY_VALUES, &mglevels[0]->X));
      else PetscCall(MatZeroEntries(mglevels[0]->X));
    } else PetscCall(VecZeroEntries(mglevels[0]->x));
    for (PetscInt i = 0; i < l - 1; i++) {
      PetscCall(PCMGMCycle_Private(pc, &mglevels[i], transpose, matapp, NULL));
      if (mglevels[i + 1]->eventinterprestrict) PetscCall(PetscLogEventBegin(mglevels[i + 1]->eventinterprestrict, 0, 0, 0, 0));
      if (matapp) PetscCall(MatMatInterpolate(mglevels[i + 1]->interpolate, mglevels[i]->X, &mglevels[i + 1]->X));
      else PetscCall(MatInterpolate(mglevels[i + 1]->interpolate, mglevels[i]->x, mglevels[i + 1]->x));
      if (mglevels[i + 1]->eventinterprestrict) PetscCall(PetscLogEventEnd(mglevels[i + 1]->eventinterprestrict, 0, 0, 0, 0));
    }
    PetscCall(PCMGMCycle_Private(pc, &mglevels[l - 1], transpose, matapp, NULL));
  } else {
    PetscCheck(!matapp, PetscObjectComm((PetscObject)pc), PETSC_ERR_SUP, "Not supported");
    PetscCall(PCMGKCycleTranspose_Private(pc, mglevels, PETSC_TRUE));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PCMGKCycle_Private(PC pc, PC_MG_Levels **mglevels, PetscBool transpose, PetscBool matapp)
{
  PetscInt l = mglevels[0]->levels;

  PetscFunctionBegin;
  if (transpose) {
    PetscCheck(!matapp, PetscObjectComm((PetscObject)pc), PETSC_ERR_SUP, "Not supported");
    PetscCall(PCMGKCycleTranspose_Private(pc, mglevels, PETSC_FALSE));
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  /* restrict the RHS through all levels to coarsest. */
  for (PetscInt i = l - 1; i > 0; i--) {
    if (mglevels[i]->eventinterprestrict) PetscCall(PetscLogEventBegin(mglevels[i]->eventinterprestrict, 0, 0, 0, 0));
    if (matapp) PetscCall(MatMatRestrict(mglevels[i]->restrct, mglevels[i]->B, &mglevels[i - 1]->B));
    else PetscCall(MatRestrict(mglevels[i]->restrct, mglevels[i]->b, mglevels[i - 1]->b));
    if (mglevels[i]->eventinterprestrict) PetscCall(PetscLogEventEnd(mglevels[i]->eventinterprestrict, 0, 0, 0, 0));
  }

  /* work our way up through the levels */
  if (matapp) {
    if (!mglevels[0]->X) PetscCall(MatDuplicate(mglevels[0]->B, MAT_DO_NOT_COPY_VALUES, &mglevels[0]->X));
    else {
      PetscCall(MatZeroEntries(mglevels[0]->X));
    }
  } else {
    PetscCall(VecZeroEntries(mglevels[0]->x));
  }
  for (PetscInt i = 0; i < l - 1; i++) {
    if (mglevels[i]->eventsmoothsolve) PetscCall(PetscLogEventBegin(mglevels[i]->eventsmoothsolve, 0, 0, 0, 0));
    if (matapp) {
      PetscCall(KSPMatSolve(mglevels[i]->smoothd, mglevels[i]->B, mglevels[i]->X));
      PetscCall(KSPCheckMatSolve(mglevels[i]->smoothd, pc, mglevels[i]->X));
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
    PetscCall(KSPCheckMatSolve(mglevels[l - 1]->smoothd, pc, mglevels[l - 1]->X));
  } else {
    PetscCall(KSPSolve(mglevels[l - 1]->smoothd, mglevels[l - 1]->b, mglevels[l - 1]->x));
    PetscCall(KSPCheckSolve(mglevels[l - 1]->smoothd, pc, mglevels[l - 1]->x));
  }
  if (mglevels[l - 1]->eventsmoothsolve) PetscCall(PetscLogEventEnd(mglevels[l - 1]->eventsmoothsolve, 0, 0, 0, 0));
  PetscFunctionReturn(PETSC_SUCCESS);
}
