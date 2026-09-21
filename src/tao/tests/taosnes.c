static char help[] = "Check TAOSNES convergence reasons on a quadratic problem.\n";

#include <petsctao.h>

static PetscErrorCode FormObjectiveGradient(Tao tao, Vec X, PetscReal *f, Vec G, PetscCtx ctx)
{
  PetscScalar dot;

  PetscFunctionBeginUser;
  PetscCall(VecDot(X, X, &dot));
  *f = 0.5 * PetscRealPart(dot);
  PetscCall(VecCopy(X, G));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode FormHessian(Tao tao, Vec X, Mat H, Mat P, PetscCtx ctx)
{
  PetscFunctionBeginUser;
  PetscCall(MatZeroEntries(H));
  PetscCall(MatShift(H, 1.0));
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv)
{
  const SNESConvergedReason snes_reasons[] = {SNES_CONVERGED_FNORM_ABS, SNES_DIVERGED_MAX_IT, SNES_CONVERGED_FNORM_RELATIVE, SNES_DIVERGED_FUNCTION_COUNT, SNES_DIVERGED_LINEAR_SOLVE};
  const TaoConvergedReason  tao_reasons[]  = {TAO_CONVERGED_GATOL, TAO_DIVERGED_MAXITS, TAO_CONVERGED_GTTOL, TAO_DIVERGED_MAXFCN, TAO_DIVERGED_USER};
  Tao                       tao;
  SNES                      snes;
  KSP                       ksp;
  PC                        pc;
  Vec                       X;
  Mat                       H;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  PetscCall(VecCreateSeq(PETSC_COMM_SELF, 2, &X));
  PetscCall(MatCreateSeqAIJ(PETSC_COMM_SELF, 2, 2, 1, NULL, &H));
  for (PetscInt i = 0; i < 2; ++i) PetscCall(MatSetValue(H, i, i, 1.0, INSERT_VALUES));
  PetscCall(MatAssemblyBegin(H, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(H, MAT_FINAL_ASSEMBLY));
  PetscCall(TaoCreate(PETSC_COMM_SELF, &tao));
  PetscCall(TaoSetType(tao, TAOSNES));
  PetscCall(TaoSetSolution(tao, X));
  PetscCall(TaoSetObjectiveAndGradient(tao, NULL, FormObjectiveGradient, NULL));
  PetscCall(TaoSetHessian(tao, H, H, FormHessian, NULL));
  PetscCall(TaoSNESGetSNES(tao, &snes));
  PetscCall(TaoSetFromOptions(tao));
  PetscCall(TaoSetUp(tao));
  PetscCall(SNESGetKSP(snes, &ksp));
  PetscCall(KSPSetType(ksp, KSPCG));
  PetscCall(KSPGetPC(ksp, &pc));
  PetscCall(PCSetType(pc, PCNONE));
  for (size_t test = 0; test < PETSC_STATIC_ARRAY_LENGTH(snes_reasons); ++test) {
    SNESConvergedReason snes_reason;
    TaoConvergedReason  tao_reason;
    PetscInt            snes_its, tao_its;

    PetscCall(VecSet(X, 1.0));
    PetscCall(SNESSetTolerances(snes, test == 2 ? 0.0 : 1.e-12, test == 2 ? 0.5 : 0.0, 0.0, test == 1 ? 0 : 10, test == 3 ? 1 : 100));
    PetscCall(KSPSetTolerances(ksp, 1.e-12, PETSC_CURRENT, PETSC_CURRENT, test == 4 ? 0 : 10));
    PetscCall(TaoSolve(tao));
    PetscCall(SNESGetConvergedReason(snes, &snes_reason));
    PetscCall(TaoGetConvergedReason(tao, &tao_reason));
    PetscCall(SNESGetIterationNumber(snes, &snes_its));
    PetscCall(TaoGetIterationNumber(tao, &tao_its));
    PetscCheck(snes_reason == snes_reasons[test], PETSC_COMM_SELF, PETSC_ERR_PLIB, "Unexpected SNES reason %s", SNESConvergedReasons[snes_reason]);
    PetscCheck(tao_reason == tao_reasons[test], PETSC_COMM_SELF, PETSC_ERR_PLIB, "SNES reason %s mapped to TAO reason %s", SNESConvergedReasons[snes_reason], TaoConvergedReasons[tao_reason]);
    PetscCheck(tao_its == snes_its, PETSC_COMM_SELF, PETSC_ERR_PLIB, "TAO and SNES iteration counts differ");
    PetscCall(PetscPrintf(PETSC_COMM_SELF, "SNES %s -> TAO %s\n", SNESConvergedReasons[snes_reason], TaoConvergedReasons[tao_reason]));
  }
  PetscCall(TaoDestroy(&tao));
  PetscCall(MatDestroy(&H));
  PetscCall(VecDestroy(&X));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

  test:
    suffix: 1
    requires: !single !complex
    args: -snes_type {{newtonls newtontr}}

TEST*/
