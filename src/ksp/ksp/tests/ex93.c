static char help[] = "Check CG quadratic objectives and stopping criteria.\n\n";

#include <petscksp.h>

typedef struct {
  Mat       A;
  Vec       b, work;
  PetscReal sign, previous;
} ObjectiveCtx;

static PetscErrorCode CheckObjective(KSP ksp, PetscInt it, PetscReal rnorm, void *ctx)
{
  ObjectiveCtx *q = (ObjectiveCtx *)ctx;
  Vec           x;
  PetscScalar   dot;
  PetscReal     objective, reported, tolerance;

  PetscFunctionBeginUser;
  PetscCall(KSPBuildSolution(ksp, NULL, &x));
  PetscCall(MatMult(q->A, x, q->work));
  PetscCall(VecDot(x, q->work, &dot));
  objective = 0.5 * PetscRealPart(dot);
  PetscCall(VecDot(x, q->b, &dot));
  objective -= PetscRealPart(dot);
  tolerance = 1000.0 * PETSC_MACHINE_EPSILON * PetscMax(1.0, PetscAbsReal(objective));
  PetscCall(KSPCGGetObjFcn(ksp, &reported));
  PetscCheck(PetscAbsReal(objective - reported) < tolerance, PETSC_COMM_WORLD, PETSC_ERR_PLIB, "Incorrect objective at iteration %" PetscInt_FMT ": reported %g, actual %g", it, (double)reported, (double)objective);
  PetscCheck(!it || q->sign * (objective - q->previous) < tolerance, PETSC_COMM_WORLD, PETSC_ERR_PLIB, "Objective changed in the wrong direction at iteration %" PetscInt_FMT, it);
  q->previous = objective;
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv)
{
  ObjectiveCtx       q;
  Vec                x, residual_vec;
  KSP                ksp;
  PC                 pc;
  PetscInt           start, end, its;
  PetscReal          guess = 0.0, radius = 0.0, target = 0.0, norm, residual;
  PetscBool          negative = PETSC_FALSE, zero_rhs = PETSC_FALSE, indefinite = PETSC_FALSE, zero_curvature = PETSC_FALSE, eigenvalues = PETSC_FALSE, check_objective = PETSC_TRUE, initial_target = PETSC_FALSE, truncate;
  KSPConvergedReason reason;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-initial_guess", &guess, NULL));
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-radius", &radius, NULL));
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-target", &target, NULL));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-negative", &negative, NULL));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-zero_rhs", &zero_rhs, NULL));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-indefinite", &indefinite, NULL));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-zero_curvature", &zero_curvature, NULL));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-check_eigenvalues", &eigenvalues, NULL));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-check_objective", &check_objective, NULL));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-initial_target", &initial_target, NULL));
  q.sign = negative ? -1.0 : 1.0;
  PetscCall(MatCreateAIJ(PETSC_COMM_WORLD, PETSC_DECIDE, PETSC_DECIDE, 3, 3, 1, NULL, 0, NULL, &q.A));
  PetscCall(MatGetOwnershipRange(q.A, &start, &end));
  for (PetscInt i = start; i < end; ++i) PetscCall(MatSetValue(q.A, i, i, q.sign * (i ? (1 << i) : (zero_curvature ? -6.0 : (indefinite ? -1.0 : 1.0))), INSERT_VALUES));
  PetscCall(MatAssemblyBegin(q.A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(q.A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatCreateVecs(q.A, &x, &q.b));
  PetscCall(VecDuplicate(x, &q.work));
  PetscCall(VecDuplicate(x, &residual_vec));
  PetscCall(VecSet(q.b, zero_rhs ? 0.0 : 1.0));
  PetscCall(KSPCreate(PETSC_COMM_WORLD, &ksp));
  PetscCall(KSPSetType(ksp, KSPCG));
  PetscCall(KSPSetOperators(ksp, q.A, q.A));
  PetscCall(KSPGetPC(ksp, &pc));
  PetscCall(PCSetType(pc, PCNONE));
  PetscCall(KSPSetInitialGuessNonzero(ksp, guess != 0.0 ? PETSC_TRUE : PETSC_FALSE));
  PetscCall(KSPSetComputeEigenvalues(ksp, eigenvalues));
  PetscCall(KSPSetTolerances(ksp, 1.e-12, 1.e-12, PETSC_CURRENT, 10));
  if (check_objective) PetscCall(KSPMonitorSet(ksp, CheckObjective, &q, NULL));
  PetscCall(KSPSetFromOptions(ksp));
  PetscCall(KSPCGSetRadius(ksp, radius));
  PetscCall(KSPCGSetObjectiveTarget(ksp, target));
  PetscCall(KSPGetConvergedNegativeCurvature(ksp, &truncate));
  if (truncate || radius) q.sign = 1.0;
  for (PetscInt solve = 0; solve < 2; ++solve) {
    PetscCall(VecSet(x, guess));
    PetscCall(KSPSolve(ksp, q.b, x));
    PetscCall(KSPGetIterationNumber(ksp, &its));
    if (check_objective) PetscCall(CheckObjective(ksp, its, 0.0, &q));
    PetscCall(KSPGetConvergedReason(ksp, &reason));
    if (initial_target) {
      PetscCheck(its == 0 && reason == KSP_CONVERGED_ATOL, PETSC_COMM_WORLD, PETSC_ERR_PLIB, "Did not accept the initial objective target: iterations %" PetscInt_FMT ", reason %s", its, KSPConvergedReasons[reason]);
      PetscCall(VecCopy(x, q.work));
      PetscCall(VecShift(q.work, -guess));
      PetscCall(VecNorm(q.work, NORM_2, &norm));
      PetscCheck(norm < 1.e-10, PETSC_COMM_WORLD, PETSC_ERR_PLIB, "Changed an initial guess that satisfies the objective target");
    } else if ((truncate || radius) && (negative || indefinite || zero_curvature)) {
      PetscCheck(reason == (truncate ? KSP_CONVERGED_NEG_CURVE : KSP_CONVERGED_STEP_LENGTH), PETSC_COMM_WORLD, PETSC_ERR_PLIB, "Unexpected curvature reason: %s", KSPConvergedReasons[reason]);
      if (negative || zero_curvature) PetscCheck(its == 1, PETSC_COMM_WORLD, PETSC_ERR_PLIB, "Did not stop at the first nonpositive curvature direction");
    } else PetscCheck(indefinite || zero_curvature ? reason == KSP_DIVERGED_INDEFINITE_MAT : reason > 0, PETSC_COMM_WORLD, PETSC_ERR_PLIB, "Unexpected CG reason: %s", KSPConvergedReasons[reason]);
    if (radius > 0.0) {
      PetscCall(VecNorm(x, NORM_2, &norm));
      PetscCheck(norm <= radius + 1.e-10, PETSC_COMM_WORLD, PETSC_ERR_PLIB, "Infeasible step norm %g", (double)norm);
      if (!initial_target)
        PetscCheck((reason == KSP_CONVERGED_STEP_LENGTH || reason == KSP_CONVERGED_NEG_CURVE) && PetscAbsReal(norm - radius) < 1.e-10, PETSC_COMM_WORLD, PETSC_ERR_PLIB, "Incorrect boundary step: norm %g, reason %s", (double)norm, KSPConvergedReasons[reason]);
    }
    if (target < 0.0) PetscCheck(q.previous <= target, PETSC_COMM_WORLD, PETSC_ERR_PLIB, "Objective target %g not reached: %g", (double)target, (double)q.previous);
    PetscCall(MatMult(q.A, x, q.work));
    PetscCall(VecAYPX(q.work, -1.0, q.b));
    PetscCall(VecNorm(q.work, NORM_2, &residual));
    if (!radius && !target && !indefinite && !zero_curvature && !(truncate && negative)) PetscCheck(residual < 1.e-10, PETSC_COMM_WORLD, PETSC_ERR_PLIB, "Linear system residual %g", (double)residual);
    PetscCall(KSPBuildResidual(ksp, NULL, residual_vec, &residual_vec));
    PetscCall(VecAXPY(residual_vec, -1.0, q.work));
    PetscCall(VecNorm(residual_vec, NORM_2, &residual));
    PetscCheck(residual < 1.e-10, PETSC_COMM_WORLD, PETSC_ERR_PLIB, "Incorrect stored CG residual %g", (double)residual);
    if (eigenvalues) {
      PetscInt  n;
      PetscReal real[3], imag[3];

      PetscCall(KSPComputeEigenvalues(ksp, 3, real, imag, &n));
      PetscCheck(n == its, PETSC_COMM_WORLD, PETSC_ERR_PLIB, "Incorrect number of Ritz values");
      for (PetscInt i = 0; i < n; ++i) {
        PetscReal expected = n == 1 ? (zero_curvature ? 0.0 : -7.0 / 3.0) : (negative ? -(1 << (2 - i)) : (1 << i));

        PetscCheck(PetscAbsReal(real[i] - expected) < 1.e-10 && imag[i] == 0.0, PETSC_COMM_WORLD, PETSC_ERR_PLIB, "Incorrect Ritz value %g + %gi, expected %g", (double)real[i], (double)imag[i], (double)expected);
      }
    }
  }
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "CG checks passed\n"));
  PetscCall(KSPDestroy(&ksp));
  PetscCall(VecDestroy(&residual_vec));
  PetscCall(VecDestroy(&q.work));
  PetscCall(VecDestroy(&q.b));
  PetscCall(VecDestroy(&x));
  PetscCall(MatDestroy(&q.A));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

  testset:
    requires: !complex double
    output_file: output/ex93.out
    test:
      suffix: linear
      nsize: {{1 2}}
      args: -initial_guess {{0 0.1}} -negative {{0 1}} -ksp_norm_type {{unpreconditioned preconditioned natural}}
    test:
      suffix: boundary
      nsize: {{1 2}}
      args: -initial_guess {{0 0.1}} -radius 0.3
    test:
      suffix: target
      args: -initial_guess {{0 0.1}} -target {{-0.2 -0.7}}
    test:
      suffix: initial_target
      nsize: {{1 2}}
      args: -initial_target -initial_guess 0.1 -target -0.2 -negative {{0 1}} -radius {{0 0.3}} -ksp_converged_neg_curve {{0 1}}
    test:
      suffix: zero_rhs
      args: -zero_rhs -initial_guess {{0 0.1}}
    test:
      suffix: preconditioner
      nsize: {{1 2}}
      args: -initial_guess 0.1 -negative {{0 1}} -pc_type jacobi
    test:
      suffix: negative_curvature
      nsize: {{1 2}}
      args: -negative -ksp_converged_neg_curve -initial_guess {{0 0.1}} -radius {{0 0.3}}
    test:
      suffix: radius_negative_curvature
      nsize: {{1 2}}
      args: -negative -initial_guess {{0 0.1}} -radius 0.3
    test:
      suffix: radius_indefinite
      nsize: {{1 2}}
      args: -indefinite -radius 10
    test:
      suffix: radius_zero_curvature
      nsize: {{1 2}}
      args: -zero_curvature -radius 0.3
    test:
      suffix: boundary_endpoint
      args: -negative -zero_rhs -initial_guess 0.1 -radius 0.3 -target -0.15 -ksp_converged_neg_curve
    test:
      suffix: indefinite
      nsize: {{1 2}}
      args: -indefinite -negative {{0 1}}
    test:
      suffix: indefinite_stop
      nsize: {{1 2}}
      args: -indefinite -ksp_converged_neg_curve -radius {{0 10}}
    test:
      suffix: zero_curvature
      nsize: {{1 2}}
      args: -zero_curvature -ksp_converged_neg_curve -radius {{0 0.3}}
    test:
      suffix: eigenvalues
      args: -check_eigenvalues -negative {{0 1}}
    test:
      suffix: curvature_eigenvalues
      args: -check_eigenvalues -negative -ksp_converged_neg_curve -radius {{0 0.3}}
    test:
      suffix: zero_eigenvalues
      args: -check_eigenvalues -zero_curvature -ksp_converged_neg_curve -radius 0.3
    test:
      suffix: single_reduction
      nsize: {{1 2}}
      args: -ksp_cg_single_reduction -check_objective 0 -initial_guess {{0 0.1}} -negative {{0 1}} -ksp_norm_type {{unpreconditioned preconditioned natural}}
    test:
      suffix: single_reduction_eigenvalues
      args: -ksp_cg_single_reduction -check_objective 0 -check_eigenvalues -negative {{0 1}}

  testset:
    requires: !complex double
    output_file: output/ex93.out
    args: -ksp_type stcg
    nsize: {{1 2}}
    test:
      suffix: stcg_linear
      args: -initial_guess {{0 0.1}} -negative {{0 1}} -ksp_converged_neg_curve 0
    test:
      suffix: stcg_boundary
      args: -initial_guess {{0 0.1}} -radius 0.3 -ksp_norm_type {{unpreconditioned preconditioned natural none}}
    test:
      suffix: stcg_target
      args: -initial_guess {{0 0.1}} -target {{-0.2 -0.7}}
    test:
      suffix: stcg_curvature
      args: -negative -initial_guess {{0 0.1}} -radius {{0 0.3}}
    test:
      suffix: stcg_endpoint
      args: -negative -zero_rhs -initial_guess 0.1 -radius 0.3 -target -0.15
    test:
      suffix: stcg_zero_rhs
      args: -zero_rhs -initial_guess {{0 0.1}}
    test:
      suffix: stcg_eigenvalues
      args: -check_eigenvalues -zero_curvature -radius 0.3

TEST*/
