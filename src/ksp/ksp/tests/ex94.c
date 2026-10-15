static char help[] = "Check the shared CG and STCG trust-region implementation.\n\n";

#include <petscksp.h>

typedef struct {
  Mat       A;
  Vec       b, diagonal, work, residual;
  PetscReal radius, previous;
  PetscInt  monitored;
  PetscBool metric, indefinite_pc;
} CheckCtx;

static PetscErrorCode ApplyPC(PC pc, Vec x, Vec y)
{
  CheckCtx *ctx;

  PetscFunctionBeginUser;
  PetscCall(PCShellGetContext(pc, &ctx));
  PetscCall(VecPointwiseDivide(y, x, ctx->diagonal));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode StepNorm(CheckCtx *ctx, Vec x, PetscReal *norm)
{
  PetscScalar dot;

  PetscFunctionBeginUser;
  if (ctx->metric) {
    PetscCall(VecPointwiseMult(ctx->work, ctx->diagonal, x));
    PetscCall(VecDot(x, ctx->work, &dot));
    *norm = PetscSqrtReal(PetscRealPart(dot));
  } else PetscCall(VecNorm(x, NORM_2, norm));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode CheckIteration(KSP ksp, PetscInt it, PetscReal rnorm, void *data)
{
  CheckCtx          *ctx = (CheckCtx *)data;
  Vec                x;
  PetscScalar        dot;
  PetscReal          obj, reported, norm;
  KSPNormType        normtype;
  KSPConvergedReason reason;

  PetscFunctionBeginUser;
  PetscCall(KSPBuildSolution(ksp, NULL, &x));
  if (ctx->radius) {
    PetscCall(StepNorm(ctx, x, &norm));
    PetscCall(KSPCGGetTRSolutionNorm(ksp, &reported));
    PetscCheck(PetscAbsReal(norm - reported) < 1.e-10, PETSC_COMM_WORLD, PETSC_ERR_PLIB, "Incorrect step norm at iteration %" PetscInt_FMT, it);
    PetscCheck(norm <= ctx->radius + 1.e-10, PETSC_COMM_WORLD, PETSC_ERR_PLIB, "Infeasible iterate");
  }
  PetscCall(MatMult(ctx->A, x, ctx->residual));
  PetscCall(VecDot(x, ctx->residual, &dot));
  obj = 0.5 * PetscRealPart(dot);
  PetscCall(VecDot(x, ctx->b, &dot));
  obj -= PetscRealPart(dot);
  PetscCall(KSPCGGetObjFcn(ksp, &reported));
  PetscCheck(PetscAbsReal(obj - reported) < 1.e-10, PETSC_COMM_WORLD, PETSC_ERR_PLIB, "Incorrect objective at iteration %" PetscInt_FMT, it);
  PetscCheck(!it || ctx->indefinite_pc || obj <= ctx->previous + 1.e-10, PETSC_COMM_WORLD, PETSC_ERR_PLIB, "Increasing objective");
  ctx->previous  = obj;
  ctx->monitored = it;

  PetscCall(VecAYPX(ctx->residual, -1.0, ctx->b));
  PetscCall(KSPGetNormType(ksp, &normtype));
  if (normtype == KSP_NORM_PRECONDITIONED || normtype == KSP_NORM_NATURAL) {
    PetscCall(VecPointwiseDivide(ctx->work, ctx->residual, ctx->diagonal));
    if (normtype == KSP_NORM_NATURAL) {
      PetscCall(VecDot(ctx->residual, ctx->work, &dot));
      norm = PetscSqrtReal(PetscAbsScalar(dot));
    } else PetscCall(VecNorm(ctx->work, NORM_2, &norm));
  } else if (normtype == KSP_NORM_UNPRECONDITIONED) PetscCall(VecNorm(ctx->residual, NORM_2, &norm));
  else norm = 0.0;
  PetscCall(KSPGetConvergedReason(ksp, &reason));
  PetscCheck((it && !reason && !rnorm) || PetscAbsReal(norm - rnorm) < 1.e-10, PETSC_COMM_WORLD, PETSC_ERR_PLIB, "Incorrect residual norm at iteration %" PetscInt_FMT ": reported %g, actual %g", it, (double)rnorm, (double)norm);
  PetscCall(KSPGetResidualNorm(ksp, &reported));
  PetscCheck(reported == rnorm, PETSC_COMM_WORLD, PETSC_ERR_PLIB, "Monitor and stored residual norms differ");
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv)
{
  CheckCtx           ctx;
  KSP                ksp;
  PC                 pc;
  Vec                x, reference;
  PetscInt           start, end, its, reference_its = 0, nhistory;
  PetscReal          guess = 0.0, target = 0.0, boundary_scale = 1.0, norm, reported, residual;
  const PetscReal   *history;
  PetscBool          negative = PETSC_FALSE, boundary_guess = PETSC_FALSE, zero_rhs = PETSC_FALSE, expect_boundary = PETSC_FALSE, api_tr_norm_type = PETSC_FALSE, axis = PETSC_FALSE;
  KSPCGTRNormType    dtype;
  KSPConvergedReason reason, reference_reason = KSP_CONVERGED_ITERATING;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  ctx.radius        = 0.45;
  ctx.indefinite_pc = PETSC_FALSE;
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-indefinite_pc", &ctx.indefinite_pc, NULL));
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-radius", &ctx.radius, NULL));
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-initial_guess", &guess, NULL));
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-target", &target, NULL));
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-boundary_scale", &boundary_scale, NULL));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-axis", &axis, NULL));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-negative", &negative, NULL));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-boundary_guess", &boundary_guess, NULL));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-zero_rhs", &zero_rhs, NULL));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-expect_boundary", &expect_boundary, NULL));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-api_tr_norm_type", &api_tr_norm_type, NULL));
  PetscCall(MatCreateAIJ(PETSC_COMM_WORLD, PETSC_DECIDE, PETSC_DECIDE, 3, 3, 3, NULL, 3, NULL, &ctx.A));
  PetscCall(MatGetOwnershipRange(ctx.A, &start, &end));
  for (PetscInt i = start; i < end; ++i) {
    PetscInt    cols[]       = {0, 1, 2};
    PetscScalar values[3][3] = {
      {negative ? -1.0 : 4.0, 1.0, 0.0},
      {1.0,                   3.0, 0.5},
      {0.0,                   0.5, 2.0}
    };

    if (axis) PetscCall(MatSetValue(ctx.A, i, i, values[i][i], INSERT_VALUES));
    else PetscCall(MatSetValues(ctx.A, 1, &i, 3, cols, values[i], INSERT_VALUES));
  }
  PetscCall(MatAssemblyBegin(ctx.A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(ctx.A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatCreateVecs(ctx.A, &x, &ctx.b));
  PetscCall(VecDuplicate(x, &ctx.diagonal));
  PetscCall(VecDuplicate(x, &ctx.work));
  PetscCall(VecDuplicate(x, &ctx.residual));
  PetscCall(VecDuplicate(x, &reference));
  for (PetscInt i = start; i < end; ++i) PetscCall(VecSetValue(ctx.diagonal, i, axis ? 4.0 : (i == 2 ? 5.0 : i + 2.0), INSERT_VALUES));
  if (ctx.indefinite_pc && !start) PetscCall(VecSetValue(ctx.diagonal, 0, -0.5, INSERT_VALUES));
  PetscCall(VecAssemblyBegin(ctx.diagonal));
  PetscCall(VecAssemblyEnd(ctx.diagonal));
  PetscCall(VecSet(ctx.b, zero_rhs ? 0.0 : 1.0));
  if (axis) {
    PetscCall(VecSet(ctx.b, 0.0));
    if (!start) PetscCall(VecSetValue(ctx.b, 0, 1.0, INSERT_VALUES));
    PetscCall(VecAssemblyBegin(ctx.b));
    PetscCall(VecAssemblyEnd(ctx.b));
  }

  for (PetscInt solver = 0; solver < 2; ++solver) {
    PetscCall(KSPCreate(PETSC_COMM_WORLD, &ksp));
    PetscCall(KSPSetType(ksp, solver ? KSPSTCG : KSPCG));
    PetscCall(KSPSetOperators(ksp, ctx.A, ctx.A));
    PetscCall(KSPGetPC(ksp, &pc));
    PetscCall(PCSetType(pc, PCSHELL));
    PetscCall(PCShellSetContext(pc, &ctx));
    PetscCall(PCShellSetApply(pc, ApplyPC));
    if (!solver) PetscCall(KSPSetConvergedNegativeCurvature(ksp, PETSC_TRUE));
    PetscCall(KSPSetTolerances(ksp, 1.e-12, 1.e-12, PETSC_CURRENT, 10));
    PetscCall(KSPSetFromOptions(ksp));
    PetscCall(KSPCGSetRadius(ksp, ctx.radius));
    PetscCall(KSPCGSetObjectiveTarget(ksp, target));
    if (api_tr_norm_type) PetscCall(KSPCGSetTRNormType(ksp, KSP_CG_TR_NORM_PRECONDITIONED));
    PetscCall(KSPCGGetTRNormType(ksp, &dtype));
    ctx.metric = (PetscBool)(dtype == KSP_CG_TR_NORM_PRECONDITIONED);
    PetscCall(KSPSetInitialGuessNonzero(ksp, (PetscBool)(guess != 0.0 || boundary_guess)));
    PetscCall(KSPSetResidualHistory(ksp, NULL, PETSC_DECIDE, PETSC_TRUE));
    PetscCall(KSPMonitorSet(ksp, CheckIteration, &ctx, NULL));
    for (PetscInt repeat = 0; repeat < 2; ++repeat) {
      PetscCall(VecSet(x, guess));
      if (boundary_guess) {
        if (!start) PetscCall(VecSetValue(x, 0, boundary_scale * ctx.radius, INSERT_VALUES));
        PetscCall(VecAssemblyBegin(x));
        PetscCall(VecAssemblyEnd(x));
      }
      PetscCall(KSPSolve(ksp, ctx.b, x));
      PetscCall(KSPGetConvergedReason(ksp, &reason));
      PetscCall(KSPGetIterationNumber(ksp, &its));
      PetscCheck(ctx.indefinite_pc ? reason == KSP_DIVERGED_INDEFINITE_PC : reason > 0, PETSC_COMM_WORLD, PETSC_ERR_PLIB, "Unexpected reason %s", KSPConvergedReasons[reason]);
      PetscCheck(ctx.monitored == its, PETSC_COMM_WORLD, PETSC_ERR_PLIB, "Final iterate was not monitored");
      PetscCall(KSPGetResidualNorm(ksp, &residual));
      PetscCall(CheckIteration(ksp, its, residual, &ctx));
      PetscCall(KSPGetResidualHistory(ksp, &history, &nhistory));
      PetscCheck(nhistory == its + 1 && history[nhistory - 1] == residual, PETSC_COMM_WORLD, PETSC_ERR_PLIB, "Incorrect residual history");
      PetscCall(KSPBuildResidual(ksp, NULL, ctx.work, &ctx.work));
      PetscCall(VecAXPY(ctx.work, -1.0, ctx.residual));
      PetscCall(VecNorm(ctx.work, NORM_2, &norm));
      PetscCheck(norm < 1.e-10, PETSC_COMM_WORLD, PETSC_ERR_PLIB, "Incorrect stored residual");
      PetscCall(StepNorm(&ctx, x, &norm));
      PetscCall(KSPCGGetTRSolutionNorm(ksp, &reported));
      PetscCheck(PetscAbsReal(norm - reported) < 1.e-10, PETSC_COMM_WORLD, PETSC_ERR_PLIB, "Incorrect final step norm");
      if (ctx.radius && (reason == KSP_CONVERGED_STEP_LENGTH || reason == KSP_CONVERGED_NEG_CURVE)) PetscCheck(reported == ctx.radius, PETSC_COMM_WORLD, PETSC_ERR_PLIB, "Boundary norm must equal the radius for TAO radius updates");
      if (ctx.indefinite_pc) {
        PetscReal step = PetscMin(1.0, ctx.radius / PetscSqrtReal(3.0));

        PetscCall(VecCopy(x, ctx.work));
        PetscCall(VecAXPY(ctx.work, -step, ctx.b));
        PetscCall(VecNorm(ctx.work, NORM_2, &residual));
        PetscCheck(its == 1 && residual < 1.e-10, PETSC_COMM_WORLD, PETSC_ERR_PLIB, "Incorrect gradient fallback for an indefinite preconditioner");
      }
      if (expect_boundary) PetscCheck(reason == KSP_CONVERGED_STEP_LENGTH && PetscAbsReal(norm - ctx.radius) < 1.e-10, PETSC_COMM_WORLD, PETSC_ERR_PLIB, "Incorrect boundary termination");
      if (negative) PetscCheck(reason == KSP_CONVERGED_NEG_CURVE, PETSC_COMM_WORLD, PETSC_ERR_PLIB, "Did not stop at negative curvature");
      if (target) PetscCheck(ctx.previous <= target, PETSC_COMM_WORLD, PETSC_ERR_PLIB, "Objective target not reached");
      if (axis) {
        PetscCall(VecCopy(x, ctx.work));
        if (!start) PetscCall(VecSetValue(ctx.work, 0, 0.0, INSERT_VALUES));
        PetscCall(VecAssemblyBegin(ctx.work));
        PetscCall(VecAssemblyEnd(ctx.work));
        PetscCall(VecNorm(ctx.work, NORM_2, &reported));
        PetscCheck(reported < 1.e-12 && PetscAbsReal(norm - PetscMin(ctx.radius, ctx.metric ? 0.5 : 0.25)) < 1.e-12, PETSC_COMM_WORLD, PETSC_ERR_PLIB, "Incorrect solution of the scalar trust-region problem");
      }
      if (!solver && !repeat) {
        PetscCall(VecCopy(x, reference));
        reference_its    = its;
        reference_reason = reason;
      } else {
        PetscCall(VecCopy(x, ctx.work));
        PetscCall(VecAXPY(ctx.work, -1.0, reference));
        PetscCall(VecNorm(ctx.work, NORM_2, &norm));
        PetscCheck(norm < 1.e-10 && its == reference_its && reason == reference_reason, PETSC_COMM_WORLD, PETSC_ERR_PLIB, "CG and STCG or repeated solves differ");
      }
    }
    PetscCall(KSPDestroy(&ksp));
  }
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Shared CG and STCG checks passed\n"));
  PetscCall(VecDestroy(&reference));
  PetscCall(VecDestroy(&ctx.residual));
  PetscCall(VecDestroy(&ctx.work));
  PetscCall(VecDestroy(&ctx.diagonal));
  PetscCall(VecDestroy(&ctx.b));
  PetscCall(VecDestroy(&x));
  PetscCall(MatDestroy(&ctx.A));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

  testset:
    requires: !complex double
    output_file: output/ex94.out
    nsize: {{1 2}}
    test:
      suffix: boundary
      args: -radius 0.45 -expect_boundary -ksp_cg_tr_norm_type {{preconditioned unpreconditioned}} -ksp_norm_type {{natural preconditioned unpreconditioned none}}
    test:
      suffix: interior
      args: -radius {{0 10}} -ksp_cg_tr_norm_type {{preconditioned unpreconditioned}} -ksp_norm_type {{natural preconditioned unpreconditioned}}
    test:
      suffix: later_boundary
      args: -radius 0.8 -expect_boundary -api_tr_norm_type
    test:
      suffix: curvature
      args: -negative -radius {{0 10}} -ksp_cg_tr_norm_type {{preconditioned unpreconditioned}} -ksp_norm_type {{natural preconditioned unpreconditioned}}
    test:
      suffix: guess
      args: -initial_guess 0.1 -radius 0.45 -expect_boundary
    test:
      suffix: inward
      args: -boundary_guess -radius 1
    test:
      suffix: target
      args: -radius 10 -target -0.4 -ksp_cg_tr_norm_type {{preconditioned unpreconditioned}}
    test:
      suffix: zero_rhs
      args: -zero_rhs -ksp_cg_tr_norm_type {{preconditioned unpreconditioned}} -ksp_norm_type {{unpreconditioned none}}
    test:
      suffix: delayed_norm
      args: -expect_boundary -ksp_check_norm_iteration 10 -ksp_norm_type {{preconditioned unpreconditioned}}
    test:
      suffix: delayed_target
      args: -radius 10 -target -0.4 -ksp_check_norm_iteration 10 -ksp_norm_type {{preconditioned unpreconditioned}}
    test:
      suffix: exact_boundary
      args: -axis -radius 0.25 -expect_boundary -ksp_norm_type {{unpreconditioned none}}
    test:
      suffix: exact_metric_boundary
      args: -axis -radius 0.5 -expect_boundary -api_tr_norm_type -ksp_norm_type {{unpreconditioned none}}
    test:
      suffix: outward
      args: -axis -radius 0.125 -boundary_guess -boundary_scale {{1 0.99999999999999 1.0000000000000002}} -expect_boundary
    test:
      suffix: inward_axis
      args: -axis -radius 1 -boundary_guess
    test:
      suffix: no_norm
      args: -axis -radius 10 -ksp_norm_type none
    test:
      suffix: gradient_fallback
      args: -indefinite_pc -radius {{0.45 10}} -ksp_norm_type {{natural preconditioned unpreconditioned none}}

TEST*/
