const char help[] = "TaoTerm coverage test comparing TaoTerm interface with traditional callbacks for Rosenbrock problem.\n\
Tests different TaoTerm configurations for L1, and HALFL2SQUARED types with various matrix and parameter options.\n";

#include <petsctao.h>
#include "../unconstrained/tutorials/rosenbrock4.h"

typedef struct {
  AppCtx    user;  /* Note: AppCtx is a pointer type in rosenbrock4.h */
  AppCtx    user2; /* Second user context for callback version */
  PetscBool test_print;
  PetscBool test_print_map;
  PetscBool use_term1;
  PetscBool term1_has_A;
  PetscBool term1_has_params;
  PetscReal term1_scale;
  PetscBool use_term2;
  PetscBool term2_has_A;
  PetscBool term2_has_params;
  PetscReal term2_scale;
  PetscReal term1_scale_callback; /* Callback-only scale options (for testing equivalence with TaoTerm scales) */
  PetscReal term2_scale_callback;
  PetscInt  map_row_size;
  PetscBool print_debug;
  TaoTerm   term1, term2;
  Vec       term1_params, term2_params;
  Vec       X_mapped1, X_mapped2, G_mapped1, G_mapped2, G_work;
  Mat       term1_A_callback; /* AIJ version for callbacks */
  Mat       term2_A_callback; /* AIJ version for callbacks */
} TestCtx;

/* Forward declarations */
static PetscErrorCode TestCtxInitialize(MPI_Comm, TestCtx *);
static PetscErrorCode TestCtxFinalize(TestCtx *);
static PetscErrorCode CreateTaoTermWithOptions(TestCtx *, TaoTerm *, Vec *, Mat *, const char *, const char *, PetscBool, PetscBool);
static PetscErrorCode FormFunctionGradient_TaoTerm(Tao, Vec, PetscReal *, Vec, void *);
static PetscErrorCode FormHessian_TaoTerm(Tao, Vec, Mat, Mat, void *);
static PetscErrorCode FormFunctionGradient_Callbacks(Tao, Vec, PetscReal *, Vec, void *);
static PetscErrorCode FormHessian_Callbacks(Tao, Vec, Mat, Mat, void *);
static PetscErrorCode CompareSolutions(Tao, Tao, TestCtx *);

int main(int argc, char **argv)
{
  TestCtx  ctx;
  Tao      tao_term, tao_callback;
  Vec      x_term, x_callback;
  Mat      H_term, H_callback;
  TaoTerm  term1, term2;
  Vec      term1_params          = NULL;
  Vec      term2_params          = NULL;
  Vec      term1_params_callback = NULL;
  Vec      term2_params_callback = NULL;
  Mat      term1_A, term2_A;
  MPI_Comm comm;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  comm = PETSC_COMM_WORLD;

  PetscCall(TestCtxInitialize(comm, &ctx));
  PetscCall(TaoCreate(comm, &tao_term));
  PetscCall(TaoSetType(tao_term, TAOLMVM));

  /* Create Rosenbrock objective using traditional TaoSet interface with user context */
  PetscCall(CreateHessian(ctx.user, &H_term));
  PetscCall(CreateVectors(ctx.user, H_term, &x_term, NULL));
  PetscCall(VecZeroEntries(x_term));
  PetscCall(TaoSetSolution(tao_term, x_term));
  PetscCall(TaoSetObjectiveAndGradient(tao_term, NULL, FormFunctionGradient_TaoTerm, &ctx));
  PetscCall(TaoSetHessian(tao_term, H_term, H_term, FormHessian_TaoTerm, &ctx));

  /* Add term 1 if requested */
  if (ctx.use_term1) {
    PetscCall(CreateTaoTermWithOptions(&ctx, &term1, &term1_params, &term1_A, "reg1_", "A1_", ctx.term1_has_A, ctx.term1_has_params));
    if (ctx.test_print) PetscCall(PetscObjectSetName((PetscObject)term1, "Regularizer TaoTerm"));
    if (ctx.test_print_map) PetscCall(PetscObjectSetName((PetscObject)term1_A, "Regularizer TaoTerm Map"));
    PetscCall(TaoAddTerm(tao_term, "reg1_", ctx.term1_scale, term1, term1_params, term1_A));
    PetscCall(TaoTermDestroy(&term1));
  }
  /* Add term 2 if requested */
  if (ctx.use_term2) {
    PetscCall(CreateTaoTermWithOptions(&ctx, &term2, &term2_params, &term2_A, "reg2_", "A2_", ctx.term2_has_A, ctx.term2_has_params));
    PetscCall(TaoAddTerm(tao_term, "reg2_", ctx.term2_scale, term2, term2_params, term2_A));
    PetscCall(TaoTermDestroy(&term2));
  }

  PetscCall(TaoSetFromOptions(tao_term));
  PetscCall(TaoSolve(tao_term));

  /* Setup Tao with traditional callbacks */
  PetscCall(TaoCreate(comm, &tao_callback));
  PetscCall(TaoSetType(tao_callback, TAOLMVM));

  /* Create Rosenbrock objective using callbacks that manually add term evaluations with user2 context */
  PetscCall(CreateHessian(ctx.user2, &H_callback));
  PetscCall(CreateVectors(ctx.user2, H_callback, &x_callback, NULL));
  PetscCall(VecZeroEntries(x_callback));
  PetscCall(TaoSetSolution(tao_callback, x_callback));
  PetscCall(TaoSetObjectiveAndGradient(tao_callback, NULL, FormFunctionGradient_Callbacks, &ctx));
  PetscCall(TaoSetHessian(tao_callback, H_callback, H_callback, FormHessian_Callbacks, &ctx));

  /* creating duplicate term1 for callback if requested */
  if (ctx.use_term1) {
    if (term1_params) {
      PetscCall(VecDuplicate(term1_params, &term1_params_callback));
      PetscCall(VecCopy(term1_params, term1_params_callback));
    }
    PetscCall(TaoTermCreate(comm, &term1));
    PetscCall(PetscObjectSetOptionsPrefix((PetscObject)term1, "reg1_"));
    PetscCall(TaoTermSetSolutionSizes(term1, PETSC_DECIDE, ctx.user->n, 1));
    PetscCall(TaoTermSetFromOptions(term1));
    /* Store for callback implementation */
    ctx.term1        = term1;
    ctx.term1_params = term1_params_callback;

    if (term1_A) PetscCall(MatDuplicate(term1_A, MAT_COPY_VALUES, &ctx.term1_A_callback));
  }
  /* creating duplicate term2 for callback if requested */
  if (ctx.use_term2) {
    if (term2_params) {
      PetscCall(VecDuplicate(term2_params, &term2_params_callback));
      PetscCall(VecCopy(term2_params, term2_params_callback));
    }
    PetscCall(TaoTermCreate(comm, &term2));
    PetscCall(PetscObjectSetOptionsPrefix((PetscObject)term2, "reg2_"));
    PetscCall(TaoTermSetSolutionSizes(term2, PETSC_DECIDE, ctx.user->n, 1));
    PetscCall(TaoTermSetFromOptions(term2));
    /* Store for callback implementation */
    ctx.term2        = term2;
    ctx.term2_params = term2_params_callback;

    if (term2_A) PetscCall(MatDuplicate(term2_A, MAT_COPY_VALUES, &ctx.term2_A_callback));
  }

  if (ctx.use_term1) {
    if (ctx.term1_has_A) PetscCall(MatCreateVecs(ctx.term1_A_callback, NULL, &ctx.X_mapped1));
    else PetscCall(VecDuplicate(x_callback, &ctx.X_mapped1));
    PetscCall(VecDuplicate(ctx.X_mapped1, &ctx.G_mapped1));
  }
  if (ctx.use_term2) {
    if (ctx.term2_has_A) PetscCall(MatCreateVecs(ctx.term2_A_callback, NULL, &ctx.X_mapped2));
    else PetscCall(VecDuplicate(x_callback, &ctx.X_mapped2));
    PetscCall(VecDuplicate(ctx.X_mapped2, &ctx.G_mapped2));
  }

  PetscCall(VecDuplicate(x_callback, &ctx.G_work));
  PetscCall(TaoSetFromOptions(tao_callback));
  if (ctx.print_debug) PetscCall(PetscPrintf(comm, "Solving Callback version \n"));
  PetscCall(TaoSolve(tao_callback));

  /* Compare solutions */
  PetscCall(CompareSolutions(tao_term, tao_callback, &ctx));

  if (ctx.use_term1) {
    PetscCall(VecDestroy(&term1_params));
    PetscCall(MatDestroy(&term1_A));
    PetscCall(MatDestroy(&ctx.term1_A_callback));
    PetscCall(VecDestroy(&ctx.term1_params));
    PetscCall(TaoTermDestroy(&term1));
  }
  if (ctx.use_term2) {
    PetscCall(VecDestroy(&term2_params));
    PetscCall(MatDestroy(&term2_A));
    PetscCall(MatDestroy(&ctx.term2_A_callback));
    PetscCall(VecDestroy(&ctx.term2_params));
    PetscCall(TaoTermDestroy(&term2));
  }
  PetscCall(TaoDestroy(&tao_term));
  PetscCall(TaoDestroy(&tao_callback));
  PetscCall(VecDestroy(&x_term));
  PetscCall(VecDestroy(&x_callback));
  PetscCall(MatDestroy(&H_term));
  PetscCall(MatDestroy(&H_callback));
  PetscCall(TestCtxFinalize(&ctx));
  PetscCall(PetscFinalize());
  return 0;
}

static PetscErrorCode TestCtxInitialize(MPI_Comm comm, TestCtx *ctx)
{
  PetscFunctionBeginUser;
  PetscCall(PetscMemzero(ctx, sizeof(TestCtx)));

  /* Initialize Rosenbrock contexts */
  PetscCall(AppCtxCreate(comm, &ctx->user));
  PetscCall(AppCtxCreate(comm, &ctx->user2));

  /* Default configuration */
  ctx->test_print           = PETSC_FALSE;
  ctx->test_print_map       = PETSC_FALSE;
  ctx->use_term1            = PETSC_FALSE;
  ctx->use_term2            = PETSC_FALSE;
  ctx->term1_has_A          = PETSC_FALSE;
  ctx->term1_has_params     = PETSC_FALSE;
  ctx->term2_has_A          = PETSC_FALSE;
  ctx->term2_has_params     = PETSC_FALSE;
  ctx->term1_scale          = 0.1;
  ctx->term2_scale          = 0.05;
  ctx->term1_scale_callback = 0;
  ctx->term2_scale_callback = 0;
  ctx->print_debug          = PETSC_FALSE;
  ctx->map_row_size         = ctx->user->n - 1;

  PetscOptionsBegin(comm, "", "TaoTerm Coverage Test Options", "TAO");
  PetscCall(PetscOptionsBool("-test_print", "Test TaoView of term with name", "", ctx->test_print, &ctx->test_print, NULL));
  PetscCall(PetscOptionsBool("-test_print_map", "Test TaoView of map of term with name", "", ctx->test_print_map, &ctx->test_print_map, NULL));
  PetscCall(PetscOptionsBool("-use_term1", "Use first additional term", "", ctx->use_term1, &ctx->use_term1, NULL));
  PetscCall(PetscOptionsBool("-use_term2", "Use second additional term", "", ctx->use_term2, &ctx->use_term2, NULL));
  PetscCall(PetscOptionsBool("-term1_has_A", "Term 1 has a map matrix A", "", ctx->term1_has_A, &ctx->term1_has_A, NULL));
  PetscCall(PetscOptionsBool("-term1_has_params", "Term 1 has parameters", "", ctx->term1_has_params, &ctx->term1_has_params, NULL));
  PetscCall(PetscOptionsBool("-term2_has_A", "Term 2 has a map matrix A", "", ctx->term2_has_A, &ctx->term2_has_A, NULL));
  PetscCall(PetscOptionsBool("-term2_has_params", "Term 2 has parameters", "", ctx->term2_has_params, &ctx->term2_has_params, NULL));
  PetscCall(PetscOptionsReal("-term1_scale", "Scaling for term 1", "", ctx->term1_scale, &ctx->term1_scale, NULL));
  PetscCall(PetscOptionsReal("-term2_scale", "Scaling for term 2", "", ctx->term2_scale, &ctx->term2_scale, NULL));
  PetscCall(PetscOptionsReal("-term1_scale_callback", "Scaling for term 1 in callback version", "", ctx->term1_scale_callback, &ctx->term1_scale_callback, NULL));
  PetscCall(PetscOptionsReal("-term2_scale_callback", "Scaling for term 2 in callback version", "", ctx->term2_scale_callback, &ctx->term2_scale_callback, NULL));
  PetscCall(PetscOptionsInt("-map_row_size", "Row size of mapping matrix", "", ctx->map_row_size, &ctx->map_row_size, NULL));
  PetscCall(PetscOptionsBool("-print_debug", "Print extra floating point for comparison", "", ctx->print_debug, &ctx->print_debug, NULL));
  PetscOptionsEnd();

  if (ctx->term1_scale_callback == 0) ctx->term1_scale_callback = ctx->term1_scale;
  if (ctx->term2_scale_callback == 0) ctx->term2_scale_callback = ctx->term2_scale;

  ctx->term1            = NULL;
  ctx->term2            = NULL;
  ctx->term1_params     = NULL;
  ctx->term2_params     = NULL;
  ctx->term1_A_callback = NULL;
  ctx->term2_A_callback = NULL;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Finalize test context */
static PetscErrorCode TestCtxFinalize(TestCtx *ctx)
{
  PetscFunctionBeginUser;
  PetscCall(VecDestroy(&ctx->X_mapped1));
  PetscCall(VecDestroy(&ctx->X_mapped2));
  PetscCall(VecDestroy(&ctx->G_mapped1));
  PetscCall(VecDestroy(&ctx->G_mapped2));
  PetscCall(VecDestroy(&ctx->G_work));
  PetscCall(AppCtxDestroy(&ctx->user));
  PetscCall(AppCtxDestroy(&ctx->user2));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode CreateTaoTermWithOptions(TestCtx *ctx, TaoTerm *term, Vec *params, Mat *A, const char *term_prefix, const char *A_prefix, PetscBool has_A, PetscBool has_params)
{
  MPI_Comm    comm = ctx->user->comm;
  PetscMPIInt size;

  PetscFunctionBeginUser;
  *term   = NULL;
  *params = NULL;
  *A      = NULL;

  PetscCallMPI(MPI_Comm_size(comm, &size));
  /* Create parameters if requested */
  if (has_params) {
    PetscCall(VecCreate(comm, params));
    PetscCall(VecSetSizes(*params, PETSC_DECIDE, has_A ? ctx->map_row_size : ctx->user->n));
    PetscCall(VecSetFromOptions(*params));
    PetscCall(VecSetRandom(*params, NULL));
  }

  /* Create map matrix A if requested */
  if (has_A) {
    PetscCall(MatCreate(comm, A));
    PetscCall(PetscObjectSetOptionsPrefix((PetscObject)*A, A_prefix));
    PetscCall(MatSetSizes(*A, PETSC_DECIDE, PETSC_DECIDE, ctx->map_row_size, ctx->user->n));
    PetscCall(MatSetType(*A, MATAIJ)); /* Set default type before SetFromOptions */
    PetscCall(MatSetFromOptions(*A));
    /* Check matrix type and set up accordingly */
    if (size == 1) PetscCall(MatSeqAIJSetPreallocation(*A, PETSC_DEFAULT, NULL));
    else PetscCall(MatMPIAIJSetPreallocation(*A, 5, NULL, 5, NULL));
    PetscCall(MatSetUp(*A));
    PetscCall(MatSetRandom(*A, NULL));
    PetscCall(MatAssemblyBegin(*A, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(*A, MAT_FINAL_ASSEMBLY));
  }
  /* Create TaoTerm, set prefix, and configure from options */
  PetscCall(TaoTermCreate(comm, term));
  PetscCall(PetscObjectSetOptionsPrefix((PetscObject)*term, term_prefix));
  PetscCall(TaoTermSetSolutionSizes(*term, PETSC_DECIDE, has_A ? ctx->map_row_size : ctx->user->n, 1));
  PetscCall(TaoTermSetFromOptions(*term));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode FormFunctionGradient_TaoTerm(Tao tao, Vec X, PetscReal *f, Vec G, void *ptr)
{
  TestCtx *ctx = (TestCtx *)ptr;

  PetscFunctionBeginUser;
  PetscCall(FormObjectiveGradient(tao, X, f, G, ctx->user));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode FormHessian_TaoTerm(Tao tao, Vec X, Mat H, Mat Hpre, void *ptr)
{
  TestCtx *ctx = (TestCtx *)ptr;

  PetscFunctionBeginUser;
  PetscCall(FormHessian(tao, X, H, Hpre, ctx->user));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Form function and gradient for callback version (Rosenbrock + terms manually) */
static PetscErrorCode FormFunctionGradient_Callbacks(Tao tao, Vec X, PetscReal *f, Vec G, void *ptr)
{
  TestCtx  *ctx = (TestCtx *)ptr;
  PetscReal f_term;

  PetscFunctionBeginUser;
  /* Compute Rosenbrock part */
  PetscCall(FormObjectiveGradient(tao, X, f, G, ctx->user2));
  /* Add term 1 contribution */
  if (ctx->use_term1 && ctx->term1) {
    /* Map X if needed */
    if (ctx->term1_A_callback) PetscCall(MatMult(ctx->term1_A_callback, X, ctx->X_mapped1));
    else PetscCall(VecCopy(X, ctx->X_mapped1));

    /* Compute term objective and gradient */
    if (ctx->term1_A_callback) {
      PetscCall(TaoTermComputeObjectiveAndGradient(ctx->term1, ctx->X_mapped1, ctx->term1_params, &f_term, ctx->G_mapped1));
      PetscCall(MatMultTranspose(ctx->term1_A_callback, ctx->G_mapped1, ctx->G_work));
      PetscCall(VecAXPY(G, ctx->term1_scale_callback, ctx->G_work));
    } else {
      PetscCall(TaoTermComputeObjectiveAndGradient(ctx->term1, ctx->X_mapped1, ctx->term1_params, &f_term, ctx->G_work));
      PetscCall(VecAXPY(G, ctx->term1_scale_callback, ctx->G_work));
    }
    *f += ctx->term1_scale_callback * f_term;
  }

  /* Add term 2 contribution */
  if (ctx->use_term2 && ctx->term2) {
    /* Map X if needed */
    if (ctx->term2_A_callback) PetscCall(MatMult(ctx->term2_A_callback, X, ctx->X_mapped2));
    else PetscCall(VecCopy(X, ctx->X_mapped2));

    /* Compute term objective and gradient */
    if (ctx->term2_A_callback) {
      PetscCall(TaoTermComputeObjectiveAndGradient(ctx->term2, ctx->X_mapped2, ctx->term2_params, &f_term, ctx->G_mapped2));
      /* Map gradient back and add to G */
      PetscCall(MatMultTranspose(ctx->term2_A_callback, ctx->G_mapped2, ctx->G_work));
      PetscCall(VecAXPY(G, ctx->term2_scale_callback, ctx->G_work));
    } else {
      PetscCall(TaoTermComputeObjectiveAndGradient(ctx->term2, ctx->X_mapped2, ctx->term2_params, &f_term, ctx->G_work));
      PetscCall(VecAXPY(G, ctx->term2_scale_callback, ctx->G_work));
    }
    *f += ctx->term2_scale_callback * f_term;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Form Hessian for callback version (Rosenbrock + terms manually) */
static PetscErrorCode FormHessian_Callbacks(Tao tao, Vec X, Mat H, Mat Hpre, void *ptr)
{
  TestCtx  *ctx = (TestCtx *)ptr;
  Mat       H_term;
  PetscInt  m, n;
  PetscBool is_assembled;

  PetscFunctionBeginUser;
  /* Compute Rosenbrock Hessian */
  PetscCall(FormHessian(tao, X, H, Hpre, ctx->user2));
  /* Add term 1 Hessian contribution */
  if (ctx->use_term1 && ctx->term1) {
    if (ctx->term1_A_callback) {
      PetscCall(MatMult(ctx->term1_A_callback, X, ctx->X_mapped1));
      PetscCall(VecGetSize(ctx->X_mapped1, &m));
      PetscCall(MatCreate(ctx->user2->comm, &H_term));
      PetscCall(MatSetSizes(H_term, PETSC_DECIDE, PETSC_DECIDE, m, m));
      PetscCall(MatSetType(H_term, MATAIJ));
      PetscCall(MatSetUp(H_term));
    } else {
      PetscCall(VecCopy(X, ctx->X_mapped1));
      PetscCall(VecGetSize(X, &n));
      PetscCall(MatCreate(ctx->user2->comm, &H_term));
      PetscCall(MatSetSizes(H_term, PETSC_DECIDE, PETSC_DECIDE, n, n));
      PetscCall(MatSetType(H_term, MATAIJ));
      PetscCall(MatSetUp(H_term));
    }
    PetscCall(MatAssembled(H_term, &is_assembled));
    if (!is_assembled) {
      PetscCall(MatAssemblyBegin(H_term, MAT_FINAL_ASSEMBLY));
      PetscCall(MatAssemblyEnd(H_term, MAT_FINAL_ASSEMBLY));
    }

    PetscCall(TaoTermComputeHessian(ctx->term1, ctx->X_mapped1, ctx->term1_params, H_term, NULL));
    if (ctx->term1_A_callback) {
      Mat H_mapped;
      /* H = H + scale * A^T * H_term * A */
      PetscCall(MatMatMult(H_term, ctx->term1_A_callback, MAT_INITIAL_MATRIX, PETSC_DETERMINE, &H_mapped));
      PetscCall(MatDestroy(&H_term));
      PetscCall(MatTransposeMatMult(ctx->term1_A_callback, H_mapped, MAT_INITIAL_MATRIX, PETSC_DETERMINE, &H_term));
      PetscCall(MatDestroy(&H_mapped));
      PetscCall(MatScale(H_term, ctx->term1_scale_callback));
      PetscCall(MatAXPY(H, 1.0, H_term, DIFFERENT_NONZERO_PATTERN));
      PetscCall(MatDestroy(&H_term));
    } else {
      PetscCall(MatScale(H_term, ctx->term1_scale_callback));
      PetscCall(MatAXPY(H, 1.0, H_term, DIFFERENT_NONZERO_PATTERN));
      PetscCall(MatDestroy(&H_term));
    }
  }

  /* Add term 2 Hessian contribution */
  if (ctx->use_term2 && ctx->term2) {
    if (ctx->term2_A_callback) {
      PetscCall(MatMult(ctx->term2_A_callback, X, ctx->X_mapped2));
      PetscCall(VecGetSize(ctx->X_mapped2, &m));
      PetscCall(MatCreate(ctx->user2->comm, &H_term));
      PetscCall(MatSetSizes(H_term, PETSC_DECIDE, PETSC_DECIDE, m, m));
      PetscCall(MatSetType(H_term, MATAIJ));
      PetscCall(MatSetUp(H_term));
    } else {
      PetscCall(VecCopy(X, ctx->X_mapped2));
      PetscCall(VecGetSize(X, &n));
      PetscCall(MatCreate(ctx->user2->comm, &H_term));
      PetscCall(MatSetSizes(H_term, PETSC_DECIDE, PETSC_DECIDE, n, n));
      PetscCall(MatSetType(H_term, MATAIJ));
      PetscCall(MatSetUp(H_term));
    }
    PetscCall(MatAssembled(H_term, &is_assembled));
    if (!is_assembled) {
      PetscCall(MatAssemblyBegin(H_term, MAT_FINAL_ASSEMBLY));
      PetscCall(MatAssemblyEnd(H_term, MAT_FINAL_ASSEMBLY));
    }

    PetscCall(TaoTermComputeHessian(ctx->term2, ctx->X_mapped2, ctx->term2_params, H_term, NULL));

    if (ctx->term2_A_callback) {
      Mat H_mapped;
      /* H = H + scale * A^T * H_term * A */
      PetscCall(MatMatMult(H_term, ctx->term2_A_callback, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &H_mapped));
      PetscCall(MatDestroy(&H_term));
      PetscCall(MatTransposeMatMult(ctx->term2_A_callback, H_mapped, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &H_term));
      PetscCall(MatDestroy(&H_mapped));
      PetscCall(MatScale(H_term, ctx->term2_scale_callback));
      PetscCall(MatAXPY(H, 1.0, H_term, DIFFERENT_NONZERO_PATTERN));
      PetscCall(MatDestroy(&H_term));
    } else {
      PetscCall(MatScale(H_term, ctx->term2_scale_callback));
      PetscCall(MatAXPY(H, 1.0, H_term, DIFFERENT_NONZERO_PATTERN));
      PetscCall(MatDestroy(&H_term));
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Compare solutions from both methods */
static PetscErrorCode CompareSolutions(Tao tao_term, Tao tao_callback, TestCtx *ctx)
{
  Vec                x_term, x_callback, diff;
  PetscReal          norm_term, norm_callback, norm_diff, rel_diff;
  PetscReal          f_term, f_callback;
  TaoConvergedReason reason_term, reason_callback;

  PetscFunctionBeginUser;
  PetscCall(TaoGetSolution(tao_term, &x_term));
  PetscCall(TaoGetSolution(tao_callback, &x_callback));

  /* Compute difference */
  PetscCall(VecDuplicate(x_term, &diff));
  PetscCall(VecCopy(x_term, diff));
  PetscCall(VecAXPY(diff, -1.0, x_callback));

  PetscCall(VecNorm(x_term, NORM_2, &norm_term));
  PetscCall(VecNorm(x_callback, NORM_2, &norm_callback));
  PetscCall(VecNorm(diff, NORM_2, &norm_diff));

  rel_diff = norm_diff / PetscMax(norm_term, 1.0e-10);

  /* Get objective values */
  PetscCall(TaoGetSolutionStatus(tao_term, NULL, &f_term, NULL, NULL, NULL, NULL));
  PetscCall(TaoGetSolutionStatus(tao_callback, NULL, &f_callback, NULL, NULL, NULL, NULL));

  /* Get convergence reasons */
  PetscCall(TaoGetConvergedReason(tao_term, &reason_term));
  PetscCall(TaoGetConvergedReason(tao_callback, &reason_callback));

  if (rel_diff < 1.0e-12) {
    if (ctx->print_debug) PetscCall(PetscPrintf(PetscObjectComm((PetscObject)tao_term), "Solutions match (relative difference < 1e-12), %6.10e\n", (double)rel_diff));
    else PetscCall(PetscPrintf(PetscObjectComm((PetscObject)tao_term), "Solutions match (relative difference < 1e-12)\n"));
  } else if (rel_diff < 1.0e-6) {
    if (ctx->print_debug) PetscCall(PetscPrintf(PetscObjectComm((PetscObject)tao_term), "Solutions approximately match (relative difference < 1e-6), %6.10e\n", (double)rel_diff));
    else PetscCall(PetscPrintf(PetscObjectComm((PetscObject)tao_term), "Solutions approximately match (relative difference < 1e-6)\n"));
  } else {
    if (ctx->print_debug) PetscCall(PetscPrintf(PetscObjectComm((PetscObject)tao_term), "Solutions differ significantly (relative difference >= 1e-6), %6.10e\n", (double)rel_diff));
    else PetscCall(PetscPrintf(PetscObjectComm((PetscObject)tao_term), "Solutions differ significantly (relative difference >= 1e-6)\n"));
  }
  PetscCall(VecDestroy(&diff));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*TEST

   build:
      requires: !complex !single !quad !defined(PETSC_USE_64BIT_INDICES) !__float128

   test:
    suffix: test_print_noparam
    args: -tao_view -tao_type nls -use_term1 -term1_has_params 0 -term1_has_A
    args: -reg1_tao_term_type l1 -test_print -test_print_map -reg1_tao_term_parameters_mode none

   test:
    suffix: test_print_p_req
    args: -tao_view -tao_type nls -use_term1 -term1_has_params 1 -term1_has_A
    args: -reg1_tao_term_type l1 -test_print -test_print_map -reg1_tao_term_parameters_mode required

# Single term tests
   # L1 with A
   testset:
      suffix: term1_l1_A
      output_file: output/taotermtest1.out
      args: -tao_type nls -use_term1 -term1_has_params {{0 1}} -term1_has_A
      args: -reg1_tao_term_type l1
      test:
         args: -term1_scale 0.012
      test:
         args: -tao_term_sum_reg1_scale 0.012 -term1_scale_callback 0.012

   # L1 without A
   testset:
      suffix: term1_l1_no_A
      output_file: output/taotermtest1.out
      args: -tao_type nls -use_term1 -term1_has_params {{0 1}}
      args: -term1_has_A 0 -reg1_tao_term_type l1
      test:
         args: -term1_scale 0.012
      test:
         args: -tao_term_sum_reg1_scale 0.012 -term1_scale_callback 0.012

   # L2 with A
   testset:
      suffix: term1_l2_A
      output_file: output/taotermtest1.out
      args: -tao_type nls -use_term1 -term1_has_params {{0 1}} -term1_has_A
      args: -reg1_tao_term_type halfl2squared
      test:
         args: -term1_scale 0.12
      test:
         args: -tao_term_sum_reg1_scale 0.12 -term1_scale_callback 0.12

   # L2 without A
   testset:
      suffix: term1_l2_no_A
      output_file: output/taotermtest1.out
      args: -tao_type nls -use_term1 -term1_has_params {{0 1}}
      args: -term1_has_A 0
      args: -reg1_tao_term_type halfl2squared
      test:
         args: -term1_scale 0.12
      test:
         args: -tao_term_sum_reg1_scale 0.12 -term1_scale_callback 0.12

# Two terms, with no mapping
   testset:
      suffix: l1_l2_no_A1_no_A2
      output_file: output/taotermtest1.out
      args: -tao_type nls -use_term1 -use_term2 -term1_has_params {{0 1}} -term2_has_params {{0 1}}
      args: -term1_has_A 0 -term2_has_A 0 -reg1_tao_term_type l1 -reg2_tao_term_type halfl2squared
      test:
         args: -term1_scale 0.123 -term2_scale 0.098
      test:
         args: -term1_scale 0.123 -tao_term_sum_reg2_scale 0.098 -term2_scale_callback 0.098
      test:
         args: -tao_term_sum_reg1_scale 0.123 -term1_scale_callback 0.123 -term2_scale 0.098
      test:
         args: -tao_term_sum_reg1_scale 0.123 -term1_scale_callback 0.123 -tao_term_sum_reg2_scale 0.098 -term2_scale_callback 0.098

# Two terms, with one mapping
   testset:
      suffix: l1_l2_no_A1_A2
      output_file: output/taotermtest1.out
      args: -tao_type nls -use_term1 -use_term2 -term1_has_params {{0 1}} -term2_has_params {{0 1}}
      args: -term1_has_A 0 -term2_has_A 1 -reg1_tao_term_type l1 -reg2_tao_term_type halfl2squared
      test:
         args: -term1_scale 0.123 -term2_scale 0.098
      test:
         args: -term1_scale 0.123 -tao_term_sum_reg2_scale 0.098 -term2_scale_callback 0.098
      test:
         args: -tao_term_sum_reg1_scale 0.123 -term1_scale_callback 0.123 -term2_scale 0.098
      test:
         args: -tao_term_sum_reg1_scale 0.123 -term1_scale_callback 0.123 -tao_term_sum_reg2_scale 0.098 -term2_scale_callback 0.098

   testset:
      suffix: l2_l1_no_A1_A2
      output_file: output/taotermtest1.out
      args: -tao_type nls -use_term1 -use_term2 -term1_has_params {{0 1}} -term2_has_params {{0 1}}
      args: -term1_has_A 0 -term2_has_A 1 -reg1_tao_term_type halfl2squared -reg2_tao_term_type l1
      test:
         args: -term1_scale 0.123 -term2_scale 0.098
      test:
         args: -term1_scale 0.123 -tao_term_sum_reg2_scale 0.098 -term2_scale_callback 0.098
      test:
         args: -tao_term_sum_reg1_scale 0.123 -term1_scale_callback 0.123 -term2_scale 0.098
      test:
         args: -tao_term_sum_reg1_scale 0.123 -term1_scale_callback 0.123 -tao_term_sum_reg2_scale 0.098 -term2_scale_callback 0.098

# Two terms: term1 has A, term2 has A
   testset:
      suffix: l1_l2_A1_A2
      output_file: output/taotermtest1.out
      args: -tao_type nls -use_term1 -use_term2 -term1_has_params {{0 1}} -term2_has_params {{0 1}}
      args: -term1_has_A 1 -term2_has_A 1 -reg1_tao_term_type l1 -reg2_tao_term_type halfl2squared
      test:
         args: -term1_scale 0.075 -term2_scale 0.098
      test:
         args: -term1_scale 0.075 -tao_term_sum_reg2_scale 0.098 -term2_scale_callback 0.098
      test:
         args: -tao_term_sum_reg1_scale 0.075 -term1_scale_callback 0.075 -term2_scale 0.098
      test:
         args: -tao_term_sum_reg1_scale 0.075 -term1_scale_callback 0.075 -tao_term_sum_reg2_scale 0.098 -term2_scale_callback 0.098

TEST*/
