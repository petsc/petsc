#include <petscregressor.h>

static char help[] = "Tests some linear PetscRegressor types with different regularizers.\n\n";

typedef struct _AppCtx {
  Mat       X;           /* Training data */
  Vec       y;           /* Target data   */
  Vec       y_predicted; /* Target data   */
  Vec       coefficients;
  PetscInt  N; /* Data size     */
  PetscBool flg_string;
  PetscBool flg_ascii;
  PetscBool flg_view_sol;
  PetscBool test_prefix;
} *AppCtx;

static PetscErrorCode DestroyCtx(AppCtx *ctx)
{
  PetscFunctionBegin;
  PetscCall(MatDestroy(&(*ctx)->X));
  PetscCall(VecDestroy(&(*ctx)->y));
  PetscCall(VecDestroy(&(*ctx)->y_predicted));
  PetscCall(PetscFree(*ctx));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TestRegressorViews(PetscRegressor regressor, AppCtx ctx)
{
  PetscRegressorType check_type;
  PetscBool          match;

  PetscFunctionBegin;
  if (ctx->flg_view_sol) {
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Training target vector is\n"));
    PetscCall(VecView(ctx->y, PETSC_VIEWER_STDOUT_WORLD));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Predicted values are\n"));
    PetscCall(VecView(ctx->y_predicted, PETSC_VIEWER_STDOUT_WORLD));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Coefficients are\n"));
    PetscCall(VecView(ctx->coefficients, PETSC_VIEWER_STDOUT_WORLD));
  }

  if (ctx->flg_string) {
    PetscViewer stringviewer;
    char        string[512];
    const char *outstring;

    PetscCall(PetscViewerStringOpen(PETSC_COMM_WORLD, string, sizeof(string), &stringviewer));
    PetscCall(PetscRegressorView(regressor, stringviewer));
    PetscCall(PetscViewerStringGetStringRead(stringviewer, &outstring, NULL));
    PetscCheck((char *)outstring == (char *)string, PETSC_COMM_WORLD, PETSC_ERR_PLIB, "String returned from viewer does not equal original string");
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Output from string viewer:%s\n", outstring));
    PetscCall(PetscViewerDestroy(&stringviewer));
  } else if (ctx->flg_ascii) PetscCall(PetscRegressorView(regressor, PETSC_VIEWER_STDOUT_WORLD));

  PetscCall(PetscRegressorGetType(regressor, &check_type));
  PetscCall(PetscStrcmp(check_type, PETSCREGRESSORLINEAR, &match));
  PetscCheck(match, PETSC_COMM_WORLD, PETSC_ERR_ARG_NOTSAMETYPE, "Regressor type is not Linear");
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TestPrefixRegressor(PetscRegressor regressor, AppCtx ctx)
{
  PetscFunctionBegin;
  if (ctx->test_prefix) {
    PetscCall(PetscRegressorSetOptionsPrefix(regressor, "sys1_"));
    PetscCall(PetscRegressorAppendOptionsPrefix(regressor, "sys2_"));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode CreateData(AppCtx ctx)
{
  PetscMPIInt rank;
  PetscInt    i;
  PetscScalar mean;

  PetscFunctionBegin;
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &rank));
  PetscCall(VecCreate(PETSC_COMM_WORLD, &ctx->y));
  PetscCall(VecSetSizes(ctx->y, PETSC_DECIDE, ctx->N));
  PetscCall(VecSetFromOptions(ctx->y));
  PetscCall(VecDuplicate(ctx->y, &ctx->y_predicted));
  PetscCall(MatCreate(PETSC_COMM_WORLD, &ctx->X));
  PetscCall(MatSetSizes(ctx->X, PETSC_DECIDE, PETSC_DECIDE, ctx->N, ctx->N));
  PetscCall(MatSetFromOptions(ctx->X));
  PetscCall(MatSetUp(ctx->X));

  if (!rank) {
    for (i = 0; i < ctx->N; i++) {
      PetscCall(VecSetValue(ctx->y, i, (PetscScalar)i, INSERT_VALUES));
      PetscCall(MatSetValue(ctx->X, i, i, 1.0, INSERT_VALUES));
    }
  }
  /* Set up a training data matrix that is the identity.
   * We do this because this gives us a special case in which we can analytically determine what the regression
   * coefficients should be for ordinary least squares, LASSO (L1 regularized), and ridge (L2 regularized) regression.
   * See details in section 6.2 of James et al.'s An Introduction to Statistical Learning (ISLR), in the subsection
   * titled "A Simple Special Case for Ridge Regression and the Lasso".
   * Note that the coefficients we generate with ridge regression (-regressor_linear_type ridge -regressor_regularizer_weight <lambda>, or, equivalently,
   * -tao_brgn_regularization_type l2pure -tao_brgn_regularizer_weight <lambda>) match those of the ISLR formula exactly.
   * For LASSO it does not match the ISLR formula: where they use lambda/2, we need to use lambda.
   * It also doesn't match what Scikit-learn does; in that case their lambda is 1/n_samples of our lambda. Apparently everyone is scaling
   * their loss function by a different value, hence the need to change what "lambda" is. But it's clear that ISLR, Scikit-learn, and we
   * are basically doing the same thing otherwise. */
  PetscCall(VecAssemblyBegin(ctx->y));
  PetscCall(VecAssemblyEnd(ctx->y));
  PetscCall(MatAssemblyBegin(ctx->X, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(ctx->X, MAT_FINAL_ASSEMBLY));
  /* Center the target vector we will train with. */
  PetscCall(VecMean(ctx->y, &mean));
  PetscCall(VecShift(ctx->y, -1.0 * mean));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode ConfigureContext(AppCtx ctx)
{
  PetscFunctionBegin;
  ctx->flg_string   = PETSC_FALSE;
  ctx->flg_ascii    = PETSC_FALSE;
  ctx->flg_view_sol = PETSC_FALSE;
  ctx->test_prefix  = PETSC_FALSE;
  ctx->N            = 10;
  PetscOptionsBegin(PETSC_COMM_WORLD, NULL, "Options for PetscRegressor ex3:", "");
  PetscCall(PetscOptionsInt("-N", "Dimension of the N x N data matrix", "ex3.c", ctx->N, &ctx->N, NULL));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-test_string_viewer", &ctx->flg_string, NULL));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-test_ascii_viewer", &ctx->flg_ascii, NULL));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-view_sols", &ctx->flg_view_sol, NULL));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-test_prefix", &ctx->test_prefix, NULL));
  PetscOptionsEnd();
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **args)
{
  AppCtx         ctx;
  PetscRegressor regressor;
  PetscScalar    intercept;

  /* Initialize PETSc */
  PetscCall(PetscInitialize(&argc, &args, (char *)0, help));

  /* Initialize problem parameters and data */
  PetscCall(PetscNew(&ctx));
  PetscCall(ConfigureContext(ctx));
  PetscCall(CreateData(ctx));

  /* Create Regressor solver with desired type and options */
  PetscCall(PetscRegressorCreate(PETSC_COMM_WORLD, &regressor));
  PetscCall(PetscRegressorSetType(regressor, PETSCREGRESSORLINEAR));
  PetscCall(PetscRegressorLinearSetType(regressor, REGRESSOR_LINEAR_OLS));
  PetscCall(PetscRegressorLinearSetFitIntercept(regressor, PETSC_FALSE));
  /* Testing prefix functions for Regressor */
  PetscCall(TestPrefixRegressor(regressor, ctx));
  /* Check for command line options */
  PetscCall(PetscRegressorSetFromOptions(regressor));
  /* Fit the regressor */
  PetscCall(PetscRegressorFit(regressor, ctx->X, ctx->y));
  /* Predict data with fitted regressor */
  PetscCall(PetscRegressorPredict(regressor, ctx->X, ctx->y_predicted));
  /* Get other desired output data */
  PetscCall(PetscRegressorLinearGetIntercept(regressor, &intercept));
  PetscCall(PetscRegressorLinearGetCoefficients(regressor, &ctx->coefficients));

  /* Testing Views, and GetTypes */
  PetscCall(TestRegressorViews(regressor, ctx));
  PetscCall(PetscRegressorDestroy(&regressor));
  PetscCall(DestroyCtx(&ctx));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   build:
      requires: !complex !single !__float128 !defined(PETSC_USE_64BIT_INDICES)

   test:
      suffix: prefix_tao
      args: -sys1_sys2_regressor_view -test_prefix

   test:
      suffix: prefix_ksp
      args: -sys1_sys2_regressor_view -test_prefix -sys1_sys2_regressor_linear_use_ksp -sys1_sys2_regressor_linear_ksp_monitor

   test:
      suffix: prefix_ksp_cholesky
      args: -sys1_sys2_regressor_view -test_prefix -sys1_sys2_regressor_linear_use_ksp -sys1_sys2_regressor_linear_pc_type cholesky
      TODO: Could not locate a solver type for factorization type CHOLESKY and matrix type normal

   test:
      suffix: prefix_ksp_suitesparse
      requires: suitesparse
      args: -sys1_sys2_regressor_view -test_prefix -sys1_sys2_regressor_linear_use_ksp -sys1_sys2_regressor_linear_pc_type qr -sys1_sys2_regressor_linear_pc_factor_mat_solver_type spqr -sys1_sys2_regressor_linear_ksp_monitor

   test:
      suffix: asciiview
      args: -test_ascii_viewer

   test:
       suffix: stringview
       args: -test_string_viewer

   test:
      suffix: ksp_intercept
      args: -regressor_linear_use_ksp -regressor_linear_fit_intercept -regressor_view

   test:
      suffix: ksp_no_intercept
      args: -regressor_linear_use_ksp -regressor_view

   test:
      suffix: lasso_1
      nsize: 1
      args: -regressor_type linear -regressor_linear_type lasso -regressor_regularizer_weight 2 -regressor_linear_fit_intercept -view_sols

   test:
      suffix: lasso_2
      nsize: 2
      args: -regressor_type linear -regressor_linear_type lasso -regressor_regularizer_weight 2 -regressor_linear_fit_intercept -view_sols

   test:
      suffix: ridge_1
      nsize: 1
      args: -regressor_type linear -regressor_linear_type ridge -regressor_regularizer_weight 2 -regressor_linear_fit_intercept -view_sols

   test:
      suffix: ridge_2
      nsize: 2
      args: -regressor_type linear -regressor_linear_type ridge -regressor_regularizer_weight 2 -regressor_linear_fit_intercept -view_sols

   test:
      suffix: ols_1
      nsize: 1
      args: -regressor_type linear -regressor_linear_type ols -regressor_linear_fit_intercept -view_sols

   test:
      suffix: ols_2
      nsize: 2
      args: -regressor_type linear -regressor_linear_type ols -regressor_linear_fit_intercept -view_sols

TEST*/
