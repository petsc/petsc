#include <../src/ml/regressor/impls/linear/linearimpl.h> /*I "petscregressor.h" I*/

const char *const PetscRegressorLinearTypes[] = {"ols", "lasso", "ridge", "RegressorLinearType", "REGRESSOR_LINEAR_", NULL};

static PetscErrorCode PetscRegressorLinearSetFitIntercept_Linear(PetscRegressor regressor, PetscBool flg)
{
  PetscRegressor_Linear *linear = (PetscRegressor_Linear *)regressor->data;

  PetscFunctionBegin;
  linear->fit_intercept = flg;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscRegressorLinearSetType_Linear(PetscRegressor regressor, PetscRegressorLinearType type)
{
  PetscRegressor_Linear *linear = (PetscRegressor_Linear *)regressor->data;

  PetscFunctionBegin;
  linear->type = type;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscRegressorLinearGetType_Linear(PetscRegressor regressor, PetscRegressorLinearType *type)
{
  PetscRegressor_Linear *linear = (PetscRegressor_Linear *)regressor->data;

  PetscFunctionBegin;
  *type = linear->type;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscRegressorLinearGetIntercept_Linear(PetscRegressor regressor, PetscScalar *intercept)
{
  PetscRegressor_Linear *linear = (PetscRegressor_Linear *)regressor->data;

  PetscFunctionBegin;
  *intercept = linear->intercept;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscRegressorLinearGetCoefficients_Linear(PetscRegressor regressor, Vec *coefficients)
{
  PetscRegressor_Linear *linear = (PetscRegressor_Linear *)regressor->data;

  PetscFunctionBegin;
  *coefficients = linear->coefficients;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscRegressorLinearGetKSP_Linear(PetscRegressor regressor, KSP *ksp)
{
  PetscRegressor_Linear *linear = (PetscRegressor_Linear *)regressor->data;

  PetscFunctionBegin;
  if (!linear->ksp) {
    PetscCall(KSPCreate(PetscObjectComm((PetscObject)regressor), &linear->ksp));
    PetscCall(PetscObjectIncrementTabLevel((PetscObject)linear->ksp, (PetscObject)regressor, 1));
    PetscCall(PetscObjectSetOptions((PetscObject)linear->ksp, ((PetscObject)regressor)->options));
  }
  *ksp = linear->ksp;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscRegressorLinearSetUseKSP_Linear(PetscRegressor regressor, PetscBool flg)
{
  PetscRegressor_Linear *linear = (PetscRegressor_Linear *)regressor->data;

  PetscFunctionBegin;
  linear->use_ksp = flg;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode EvaluateResidual(Tao tao, Vec x, Vec f, void *ptr)
{
  PetscRegressor_Linear *linear = (PetscRegressor_Linear *)ptr;

  PetscFunctionBegin;
  /* Evaluate f = A * x - b */
  PetscCall(MatMult(linear->X, x, f));
  PetscCall(VecAXPY(f, -1.0, linear->rhs));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode EvaluateJacobian(Tao tao, Vec x, Mat J, Mat Jpre, void *ptr)
{
  /* The TAOBRGN API expects us to pass an EvaluateJacobian() routine to it, but in this case it is a dummy function.
     Denoting our data matrix as X, for linear least squares J[m][n] = df[m]/dx[n] = X[m][n]. */
  PetscFunctionBegin;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscRegressorSetUp_Linear(PetscRegressor regressor)
{
  PetscInt               M, N;
  PetscRegressor_Linear *linear = (PetscRegressor_Linear *)regressor->data;
  KSP                    ksp;
  Tao                    tao;

  PetscFunctionBegin;
  PetscCall(MatGetSize(regressor->training, &M, &N));

  if (linear->fit_intercept) {
    /* If we are fitting the intercept, we need to make A a composite matrix using MATCENTERING to preserve sparsity.
     * Though there might be some cases we don't want to do this for, depending on what kind of matrix is passed in. (Probably bad idea for dense?)
     * We will also need to ensure that the right-hand side passed to the KSP is also mean-centered, since we
     * intend to compute the intercept separately from regression coefficients (that is, we will not be adding a
     * column of all 1s to our design matrix). */
    PetscCall(MatCreateCentering(PetscObjectComm((PetscObject)regressor), PETSC_DECIDE, M, &linear->C));
    PetscCall(MatCreate(PetscObjectComm((PetscObject)regressor), &linear->X));
    PetscCall(MatSetSizes(linear->X, PETSC_DECIDE, PETSC_DECIDE, M, N));
    PetscCall(MatSetType(linear->X, MATCOMPOSITE));
    PetscCall(MatCompositeSetType(linear->X, MAT_COMPOSITE_MULTIPLICATIVE));
    PetscCall(MatCompositeAddMat(linear->X, regressor->training));
    PetscCall(MatCompositeAddMat(linear->X, linear->C));
    PetscCall(VecDuplicate(regressor->target, &linear->rhs));
    PetscCall(MatMult(linear->C, regressor->target, linear->rhs));
  } else {
    // When not fitting intercept, we assume that the input data are already centered.
    linear->X   = regressor->training;
    linear->rhs = regressor->target;

    PetscCall(PetscObjectReference((PetscObject)linear->X));
    PetscCall(PetscObjectReference((PetscObject)linear->rhs));
  }

  if (linear->coefficients) PetscCall(VecDestroy(&linear->coefficients));

  if (linear->use_ksp) {
    PetscCheck(linear->type == REGRESSOR_LINEAR_OLS, PetscObjectComm((PetscObject)regressor), PETSC_ERR_ARG_WRONGSTATE, "KSP can be used to fit a linear regressor only when its type is OLS");

    if (!linear->ksp) PetscCall(PetscRegressorLinearGetKSP(regressor, &linear->ksp));
    ksp = linear->ksp;

    PetscCall(MatCreateVecs(linear->X, &linear->coefficients, NULL));
    /* Set up the KSP to solve the least squares problem (without solving for intercept, as this is done separately) using KSPLSQR. */
    PetscCall(MatCreateNormal(linear->X, &linear->XtX));
    PetscCall(KSPSetType(ksp, KSPLSQR));
    PetscCall(KSPSetOperators(ksp, linear->X, linear->XtX));
    PetscCall(KSPSetOptionsPrefix(ksp, ((PetscObject)regressor)->prefix));
    PetscCall(KSPAppendOptionsPrefix(ksp, "regressor_linear_"));
    PetscCall(KSPSetFromOptions(ksp));
  } else {
    /* Note: Currently implementation creates TAO inside of implementations.
      * Thus, all the prefix jobs are done inside implementations, not in interface */
    const char *prefix;

    if (!regressor->tao) PetscCall(PetscRegressorGetTao(regressor, &tao));

    PetscCall(MatCreateVecs(linear->X, &linear->coefficients, &linear->residual));
    /* Set up the TAO object to solve the (regularized) least squares problem (without solving for intercept, which is done separately) using TAOBRGN. */
    PetscCall(TaoSetType(tao, TAOBRGN));
    PetscCall(TaoSetSolution(tao, linear->coefficients));
    PetscCall(TaoSetResidualRoutine(tao, linear->residual, EvaluateResidual, linear));
    PetscCall(TaoSetJacobianResidualRoutine(tao, linear->X, linear->X, EvaluateJacobian, linear));
    // Set the regularization type and weight for the BRGN as linear->type dictates:
    // TODO BRGN needs to be BRGNSetRegularizationType
    // PetscOptionsSetValue no longer works due to functioning prefix system
    PetscCall(PetscRegressorGetOptionsPrefix(regressor, &prefix));
    PetscCall(TaoSetOptionsPrefix(regressor->tao, prefix));
    PetscCall(TaoAppendOptionsPrefix(tao, "regressor_linear_"));
    switch (linear->type) {
    case REGRESSOR_LINEAR_OLS:
      regressor->regularizer_weight = 0.0; // OLS, by definition, uses a regularizer weight of 0
      break;
    case REGRESSOR_LINEAR_LASSO:
      PetscCall(TaoBRGNSetRegularizationType(regressor->tao, TAOBRGN_REGULARIZATION_L1DICT));
      break;
    case REGRESSOR_LINEAR_RIDGE:
      PetscCall(TaoBRGNSetRegularizationType(regressor->tao, TAOBRGN_REGULARIZATION_L2PURE));
      break;
    default:
      break;
    }
    if (!linear->use_ksp) PetscCall(TaoBRGNSetRegularizerWeight(tao, regressor->regularizer_weight));
    PetscCall(TaoSetFromOptions(tao));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscRegressorReset_Linear(PetscRegressor regressor)
{
  PetscRegressor_Linear *linear = (PetscRegressor_Linear *)regressor->data;

  PetscFunctionBegin;
  /* Destroy the PETSc objects associated with the linear regressor implementation. */
  linear->ksp_its     = 0;
  linear->ksp_tot_its = 0;

  PetscCall(MatDestroy(&linear->X));
  PetscCall(MatDestroy(&linear->XtX));
  PetscCall(MatDestroy(&linear->C));
  PetscCall(KSPDestroy(&linear->ksp));
  PetscCall(VecDestroy(&linear->coefficients));
  PetscCall(VecDestroy(&linear->rhs));
  PetscCall(VecDestroy(&linear->residual));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscRegressorDestroy_Linear(PetscRegressor regressor)
{
  PetscFunctionBegin;
  PetscCall(PetscObjectComposeFunction((PetscObject)regressor, "PetscRegressorLinearSetFitIntercept_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)regressor, "PetscRegressorLinearSetUseKSP_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)regressor, "PetscRegressorLinearGetKSP_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)regressor, "PetscRegressorLinearGetCoefficients_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)regressor, "PetscRegressorLinearGetIntercept_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)regressor, "PetscRegressorLinearSetType_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)regressor, "PetscRegressorLinearGetType_C", NULL));
  PetscCall(PetscRegressorReset_Linear(regressor));
  PetscCall(PetscFree(regressor->data));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscRegressorLinearSetFitIntercept - Set a flag to indicate that the intercept (also known as the "bias" or "offset") should
  be calculated; data are assumed to be mean-centered if false.

  Logically Collective

  Input Parameters:
+ regressor - the `PetscRegressor` context
- flg       - `PETSC_TRUE` to calculate the intercept, `PETSC_FALSE` to assume mean-centered data (default is `PETSC_TRUE`)

  Level: intermediate

  Options Database Key:
. regressor_linear_fit_intercept <true,false> - fit the intercept

  Note:
  If the user indicates that the intercept should not be calculated, the intercept will be set to zero.

.seealso: `PetscRegressor`, `PetscRegressorFit()`
@*/
PetscErrorCode PetscRegressorLinearSetFitIntercept(PetscRegressor regressor, PetscBool flg)
{
  PetscFunctionBegin;
  /* TODO: Add companion PetscRegressorLinearGetFitIntercept(), and put it in the .seealso: */
  PetscValidHeaderSpecific(regressor, PETSCREGRESSOR_CLASSID, 1);
  PetscValidLogicalCollectiveBool(regressor, flg, 2);
  PetscTryMethod(regressor, "PetscRegressorLinearSetFitIntercept_C", (PetscRegressor, PetscBool), (regressor, flg));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscRegressorLinearSetUseKSP - Set a flag to indicate that a `KSP` object, instead of a `Tao` one, should be used
  to fit the linear regressor

  Logically Collective

  Input Parameters:
+ regressor - the `PetscRegressor` context
- flg       - `PETSC_TRUE` to use a `KSP`, `PETSC_FALSE` to use a `Tao` object (default is false)

  Options Database Key:
. regressor_linear_use_ksp <true,false> - use `KSP`

  Level: intermediate

  Notes:
  `KSPLSQR` with no preconditioner is used to solve the normal equations by default.

  For sequential `MATSEQAIJ` sparse matrices QR factorization a `PCType` of `PCQR` can be used to solve the least-squares system with a `MatSolverType` of
  `MATSOLVERSPQR`, using, for example,
.vb
  -ksp_type none -pc_type qr -pc_factor_mat_solver_type sp
.ve
  if centering, `PetscRegressorLinearSetFitIntercept()`, is not used.

  Developer Notes:
  It should be possible to use Cholesky (and any other preconditioners) to solve the normal equations.

  It should be possible to use QR if centering is used. See ml/regressor/ex1.c and ex2.c

  It should be possible to use dense SVD `PCSVD` and dense qr directly on the rectangular matrix to solve the least squares problem.

  Adding the above support seems to require a refactorization of how least squares problems are solved with PETSc in `KSPLSQR`

.seealso: `PetscRegressor`, `PetscRegressorLinearGetKSP()`, `KSPLSQR`, `PCQR`, `MATSOLVERSPQR`, `MatSolverType`, `MATSEQDENSE`, `PCSVD`
@*/
PetscErrorCode PetscRegressorLinearSetUseKSP(PetscRegressor regressor, PetscBool flg)
{
  PetscFunctionBegin;
  /* TODO: Add companion PetscRegressorLinearGetUseKSP(), and put it in the .seealso: */
  PetscValidHeaderSpecific(regressor, PETSCREGRESSOR_CLASSID, 1);
  PetscValidLogicalCollectiveBool(regressor, flg, 2);
  PetscTryMethod(regressor, "PetscRegressorLinearSetUseKSP_C", (PetscRegressor, PetscBool), (regressor, flg));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscRegressorSetFromOptions_Linear(PetscRegressor regressor, PetscOptionItems PetscOptionsObject)
{
  PetscBool              set, flg = PETSC_FALSE;
  PetscRegressor_Linear *linear = (PetscRegressor_Linear *)regressor->data;

  PetscFunctionBegin;
  PetscOptionsHeadBegin(PetscOptionsObject, "PetscRegressor options for linear regressors");
  PetscCall(PetscOptionsBool("-regressor_linear_fit_intercept", "Calculate intercept for linear model", "PetscRegressorLinearSetFitIntercept", flg, &flg, &set));
  if (set) PetscCall(PetscRegressorLinearSetFitIntercept(regressor, flg));
  PetscCall(PetscOptionsBool("-regressor_linear_use_ksp", "Use KSP instead of TAO for linear model fitting problem", "PetscRegressorLinearSetFitIntercept", flg, &flg, &set));
  if (set) PetscCall(PetscRegressorLinearSetUseKSP(regressor, flg));
  PetscCall(PetscOptionsEnum("-regressor_linear_type", "Linear regression method", "PetscRegressorLinearTypes", PetscRegressorLinearTypes, (PetscEnum)linear->type, (PetscEnum *)&linear->type, NULL));
  PetscOptionsHeadEnd();
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscRegressorView_Linear(PetscRegressor regressor, PetscViewer viewer)
{
  PetscBool              isascii;
  PetscRegressor_Linear *linear = (PetscRegressor_Linear *)regressor->data;

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERASCII, &isascii));
  if (isascii) {
    PetscCall(PetscViewerASCIIPushTab(viewer));
    PetscCall(PetscViewerASCIIPrintf(viewer, "PetscRegressor Linear Type: %s\n", PetscRegressorLinearTypes[linear->type]));
    if (linear->ksp) {
      PetscCall(KSPView(linear->ksp, viewer));
      PetscCall(PetscViewerASCIIPrintf(viewer, "total KSP iterations: %" PetscInt_FMT "\n", linear->ksp_tot_its));
    }
    if (linear->fit_intercept) PetscCall(PetscViewerASCIIPrintf(viewer, "Intercept=%g\n", (double)linear->intercept));
    PetscCall(PetscViewerASCIIPopTab(viewer));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscRegressorLinearGetKSP - Returns the `KSP` context for a `PETSCREGRESSORLINEAR` object.

  Not Collective, but if the `PetscRegressor` is parallel, then the `KSP` object is parallel

  Input Parameter:
. regressor - the `PetscRegressor` context

  Output Parameter:
. ksp - the `KSP` context

  Level: beginner

  Note:
  This routine will always return a `KSP`, but, depending on the type of the linear regressor and the options that are set, the regressor may actually use a `Tao` object instead of this `KSP`.

.seealso: `PetscRegressorGetTao()`
@*/
PetscErrorCode PetscRegressorLinearGetKSP(PetscRegressor regressor, KSP *ksp)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(regressor, PETSCREGRESSOR_CLASSID, 1);
  PetscAssertPointer(ksp, 2);
  PetscUseMethod(regressor, "PetscRegressorLinearGetKSP_C", (PetscRegressor, KSP *), (regressor, ksp));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscRegressorLinearGetCoefficients - Get a vector of the fitted coefficients from a linear regression model

  Not Collective but the vector is parallel

  Input Parameter:
. regressor - the `PetscRegressor` context

  Output Parameter:
. coefficients - the vector of the coefficients

  Level: beginner

.seealso: `PetscRegressor`, `PetscRegressorLinearGetIntercept()`, `PETSCREGRESSORLINEAR`, `Vec`
@*/
PETSC_EXTERN PetscErrorCode PetscRegressorLinearGetCoefficients(PetscRegressor regressor, Vec *coefficients)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(regressor, PETSCREGRESSOR_CLASSID, 1);
  PetscAssertPointer(coefficients, 2);
  PetscUseMethod(regressor, "PetscRegressorLinearGetCoefficients_C", (PetscRegressor, Vec *), (regressor, coefficients));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscRegressorLinearGetIntercept - Get the intercept from a linear regression model

  Not Collective

  Input Parameter:
. regressor - the `PetscRegressor` context

  Output Parameter:
. intercept - the intercept

  Level: beginner

.seealso: `PetscRegressor`, `PetscRegressorLinearSetFitIntercept()`, `PetscRegressorLinearGetCoefficients()`, `PETSCREGRESSORLINEAR`
@*/
PETSC_EXTERN PetscErrorCode PetscRegressorLinearGetIntercept(PetscRegressor regressor, PetscScalar *intercept)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(regressor, PETSCREGRESSOR_CLASSID, 1);
  PetscAssertPointer(intercept, 2);
  PetscUseMethod(regressor, "PetscRegressorLinearGetIntercept_C", (PetscRegressor, PetscScalar *), (regressor, intercept));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PetscRegressorLinearSetType - Sets the type of linear regression to be performed

  Logically Collective

  Input Parameters:
+ regressor - the `PetscRegressor` context (should be of type `PETSCREGRESSORLINEAR`)
- type      - a known linear regression method

  Options Database Key:
. -regressor_linear_type - Sets the linear regression method; use -help for a list of available methods
   (for instance "-regressor_linear_type ols" or "-regressor_linear_type lasso")

  Level: intermediate

.seealso: `PetscRegressorLinearGetType()`, `PetscRegressorLinearType`, `PetscRegressorSetType()`, `REGRESSOR_LINEAR_OLS`,
          `REGRESSOR_LINEAR_LASSO`, `REGRESSOR_LINEAR_RIDGE`
@*/
PetscErrorCode PetscRegressorLinearSetType(PetscRegressor regressor, PetscRegressorLinearType type)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(regressor, PETSCREGRESSOR_CLASSID, 1);
  PetscValidLogicalCollectiveEnum(regressor, type, 2);
  PetscTryMethod(regressor, "PetscRegressorLinearSetType_C", (PetscRegressor, PetscRegressorLinearType), (regressor, type));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscRegressorLinearGetType - Return the type for the `PETSCREGRESSORLINEAR` solver

  Input Parameter:
. regressor - the `PetscRegressor` solver context

  Output Parameter:
. type - `PETSCREGRESSORLINEAR` type

  Level: advanced

.seealso: `PetscRegressor`, `PETSCREGRESSORLINEAR`, `PetscRegressorLinearSetType()`, `PetscRegressorLinearType`
@*/
PetscErrorCode PetscRegressorLinearGetType(PetscRegressor regressor, PetscRegressorLinearType *type)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(regressor, PETSCREGRESSOR_CLASSID, 1);
  PetscAssertPointer(type, 2);
  PetscUseMethod(regressor, "PetscRegressorLinearGetType_C", (PetscRegressor, PetscRegressorLinearType *), (regressor, type));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscRegressorFit_Linear(PetscRegressor regressor)
{
  PetscRegressor_Linear *linear = (PetscRegressor_Linear *)regressor->data;
  KSP                    ksp;
  PetscScalar            target_mean, *column_means_global, *column_means_local, column_means_dot_coefficients;
  Vec                    column_means;
  PetscInt               m, N, istart, i, kspits;

  PetscFunctionBegin;
  if (linear->use_ksp) PetscCall(PetscRegressorLinearGetKSP(regressor, &linear->ksp));
  ksp = linear->ksp;

  /* Solve the least-squares problem (previously set up in PetscRegressorSetUp_Linear()) without finding the intercept. */
  if (linear->use_ksp) {
    PetscCall(KSPSolve(ksp, linear->rhs, linear->coefficients));
    PetscCall(KSPGetIterationNumber(ksp, &kspits));
    linear->ksp_its += kspits;
    linear->ksp_tot_its += kspits;
  } else {
    PetscCall(TaoSolve(regressor->tao));
  }

  /* Calculate the intercept. */
  if (linear->fit_intercept) {
    PetscCall(MatGetSize(regressor->training, NULL, &N));
    PetscCall(PetscMalloc1(N, &column_means_global));
    PetscCall(VecMean(regressor->target, &target_mean));
    /* We need the means of all columns of regressor->training, placed into a Vec compatible with linear->coefficients.
     * Note the potential scalability issue: MatGetColumnMeans() computes means of ALL columns. */
    PetscCall(MatGetColumnMeans(regressor->training, column_means_global));
    /* TODO: Calculation of the Vec and matrix column means should probably go into the SetUp phase, and also be placed
     *       into a routine that is callable from outside of PetscRegressorFit_Linear(), because we'll want to do the same
     *       thing for other models, such as ridge and LASSO regression, and should avoid code duplication.
     *       What we are calling 'target_mean' and 'column_means' should be stashed in the base linear regressor struct,
     *       and perhaps renamed to make it clear they are offsets that should be applied (though the current naming
     *       makes sense since it makes it clear where these come from.) */
    PetscCall(VecDuplicate(linear->coefficients, &column_means));
    PetscCall(VecGetLocalSize(column_means, &m));
    PetscCall(VecGetOwnershipRange(column_means, &istart, NULL));
    PetscCall(VecGetArrayWrite(column_means, &column_means_local));
    for (i = 0; i < m; i++) column_means_local[i] = column_means_global[istart + i];
    PetscCall(VecRestoreArrayWrite(column_means, &column_means_local));
    PetscCall(VecDot(column_means, linear->coefficients, &column_means_dot_coefficients));
    PetscCall(VecDestroy(&column_means));
    PetscCall(PetscFree(column_means_global));
    linear->intercept = target_mean - column_means_dot_coefficients;
  } else {
    linear->intercept = 0.0;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscRegressorPredict_Linear(PetscRegressor regressor, Mat X, Vec y)
{
  PetscRegressor_Linear *linear = (PetscRegressor_Linear *)regressor->data;

  PetscFunctionBegin;
  PetscCall(MatMult(X, linear->coefficients, y));
  PetscCall(VecShift(y, linear->intercept));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*MC
     PETSCREGRESSORLINEAR - Linear regression model (ordinary least squares or regularized variants)

   Options Database:
+  -regressor_linear_fit_intercept - Calculate the intercept for the linear model
-  -regressor_linear_use_ksp       - Use `KSP` instead of `Tao` for linear model fitting (non-regularized variants only)

   Level: beginner

   Notes:
   By "linear" we mean that the model is linear in its coefficients, but not necessarily in its input features.
   One can use the linear regressor to fit polynomial functions by training the model with a design matrix that
   is a nonlinear function of the input data.

   This is the default regressor in `PetscRegressor`.

.seealso: `PetscRegressorCreate()`, `PetscRegressor`, `PetscRegressorSetType()`
M*/
PETSC_EXTERN PetscErrorCode PetscRegressorCreate_Linear(PetscRegressor regressor)
{
  PetscRegressor_Linear *linear;

  PetscFunctionBegin;
  PetscCall(PetscNew(&linear));
  regressor->data = (void *)linear;

  regressor->ops->setup          = PetscRegressorSetUp_Linear;
  regressor->ops->reset          = PetscRegressorReset_Linear;
  regressor->ops->destroy        = PetscRegressorDestroy_Linear;
  regressor->ops->setfromoptions = PetscRegressorSetFromOptions_Linear;
  regressor->ops->view           = PetscRegressorView_Linear;
  regressor->ops->fit            = PetscRegressorFit_Linear;
  regressor->ops->predict        = PetscRegressorPredict_Linear;

  linear->intercept     = 0.0;
  linear->fit_intercept = PETSC_TRUE;  /* Default to calculating the intercept. */
  linear->use_ksp       = PETSC_FALSE; /* Do not default to using KSP for solving the model-fitting problem (use TAO instead). */
  linear->type          = REGRESSOR_LINEAR_OLS;
  /* Above, manually set the default linear regressor type.
       We don't use PetscRegressorLinearSetType() here, because that expects the SetUp event to already have happened. */

  PetscCall(PetscObjectComposeFunction((PetscObject)regressor, "PetscRegressorLinearSetFitIntercept_C", PetscRegressorLinearSetFitIntercept_Linear));
  PetscCall(PetscObjectComposeFunction((PetscObject)regressor, "PetscRegressorLinearSetUseKSP_C", PetscRegressorLinearSetUseKSP_Linear));
  PetscCall(PetscObjectComposeFunction((PetscObject)regressor, "PetscRegressorLinearGetKSP_C", PetscRegressorLinearGetKSP_Linear));
  PetscCall(PetscObjectComposeFunction((PetscObject)regressor, "PetscRegressorLinearGetCoefficients_C", PetscRegressorLinearGetCoefficients_Linear));
  PetscCall(PetscObjectComposeFunction((PetscObject)regressor, "PetscRegressorLinearGetIntercept_C", PetscRegressorLinearGetIntercept_Linear));
  PetscCall(PetscObjectComposeFunction((PetscObject)regressor, "PetscRegressorLinearSetType_C", PetscRegressorLinearSetType_Linear));
  PetscCall(PetscObjectComposeFunction((PetscObject)regressor, "PetscRegressorLinearGetType_C", PetscRegressorLinearGetType_Linear));
  PetscFunctionReturn(PETSC_SUCCESS);
}
