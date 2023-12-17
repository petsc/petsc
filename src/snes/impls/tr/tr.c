#include <../src/snes/impls/tr/trimpl.h> /*I   "petscsnes.h"   I*/

typedef struct {
  SNES snes;
  PetscErrorCode (*convtest)(KSP, PetscInt, PetscReal, KSPConvergedReason *, void *);
  PetscErrorCode (*convdestroy)(void *);
  void *convctx;
} SNES_TR_KSPConverged_Ctx;

const char *const SNESNewtonTRFallbackTypes[] = {"NEWTON", "CAUCHY", "DOGLEG", "SNESNewtonTRFallbackType", "SNES_TR_FALLBACK_", NULL};
const char *const SNESNewtonTRQNTypes[]       = {"NONE", "SAME", "DIFFERENT", "SNESNewtonTRQNType", "SNES_TR_QN_", NULL};

static PetscErrorCode SNESComputeJacobian_MATLMVM(SNES snes, Vec X, Mat J, Mat B, void *dummy)
{
  PetscFunctionBegin;
  // PetscCall(MatLMVMSymBroydenSetDelta(B, _some_delta));
  PetscCall(MatLMVMUpdate(B, X, snes->vec_func));
  PetscCall(MatAssemblyBegin(B, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(B, MAT_FINAL_ASSEMBLY));
  if (J != B) {
    // PetscCall(MatLMVMSymBroydenSetDelta(J, _some_delta));
    PetscCall(MatLMVMUpdate(J, X, snes->vec_func));
    PetscCall(MatAssemblyBegin(J, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(J, MAT_FINAL_ASSEMBLY));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode SNESTR_KSPConverged_Private(KSP ksp, PetscInt n, PetscReal rnorm, KSPConvergedReason *reason, void *cctx)
{
  SNES_TR_KSPConverged_Ctx *ctx  = (SNES_TR_KSPConverged_Ctx *)cctx;
  SNES                      snes = ctx->snes;
  SNES_NEWTONTR            *neP  = (SNES_NEWTONTR *)snes->data;
  Vec                       x;
  PetscReal                 nrm;

  PetscFunctionBegin;
  PetscCall((*ctx->convtest)(ksp, n, rnorm, reason, ctx->convctx));
  if (*reason) PetscCall(PetscInfo(snes, "Default or user provided convergence test KSP iterations=%" PetscInt_FMT ", rnorm=%g\n", n, (double)rnorm));
  /* Determine norm of solution */
  PetscCall(KSPBuildSolution(ksp, NULL, &x));
  PetscCall(VecNorm(x, neP->norm, &nrm));
  if (nrm >= neP->delta) {
    PetscCall(PetscInfo(snes, "Ending linear iteration early, delta=%g, length=%g\n", (double)neP->delta, (double)nrm));
    *reason = KSP_CONVERGED_STEP_LENGTH;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode SNESTR_KSPConverged_Destroy(void *cctx)
{
  SNES_TR_KSPConverged_Ctx *ctx = (SNES_TR_KSPConverged_Ctx *)cctx;

  PetscFunctionBegin;
  PetscCall((*ctx->convdestroy)(ctx->convctx));
  PetscCall(PetscFree(ctx));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode SNESTR_Converged_Private(SNES snes, PetscInt it, PetscReal xnorm, PetscReal pnorm, PetscReal fnorm, SNESConvergedReason *reason, void *dummy)
{
  SNES_NEWTONTR *neP = (SNES_NEWTONTR *)snes->data;

  PetscFunctionBegin;
  *reason = SNES_CONVERGED_ITERATING;
  if (neP->delta < snes->deltatol) {
    PetscCall(PetscInfo(snes, "Diverged due to too small a trust region %g<%g\n", (double)neP->delta, (double)snes->deltatol));
    *reason = SNES_DIVERGED_TR_DELTA;
  } else if (snes->nfuncs >= snes->max_funcs && snes->max_funcs >= 0) {
    PetscCall(PetscInfo(snes, "Exceeded maximum number of function evaluations: %" PetscInt_FMT "\n", snes->max_funcs));
    *reason = SNES_DIVERGED_FUNCTION_COUNT;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  SNESNewtonTRSetNormType - Specify the type of norm to use for the computation of the trust region.

  Input Parameters:
+ snes - the nonlinear solver object
- norm - the norm type

  Level: intermediate

.seealso: `SNESNEWTONTR`, `NormType`
@*/
PetscErrorCode SNESNewtonTRSetNormType(SNES snes, NormType norm)
{
  PetscBool flg;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes, SNES_CLASSID, 1);
  PetscValidLogicalCollectiveEnum(snes, norm, 2);
  PetscCall(PetscObjectTypeCompare((PetscObject)snes, SNESNEWTONTR, &flg));
  if (flg) {
    SNES_NEWTONTR *tr = (SNES_NEWTONTR *)snes->data;

    tr->norm = norm;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  SNESNewtonTRSetQNType - Specify to use a quasi-Newton model.

  Input Parameters:
+ snes - the nonlinear solver object
- use  - the type of approximations to be used

  Level: intermediate

  Notes:
  Options for the approximations can be set with the snes_tr_qn_ and snes_tr_qn_pre_ prefixes.

.seealso: `SNESNEWTONTR`, `SNESNewtonTRQNType`, `MATLMVM`
@*/
PetscErrorCode SNESNewtonTRSetQNType(SNES snes, SNESNewtonTRQNType use)
{
  PetscBool flg;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes, SNES_CLASSID, 1);
  PetscValidLogicalCollectiveEnum(snes, use, 2);
  PetscCall(PetscObjectTypeCompare((PetscObject)snes, SNESNEWTONTR, &flg));
  if (flg) {
    SNES_NEWTONTR *tr = (SNES_NEWTONTR *)snes->data;

    tr->qn = use;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  SNESNewtonTRSetFallbackType - Set the type of fallback to use if the solution of the trust region subproblem is outside the radius

  Input Parameters:
+ snes  - the nonlinear solver object
- ftype - the fallback type, see `SNESNewtonTRFallbackType`

  Level: intermediate

.seealso: [](ch_snes), `SNESNEWTONTR`, `SNESNewtonTRPreCheck()`, `SNESNewtonTRGetPreCheck()`, `SNESNewtonTRSetPreCheck()`,
          `SNESNewtonTRSetPostCheck()`, `SNESNewtonTRGetPostCheck()`
@*/
PetscErrorCode SNESNewtonTRSetFallbackType(SNES snes, SNESNewtonTRFallbackType ftype)
{
  SNES_NEWTONTR *tr = (SNES_NEWTONTR *)snes->data;
  PetscBool      flg;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes, SNES_CLASSID, 1);
  PetscValidLogicalCollectiveEnum(snes, ftype, 2);
  PetscCall(PetscObjectTypeCompare((PetscObject)snes, SNESNEWTONTR, &flg));
  if (flg) tr->fallback = ftype;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  SNESNewtonTRSetPreCheck - Sets a user function that is called before the search step has been determined.
  Allows the user a chance to change or override the trust region decision.

  Logically Collective

  Input Parameters:
+ snes - the nonlinear solver object
. func - [optional] function evaluation routine, for the calling sequence see `SNESNewtonTRPreCheck()`
- ctx  - [optional] user-defined context for private data for the function evaluation routine (may be `NULL`)

  Level: deprecated (since 3.19)

  Note:
  This function is called BEFORE the function evaluation within the solver.

.seealso: [](ch_snes), `SNESNEWTONTR`, `SNESNewtonTRPreCheck()`, `SNESNewtonTRGetPreCheck()`, `SNESNewtonTRSetPostCheck()`, `SNESNewtonTRGetPostCheck()`,
@*/
PetscErrorCode SNESNewtonTRSetPreCheck(SNES snes, PetscErrorCode (*func)(SNES, Vec, Vec, PetscBool *, void *), void *ctx)
{
  SNES_NEWTONTR *tr = (SNES_NEWTONTR *)snes->data;
  PetscBool      flg;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes, SNES_CLASSID, 1);
  PetscCall(PetscObjectTypeCompare((PetscObject)snes, SNESNEWTONTR, &flg));
  if (flg) {
    if (func) tr->precheck = func;
    if (ctx) tr->precheckctx = ctx;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  SNESNewtonTRGetPreCheck - Gets the pre-check function

  Deprecated use `SNESNEWTONDCTRDC`

  Not Collective

  Input Parameter:
. snes - the nonlinear solver context

  Output Parameters:
+ func - [optional] function evaluation routine, for the calling sequence see `SNESNewtonTRPreCheck()`
- ctx  - [optional] user-defined context for private data for the function evaluation routine (may be `NULL`)

  Level: deprecated (since 3.19)

.seealso: [](ch_snes), `SNESNEWTONTR`, `SNESNewtonTRSetPreCheck()`, `SNESNewtonTRPreCheck()`
@*/
PetscErrorCode SNESNewtonTRGetPreCheck(SNES snes, PetscErrorCode (**func)(SNES, Vec, Vec, PetscBool *, void *), void **ctx)
{
  SNES_NEWTONTR *tr = (SNES_NEWTONTR *)snes->data;
  PetscBool      flg;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes, SNES_CLASSID, 1);
  PetscCall(PetscObjectTypeCompare((PetscObject)snes, SNESNEWTONTR, &flg));
  PetscAssert(flg, PetscObjectComm((PetscObject)snes), PETSC_ERR_ARG_WRONG, "Not for type %s", ((PetscObject)snes)->type_name);
  if (func) *func = tr->precheck;
  if (ctx) *ctx = tr->precheckctx;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  SNESNewtonTRSetPostCheck - Sets a user function that is called after the search step has been determined but before the next
  function evaluation. Allows the user a chance to change or override the internal decision of the solver

  Logically Collective

  Input Parameters:
+ snes - the nonlinear solver object
. func - [optional] function evaluation routine, for the calling sequence see `SNESNewtonTRPostCheck()`
- ctx  - [optional] user-defined context for private data for the function evaluation routine (may be `NULL`)

  Level: deprecated (since 3.19)

  Note:
  This function is called BEFORE the function evaluation within the solver while the function set in
  `SNESLineSearchSetPostCheck()` is called AFTER the function evaluation.

.seealso: [](ch_snes), `SNESNEWTONTR`, `SNESNewtonTRPostCheck()`, `SNESNewtonTRGetPostCheck()`, `SNESNewtonTRSetPreCheck()`, `SNESNewtonTRGetPreCheck()`
@*/
PetscErrorCode SNESNewtonTRSetPostCheck(SNES snes, PetscErrorCode (*func)(SNES, Vec, Vec, Vec, PetscBool *, PetscBool *, void *), void *ctx)
{
  SNES_NEWTONTR *tr = (SNES_NEWTONTR *)snes->data;
  PetscBool      flg;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes, SNES_CLASSID, 1);
  PetscCall(PetscObjectTypeCompare((PetscObject)snes, SNESNEWTONTR, &flg));
  if (flg) {
    if (func) tr->postcheck = func;
    if (ctx) tr->postcheckctx = ctx;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  SNESNewtonTRGetPostCheck - Gets the post-check function

  Not Collective

  Input Parameter:
. snes - the nonlinear solver context

  Output Parameters:
+ func - [optional] function evaluation routine, for the calling sequence see `SNESNewtonTRPostCheck()`
- ctx  - [optional] user-defined context for private data for the function evaluation routine (may be `NULL`)

  Level: intermediate

.seealso: [](ch_snes), `SNESNEWTONTR`, `SNESNewtonTRSetPostCheck()`, `SNESNewtonTRPostCheck()`
@*/
PetscErrorCode SNESNewtonTRGetPostCheck(SNES snes, PetscErrorCode (**func)(SNES, Vec, Vec, Vec, PetscBool *, PetscBool *, void *), void **ctx)
{
  SNES_NEWTONTR *tr = (SNES_NEWTONTR *)snes->data;
  PetscBool      flg;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes, SNES_CLASSID, 1);
  PetscCall(PetscObjectTypeCompare((PetscObject)snes, SNESNEWTONTR, &flg));
  PetscAssert(flg, PetscObjectComm((PetscObject)snes), PETSC_ERR_ARG_WRONG, "Not for type %s", ((PetscObject)snes)->type_name);
  if (func) *func = tr->postcheck;
  if (ctx) *ctx = tr->postcheckctx;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  SNESNewtonTRPreCheck - Runs the precheck routine

  Logically Collective

  Input Parameters:
+ snes - the solver
. X    - The last solution
- Y    - The step direction

  Output Parameter:
. changed_Y - Indicator that the step direction `Y` has been changed.

  Level: intermediate

.seealso: [](ch_snes), `SNESNEWTONTR`, `SNESNewtonTRSetPreCheck()`, `SNESNewtonTRGetPreCheck()`, `SNESNewtonTRPostCheck()`
@*/
PetscErrorCode SNESNewtonTRPreCheck(SNES snes, Vec X, Vec Y, PetscBool *changed_Y)
{
  SNES_NEWTONTR *tr = (SNES_NEWTONTR *)snes->data;
  PetscBool      flg;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes, SNES_CLASSID, 1);
  PetscCall(PetscObjectTypeCompare((PetscObject)snes, SNESNEWTONTR, &flg));
  PetscAssert(flg, PetscObjectComm((PetscObject)snes), PETSC_ERR_ARG_WRONG, "Not for type %s", ((PetscObject)snes)->type_name);
  *changed_Y = PETSC_FALSE;
  if (tr->precheck) {
    PetscCall((*tr->precheck)(snes, X, Y, changed_Y, tr->precheckctx));
    PetscValidLogicalCollectiveBool(snes, *changed_Y, 4);
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  SNESNewtonTRPostCheck - Runs the postcheck routine

  Logically Collective

  Input Parameters:
+ snes - the solver
. X    - The last solution
. Y    - The full step direction
- W    - The updated solution, W = X - Y

  Output Parameters:
+ changed_Y - indicator if step has been changed
- changed_W - Indicator if the new candidate solution W has been changed.

  Note:
  If Y is changed then W is recomputed as X - Y

  Level: intermediate

.seealso: [](ch_snes), `SNESNEWTONTR`, `SNESNewtonTRSetPostCheck()`, `SNESNewtonTRGetPostCheck()`, `SNESNewtonTRPreCheck()`
@*/
PetscErrorCode SNESNewtonTRPostCheck(SNES snes, Vec X, Vec Y, Vec W, PetscBool *changed_Y, PetscBool *changed_W)
{
  SNES_NEWTONTR *tr = (SNES_NEWTONTR *)snes->data;
  PetscBool      flg;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes, SNES_CLASSID, 1);
  PetscCall(PetscObjectTypeCompare((PetscObject)snes, SNESNEWTONTR, &flg));
  PetscAssert(flg, PetscObjectComm((PetscObject)snes), PETSC_ERR_ARG_WRONG, "Not for type %s", ((PetscObject)snes)->type_name);
  *changed_Y = PETSC_FALSE;
  *changed_W = PETSC_FALSE;
  if (tr->postcheck) {
    PetscCall((*tr->postcheck)(snes, X, Y, W, changed_Y, changed_W, tr->postcheckctx));
    PetscValidLogicalCollectiveBool(snes, *changed_Y, 5);
    PetscValidLogicalCollectiveBool(snes, *changed_W, 6);
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* stable implementation of roots of a*x^2 + b*x + c = 0 */
static inline void PetscQuadraticRoots(PetscReal a, PetscReal b, PetscReal c, PetscReal *xm, PetscReal *xp)
{
  PetscReal temp = -0.5 * (b + PetscCopysignReal(1.0, b) * PetscSqrtReal(b * b - 4 * a * c));
  PetscReal x1   = temp / a;
  PetscReal x2   = c / temp;
  *xm            = PetscMin(x1, x2);
  *xp            = PetscMax(x1, x2);
}

/* Computes the quadratic model difference */
static PetscErrorCode SNESNewtonTRQuadraticDelta(SNES snes, Mat J, PetscBool has_objective, Vec Y, Vec GradF, Vec W, PetscReal *yTHy_, PetscReal *gTy_, PetscReal *deltaqm)
{
  PetscReal yTHy, gTy;

  PetscFunctionBegin;
  PetscCall(MatMult(J, Y, W));
  if (has_objective) PetscCall(VecDotRealPart(Y, W, &yTHy));
  else PetscCall(VecDotRealPart(W, W, &yTHy)); /* Gauss-Newton approximation J^t * J */
  PetscCall(VecDotRealPart(GradF, Y, &gTy));
  *deltaqm = -(-(gTy) + 0.5 * (yTHy)); /* difference in quadratic model, -gTy because SNES solves it this way */
  if (yTHy_) *yTHy_ = yTHy;
  if (gTy_) *gTy_ = gTy;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Computes the new objective given X = Xk, Y = direction
   W work vector, on output W = X - Y
   G work vector, on output G = SNESFunction(W) */
static PetscErrorCode SNESNewtonTRObjective(SNES snes, PetscBool has_objective, Vec X, Vec Y, Vec W, Vec G, PetscReal *gnorm, PetscReal *fkp1)
{
  PetscBool changed_y, changed_w;

  PetscFunctionBegin;
  /* TODO: we can add a linesearch here */
  PetscCall(SNESNewtonTRPreCheck(snes, X, Y, &changed_y));
  PetscCall(VecWAXPY(W, -1.0, Y, X)); /* Xkp1 */
  PetscCall(SNESNewtonTRPostCheck(snes, X, Y, W, &changed_y, &changed_w));
  if (changed_y && !changed_w) PetscCall(VecWAXPY(W, -1.0, Y, X));

  PetscCall(SNESComputeFunction(snes, W, G)); /*  F(Xkp1) = G */
  PetscCall(VecNorm(G, NORM_2, gnorm));
  if (has_objective) PetscCall(SNESComputeObjective(snes, W, fkp1));
  else *fkp1 = 0.5 * PetscSqr(*gnorm);
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode SNESSetUpQN_NEWTONTR(SNES snes)
{
  SNES_NEWTONTR *tr = (SNES_NEWTONTR *)snes->data;

  PetscFunctionBegin;
  PetscCall(MatDestroy(&tr->qnB));
  PetscCall(MatDestroy(&tr->qnB_pre));
  if (tr->qn) {
    PetscInt    n, N;
    const char *optionsprefix;
    Mat         B;

    PetscCall(MatCreate(PetscObjectComm((PetscObject)snes), &B));
    PetscCall(SNESGetOptionsPrefix(snes, &optionsprefix));
    PetscCall(MatSetOptionsPrefix(B, "snes_tr_qn_"));
    PetscCall(MatAppendOptionsPrefix(B, optionsprefix));
    PetscCall(MatSetType(B, MATLMVMBFGS));
    PetscCall(VecGetLocalSize(snes->vec_sol, &n));
    PetscCall(VecGetSize(snes->vec_sol, &N));
    PetscCall(MatSetSizes(B, n, n, N, N));
    PetscCall(MatSetUp(B));
    PetscCall(MatSetFromOptions(B));
    PetscCall(MatLMVMAllocate(B, snes->vec_sol, snes->vec_func));
    tr->qnB = B;
    if (tr->qn == SNES_TR_QN_DIFFERENT) {
      PetscCall(MatCreate(PetscObjectComm((PetscObject)snes), &B));
      PetscCall(SNESGetOptionsPrefix(snes, &optionsprefix));
      PetscCall(MatSetOptionsPrefix(B, "snes_tr_qn_pre_"));
      PetscCall(MatAppendOptionsPrefix(B, optionsprefix));
      PetscCall(MatSetType(B, MATLMVMBFGS));
      PetscCall(MatSetSizes(B, n, n, N, N));
      PetscCall(MatSetUp(B));
      PetscCall(MatSetFromOptions(B));
      PetscCall(MatLMVMAllocate(B, snes->vec_sol, snes->vec_func));
      tr->qnB_pre = B;
    } else {
      PetscCall(PetscObjectReference((PetscObject)tr->qnB));
      tr->qnB_pre = tr->qnB;
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
   SNESSolve_NEWTONTR - Implements Newton's Method with trust-region subproblem and adds dogleg Cauchy
   (Steepest Descent direction) step and direction if the trust region is not satisfied for solving system of
   nonlinear equations

*/
static PetscErrorCode SNESSolve_NEWTONTR(SNES snes)
{
  SNES_NEWTONTR            *neP = (SNES_NEWTONTR *)snes->data;
  Vec                       X, F, Y, G, W, GradF, YU, Yc;
  PetscInt                  maxits, lits;
  PetscReal                 rho, fnorm, gnorm = 0.0, xnorm = 0.0, delta, ynorm;
  PetscReal                 deltaM, fk, fkp1, deltaqm = 0.0, gTy = 0.0, yTHy = 0.0;
  PetscReal                 auk, tauk, gfnorm, gfnorm_k, ycnorm, gTBg, objmin = 0.0, beta_k = 1.0;
  PC                        pc;
  Mat                       J, Jp;
  PetscBool                 already_done = PETSC_FALSE, on_boundary;
  PetscBool                 clear_converged_test, rho_satisfied, has_objective;
  SNES_TR_KSPConverged_Ctx *ctx;
  void                     *convctx;

  PetscErrorCode (*convtest)(KSP, PetscInt, PetscReal, KSPConvergedReason *, void *), (*convdestroy)(void *);
  PetscErrorCode (*objective)(SNES, Vec, PetscReal *, void *);

  PetscFunctionBegin;
  PetscCall(SNESGetObjective(snes, &objective, NULL));
  has_objective = objective ? PETSC_TRUE : PETSC_FALSE;

  maxits = snes->max_its;                                   /* maximum number of iterations */
  X      = snes->vec_sol;                                   /* solution vector */
  F      = snes->vec_func;                                  /* residual vector */
  Y      = snes->vec_sol_update;                            /* update vector */
  G      = snes->work[0];                                   /* updated residual */
  W      = snes->work[1];                                   /* temporary vector */
  GradF  = !has_objective ? snes->work[2] : snes->vec_func; /* grad f = J^T F */
  YU     = snes->work[3];                                   /* work vector for dogleg method */
  Yc     = snes->work[4];                                   /* Cauchy point */

  PetscCheck(!snes->xl && !snes->xu && !snes->ops->computevariablebounds, PetscObjectComm((PetscObject)snes), PETSC_ERR_ARG_WRONGSTATE, "SNES solver %s does not support bounds", ((PetscObject)snes)->type_name);

  PetscCall(PetscObjectSAWsTakeAccess((PetscObject)snes));
  snes->iter = 0;
  PetscCall(PetscObjectSAWsGrantAccess((PetscObject)snes));

  /* setup QN matrices if needed */
  PetscCall(SNESSetUpQN_NEWTONTR(snes));

  /* Set the linear stopping criteria to use the More' trick if needed */
  clear_converged_test = PETSC_FALSE;
  PetscCall(SNESGetKSP(snes, &snes->ksp));
  PetscCall(KSPGetConvergenceTest(snes->ksp, &convtest, &convctx, &convdestroy));
  if (convtest != SNESTR_KSPConverged_Private) {
    clear_converged_test = PETSC_TRUE;
    PetscCall(PetscNew(&ctx));
    ctx->snes = snes;
    PetscCall(KSPGetAndClearConvergenceTest(snes->ksp, &ctx->convtest, &ctx->convctx, &ctx->convdestroy));
    PetscCall(KSPSetConvergenceTest(snes->ksp, SNESTR_KSPConverged_Private, ctx, SNESTR_KSPConverged_Destroy));
    PetscCall(PetscInfo(snes, "Using Krylov convergence test SNESTR_KSPConverged_Private\n"));
  }

  if (!snes->vec_func_init_set) {
    PetscCall(SNESComputeFunction(snes, X, F)); /* F(X) */
  } else snes->vec_func_init_set = PETSC_FALSE;

  PetscCall(VecNorm(F, NORM_2, &fnorm)); /* fnorm <- || F || */
  SNESCheckFunctionNorm(snes, fnorm);
  PetscCall(VecNorm(X, NORM_2, &xnorm)); /* xnorm <- || X || */

  PetscCall(PetscObjectSAWsTakeAccess((PetscObject)snes));
  snes->norm = fnorm;
  PetscCall(PetscObjectSAWsGrantAccess((PetscObject)snes));
  delta      = neP->delta0;
  deltaM     = neP->deltaM;
  neP->delta = delta;
  PetscCall(SNESLogConvergenceHistory(snes, fnorm, 0));

  /* test convergence */
  rho_satisfied = PETSC_FALSE;
  PetscCall(SNESConverged(snes, 0, 0.0, 0.0, fnorm));
  PetscCall(SNESMonitor(snes, 0, fnorm));
  if (snes->reason) PetscFunctionReturn(PETSC_SUCCESS);

  if (has_objective) PetscCall(SNESComputeObjective(snes, X, &fk));
  else fk = 0.5 * PetscSqr(fnorm); /* obj(x) = 0.5 * ||F(x)||^2 */

  /* hook state vector to BFGS preconditioner */
  PetscCall(KSPGetPC(snes->ksp, &pc));
  PetscCall(PCLMVMSetUpdateVec(pc, X));

  if (neP->kmdc) PetscCall(KSPSetComputeEigenvalues(snes->ksp, PETSC_TRUE));

  while (snes->iter < maxits) {
    /* calculating Jacobian and GradF of minimization function only once */
    if (!already_done) {
      /* Call general purpose update function */
      PetscTryTypeMethod(snes, update, snes->iter);

      /* apply the nonlinear preconditioner */
      if (snes->npc && snes->npcside == PC_RIGHT) {
        SNESConvergedReason reason;

        PetscCall(SNESSetInitialFunction(snes->npc, F));
        PetscCall(PetscLogEventBegin(SNES_NPCSolve, snes->npc, X, snes->vec_rhs, 0));
        PetscCall(SNESSolve(snes->npc, snes->vec_rhs, X));
        PetscCall(PetscLogEventEnd(SNES_NPCSolve, snes->npc, X, snes->vec_rhs, 0));
        PetscCall(SNESGetConvergedReason(snes->npc, &reason));
        if (reason < 0 && reason != SNES_DIVERGED_MAX_IT && reason != SNES_DIVERGED_TR_DELTA) {
          snes->reason = SNES_DIVERGED_INNER;
          PetscFunctionReturn(PETSC_SUCCESS);
        }
        // XXX
        PetscCall(SNESGetNPCFunction(snes, F, &fnorm));
      } else if (snes->ops->update) { /* if update is present, recompute objective function and function norm */
        PetscCall(SNESComputeFunction(snes, X, F));
      }

      /* Jacobian */
      J  = NULL;
      Jp = NULL;
      if (!neP->qnB) {
        PetscCall(SNESComputeJacobian(snes, X, snes->jacobian, snes->jacobian_pre));
        J  = snes->jacobian;
        Jp = snes->jacobian_pre;
      } else { /* QN model */
        PetscCall(SNESComputeJacobian_MATLMVM(snes, X, neP->qnB, neP->qnB_pre, NULL));
        J  = neP->qnB;
        Jp = neP->qnB_pre;
      }
      SNESCheckJacobianDomainerror(snes);

      /* objective function */
      PetscCall(VecNorm(F, NORM_2, &fnorm));
      if (has_objective) PetscCall(SNESComputeObjective(snes, X, &fk));
      else fk = 0.5 * PetscSqr(fnorm); /* obj(x) = 0.5 * ||F(x)||^2 */

      /* GradF */
      if (has_objective) gfnorm = fnorm;
      else {
        PetscCall(MatMultTranspose(J, F, GradF)); /* grad f = J^T F */
        PetscCall(VecNorm(GradF, NORM_2, &gfnorm));
      }
      PetscCall(VecNorm(GradF, neP->norm, &gfnorm_k));
    }
    already_done = PETSC_TRUE;

    /* solve trust-region subproblem */

    /* first compute Cauchy Point */
    PetscCall(MatMult(J, GradF, W));
    if (has_objective) PetscCall(VecDotRealPart(GradF, W, &gTBg));
    else PetscCall(VecDotRealPart(W, W, &gTBg)); /* B = J^t * J */
    /* Eqs 4.11 and 4.12 in Nocedal and Wright 2nd Edition (4.7 and 4.8 in 1st Edition) */
    auk = delta / gfnorm_k;
    if (gTBg < 0.0) tauk = 1.0;
    else tauk = PetscMin(gfnorm * gfnorm * gfnorm_k / (delta * gTBg), 1);
    auk *= tauk;
    ycnorm = auk * gfnorm;
    PetscCall(VecAXPBY(Yc, auk, 0.0, GradF));

    on_boundary = PETSC_FALSE;
    if (tauk != 1.0) {
      KSPConvergedReason reason;

      /* sufficient decrease (see 6.3.27 in Conn, Gould, Toint "Trust Region Methods")
         beta_k the largest eigenvalue of the Hessian. Here we use the previous estimated value */
      objmin = -neP->kmdc * gnorm * PetscMin(gnorm / beta_k, delta);
      PetscCall(KSPCGSetObjectiveTarget(snes->ksp, objmin));

      /* specify radius if looking for Newton step and trust region norm is the l2 norm */
      PetscCall(KSPCGSetRadius(snes->ksp, neP->fallback == SNES_TR_FALLBACK_NEWTON && neP->norm == NORM_2 ? delta : 0.0));
      PetscCall(KSPSetOperators(snes->ksp, J, Jp));
      PetscCall(KSPSolve(snes->ksp, F, Y));
      SNESCheckKSPSolve(snes);
      PetscCall(KSPGetIterationNumber(snes->ksp, &lits));
      PetscCall(KSPGetConvergedReason(snes->ksp, &reason));
      on_boundary = (PetscBool)(reason == KSP_CONVERGED_STEP_LENGTH);
      PetscCall(PetscInfo(snes, "iter=%" PetscInt_FMT ", linear solve iterations=%" PetscInt_FMT "\n", snes->iter, lits));
      if (neP->kmdc) { /* update estimated Hessian largest eigenvalue */
        PetscReal emax, emin;
        PetscCall(KSPComputeExtremeSingularValues(snes->ksp, &emax, &emin));
        if (emax > 0.0) beta_k = emax + 1;
      }
    } else { /* Cauchy point is on the boundary, accept it */
      on_boundary = PETSC_TRUE;
      PetscCall(VecCopy(Yc, Y));
      PetscCall(PetscInfo(snes, "CP evaluated on boundary. delta: %g, ycnorm: %g, gTBg: %g\n", (double)delta, (double)ycnorm, (double)gTBg));
    }
    PetscCall(VecNorm(Y, neP->norm, &ynorm));

    /* decide what to do when the update is outside of trust region */
    if (ynorm > delta || ynorm == 0.0) {
      SNESNewtonTRFallbackType fallback = ynorm > 0.0 ? neP->fallback : SNES_TR_FALLBACK_CAUCHY;

      PetscCheck(neP->norm == NORM_2 || fallback != SNES_TR_FALLBACK_DOGLEG, PetscObjectComm((PetscObject)snes), PETSC_ERR_SUP, "DOGLEG without l2 norm not implemented");
      switch (fallback) {
      case SNES_TR_FALLBACK_NEWTON:
        auk = delta / ynorm;
        PetscCall(VecScale(Y, auk));
        PetscCall(PetscInfo(snes, "SN evaluated. delta: %g, ynorm: %g\n", (double)delta, (double)ynorm));
        break;
      case SNES_TR_FALLBACK_CAUCHY:
      case SNES_TR_FALLBACK_DOGLEG:
        if (fallback == SNES_TR_FALLBACK_CAUCHY || gTBg <= 0.0) {
          PetscCall(VecCopy(Yc, Y));
          PetscCall(PetscInfo(snes, "CP evaluated. delta: %g, ynorm: %g, ycnorm: %g, gTBg: %g\n", (double)delta, (double)ynorm, (double)ycnorm, (double)gTBg));
        } else { /* take linear combination of Cauchy and Newton direction and step */
          auk = gfnorm * gfnorm / gTBg;
          if (gfnorm_k * auk >= delta) { /* first leg: Cauchy point outside of trust region */
            PetscCall(VecAXPBY(Y, delta / gfnorm_k, 0.0, GradF));
            PetscCall(PetscInfo(snes, "CP evaluated (outside region). delta: %g, ynorm: %g, ycnorm: %g\n", (double)delta, (double)ynorm, (double)ycnorm));
          } else { /* second leg */
            PetscReal c0, c1, c2, tau = 0.0, tpos, tneg;
            PetscBool noroots;

            /* Find solutions of (Eq. 4.16 in Nocedal and Wright)
                 ||p_U + lambda * (p_B - p_U)||^2 - delta^2 = 0,
               where p_U  the Cauchy direction, p_B the Newton direction */
            PetscCall(VecAXPBY(YU, auk, 0.0, GradF));
            PetscCall(VecAXPY(Y, -1.0, YU));
            PetscCall(VecNorm(Y, NORM_2, &c0));
            PetscCall(VecDotRealPart(YU, Y, &c1));
            c0 = PetscSqr(c0);
            c2 = PetscSqr(ycnorm) - PetscSqr(delta);
            PetscQuadraticRoots(c0, 2 * c1, c2, &tneg, &tpos);

            /* In principle the DL strategy as a unique solution in [0,1]
               here we check that for some reason we numerically failed
               to compute it. In that case, we use the Cauchy point */
            noroots = PetscIsInfOrNanReal(tneg);
            if (!noroots) {
              if (tpos > 1) {
                if (tneg >= 0 && tneg <= 1) {
                  tau = tneg;
                } else noroots = PETSC_TRUE;
              } else if (tpos >= 0) {
                tau = tpos;
              } else noroots = PETSC_TRUE;
            }
            if (noroots) { /* No roots, select Cauchy point */
              PetscCall(VecCopy(Yc, Y));
            } else {
              PetscCall(VecAXPBY(Y, 1.0, tau, YU));
            }
            PetscCall(PetscInfo(snes, "%s evaluated. roots: (%g, %g), tau %g, ynorm: %g, ycnorm: %g, gTBg: %g\n", noroots ? "CP" : "DL", (double)tneg, (double)tpos, (double)tau, (double)ynorm, (double)ycnorm, (double)gTBg));
          }
        }
        break;
      default:
        SETERRQ(PetscObjectComm((PetscObject)snes), PETSC_ERR_SUP, "Unknown fallback mode");
        break;
      }
    }

    /* compute the quadratic model difference */
    PetscCall(SNESNewtonTRQuadraticDelta(snes, J, has_objective, Y, GradF, W, &yTHy, &gTy, &deltaqm));

    /* Compute new objective function */
    PetscCall(SNESNewtonTRObjective(snes, has_objective, X, Y, W, G, &gnorm, &fkp1));
    if (PetscIsInfOrNanReal(fkp1)) rho = neP->eta1;
    else {
      if (deltaqm > 0.0) rho = (fk - fkp1) / deltaqm; /* actual improvement over predicted improvement */
      else rho = neP->eta1;                           /*  no reduction in quadratic model, step must be rejected */
    }

    PetscCall(VecNorm(Y, neP->norm, &ynorm));
    PetscCall(PetscInfo(snes, "rho=%g, delta=%g, fk=%g, fkp1=%g, deltaqm=%g, gTy=%g, yTHy=%g, ynormk=%g\n", (double)rho, (double)delta, (double)fk, (double)fkp1, (double)deltaqm, (double)gTy, (double)yTHy, (double)ynorm));

    /* update the size of the trust region */
    if (rho < neP->eta2) delta *= neP->t1;                     /* shrink the region */
    else if (rho > neP->eta3 && on_boundary) delta *= neP->t2; /* expand the region */
    delta = PetscMin(delta, deltaM);                           /* but not greater than deltaM */

    /* log 2-norm of update for moniroting routines */
    PetscCall(VecNorm(Y, NORM_2, &ynorm));

    /* decide on new step */
    neP->delta = delta;
    if (rho > neP->eta1) {
      rho_satisfied = PETSC_TRUE;
    } else {
      rho_satisfied = PETSC_FALSE;
      PetscCall(PetscInfo(snes, "Trying again in smaller region\n"));
      /* check to see if progress is hopeless */
      PetscCall(SNESTR_Converged_Private(snes, snes->iter, xnorm, ynorm, fnorm, &snes->reason, snes->cnvP));
      if (!snes->reason) PetscCall(SNESConverged(snes, snes->iter, xnorm, ynorm, fnorm));
      if (snes->reason == SNES_CONVERGED_SNORM_RELATIVE) snes->reason = SNES_DIVERGED_TR_DELTA;
      snes->numFailures++;
      /* We're not progressing, so return with the current iterate */
      if (snes->reason) break;
    }
    if (rho_satisfied) {
      /* Update function values */
      already_done = PETSC_FALSE;
      fnorm        = gnorm;
      fk           = fkp1;

      /* New residual and linearization point */
      PetscCall(VecCopy(G, F));
      PetscCall(VecCopy(W, X));

      /* Monitor convergence */
      PetscCall(PetscObjectSAWsTakeAccess((PetscObject)snes));
      snes->iter++;
      snes->norm  = fnorm;
      snes->xnorm = xnorm;
      snes->ynorm = ynorm;
      PetscCall(PetscObjectSAWsGrantAccess((PetscObject)snes));
      PetscCall(SNESLogConvergenceHistory(snes, snes->norm, lits));

      /* Test for convergence, xnorm = || X || */
      PetscCall(VecNorm(X, NORM_2, &xnorm));
      PetscCall(SNESConverged(snes, snes->iter, xnorm, ynorm, fnorm));
      PetscCall(SNESMonitor(snes, snes->iter, snes->norm));
      if (snes->reason) break;
    }
  }

  if (clear_converged_test) {
    PetscCall(KSPGetAndClearConvergenceTest(snes->ksp, &ctx->convtest, &ctx->convctx, &ctx->convdestroy));
    PetscCall(PetscFree(ctx));
    PetscCall(KSPSetConvergenceTest(snes->ksp, convtest, convctx, convdestroy));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode SNESSetUp_NEWTONTR(SNES snes)
{
  PetscFunctionBegin;
  PetscCall(SNESSetWorkVecs(snes, 5));
  PetscCall(SNESSetUpMatrices(snes));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode SNESReset_NEWTONTR(SNES snes)
{
  SNES_NEWTONTR *tr = (SNES_NEWTONTR *)snes->data;

  PetscFunctionBegin;
  PetscCall(MatDestroy(&tr->qnB));
  PetscCall(MatDestroy(&tr->qnB_pre));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode SNESDestroy_NEWTONTR(SNES snes)
{
  PetscFunctionBegin;
  PetscCall(SNESReset_NEWTONTR(snes));
  PetscCall(PetscFree(snes->data));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode SNESSetFromOptions_NEWTONTR(SNES snes, PetscOptionItems *PetscOptionsObject)
{
  SNES_NEWTONTR           *ctx = (SNES_NEWTONTR *)snes->data;
  SNESNewtonTRQNType       qn;
  SNESNewtonTRFallbackType fallback;
  NormType                 norm;
  PetscReal                deltatol;
  PetscBool                flg;

  PetscFunctionBegin;
  PetscOptionsHeadBegin(PetscOptionsObject, "SNES trust region options for nonlinear equations");
  PetscCall(PetscOptionsReal("-snes_tr_eta1", "eta1", "None", ctx->eta1, &ctx->eta1, NULL));
  PetscCall(PetscOptionsReal("-snes_tr_eta2", "eta2", "None", ctx->eta2, &ctx->eta2, NULL));
  PetscCall(PetscOptionsReal("-snes_tr_eta3", "eta3", "None", ctx->eta3, &ctx->eta3, NULL));
  PetscCall(PetscOptionsReal("-snes_tr_t1", "t1", "None", ctx->t1, &ctx->t1, NULL));
  PetscCall(PetscOptionsReal("-snes_tr_t2", "t2", "None", ctx->t2, &ctx->t2, NULL));
  PetscCall(PetscOptionsReal("-snes_tr_deltaM", "deltaM", "None", ctx->deltaM, &ctx->deltaM, NULL));
  PetscCall(PetscOptionsReal("-snes_tr_delta0", "delta0", "None", ctx->delta0, &ctx->delta0, NULL));
  PetscCall(PetscOptionsReal("-snes_tr_kmdc", "sufficient decrease parameter", "None", ctx->kmdc, &ctx->kmdc, NULL));

  deltatol = snes->deltatol;
  PetscCall(PetscOptionsReal("-snes_tr_tol", "Trust region tolerance", "SNESSetTrustRegionTolerance", deltatol, &deltatol, &flg));
  if (flg) PetscCall(SNESSetTrustRegionTolerance(snes, deltatol));

  fallback = ctx->fallback;
  PetscCall(PetscOptionsEnum("-snes_tr_fallback_type", "Type of fallback if subproblem solution is outside of the trust region", "SNESNewtonTRSetFallbackType", SNESNewtonTRFallbackTypes, (PetscEnum)fallback, (PetscEnum *)&fallback, &flg));
  if (flg) PetscCall(SNESNewtonTRSetFallbackType(snes, fallback));

  qn = ctx->qn;
  PetscCall(PetscOptionsEnum("-snes_tr_qn", "Use Quasi-Newton approximations for the model", "SNESNewtonTRSetQNType", SNESNewtonTRQNTypes, (PetscEnum)qn, (PetscEnum *)&qn, &flg));
  if (flg) PetscCall(SNESNewtonTRSetQNType(snes, qn));

  norm = ctx->norm;
  PetscCall(PetscOptionsEnum("-snes_tr_norm_type", "Type of norm for trust region bounds", "SNESNewtonTRSetNormType", NormTypes, (PetscEnum)norm, (PetscEnum *)&norm, &flg));
  if (flg) PetscCall(SNESNewtonTRSetNormType(snes, norm));

  PetscOptionsHeadEnd();
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode SNESView_NEWTONTR(SNES snes, PetscViewer viewer)
{
  SNES_NEWTONTR *tr = (SNES_NEWTONTR *)snes->data;
  PetscBool      iascii;

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERASCII, &iascii));
  if (iascii) {
    PetscCall(PetscViewerASCIIPrintf(viewer, "  Trust region tolerance %g\n", (double)snes->deltatol));
    PetscCall(PetscViewerASCIIPrintf(viewer, "  eta1=%g, eta2=%g, eta3=%g\n", (double)tr->eta1, (double)tr->eta2, (double)tr->eta3));
    PetscCall(PetscViewerASCIIPrintf(viewer, "  delta0=%g, t1=%g, t2=%g, deltaM=%g\n", (double)tr->delta0, (double)tr->t1, (double)tr->t2, (double)tr->deltaM));
    PetscCall(PetscViewerASCIIPrintf(viewer, "  kmdc=%g\n", (double)tr->kmdc));
    PetscCall(PetscViewerASCIIPrintf(viewer, "  fallback=%s\n", SNESNewtonTRFallbackTypes[tr->fallback]));
    if (tr->qn) PetscCall(PetscViewerASCIIPrintf(viewer, "  qn=%s\n", SNESNewtonTRQNTypes[tr->qn]));
    if (tr->norm != NORM_2) PetscCall(PetscViewerASCIIPrintf(viewer, "  norm=%s\n", NormTypes[tr->norm]));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*MC
   SNESNEWTONTR - Newton based nonlinear solver that uses trust-region dogleg method with Cauchy direction {cite}`nocedal2006numerical`

   Options Database Keys:
+  -snes_tr_tol <tol>                            - trust region tolerance
.  -snes_tr_eta1 <eta1>                          - trust region parameter eta1 <= eta2, rho > eta1 breaks out of the inner iteration (default: eta1=0.001)
.  -snes_tr_eta2 <eta2>                          - trust region parameter, rho <= eta2 shrinks the trust region (default: eta2=0.25)
.  -snes_tr_eta3 <eta3>                          - trust region parameter eta3 > eta2, rho >= eta3 expands the trust region (default: eta3=0.75)
.  -snes_tr_t1 <t1>                              - trust region parameter, shrinking factor of trust region (default: 0.25)
.  -snes_tr_t2 <t2>                              - trust region parameter, expanding factor of trust region (default: 2.0)
.  -snes_tr_deltaM <deltaM>                      - trust region parameter, max size of trust region (default: MAX_REAL)
.  -snes_tr_delta0 <delta0>                      - trust region parameter, initial size of trust region (default: 0.2)
-  -snes_tr_fallback_type <newton,cauchy,dogleg> - Solution strategy to test reduction when step is outside of trust region. Can use scaled Newton direction, Cauchy point (Steepest Descent direction) or dogleg method.
    Level: deprecated

.seealso: [](ch_snes), `SNESCreate()`, `SNES`, `SNESSetType()`, `SNESNEWTONLS`, `SNESSetTrustRegionTolerance()`,
          `SNESNewtonTRPreCheck()`, `SNESNewtonTRGetPreCheck()`, `SNESNewtonTRSetPostCheck()`, `SNESNewtonTRGetPostCheck()`,
          `SNESNewtonTRSetPreCheck()`, `SNESNewtonTRSetFallbackType()`, `SNESNewtonTRSetQNType()`
M*/
PETSC_EXTERN PetscErrorCode SNESCreate_NEWTONTR(SNES snes)
{
  SNES_NEWTONTR *neP;

  PetscFunctionBegin;
  snes->ops->setup          = SNESSetUp_NEWTONTR;
  snes->ops->solve          = SNESSolve_NEWTONTR;
  snes->ops->reset          = SNESReset_NEWTONTR;
  snes->ops->destroy        = SNESDestroy_NEWTONTR;
  snes->ops->setfromoptions = SNESSetFromOptions_NEWTONTR;
  snes->ops->view           = SNESView_NEWTONTR;

  snes->stol    = 0.0;
  snes->usesksp = PETSC_TRUE;
  snes->npcside = PC_RIGHT;
  snes->usesnpc = PETSC_TRUE;

  snes->alwayscomputesfinalresidual = PETSC_TRUE;

  PetscCall(PetscNew(&neP));
  snes->data    = (void *)neP;
  neP->delta    = 0.0;
  neP->delta0   = 0.2;
  neP->eta1     = 0.001;
  neP->eta2     = 0.25;
  neP->eta3     = 0.75;
  neP->t1       = 0.25;
  neP->t2       = 2.0;
  neP->deltaM   = 1.e10;
  neP->norm     = NORM_2;
  neP->fallback = SNES_TR_FALLBACK_NEWTON;
  neP->kmdc     = 0.0; /* by default do not use sufficient decrease */
  PetscFunctionReturn(PETSC_SUCCESS);
}
