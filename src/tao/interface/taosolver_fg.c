#include <petsc/private/taoimpl.h> /*I "petsctao.h" I*/

/*@
  TaoSetSolution - Sets the vector holding the initial guess for the solve

  Logically Collective

  Input Parameters:
+ tao - the `Tao` context
- x0  - the initial guess

  Level: beginner

.seealso: [](ch_tao), `Tao`, `TaoCreate()`, `TaoSolve()`, `TaoGetSolution()`
@*/
PetscErrorCode TaoSetSolution(Tao tao, Vec x0)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao, TAO_CLASSID, 1);
  if (x0) PetscValidHeaderSpecific(x0, VEC_CLASSID, 2);
  PetscCall(PetscObjectReference((PetscObject)x0));
  PetscCall(VecDestroy(&tao->solution));
  tao->solution = x0;
  if (x0) PetscCall(TaoTermSetSolutionTemplate(tao->callbacks, x0));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode TaoTestGradient_Internal(Tao tao, Vec x, Vec g1, PetscViewer viewer, PetscViewer mviewer)
{
  Vec         g2, g3;
  PetscReal   hcnorm, fdnorm, hcmax, fdmax, diffmax, diffnorm;
  PetscScalar dot;

  PetscFunctionBegin;
  PetscCall(VecDuplicate(x, &g2));
  PetscCall(VecDuplicate(x, &g3));

  /* Compute finite difference gradient, assume the gradient is already computed by TaoComputeGradient() and put into g1 */
  PetscCall(TaoDefaultComputeGradient(tao, x, g2, NULL));

  PetscCall(VecNorm(g2, NORM_2, &fdnorm));
  PetscCall(VecNorm(g1, NORM_2, &hcnorm));
  PetscCall(VecNorm(g2, NORM_INFINITY, &fdmax));
  PetscCall(VecNorm(g1, NORM_INFINITY, &hcmax));
  PetscCall(VecDot(g1, g2, &dot));
  PetscCall(VecCopy(g1, g3));
  PetscCall(VecAXPY(g3, -1.0, g2));
  PetscCall(VecNorm(g3, NORM_2, &diffnorm));
  PetscCall(VecNorm(g3, NORM_INFINITY, &diffmax));
  PetscCall(PetscViewerASCIIPrintf(viewer, "  ||Gfd|| %g, ||G|| = %g, angle cosine = (Gfd'G)/||Gfd||||G|| = %g\n", (double)fdnorm, (double)hcnorm, (double)(PetscRealPart(dot) / (fdnorm * hcnorm))));
  PetscCall(PetscViewerASCIIPrintf(viewer, "  2-norm ||G - Gfd||/||G|| = %g, ||G - Gfd|| = %g\n", (double)(diffnorm / PetscMax(hcnorm, fdnorm)), (double)diffnorm));
  PetscCall(PetscViewerASCIIPrintf(viewer, "  max-norm ||G - Gfd||/||G|| = %g, ||G - Gfd|| = %g\n", (double)(diffmax / PetscMax(hcmax, fdmax)), (double)diffmax));

  if (mviewer) {
    PetscCall(PetscViewerASCIIPrintf(viewer, "  Hand-coded gradient ----------\n"));
    PetscCall(VecView(g1, mviewer));
    PetscCall(PetscViewerASCIIPrintf(viewer, "  Finite difference gradient ----------\n"));
    PetscCall(VecView(g2, mviewer));
    PetscCall(PetscViewerASCIIPrintf(viewer, "  Hand-coded minus finite-difference gradient ----------\n"));
    PetscCall(VecView(g3, mviewer));
  }
  PetscCall(VecDestroy(&g2));
  PetscCall(VecDestroy(&g3));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode TaoTestGradient(Tao tao, Vec x, Vec g1)
{
  PetscBool         complete_print = PETSC_FALSE, test = PETSC_FALSE;
  MPI_Comm          comm;
  PetscViewer       viewer, mviewer;
  PetscViewerFormat format;
  PetscInt          tabs;
  static PetscBool  directionsprinted = PETSC_FALSE;

  PetscFunctionBegin;
  PetscObjectOptionsBegin((PetscObject)tao);
  PetscCall(PetscOptionsName("-tao_test_gradient", "Compare hand-coded and finite difference Gradients", "None", &test));
  PetscCall(PetscOptionsViewer("-tao_test_gradient_view", "View difference between hand-coded and finite difference Gradients element entries", "None", &mviewer, &format, &complete_print));
  PetscOptionsEnd();
  if (!test) {
    if (complete_print) PetscCall(PetscViewerDestroy(&mviewer));
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  PetscCall(PetscObjectGetComm((PetscObject)tao, &comm));
  PetscCall(PetscViewerASCIIGetStdout(comm, &viewer));
  PetscCall(PetscViewerASCIIGetTab(viewer, &tabs));
  PetscCall(PetscViewerASCIISetTab(viewer, ((PetscObject)tao)->tablevel));
  PetscCall(PetscViewerASCIIPrintf(viewer, "  ---------- Testing Gradient -------------\n"));
  if (!complete_print && !directionsprinted) {
    PetscCall(PetscViewerASCIIPrintf(viewer, "  Run with -tao_test_gradient_view and optionally -tao_test_gradient <threshold> to show difference\n"));
    PetscCall(PetscViewerASCIIPrintf(viewer, "    of hand-coded and finite difference gradient entries greater than <threshold>.\n"));
  }
  if (!directionsprinted) {
    PetscCall(PetscViewerASCIIPrintf(viewer, "  Testing hand-coded Gradient, if (for double precision runs) ||G - Gfd||/||G|| is\n"));
    PetscCall(PetscViewerASCIIPrintf(viewer, "    O(1.e-8), the hand-coded Gradient is probably correct.\n"));
    directionsprinted = PETSC_TRUE;
  }
  if (complete_print) PetscCall(PetscViewerPushFormat(mviewer, format));
  PetscCall(TaoTestGradient_Internal(tao, x, g1, viewer, complete_print ? mviewer : NULL));
  if (complete_print) {
    PetscCall(PetscViewerPopFormat(mviewer));
    PetscCall(PetscViewerDestroy(&mviewer));
  }
  PetscCall(PetscViewerASCIISetTab(viewer, tabs));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TaoComputeGradient - Computes the gradient of the objective function

  Collective

  Input Parameters:
+ tao - the `Tao` context
- X   - input vector

  Output Parameter:
. G - gradient vector

  Options Database Keys:
+ -tao_test_gradient      - compare the user provided gradient with one compute via finite differences to check for errors
- -tao_test_gradient_view - display the user provided gradient, the finite difference gradient and the difference between them to help users detect the location of errors in the user provided gradient

  Level: developer

  Note:
  `TaoComputeGradient()` is typically used within the implementation of the optimization method,
  so most users would not generally call this routine themselves.

.seealso: [](ch_tao), `TaoComputeObjective()`, `TaoComputeObjectiveAndGradient()`, `TaoSetGradient()`
@*/
PetscErrorCode TaoComputeGradient(Tao tao, Vec X, Vec G)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao, TAO_CLASSID, 1);
  PetscValidHeaderSpecific(X, VEC_CLASSID, 2);
  PetscValidHeaderSpecific(G, VEC_CLASSID, 3);
  PetscCheckSameComm(tao, 1, X, 2);
  PetscCheckSameComm(tao, 1, G, 3);
  PetscCall(TaoTermMappingComputeGradient(&tao->objective_term, X, tao->objective_parameters, INSERT_VALUES, G));
  tao->ngrads++;
  PetscCall(TaoTestGradient(tao, X, G));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TaoComputeObjective - Computes the objective function value at a given point

  Collective

  Input Parameters:
+ tao - the `Tao` context
- X   - input vector

  Output Parameter:
. f - Objective value at X

  Level: developer

  Note:
  `TaoComputeObjective()` is typically used within the implementation of the optimization algorithm
  so most users would not generally call this routine themselves.

.seealso: [](ch_tao), `Tao`, `TaoComputeGradient()`, `TaoComputeObjectiveAndGradient()`, `TaoSetObjective()`
@*/
PetscErrorCode TaoComputeObjective(Tao tao, Vec X, PetscReal *f)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao, TAO_CLASSID, 1);
  PetscValidHeaderSpecific(X, VEC_CLASSID, 2);
  PetscAssertPointer(f, 3);
  PetscCheckSameComm(tao, 1, X, 2);
  PetscCall(TaoTermMappingComputeObjective(&tao->objective_term, X, tao->objective_parameters, INSERT_VALUES, f));
  tao->nfuncs++;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TaoComputeObjectiveAndGradient - Computes the objective function value at a given point

  Collective

  Input Parameters:
+ tao - the `Tao` context
- X   - input vector

  Output Parameters:
+ f - Objective value at `X`
- G - Gradient vector at `X`

  Level: developer

  Note:
  `TaoComputeObjectiveAndGradient()` is typically used within the implementation of the optimization algorithm,
  so most users would not generally call this routine themselves.

.seealso: [](ch_tao), `TaoComputeGradient()`, `TaoSetObjective()`
@*/
PetscErrorCode TaoComputeObjectiveAndGradient(Tao tao, Vec X, PetscReal *f, Vec G)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao, TAO_CLASSID, 1);
  PetscValidHeaderSpecific(X, VEC_CLASSID, 2);
  PetscAssertPointer(f, 3);
  PetscValidHeaderSpecific(G, VEC_CLASSID, 4);
  PetscCheckSameComm(tao, 1, X, 2);
  PetscCheckSameComm(tao, 1, G, 4);
  PetscCall(TaoTermMappingComputeObjectiveAndGradient(&tao->objective_term, X, tao->objective_parameters, INSERT_VALUES, f, G));
  tao->nfuncgrads++;
  PetscCall(TaoTestGradient(tao, X, G));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  TaoSetObjective - Sets the function evaluation routine for minimization

  Logically Collective

  Input Parameters:
+ tao  - the `Tao` context
. func - the objective function
- ctx  - [optional] user-defined context for private data for the function evaluation
        routine (may be `NULL`)

  Calling sequence of `func`:
+ tao - the optimizer
. x   - input vector
. f   - function value
- ctx - [optional] user-defined function context

  Level: beginner

.seealso: [](ch_tao), `TaoSetGradient()`, `TaoSetHessian()`, `TaoSetObjectiveAndGradient()`, `TaoGetObjective()`
@*/
PetscErrorCode TaoSetObjective(Tao tao, PetscErrorCode (*func)(Tao tao, Vec x, PetscReal *f, PetscCtx ctx), PetscCtx ctx)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao, TAO_CLASSID, 1);
  PetscCall(TaoTermCallbacksSetObjective(tao->callbacks, func, ctx));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  TaoGetObjective - Gets the function evaluation routine for the function to be minimized

  Not Collective

  Input Parameter:
. tao - the `Tao` context

  Output Parameters:
+ func - the objective function
- ctx  - the user-defined context for private data for the function evaluation

  Calling sequence of `func`:
+ tao - the optimizer
. x   - input vector
. f   - function value
- ctx - [optional] user-defined function context

  Level: beginner

  Notes:
  In addition to specifying an objective function using callbacks such as
  `TaoSetObjective()` and `TaoSetGradient()`, users can specify
  objective functions with `TaoAddTerm()`.

  `TaoGetObjective()` will always return the callback specified with
  `TaoSetObjective()`, even if the objective function has been changed by
  calling `TaoAddTerm()`.

.seealso: [](ch_tao), `Tao`, `TaoSetGradient()`, `TaoSetHessian()`, `TaoSetObjective()`
@*/
PetscErrorCode TaoGetObjective(Tao tao, PetscErrorCode (**func)(Tao tao, Vec x, PetscReal *f, PetscCtx ctx), PetscCtxRt ctx)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao, TAO_CLASSID, 1);
  if (func || ctx) PetscCall(TaoTermCallbacksGetObjective(tao->callbacks, func, ctx));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  TaoSetResidualRoutine - Sets the residual evaluation routine for least-square applications

  Logically Collective

  Input Parameters:
+ tao  - the `Tao` context
. res  - the residual vector
. func - the residual evaluation routine
- ctx  - [optional] user-defined context for private data for the function evaluation
         routine (may be `NULL`)

  Calling sequence of `func`:
+ tao - the optimizer
. x   - input vector
. res - function value vector
- ctx - [optional] user-defined function context

  Level: beginner

.seealso: [](ch_tao), `Tao`, `TaoSetObjective()`, `TaoSetJacobianRoutine()`
@*/
PetscErrorCode TaoSetResidualRoutine(Tao tao, Vec res, PetscErrorCode (*func)(Tao tao, Vec x, Vec res, PetscCtx ctx), PetscCtx ctx)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao, TAO_CLASSID, 1);
  PetscValidHeaderSpecific(res, VEC_CLASSID, 2);
  PetscCall(PetscObjectReference((PetscObject)res));
  if (tao->ls_res) PetscCall(VecDestroy(&tao->ls_res));
  tao->ls_res               = res;
  tao->user_lsresP          = ctx;
  tao->ops->computeresidual = func;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TaoSetResidualWeights - Give weights for the residual values. A vector can be used if only diagonal terms are used, otherwise a matrix can be give.

  Collective

  Input Parameters:
+ tao     - the `Tao` context
. sigma_v - vector of weights (diagonal terms only)
. n       - the number of weights (if using off-diagonal)
. rows    - index list of rows for `sigma_v`
. cols    - index list of columns for `sigma_v`
- vals    - array of weights

  Level: intermediate

  Notes:
  If this function is not provided, or if `sigma_v` and `vals` are both `NULL`, then the
  identity matrix will be used for weights.

  Either `sigma_v` or `vals` should be `NULL`

.seealso: [](ch_tao), `Tao`, `TaoSetResidualRoutine()`
@*/
PetscErrorCode TaoSetResidualWeights(Tao tao, Vec sigma_v, PetscInt n, PetscInt *rows, PetscInt *cols, PetscReal *vals)
{
  PetscInt i;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao, TAO_CLASSID, 1);
  if (sigma_v) PetscValidHeaderSpecific(sigma_v, VEC_CLASSID, 2);
  PetscCall(PetscObjectReference((PetscObject)sigma_v));
  PetscCall(VecDestroy(&tao->res_weights_v));
  tao->res_weights_v = sigma_v;
  if (vals) {
    PetscCall(PetscFree(tao->res_weights_rows));
    PetscCall(PetscFree(tao->res_weights_cols));
    PetscCall(PetscFree(tao->res_weights_w));
    PetscCall(PetscMalloc1(n, &tao->res_weights_rows));
    PetscCall(PetscMalloc1(n, &tao->res_weights_cols));
    PetscCall(PetscMalloc1(n, &tao->res_weights_w));
    tao->res_weights_n = n;
    for (i = 0; i < n; i++) {
      tao->res_weights_rows[i] = rows[i];
      tao->res_weights_cols[i] = cols[i];
      tao->res_weights_w[i]    = vals[i];
    }
  } else {
    tao->res_weights_n    = 0;
    tao->res_weights_rows = NULL;
    tao->res_weights_cols = NULL;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TaoComputeResidual - Computes a least-squares residual vector at a given point

  Collective

  Input Parameters:
+ tao - the `Tao` context
- X   - input vector

  Output Parameter:
. F - Objective vector at `X`

  Level: advanced

  Notes:
  `TaoComputeResidual()` is typically used within the implementation of the optimization algorithm,
  so most users would not generally call this routine themselves.

.seealso: [](ch_tao), `Tao`, `TaoSetResidualRoutine()`
@*/
PetscErrorCode TaoComputeResidual(Tao tao, Vec X, Vec F)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao, TAO_CLASSID, 1);
  PetscValidHeaderSpecific(X, VEC_CLASSID, 2);
  PetscValidHeaderSpecific(F, VEC_CLASSID, 3);
  PetscCheckSameComm(tao, 1, X, 2);
  PetscCheckSameComm(tao, 1, F, 3);
  PetscCheck(tao->ops->computeresidual, PetscObjectComm((PetscObject)tao), PETSC_ERR_ARG_WRONGSTATE, "TaoSetResidualRoutine() has not been called");
  PetscCall(PetscLogEventBegin(TAO_ResidualEval, tao, X, NULL, NULL));
  PetscCallBack("Tao callback least-squares residual", (*tao->ops->computeresidual)(tao, X, F, tao->user_lsresP));
  PetscCall(PetscLogEventEnd(TAO_ResidualEval, tao, X, NULL, NULL));
  tao->nfuncs++;
  PetscCall(PetscInfo(tao, "TAO least-squares residual evaluation.\n"));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  TaoSetGradient - Sets the gradient evaluation routine for the function to be optimized

  Logically Collective

  Input Parameters:
+ tao  - the `Tao` context
. g    - [optional] the vector to internally hold the gradient computation
. func - the gradient function
- ctx  - [optional] user-defined context for private data for the gradient evaluation
        routine (may be `NULL`)

  Calling sequence of `func`:
+ tao - the optimization solver
. x   - input vector
. g   - gradient value (output)
- ctx - [optional] user-defined function context

  Level: beginner

.seealso: [](ch_tao), `Tao`, `TaoSolve()`, `TaoSetObjective()`, `TaoSetHessian()`, `TaoSetObjectiveAndGradient()`, `TaoGetGradient()`
@*/
PetscErrorCode TaoSetGradient(Tao tao, Vec g, PetscErrorCode (*func)(Tao tao, Vec x, Vec g, PetscCtx ctx), PetscCtx ctx)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao, TAO_CLASSID, 1);
  if (g) {
    PetscValidHeaderSpecific(g, VEC_CLASSID, 2);
    PetscCheckSameComm(tao, 1, g, 2);
    PetscCall(PetscObjectReference((PetscObject)g));
    PetscCall(VecDestroy(&tao->gradient));
    tao->gradient = g;
  }
  PetscCall(TaoTermCallbacksSetGradient(tao->callbacks, func, ctx));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  TaoGetGradient - Gets the gradient evaluation routine for the function being optimized

  Not Collective

  Input Parameter:
. tao - the `Tao` context

  Output Parameters:
+ g    - the vector to internally hold the gradient computation
. func - the gradient function
- ctx  - user-defined context for private data for the gradient evaluation routine

  Calling sequence of `func`:
+ tao - the optimizer
. x   - input vector
. g   - gradient value (output)
- ctx - [optional] user-defined function context

  Level: beginner

  Notes:
  In addition to specifying an objective function using callbacks such as
  `TaoSetObjective()` and `TaoSetGradient()`, users can specify
  objective functions with `TaoAddTerm()`.

  `TaoGetGradient()` will always return the callback specified with
  `TaoSetGradient()`, even if the objective function has been changed by
  calling `TaoAddTerm()`.

.seealso: [](ch_tao), `Tao`, `TaoSetObjective()`, `TaoSetHessian()`, `TaoSetObjectiveAndGradient()`, `TaoSetGradient()`
@*/
PetscErrorCode TaoGetGradient(Tao tao, Vec *g, PetscErrorCode (**func)(Tao tao, Vec x, Vec g, PetscCtx ctx), PetscCtxRt ctx)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao, TAO_CLASSID, 1);
  if (g) *g = tao->gradient;
  if (func || ctx) PetscCall(TaoTermCallbacksGetGradient(tao->callbacks, func, ctx));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  TaoSetObjectiveAndGradient - Sets a combined objective function and gradient evaluation routine for the function to be optimized

  Logically Collective

  Input Parameters:
+ tao  - the `Tao` context
. g    - [optional] the vector to internally hold the gradient computation
. func - the gradient function
- ctx  - [optional] user-defined context for private data for the gradient evaluation
        routine (may be `NULL`)

  Calling sequence of `func`:
+ tao - the optimization object
. x   - input vector
. f   - objective value (output)
. g   - gradient value (output)
- ctx - [optional] user-defined function context

  Level: beginner

  Note:
  For some optimization methods using a combined function can be more efficient.

.seealso: [](ch_tao), `Tao`, `TaoSolve()`, `TaoSetObjective()`, `TaoSetHessian()`, `TaoSetGradient()`, `TaoGetObjectiveAndGradient()`
@*/
PetscErrorCode TaoSetObjectiveAndGradient(Tao tao, Vec g, PetscErrorCode (*func)(Tao tao, Vec x, PetscReal *f, Vec g, PetscCtx ctx), PetscCtx ctx)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao, TAO_CLASSID, 1);
  if (g) {
    PetscValidHeaderSpecific(g, VEC_CLASSID, 2);
    PetscCheckSameComm(tao, 1, g, 2);
    PetscCall(PetscObjectReference((PetscObject)g));
    PetscCall(VecDestroy(&tao->gradient));
    tao->gradient = g;
  }
  PetscCall(TaoTermCallbacksSetObjectiveAndGradient(tao->callbacks, func, ctx));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  TaoGetObjectiveAndGradient - Gets the combined objective function and gradient evaluation routine for the function to be optimized

  Not Collective

  Input Parameter:
. tao - the `Tao` context

  Output Parameters:
+ g    - the vector to internally hold the gradient computation
. func - the gradient function
- ctx  - user-defined context for private data for the gradient evaluation routine

  Calling sequence of `func`:
+ tao - the optimizer
. x   - input vector
. f   - objective value (output)
. g   - gradient value (output)
- ctx - [optional] user-defined function context

  Level: beginner

  Note:
  In addition to specifying an objective function using callbacks such as
  `TaoSetObjectiveAndGradient()`, users can specify
  objective functions with `TaoAddTerm()`.

  `TaoGetObjectiveAndGradient()` will always return the callback specified with
  `TaoSetObjectiveAndGradient()`, even if the objective function has been changed by
  calling `TaoAddTerm()`.

.seealso: [](ch_tao), `Tao`, `TaoSolve()`, `TaoSetObjective()`, `TaoSetGradient()`, `TaoSetHessian()`, `TaoSetObjectiveAndGradient()`
@*/
PetscErrorCode TaoGetObjectiveAndGradient(Tao tao, Vec *g, PetscErrorCode (**func)(Tao tao, Vec x, PetscReal *f, Vec g, PetscCtx ctx), PetscCtxRt ctx)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao, TAO_CLASSID, 1);
  if (g) *g = tao->gradient;
  if (func || ctx) PetscCall(TaoTermCallbacksGetObjectiveAndGradient(tao->callbacks, func, ctx));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TaoIsObjectiveDefined - Checks to see if the user has
  declared an objective-only routine.  Useful for determining when
  it is appropriate to call `TaoComputeObjective()` or
  `TaoComputeObjectiveAndGradient()`

  Not Collective

  Input Parameter:
. tao - the `Tao` context

  Output Parameter:
. flg - `PETSC_TRUE` if the `Tao` has this routine `PETSC_FALSE` otherwise

  Level: developer

  Note:
  If the objective of `Tao` has been altered via `TaoAddTerm()`, it will
  return whether the summation of all terms has this routine.

.seealso: [](ch_tao), `Tao`, `TaoSetObjective()`, `TaoIsGradientDefined()`, `TaoIsObjectiveAndGradientDefined()`
@*/
PetscErrorCode TaoIsObjectiveDefined(Tao tao, PetscBool *flg)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao, TAO_CLASSID, 1);
  PetscCall(TaoTermIsObjectiveDefined(tao->objective_term.term, flg));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TaoIsGradientDefined - Checks to see if the user has
  declared a gradient-only routine.  Useful for determining when
  it is appropriate to call `TaoComputeGradient()` or
  `TaoComputeObjectiveAndGradient()`

  Not Collective

  Input Parameter:
. tao - the `Tao` context

  Output Parameter:
. flg - `PETSC_TRUE` if the objective `TaoTerm` has this routine `PETSC_FALSE` otherwise

  Level: developer

  Note:
  If the objective of `Tao` has been altered via `TaoAddTerm()`, it will
  return whether the summation of all terms has this routine.

.seealso: [](ch_tao), `TaoSetGradient()`, `TaoIsObjectiveDefined()`, `TaoIsObjectiveAndGradientDefined()`
@*/
PetscErrorCode TaoIsGradientDefined(Tao tao, PetscBool *flg)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao, TAO_CLASSID, 1);
  PetscCall(TaoTermIsGradientDefined(tao->objective_term.term, flg));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TaoIsObjectiveAndGradientDefined - Checks to see if the user has
  declared a joint objective/gradient routine.  Useful for determining when
  it is appropriate to call `TaoComputeObjectiveAndGradient()`

  Not Collective

  Input Parameter:
. tao - the `Tao` context

  Output Parameter:
. flg - `PETSC_TRUE` if the objective `TaoTerm` has this routine `PETSC_FALSE` otherwise

  Level: developer

  Note:
  If the objective of `Tao` has been altered via `TaoAddTerm()`, it will
  return whether the summation of all terms has this routine.

.seealso: [](ch_tao), `TaoSetObjectiveAndGradient()`, `TaoIsObjectiveDefined()`, `TaoIsGradientDefined()`
@*/
PetscErrorCode TaoIsObjectiveAndGradientDefined(Tao tao, PetscBool *flg)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao, TAO_CLASSID, 1);
  PetscCall(TaoTermIsObjectiveAndGradientDefined(tao->objective_term.term, flg));
  PetscFunctionReturn(PETSC_SUCCESS);
}
