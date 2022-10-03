#include <petsc/private/taoimpl.h> /*I "petsctao.h" I*/

/*@
  TaoSetSolution - Sets the vector holding the initial guess for the solve

  Logically collective on tao

  Input Parameters:
+ tao - the Tao context
- x0  - the initial guess

  Level: beginner
.seealso: `Tao`, `TaoCreate()`, `TaoSolve()`, `TaoGetSolution()`
@*/
PetscErrorCode TaoSetSolution(Tao tao, Vec x0)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao, TAO_CLASSID, 1);
  if (x0) PetscValidHeaderSpecific(x0, VEC_CLASSID, 2);
  PetscCall(PetscObjectReference((PetscObject)x0));
  PetscCall(VecDestroy(&tao->solution));
  tao->solution = x0;
  PetscFunctionReturn(0);
}

PetscErrorCode TaoTestGradient(Tao tao, Vec x, Vec g1)
{
  Vec               g2, g3;
  PetscBool         complete_print = PETSC_FALSE, test = PETSC_FALSE;
  PetscReal         hcnorm, fdnorm, hcmax, fdmax, diffmax, diffnorm;
  PetscScalar       dot;
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
    PetscFunctionReturn(0);
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

  if (complete_print) {
    PetscCall(PetscViewerASCIIPrintf(viewer, "  Hand-coded gradient ----------\n"));
    PetscCall(VecView(g1, mviewer));
    PetscCall(PetscViewerASCIIPrintf(viewer, "  Finite difference gradient ----------\n"));
    PetscCall(VecView(g2, mviewer));
    PetscCall(PetscViewerASCIIPrintf(viewer, "  Hand-coded minus finite-difference gradient ----------\n"));
    PetscCall(VecView(g3, mviewer));
  }
  PetscCall(VecDestroy(&g2));
  PetscCall(VecDestroy(&g3));

  if (complete_print) {
    PetscCall(PetscViewerPopFormat(mviewer));
    PetscCall(PetscViewerDestroy(&mviewer));
  }
  PetscCall(PetscViewerASCIISetTab(viewer, tabs));
  PetscFunctionReturn(0);
}

/*@
  TaoComputeGradient - Computes the gradient of the objective function

  Collective on tao

  Input Parameters:
+ tao - the Tao context
- X - input vector

  Output Parameter:
. G - gradient vector

  Options Database Keys:
+    -tao_test_gradient - compare the user provided gradient with one compute via finite differences to check for errors
-    -tao_test_gradient_view - display the user provided gradient, the finite difference gradient and the difference between them to help users detect the location of errors in the user provided gradient

  Note:
    `TaoComputeGradient()` is typically used within the implementation of the optimization method,
  so most users would not generally call this routine themselves.

  Level: developer

.seealso: `TaoComputeObjective()`, `TaoComputeObjectiveAndGradient()`, `TaoSetGradient()`
@*/
PetscErrorCode TaoComputeGradient(Tao tao, Vec X, Vec G)
{
  PetscReal dummy;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao, TAO_CLASSID, 1);
  PetscValidHeaderSpecific(X, VEC_CLASSID, 2);
  PetscValidHeaderSpecific(G, VEC_CLASSID, 3);
  PetscCheckSameComm(tao, 1, X, 2);
  PetscCheckSameComm(tao, 1, G, 3);
  PetscCall(VecLockReadPush(X));
  if (tao->ops->computegradient) {
    PetscCall(PetscLogEventBegin(TAO_GradientEval, tao, X, G, NULL));
    PetscCallBack("Tao callback gradient", (*tao->ops->computegradient)(tao, X, G, tao->user_gradP));
    PetscCall(PetscLogEventEnd(TAO_GradientEval, tao, X, G, NULL));
    tao->ngrads++;
  } else if (tao->ops->computeobjectiveandgradient) {
    PetscCall(PetscLogEventBegin(TAO_ObjGradEval, tao, X, G, NULL));
    PetscCallBack("Tao callback objective/gradient", (*tao->ops->computeobjectiveandgradient)(tao, X, &dummy, G, tao->user_objgradP));
    PetscCall(PetscLogEventEnd(TAO_ObjGradEval, tao, X, G, NULL));
    tao->nfuncgrads++;
  } else SETERRQ(PetscObjectComm((PetscObject)tao), PETSC_ERR_ARG_WRONGSTATE, "TaoSetGradient() has not been called");
  PetscCall(VecLockReadPop(X));

  PetscCall(TaoTestGradient(tao, X, G));
  PetscFunctionReturn(0);
}

/*@
  TaoComputeObjective - Computes the objective function value at a given point

  Collective on tao

  Input Parameters:
+ tao - the Tao context
- X - input vector

  Output Parameter:
. f - Objective value at X

  Note:
    `TaoComputeObjective()` is typically used within the implementation of the optimization algorithm
  so most users would not generally call this routine themselves.

  Level: developer

.seealso: `Tao`, `TaoComputeGradient()`, `TaoComputeObjectiveAndGradient()`, `TaoSetObjective()`
@*/
PetscErrorCode TaoComputeObjective(Tao tao, Vec X, PetscReal *f)
{
  Vec temp;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao, TAO_CLASSID, 1);
  PetscValidHeaderSpecific(X, VEC_CLASSID, 2);
  PetscCheckSameComm(tao, 1, X, 2);
  PetscCall(VecLockReadPush(X));
  if (tao->ops->computeobjective) {
    PetscCall(PetscLogEventBegin(TAO_ObjectiveEval, tao, X, NULL, NULL));
    PetscCallBack("Tao callback objective", (*tao->ops->computeobjective)(tao, X, f, tao->user_objP));
    PetscCall(PetscLogEventEnd(TAO_ObjectiveEval, tao, X, NULL, NULL));
    tao->nfuncs++;
  } else if (tao->ops->computeobjectiveandgradient) {
    PetscCall(PetscInfo(tao, "Duplicating variable vector in order to call func/grad routine\n"));
    PetscCall(VecDuplicate(X, &temp));
    PetscCall(PetscLogEventBegin(TAO_ObjGradEval, tao, X, NULL, NULL));
    PetscCallBack("Tao callback objective/gradient", (*tao->ops->computeobjectiveandgradient)(tao, X, f, temp, tao->user_objgradP));
    PetscCall(PetscLogEventEnd(TAO_ObjGradEval, tao, X, NULL, NULL));
    PetscCall(VecDestroy(&temp));
    tao->nfuncgrads++;
  } else SETERRQ(PetscObjectComm((PetscObject)tao), PETSC_ERR_ARG_WRONGSTATE, "TaoSetObjective() has not been called");
  PetscCall(PetscInfo(tao, "TAO Function evaluation: %20.19e\n", (double)(*f)));
  PetscCall(VecLockReadPop(X));
  PetscFunctionReturn(0);
}

/*@
  TaoComputeObjectiveAndGradient - Computes the objective function value at a given point

  Collective on tao

  Input Parameters:
+ tao - the Tao context
- X - input vector

  Output Parameters:
+ f - Objective value at X
- g - Gradient vector at X

  Note:
    `TaoComputeObjectiveAndGradient()` is typically used within the implementation of the optimization algorithm,
  so most users would not generally call this routine themselves.

  Level: developer

.seealso: `TaoComputeGradient()`, `TaoComputeObjectiveAndGradient()`, `TaoSetObjective()`
@*/
PetscErrorCode TaoComputeObjectiveAndGradient(Tao tao, Vec X, PetscReal *f, Vec G)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao, TAO_CLASSID, 1);
  PetscValidHeaderSpecific(X, VEC_CLASSID, 2);
  PetscValidHeaderSpecific(G, VEC_CLASSID, 4);
  PetscCheckSameComm(tao, 1, X, 2);
  PetscCheckSameComm(tao, 1, G, 4);
  PetscCall(VecLockReadPush(X));
  if (tao->ops->computeobjectiveandgradient) {
    PetscCall(PetscLogEventBegin(TAO_ObjGradEval, tao, X, G, NULL));
    if (tao->ops->computegradient == TaoDefaultComputeGradient) {
      PetscCall(TaoComputeObjective(tao, X, f));
      PetscCall(TaoDefaultComputeGradient(tao, X, G, NULL));
    } else PetscCallBack("Tao callback objective/gradient", (*tao->ops->computeobjectiveandgradient)(tao, X, f, G, tao->user_objgradP));
    PetscCall(PetscLogEventEnd(TAO_ObjGradEval, tao, X, G, NULL));
    tao->nfuncgrads++;
  } else if (tao->ops->computeobjective && tao->ops->computegradient) {
    PetscCall(PetscLogEventBegin(TAO_ObjectiveEval, tao, X, NULL, NULL));
    PetscCallBack("Tao callback objective", (*tao->ops->computeobjective)(tao, X, f, tao->user_objP));
    PetscCall(PetscLogEventEnd(TAO_ObjectiveEval, tao, X, NULL, NULL));
    tao->nfuncs++;
    PetscCall(PetscLogEventBegin(TAO_GradientEval, tao, X, G, NULL));
    PetscCallBack("Tao callback gradient", (*tao->ops->computegradient)(tao, X, G, tao->user_gradP));
    PetscCall(PetscLogEventEnd(TAO_GradientEval, tao, X, G, NULL));
    tao->ngrads++;
  } else SETERRQ(PetscObjectComm((PetscObject)tao), PETSC_ERR_ARG_WRONGSTATE, "TaoSetObjective() or TaoSetGradient() not set");
  PetscCall(PetscInfo(tao, "TAO Function evaluation: %20.19e\n", (double)(*f)));
  PetscCall(VecLockReadPop(X));

  PetscCall(TaoTestGradient(tao, X, G));
  PetscFunctionReturn(0);
}

/*@C
  TaoSetObjective - Sets the function evaluation routine for minimization

  Logically collective on tao

  Input Parameters:
+ tao - the Tao context
. func - the objective function
- ctx - [optional] user-defined context for private data for the function evaluation
        routine (may be NULL)

  Calling sequence of func:
$      func (Tao tao, Vec x, PetscReal *f, void *ctx);

+ x - input vector
. f - function value
- ctx - [optional] user-defined function context

  Level: beginner

.seealso: `TaoSetGradient()`, `TaoSetHessian()`, `TaoSetObjectiveAndGradient()`, `TaoGetObjective()`
@*/
PetscErrorCode TaoSetObjective(Tao tao, PetscErrorCode (*func)(Tao, Vec, PetscReal *, void *), void *ctx)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao, TAO_CLASSID, 1);
  if (ctx) tao->user_objP = ctx;
  if (func) tao->ops->computeobjective = func;
  PetscFunctionReturn(0);
}

/*@C
  TaoGetObjective - Gets the function evaluation routine for the function to be minimized

  Not collective

  Input Parameter:
. tao - the Tao context

  Output Parameters
+ func - the objective function
- ctx - the user-defined context for private data for the function evaluation

  Calling sequence of func:
$      func (Tao tao, Vec x, PetscReal *f, void *ctx);

+ x - input vector
. f - function value
- ctx - [optional] user-defined function context

  Level: beginner

.seealso: `Tao`, `TaoSetGradient()`, `TaoSetHessian()`, `TaoSetObjective()`
@*/
PetscErrorCode TaoGetObjective(Tao tao, PetscErrorCode (**func)(Tao, Vec, PetscReal *, void *), void **ctx)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao, TAO_CLASSID, 1);
  if (func) *func = tao->ops->computeobjective;
  if (ctx) *ctx = tao->user_objP;
  PetscFunctionReturn(0);
}

/*@C
  TaoSetResidualRoutine - Sets the residual evaluation routine for least-square applications

  Logically collective on tao

  Input Parameters:
+ tao - the Tao context
. func - the residual evaluation routine
- ctx - [optional] user-defined context for private data for the function evaluation
        routine (may be NULL)

  Calling sequence of func:
$      func (Tao tao, Vec x, Vec f, void *ctx);

+ x - input vector
. f - function value vector
- ctx - [optional] user-defined function context

  Level: beginner

.seealso: `Tao`, `TaoSetObjective()`, `TaoSetJacobianRoutine()`
@*/
PetscErrorCode TaoSetResidualRoutine(Tao tao, Vec res, PetscErrorCode (*func)(Tao, Vec, Vec, void *), void *ctx)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao, TAO_CLASSID, 1);
  PetscValidHeaderSpecific(res, VEC_CLASSID, 2);
  PetscCall(PetscObjectReference((PetscObject)res));
  if (tao->ls_res) PetscCall(VecDestroy(&tao->ls_res));
  tao->ls_res               = res;
  tao->user_lsresP          = ctx;
  tao->ops->computeresidual = func;

  PetscFunctionReturn(0);
}

/*@
  TaoSetResidualWeights - Give weights for the residual values. A vector can be used if only diagonal terms are used, otherwise a matrix can be give.
   If this function is not provided, or if sigma_v and sigma_w are both NULL, then the identity matrix will be used for weights.

  Collective on tao

  Input Parameters:
+ tao - the Tao context
. sigma_v - vector of weights (diagonal terms only)
. n       - the number of weights (if using off-diagonal)
. rows    - index list of rows for sigma_w
. cols    - index list of columns for sigma_w
- vals - array of weights

  Note: Either sigma_v or sigma_w (or both) should be NULL

  Level: intermediate

.seealso: `Tao`, `TaoSetResidualRoutine()`
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
  PetscFunctionReturn(0);
}

/*@
  TaoComputeResidual - Computes a least-squares residual vector at a given point

  Collective on tao

  Input Parameters:
+ tao - the Tao context
- X - input vector

  Output Parameter:
. f - Objective vector at X

  Notes:
    `TaoComputeResidual()` is typically used within the implementation of the optimization algorithm,
  so most users would not generally call this routine themselves.

  Level: advanced

.seealso: `Tao`, `TaoSetResidualRoutine()`
@*/
PetscErrorCode TaoComputeResidual(Tao tao, Vec X, Vec F)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao, TAO_CLASSID, 1);
  PetscValidHeaderSpecific(X, VEC_CLASSID, 2);
  PetscValidHeaderSpecific(F, VEC_CLASSID, 3);
  PetscCheckSameComm(tao, 1, X, 2);
  PetscCheckSameComm(tao, 1, F, 3);
  if (tao->ops->computeresidual) {
    PetscCall(PetscLogEventBegin(TAO_ObjectiveEval, tao, X, NULL, NULL));
    PetscCallBack("Tao callback least-squares residual", (*tao->ops->computeresidual)(tao, X, F, tao->user_lsresP));
    PetscCall(PetscLogEventEnd(TAO_ObjectiveEval, tao, X, NULL, NULL));
    tao->nfuncs++;
  } else SETERRQ(PetscObjectComm((PetscObject)tao), PETSC_ERR_ARG_WRONGSTATE, "TaoSetResidualRoutine() has not been called");
  PetscCall(PetscInfo(tao, "TAO least-squares residual evaluation.\n"));
  PetscFunctionReturn(0);
}

/*@C
  TaoSetGradient - Sets the gradient evaluation routine for the function to be optimized

  Logically collective on tao

  Input Parameters:
+ tao - the Tao context
. g - [optional] the vector to internally hold the gradient computation
. func - the gradient function
- ctx - [optional] user-defined context for private data for the gradient evaluation
        routine (may be NULL)

  Calling sequence of func:
$      func (Tao tao, Vec x, Vec g, void *ctx);

+ x - input vector
. g - gradient value (output)
- ctx - [optional] user-defined function context

  Level: beginner

.seealso: `Tao`, `TaoSolve()`, `TaoSetObjective()`, `TaoSetHessian()`, `TaoSetObjectiveAndGradient()`, `TaoGetGradient()`
@*/
PetscErrorCode TaoSetGradient(Tao tao, Vec g, PetscErrorCode (*func)(Tao, Vec, Vec, void *), void *ctx)
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
  if (func) tao->ops->computegradient = func;
  if (ctx) tao->user_gradP = ctx;
  PetscFunctionReturn(0);
}

/*@C
  TaoGetGradient - Gets the gradient evaluation routine for the function being optimized

  Not collective

  Input Parameter:
. tao - the Tao context

  Output Parameters:
+ g - the vector to internally hold the gradient computation
. func - the gradient function
- ctx - user-defined context for private data for the gradient evaluation routine

  Calling sequence of func:
$      func (Tao tao, Vec x, Vec g, void *ctx);

+ x - input vector
. g - gradient value (output)
- ctx - [optional] user-defined function context

  Level: beginner

.seealso: `Tao`, `TaoSetObjective()`, `TaoSetHessian()`, `TaoSetObjectiveAndGradient()`, `TaoSetGradient()`
@*/
PetscErrorCode TaoGetGradient(Tao tao, Vec *g, PetscErrorCode (**func)(Tao, Vec, Vec, void *), void **ctx)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao, TAO_CLASSID, 1);
  if (g) *g = tao->gradient;
  if (func) *func = tao->ops->computegradient;
  if (ctx) *ctx = tao->user_gradP;
  PetscFunctionReturn(0);
}

/*@C
  TaoSetObjectiveAndGradient - Sets a combined objective function and gradient evaluation routine for the function to be optimized

  Logically collective on tao

  Input Parameters:
+ tao - the Tao context
. g - [optional] the vector to internally hold the gradient computation
. func - the gradient function
- ctx - [optional] user-defined context for private data for the gradient evaluation
        routine (may be NULL)

  Calling sequence of func:
$      func (Tao tao, Vec x, PetscReal *f, Vec g, void *ctx);

+ x - input vector
. f - objective value (output)
. g - gradient value (output)
- ctx - [optional] user-defined function context

  Level: beginner

  Note:
  For some optimization methods using a combined function can be more eifficient.

.seealso: `Tao`, `TaoSolve()`, `TaoSetObjective()`, `TaoSetHessian()`, `TaoSetGradient()`, `TaoGetObjectiveAndGradient()`
@*/
PetscErrorCode TaoSetObjectiveAndGradient(Tao tao, Vec g, PetscErrorCode (*func)(Tao, Vec, PetscReal *, Vec, void *), void *ctx)
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
  if (ctx) tao->user_objgradP = ctx;
  if (func) tao->ops->computeobjectiveandgradient = func;
  PetscFunctionReturn(0);
}

/*@C
  TaoGetObjectiveAndGradient - Gets the combined objective function and gradient evaluation routine for the function to be optimized

  Not collective

  Input Parameter:
. tao - the Tao context

  Output Parameters:
+ g - the vector to internally hold the gradient computation
. func - the gradient function
- ctx - user-defined context for private data for the gradient evaluation routine

  Calling sequence of func:
$      func (Tao tao, Vec x, PetscReal *f, Vec g, void *ctx);

+ x - input vector
. f - objective value (output)
. g - gradient value (output)
- ctx - [optional] user-defined function context

  Level: beginner

.seealso: `Tao`, `TaoSolve()`, `TaoSetObjective()`, `TaoSetGradient()`, `TaoSetHessian()`, `TaoSetObjectiveAndGradient()`
@*/
PetscErrorCode TaoGetObjectiveAndGradient(Tao tao, Vec *g, PetscErrorCode (**func)(Tao, Vec, PetscReal *, Vec, void *), void **ctx)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao, TAO_CLASSID, 1);
  if (g) *g = tao->gradient;
  if (func) *func = tao->ops->computeobjectiveandgradient;
  if (ctx) *ctx = tao->user_objgradP;
  PetscFunctionReturn(0);
}

/*@
  TaoIsObjectiveDefined - Checks to see if the user has
  declared an objective-only routine.  Useful for determining when
  it is appropriate to call `TaoComputeObjective()` or
  `TaoComputeObjectiveAndGradient()`

  Not collective

  Input Parameter:
. tao - the Tao context

  Output Parameter:
. flg - `PETSC_TRUE` if function routine is set by user, `PETSC_FALSE` otherwise

  Level: developer

.seealso: `Tao`, `TaoSetObjective()`, `TaoIsGradientDefined()`, `TaoIsObjectiveAndGradientDefined()`
@*/
PetscErrorCode TaoIsObjectiveDefined(Tao tao, PetscBool *flg)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao, TAO_CLASSID, 1);
  if (tao->ops->computeobjective == NULL) *flg = PETSC_FALSE;
  else *flg = PETSC_TRUE;
  PetscFunctionReturn(0);
}

/*@
  TaoIsGradientDefined - Checks to see if the user has
  declared an objective-only routine.  Useful for determining when
  it is appropriate to call `TaoComputeGradient()` or
  `TaoComputeGradientAndGradient()`

  Not Collective

  Input Parameter:
. tao - the Tao context

  Output Parameter:
. flg - `PETSC_TRUE` if function routine is set by user, `PETSC_FALSE` otherwise

  Level: developer

.seealso: `TaoSetGradient()`, `TaoIsObjectiveDefined()`, `TaoIsObjectiveAndGradientDefined()`
@*/
PetscErrorCode TaoIsGradientDefined(Tao tao, PetscBool *flg)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao, TAO_CLASSID, 1);
  if (tao->ops->computegradient == NULL) *flg = PETSC_FALSE;
  else *flg = PETSC_TRUE;
  PetscFunctionReturn(0);
}

/*@
  TaoIsObjectiveAndGradientDefined - Checks to see if the user has
  declared a joint objective/gradient routine.  Useful for determining when
  it is appropriate to call `TaoComputeObjective()` or
  `TaoComputeObjectiveAndGradient()`

  Not Collective

  Input Parameter:
. tao - the Tao context

  Output Parameter:
. flg - `PETSC_TRUE` if function routine is set by user, `PETSC_FALSE` otherwise

  Level: developer

.seealso: `TaoSetObjectiveAndGradient()`, `TaoIsObjectiveDefined()`, `TaoIsGradientDefined()`
@*/
PetscErrorCode TaoIsObjectiveAndGradientDefined(Tao tao, PetscBool *flg)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao, TAO_CLASSID, 1);
  if (tao->ops->computeobjectiveandgradient == NULL) *flg = PETSC_FALSE;
  else *flg = PETSC_TRUE;
  PetscFunctionReturn(0);
}
