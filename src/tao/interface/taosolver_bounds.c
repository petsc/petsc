#include <petsc/private/taoimpl.h> /*I "petsctao.h" I*/

/*@
  TaoSetVariableBounds - Sets the upper and lower bounds for the optimization problem

  Logically Collective

  Input Parameters:
+ tao - the `Tao` context
. XL  - vector of lower bounds
- XU  - vector of upper bounds

  Level: beginner

.seealso: [](chapter_tao), `Tao`, `TaoSetObjective()`, `TaoSetHessian()`, `TaoSetObjectiveAndGradient()`, `TaoGetVariableBounds()`
@*/
PetscErrorCode TaoSetVariableBounds(Tao tao, Vec XL, Vec XU)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao, TAO_CLASSID, 1);
  if (XL) PetscValidHeaderSpecific(XL, VEC_CLASSID, 2);
  if (XU) PetscValidHeaderSpecific(XU, VEC_CLASSID, 3);
  PetscCall(PetscObjectReference((PetscObject)XL));
  PetscCall(PetscObjectReference((PetscObject)XU));
  PetscCall(VecDestroy(&tao->XL));
  PetscCall(VecDestroy(&tao->XU));
  tao->XL      = XL;
  tao->XU      = XU;
  tao->bounded = (PetscBool)(XL || XU);
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  TaoSetVariableBoundsRoutine - Sets a function to be used to compute lower and upper variable bounds for the optimization

  Logically Collective

  Input Parameters:
+ tao - the `Tao` context
. func - the bounds computation routine
- ctx - [optional] user-defined context for private data for the bounds computation (may be `NULL`)

  Calling sequence of `func`:
$ PetscErrorCode func (Tao tao, Vec xl, Vec xu, void *ctx);
+ tao - the `Tao` solver
. xl  - vector of lower bounds
. xu  - vector of upper bounds
- ctx - the (optional) user-defined function context

  Level: beginner

  Note:
  The func passed to `TaoSetVariableBoundsRoutine()` takes precedence over any values set in `TaoSetVariableBounds()`.

.seealso: [](chapter_tao), `Tao`, `TaoSetObjective()`, `TaoSetHessian()`, `TaoSetObjectiveAndGradient()`, `TaoSetVariableBounds()`
@*/
PetscErrorCode TaoSetVariableBoundsRoutine(Tao tao, PetscErrorCode (*func)(Tao, Vec, Vec, void *), void *ctx)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao, TAO_CLASSID, 1);
  tao->user_boundsP       = ctx;
  tao->ops->computebounds = func;
  tao->bounded            = func ? PETSC_TRUE : PETSC_FALSE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TaoGetVariableBounds - Gets the upper and lower bounds vectors set with `TaoSetVariableBounds()`

  Not Collective

  Input Parameter:
. tao - the `Tao` context

  Output Parameters:
+ XL  - vector of lower bounds
- XU  - vector of upper bounds

  Level: beginner

.seealso: [](chapter_tao), `Tao`, `TaoSetObjective()`, `TaoSetHessian()`, `TaoSetObjectiveAndGradient()`, `TaoSetVariableBounds()`
@*/
PetscErrorCode TaoGetVariableBounds(Tao tao, Vec *XL, Vec *XU)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao, TAO_CLASSID, 1);
  if (XL) *XL = tao->XL;
  if (XU) *XU = tao->XU;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
   TaoComputeVariableBounds - Compute the variable bounds using the
   routine set by `TaoSetVariableBoundsRoutine()`.

   Collective

   Input Parameter:
.  tao - the `Tao` context

   Level: developer

.seealso: [](chapter_tao), `Tao`, `TaoSetVariableBoundsRoutine()`, `TaoSetVariableBounds()`
@*/
PetscErrorCode TaoComputeVariableBounds(Tao tao)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao, TAO_CLASSID, 1);
  if (tao->ops->computebounds) {
    if (!tao->XL) {
      PetscCall(VecDuplicate(tao->solution, &tao->XL));
      PetscCall(VecSet(tao->XL, PETSC_NINFINITY));
    }
    if (!tao->XU) {
      PetscCall(VecDuplicate(tao->solution, &tao->XU));
      PetscCall(VecSet(tao->XU, PETSC_INFINITY));
    }
    PetscCallBack("Tao callback variable bounds", (*tao->ops->computebounds)(tao, tao->XL, tao->XU, tao->user_boundsP));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TaoSetInequalityBounds - Sets the upper and lower bounds

  Logically Collective

  Input Parameters:
+ tao - the `Tao` context
. IL  - vector of lower bounds
- IU  - vector of upper bounds

  Level: beginner

.seealso: [](chapter_tao), `Tao`, `TaoSetObjective()`, `TaoSetHessian()`, `TaoSetObjectiveAndGradient()`, `TaoGetInequalityBounds()`
@*/
PetscErrorCode TaoSetInequalityBounds(Tao tao, Vec IL, Vec IU)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao, TAO_CLASSID, 1);
  if (IL) PetscValidHeaderSpecific(IL, VEC_CLASSID, 2);
  if (IU) PetscValidHeaderSpecific(IU, VEC_CLASSID, 3);
  PetscCall(PetscObjectReference((PetscObject)IL));
  PetscCall(PetscObjectReference((PetscObject)IU));
  PetscCall(VecDestroy(&tao->IL));
  PetscCall(VecDestroy(&tao->IU));
  tao->IL               = IL;
  tao->IU               = IU;
  tao->ineq_doublesided = (PetscBool)(IL || IU);
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TaoGetInequalityBounds - Gets the upper and lower bounds set via `TaoSetInequalityBounds()`

  Logically Collective

  Input Parameter:
. tao - the `Tao` context

  Output Parameters:
+ IL  - vector of lower bounds
- IU  - vector of upper bounds

  Level: beginner

.seealso: [](chapter_tao), `TaoSetObjective()`, `TaoSetHessian()`, `TaoSetObjectiveAndGradient()`, `TaoSetInequalityBounds()`
@*/
PetscErrorCode TaoGetInequalityBounds(Tao tao, Vec *IL, Vec *IU)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao, TAO_CLASSID, 1);
  if (IL) *IL = tao->IL;
  if (IU) *IU = tao->IU;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
   TaoComputeConstraints - Compute the variable bounds using the
   routine set by `TaoSetConstraintsRoutine()`.

   Collective

   Input Parameters:
+  tao - the `Tao` context
-  X - location to evaluate the constraints

   Output Parameter:
.  C - the constraints

   Level: developer

.seealso: [](chapter_tao), `Tao`, `TaoSetConstraintsRoutine()`, `TaoComputeJacobian()`
@*/
PetscErrorCode TaoComputeConstraints(Tao tao, Vec X, Vec C)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao, TAO_CLASSID, 1);
  PetscValidHeaderSpecific(X, VEC_CLASSID, 2);
  PetscValidHeaderSpecific(C, VEC_CLASSID, 3);
  PetscCheckSameComm(tao, 1, X, 2);
  PetscCheckSameComm(tao, 1, C, 3);
  PetscCall(PetscLogEventBegin(TAO_ConstraintsEval, tao, X, C, NULL));
  PetscCallBack("Tao callback constraints", (*tao->ops->computeconstraints)(tao, X, C, tao->user_conP));
  PetscCall(PetscLogEventEnd(TAO_ConstraintsEval, tao, X, C, NULL));
  tao->nconstraints++;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  TaoSetConstraintsRoutine - Sets a function to be used to compute constraints.  Tao only handles constraints under certain conditions, see [](chapter_tao) for details

  Logically Collective

  Input Parameters:
+ tao - the `Tao` context
. c   - A vector that will be used to store constraint evaluation
. func - the bounds computation routine
- ctx - [optional] user-defined context for private data for the constraints computation (may be `NULL`)

  Calling sequence of `func`:
$ PetscErrorCode func(Tao tao, Vec x, Vec c, void *ctx);
+ tao - the `Tao` solver
. x   - point to evaluate constraints
. c   - vector constraints evaluated at `x`
- ctx - the (optional) user-defined function context

  Level: intermediate

.seealso: [](chapter_tao), `Tao`, `TaoSetObjective()`, `TaoSetHessian()`, `TaoSetObjectiveAndGradient()`, `TaoSetVariablevBounds()`
@*/
PetscErrorCode TaoSetConstraintsRoutine(Tao tao, Vec c, PetscErrorCode (*func)(Tao, Vec, Vec, void *), void *ctx)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao, TAO_CLASSID, 1);
  if (c) PetscValidHeaderSpecific(c, VEC_CLASSID, 2);
  PetscCall(PetscObjectReference((PetscObject)c));
  PetscCall(VecDestroy(&tao->constraints));
  tao->constrained             = func ? PETSC_TRUE : PETSC_FALSE;
  tao->constraints             = c;
  tao->user_conP               = ctx;
  tao->ops->computeconstraints = func;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TaoComputeDualVariables - Computes the dual vectors corresponding to the bounds
  of the variables

  Collective

  Input Parameter:
. tao - the `Tao` context

  Output Parameters:
+ DL - dual variable vector for the lower bounds
- DU - dual variable vector for the upper bounds

  Level: advanced

  Note:
  DL and DU should be created before calling this routine.  If calling
  this routine after using an unconstrained solver, `DL` and `DU` are set to all
  zeros.

.seealso: [](chapter_tao), `Tao`, `TaoComputeObjective()`, `TaoSetVariableBounds()`
@*/
PetscErrorCode TaoComputeDualVariables(Tao tao, Vec DL, Vec DU)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao, TAO_CLASSID, 1);
  PetscValidHeaderSpecific(DL, VEC_CLASSID, 2);
  PetscValidHeaderSpecific(DU, VEC_CLASSID, 3);
  PetscCheckSameComm(tao, 1, DL, 2);
  PetscCheckSameComm(tao, 1, DU, 3);
  if (tao->ops->computedual) {
    PetscUseTypeMethod(tao, computedual, DL, DU);
  } else {
    PetscCall(VecSet(DL, 0.0));
    PetscCall(VecSet(DU, 0.0));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TaoGetDualVariables - Gets the dual vectors

  Collective

  Input Parameter:
. tao - the `Tao` context

  Output Parameters:
+ DE - dual variable vector for the lower bounds
- DI - dual variable vector for the upper bounds

  Level: advanced

.seealso: [](chapter_tao), `Tao`, `TaoComputeDualVariables()`
@*/
PetscErrorCode TaoGetDualVariables(Tao tao, Vec *DE, Vec *DI)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao, TAO_CLASSID, 1);
  if (DE) *DE = tao->DE;
  if (DI) *DI = tao->DI;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  TaoSetEqualityConstraintsRoutine - Sets a function to be used to compute constraints.  Tao only handles constraints under certain conditions, see [](chapter_tao) for details

  Logically Collective

  Input Parameters:
+ tao - the `Tao` context
. ce   - A vector that will be used to store equality constraint evaluation
. func - the bounds computation routine
- ctx - [optional] user-defined context for private data for the equality constraints computation (may be `NULL`)

  Calling sequence of `func`:
$ PetscErrorCode func(Tao tao, Vec x, Vec ce, void *ctx);
+ tao - the `Tao` solver
. x   - point to evaluate equality constraints
. ce   - vector of equality constraints evaluated at x
- ctx - the (optional) user-defined function context

  Level: intermediate

.seealso: [](chapter_tao), `Tao`, `TaoSetObjective()`, `TaoSetHessian()`, `TaoSetObjectiveAndGradient()`, `TaoSetVariableBounds()`
@*/
PetscErrorCode TaoSetEqualityConstraintsRoutine(Tao tao, Vec ce, PetscErrorCode (*func)(Tao, Vec, Vec, void *), void *ctx)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao, TAO_CLASSID, 1);
  if (ce) PetscValidHeaderSpecific(ce, VEC_CLASSID, 2);
  PetscCall(PetscObjectReference((PetscObject)ce));
  PetscCall(VecDestroy(&tao->constraints_equality));
  tao->eq_constrained                  = func ? PETSC_TRUE : PETSC_FALSE;
  tao->constraints_equality            = ce;
  tao->user_con_equalityP              = ctx;
  tao->ops->computeequalityconstraints = func;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  TaoSetInequalityConstraintsRoutine - Sets a function to be used to compute constraints.  Tao only handles constraints under certain conditions, see [](chapter_tao) for details

  Logically Collective

  Input Parameters:
+ tao - the `Tao` context
. ci   - A vector that will be used to store inequality constraint evaluation
. func - the bounds computation routine
- ctx - [optional] user-defined context for private data for the inequality constraints computation (may be `NULL`)

  Calling sequence of `func`:
$ PetscErrorCode func(Tao tao, Vec x, Vec ci, void *ctx);
+ tao - the `Tao` solver
. x   - point to evaluate inequality constraints
. ci   - vector of inequality constraints evaluated at x
- ctx - the (optional) user-defined function context

  Level: intermediate

.seealso: [](chapter_tao), `Tao, `TaoSetObjective()`, `TaoSetHessian()`, `TaoSetObjectiveAndGradient()`, `TaoSetVariableBounds()`
@*/
PetscErrorCode TaoSetInequalityConstraintsRoutine(Tao tao, Vec ci, PetscErrorCode (*func)(Tao, Vec, Vec, void *), void *ctx)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao, TAO_CLASSID, 1);
  if (ci) PetscValidHeaderSpecific(ci, VEC_CLASSID, 2);
  PetscCall(PetscObjectReference((PetscObject)ci));
  PetscCall(VecDestroy(&tao->constraints_inequality));
  tao->constraints_inequality            = ci;
  tao->ineq_constrained                  = func ? PETSC_TRUE : PETSC_FALSE;
  tao->user_con_inequalityP              = ctx;
  tao->ops->computeinequalityconstraints = func;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
   TaoComputeEqualityConstraints - Compute the variable bounds using the
   routine set by `TaoSetEqualityConstraintsRoutine()`.

   Collective

   Input Parameter:
.  tao - the `Tao` context

   Output Parameters:
+  X - point the equality constraints were evaluated on
-  CE   - vector of equality constraints evaluated at X

   Level: developer

.seealso: [](chapter_tao), `Tao`, `TaoSetEqualityConstraintsRoutine()`, `TaoComputeJacobianEquality()`, `TaoComputeInequalityConstraints()`
@*/
PetscErrorCode TaoComputeEqualityConstraints(Tao tao, Vec X, Vec CE)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao, TAO_CLASSID, 1);
  PetscValidHeaderSpecific(X, VEC_CLASSID, 2);
  PetscValidHeaderSpecific(CE, VEC_CLASSID, 3);
  PetscCheckSameComm(tao, 1, X, 2);
  PetscCheckSameComm(tao, 1, CE, 3);
  PetscCall(PetscLogEventBegin(TAO_ConstraintsEval, tao, X, CE, NULL));
  PetscCallBack("Tao callback equality constraints", (*tao->ops->computeequalityconstraints)(tao, X, CE, tao->user_con_equalityP));
  PetscCall(PetscLogEventEnd(TAO_ConstraintsEval, tao, X, CE, NULL));
  tao->nconstraints++;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
   TaoComputeInequalityConstraints - Compute the variable bounds using the
   routine set by `TaoSetInequalityConstraintsRoutine()`.

   Collective

   Input Parameter:
.  tao - the `Tao` context

   Output Parameters:
+  X - point the inequality constraints were evaluated on
-  CE   - vector of inequality constraints evaluated at X

   Level: developer

.seealso: [](chapter_tao), `Tao`, `TaoSetInequalityConstraintsRoutine()`, `TaoComputeJacobianInequality()`, `TaoComputeEqualityConstraints()`
@*/
PetscErrorCode TaoComputeInequalityConstraints(Tao tao, Vec X, Vec CI)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao, TAO_CLASSID, 1);
  PetscValidHeaderSpecific(X, VEC_CLASSID, 2);
  PetscValidHeaderSpecific(CI, VEC_CLASSID, 3);
  PetscCheckSameComm(tao, 1, X, 2);
  PetscCheckSameComm(tao, 1, CI, 3);
  PetscCall(PetscLogEventBegin(TAO_ConstraintsEval, tao, X, CI, NULL));
  PetscCallBack("Tao callback inequality constraints", (*tao->ops->computeinequalityconstraints)(tao, X, CI, tao->user_con_inequalityP));
  PetscCall(PetscLogEventEnd(TAO_ConstraintsEval, tao, X, CI, NULL));
  tao->nconstraints++;
  PetscFunctionReturn(PETSC_SUCCESS);
}
