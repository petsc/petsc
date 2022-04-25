#include <petsc/private/taoimpl.h> /*I "petsctao.h" I*/

/*@
  TaoSetVariableBounds - Sets the upper and lower bounds

  Logically collective on Tao

  Input Parameters:
+ tao - the Tao context
. XL  - vector of lower bounds
- XU  - vector of upper bounds

  Level: beginner

.seealso: `TaoSetObjective()`, `TaoSetHessian()`, `TaoSetObjectiveAndGradient()`, `TaoGetVariableBounds()`
@*/
PetscErrorCode TaoSetVariableBounds(Tao tao, Vec XL, Vec XU)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao,TAO_CLASSID,1);
  if (XL) PetscValidHeaderSpecific(XL,VEC_CLASSID,2);
  if (XU) PetscValidHeaderSpecific(XU,VEC_CLASSID,3);
  PetscCall(PetscObjectReference((PetscObject)XL));
  PetscCall(PetscObjectReference((PetscObject)XU));
  PetscCall(VecDestroy(&tao->XL));
  PetscCall(VecDestroy(&tao->XU));
  tao->XL = XL;
  tao->XU = XU;
  tao->bounded = (PetscBool)(XL || XU);
  PetscFunctionReturn(0);
}

/*@C
  TaoSetVariableBoundsRoutine - Sets a function to be used to compute variable bounds

  Logically collective on Tao

  Input Parameters:
+ tao - the Tao context
. func - the bounds computation routine
- ctx - [optional] user-defined context for private data for the bounds computation (may be NULL)

  Calling sequence of func:
$      func (Tao tao, Vec xl, Vec xu);

+ tao - the Tao
. xl  - vector of lower bounds
. xu  - vector of upper bounds
- ctx - the (optional) user-defined function context

  Level: beginner

.seealso: `TaoSetObjective()`, `TaoSetHessian()`, `TaoSetObjectiveAndGradient()`, `TaoSetVariableBounds()`

Note: The func passed in to TaoSetVariableBoundsRoutine() takes
precedence over any values set in TaoSetVariableBounds().

@*/
PetscErrorCode TaoSetVariableBoundsRoutine(Tao tao, PetscErrorCode (*func)(Tao, Vec, Vec, void*), void *ctx)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao,TAO_CLASSID,1);
  tao->user_boundsP = ctx;
  tao->ops->computebounds = func;
  tao->bounded = func ? PETSC_TRUE : PETSC_FALSE;
  PetscFunctionReturn(0);
}

/*@
  TaoGetVariableBounds - Gets the upper and lower bounds vectors set with TaoSetVariableBounds

  Not collective

  Input Parameter:
. tao - the Tao context

  Output Parametrs:
+ XL  - vector of lower bounds
- XU  - vector of upper bounds

  Level: beginner

.seealso: `TaoSetObjective()`, `TaoSetHessian()`, `TaoSetObjectiveAndGradient()`, `TaoSetVariableBounds()`
@*/
PetscErrorCode TaoGetVariableBounds(Tao tao, Vec *XL, Vec *XU)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao,TAO_CLASSID,1);
  if (XL) *XL = tao->XL;
  if (XU) *XU = tao->XU;
  PetscFunctionReturn(0);
}

/*@C
   TaoComputeVariableBounds - Compute the variable bounds using the
   routine set by TaoSetVariableBoundsRoutine().

   Collective on Tao

   Input Parameter:
.  tao - the Tao context

   Level: developer

.seealso: `TaoSetVariableBoundsRoutine()`, `TaoSetVariableBounds()`
@*/

PetscErrorCode TaoComputeVariableBounds(Tao tao)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao,TAO_CLASSID,1);
  if (!tao->XL || !tao->XU) {
    PetscCheck(tao->solution,PetscObjectComm((PetscObject)tao),PETSC_ERR_ORDER,"TaoSetSolution must be called before TaoComputeVariableBounds");
    PetscCall(VecDuplicate(tao->solution, &tao->XL));
    PetscCall(VecSet(tao->XL, PETSC_NINFINITY));
    PetscCall(VecDuplicate(tao->solution, &tao->XU));
    PetscCall(VecSet(tao->XU, PETSC_INFINITY));
  }
  if (tao->ops->computebounds) {
    PetscStackPush("Tao compute variable bounds");
    PetscCall((*tao->ops->computebounds)(tao,tao->XL,tao->XU,tao->user_boundsP));
    PetscStackPop;
  }
  PetscFunctionReturn(0);
}

/*@
  TaoSetInequalityBounds - Sets the upper and lower bounds

  Logically collective on Tao

  Input Parameters:
+ tao - the Tao context
. IL  - vector of lower bounds
- IU  - vector of upper bounds

  Level: beginner

.seealso: `TaoSetObjective()`, `TaoSetHessian()`, `TaoSetObjectiveAndGradient()`, `TaoGetInequalityBounds()`
@*/
PetscErrorCode TaoSetInequalityBounds(Tao tao, Vec IL, Vec IU)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao,TAO_CLASSID,1);
  if (IL) PetscValidHeaderSpecific(IL,VEC_CLASSID,2);
  if (IU) PetscValidHeaderSpecific(IU,VEC_CLASSID,3);
  PetscCall(PetscObjectReference((PetscObject)IL));
  PetscCall(PetscObjectReference((PetscObject)IU));
  PetscCall(VecDestroy(&tao->IL));
  PetscCall(VecDestroy(&tao->IU));
  tao->IL = IL;
  tao->IU = IU;
  tao->ineq_doublesided = (PetscBool)(IL || IU);
  PetscFunctionReturn(0);
}

/*@
  TaoGetInequalityBounds - Gets the upper and lower bounds set via TaoSetInequalityBounds

  Logically collective on Tao

  Input Parameter:
. tao - the Tao context

  Output Parameters:
+ IL  - vector of lower bounds
- IU  - vector of upper bounds

  Level: beginner

.seealso: `TaoSetObjective()`, `TaoSetHessian()`, `TaoSetObjectiveAndGradient()`, `TaoSetInequalityBounds()`
@*/
PetscErrorCode TaoGetInequalityBounds(Tao tao, Vec *IL, Vec *IU)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao,TAO_CLASSID,1);
  if (IL) *IL = tao->IL;
  if (IU) *IU = tao->IU;
  PetscFunctionReturn(0);
}

/*@C
   TaoComputeConstraints - Compute the variable bounds using the
   routine set by TaoSetConstraintsRoutine().

   Collective on Tao

   Input Parameters:
.  tao - the Tao context

   Level: developer

.seealso: `TaoSetConstraintsRoutine()`, `TaoComputeJacobian()`
@*/

PetscErrorCode TaoComputeConstraints(Tao tao, Vec X, Vec C)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao,TAO_CLASSID,1);
  PetscValidHeaderSpecific(X,VEC_CLASSID,2);
  PetscValidHeaderSpecific(C,VEC_CLASSID,3);
  PetscCheckSameComm(tao,1,X,2);
  PetscCheckSameComm(tao,1,C,3);
  PetscCheck(tao->ops->computeconstraints,PetscObjectComm((PetscObject)tao),PETSC_ERR_ARG_WRONGSTATE,"TaoSetConstraintsRoutine() has not been called");
  PetscCall(PetscLogEventBegin(TAO_ConstraintsEval,tao,X,C,NULL));
  PetscStackPush("Tao constraints evaluation routine");
  PetscCall((*tao->ops->computeconstraints)(tao,X,C,tao->user_conP));
  PetscStackPop;
  PetscCall(PetscLogEventEnd(TAO_ConstraintsEval,tao,X,C,NULL));
  tao->nconstraints++;
  PetscFunctionReturn(0);
}

/*@C
  TaoSetConstraintsRoutine - Sets a function to be used to compute constraints.  TAO only handles constraints under certain conditions, see manual for details

  Logically collective on Tao

  Input Parameters:
+ tao - the Tao context
. c   - A vector that will be used to store constraint evaluation
. func - the bounds computation routine
- ctx - [optional] user-defined context for private data for the constraints computation (may be NULL)

  Calling sequence of func:
$      func (Tao tao, Vec x, Vec c, void *ctx);

+ tao - the Tao
. x   - point to evaluate constraints
. c   - vector constraints evaluated at x
- ctx - the (optional) user-defined function context

  Level: intermediate

.seealso: `TaoSetObjective()`, `TaoSetHessian()`, `TaoSetObjectiveAndGradient()`, `TaoSetVariablevBounds()`

@*/
PetscErrorCode TaoSetConstraintsRoutine(Tao tao, Vec c, PetscErrorCode (*func)(Tao, Vec, Vec, void*), void *ctx)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao,TAO_CLASSID,1);
  if (c) PetscValidHeaderSpecific(c,VEC_CLASSID,2);
  PetscCall(PetscObjectReference((PetscObject)c));
  PetscCall(VecDestroy(&tao->constraints));
  tao->constrained = func ? PETSC_TRUE : PETSC_FALSE;
  tao->constraints = c;
  tao->user_conP = ctx;
  tao->ops->computeconstraints = func;
  PetscFunctionReturn(0);
}

/*@
  TaoComputeDualVariables - Computes the dual vectors corresponding to the bounds
  of the variables

  Collective on Tao

  Input Parameter:
. tao - the Tao context

  Output Parameters:
+ DL - dual variable vector for the lower bounds
- DU - dual variable vector for the upper bounds

  Level: advanced

  Note:
  DL and DU should be created before calling this routine.  If calling
  this routine after using an unconstrained solver, DL and DU are set to all
  zeros.

  Level: advanced

.seealso: `TaoComputeObjective()`, `TaoSetVariableBounds()`
@*/
PetscErrorCode TaoComputeDualVariables(Tao tao, Vec DL, Vec DU)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao,TAO_CLASSID,1);
  PetscValidHeaderSpecific(DL,VEC_CLASSID,2);
  PetscValidHeaderSpecific(DU,VEC_CLASSID,3);
  PetscCheckSameComm(tao,1,DL,2);
  PetscCheckSameComm(tao,1,DU,3);
  if (tao->ops->computedual) {
    PetscCall((*tao->ops->computedual)(tao,DL,DU));
  } else {
    PetscCall(VecSet(DL,0.0));
    PetscCall(VecSet(DU,0.0));
  }
  PetscFunctionReturn(0);
}

/*@
  TaoGetDualVariables - Gets pointers to the dual vectors

  Collective on Tao

  Input Parameter:
. tao - the Tao context

  Output Parameters:
+ DE - dual variable vector for the lower bounds
- DI - dual variable vector for the upper bounds

  Level: advanced

.seealso: `TaoComputeDualVariables()`
@*/
PetscErrorCode TaoGetDualVariables(Tao tao, Vec *DE, Vec *DI)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao,TAO_CLASSID,1);
  if (DE) *DE = tao->DE;
  if (DI) *DI = tao->DI;
  PetscFunctionReturn(0);
}

/*@C
  TaoSetEqualityConstraintsRoutine - Sets a function to be used to compute constraints.  TAO only handles constraints under certain conditions, see manual for details

  Logically collective on Tao

  Input Parameters:
+ tao - the Tao context
. ce   - A vector that will be used to store equality constraint evaluation
. func - the bounds computation routine
- ctx - [optional] user-defined context for private data for the equality constraints computation (may be NULL)

  Calling sequence of func:
$      func (Tao tao, Vec x, Vec ce, void *ctx);

+ tao - the Tao
. x   - point to evaluate equality constraints
. ce   - vector of equality constraints evaluated at x
- ctx - the (optional) user-defined function context

  Level: intermediate

.seealso: `TaoSetObjective()`, `TaoSetHessian()`, `TaoSetObjectiveAndGradient()`, `TaoSetVariableBounds()`

@*/
PetscErrorCode TaoSetEqualityConstraintsRoutine(Tao tao, Vec ce, PetscErrorCode (*func)(Tao, Vec, Vec, void*), void *ctx)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao,TAO_CLASSID,1);
  if (ce) PetscValidHeaderSpecific(ce,VEC_CLASSID,2);
  PetscCall(PetscObjectReference((PetscObject)ce));
  PetscCall(VecDestroy(&tao->constraints_equality));
  tao->eq_constrained = func ? PETSC_TRUE : PETSC_FALSE;
  tao->constraints_equality = ce;
  tao->user_con_equalityP = ctx;
  tao->ops->computeequalityconstraints = func;
  PetscFunctionReturn(0);
}

/*@C
  TaoSetInequalityConstraintsRoutine - Sets a function to be used to compute constraints.  TAO only handles constraints under certain conditions, see manual for details

  Logically collective on Tao

  Input Parameters:
+ tao - the Tao context
. ci   - A vector that will be used to store inequality constraint evaluation
. func - the bounds computation routine
- ctx - [optional] user-defined context for private data for the inequality constraints computation (may be NULL)

  Calling sequence of func:
$      func (Tao tao, Vec x, Vec ci, void *ctx);

+ tao - the Tao
. x   - point to evaluate inequality constraints
. ci   - vector of inequality constraints evaluated at x
- ctx - the (optional) user-defined function context

  Level: intermediate

.seealso: `TaoSetObjective()`, `TaoSetHessian()`, `TaoSetObjectiveAndGradient()`, `TaoSetVariableBounds()`

@*/
PetscErrorCode TaoSetInequalityConstraintsRoutine(Tao tao, Vec ci, PetscErrorCode (*func)(Tao, Vec, Vec, void*), void *ctx)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao,TAO_CLASSID,1);
  if (ci) PetscValidHeaderSpecific(ci,VEC_CLASSID,2);
  PetscCall(PetscObjectReference((PetscObject)ci));
  PetscCall(VecDestroy(&tao->constraints_inequality));
  tao->constraints_inequality = ci;
  tao->ineq_constrained = func ? PETSC_TRUE : PETSC_FALSE;
  tao->user_con_inequalityP = ctx;
  tao->ops->computeinequalityconstraints = func;
  PetscFunctionReturn(0);
}

/*@C
   TaoComputeEqualityConstraints - Compute the variable bounds using the
   routine set by TaoSetEqualityConstraintsRoutine().

   Collective on Tao

   Input Parameters:
.  tao - the Tao context

   Level: developer

.seealso: `TaoSetEqualityConstraintsRoutine()`, `TaoComputeJacobianEquality()`
@*/

PetscErrorCode TaoComputeEqualityConstraints(Tao tao, Vec X, Vec CE)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao,TAO_CLASSID,1);
  PetscValidHeaderSpecific(X,VEC_CLASSID,2);
  PetscValidHeaderSpecific(CE,VEC_CLASSID,3);
  PetscCheckSameComm(tao,1,X,2);
  PetscCheckSameComm(tao,1,CE,3);
  PetscCheck(tao->ops->computeequalityconstraints,PetscObjectComm((PetscObject)tao),PETSC_ERR_ARG_WRONGSTATE,"TaoSetEqualityConstraintsRoutine() has not been called");
  PetscCall(PetscLogEventBegin(TAO_ConstraintsEval,tao,X,CE,NULL));
  PetscStackPush("Tao equality constraints evaluation routine");
  PetscCall((*tao->ops->computeequalityconstraints)(tao,X,CE,tao->user_con_equalityP));
  PetscStackPop;
  PetscCall(PetscLogEventEnd(TAO_ConstraintsEval,tao,X,CE,NULL));
  tao->nconstraints++;
  PetscFunctionReturn(0);
}

/*@C
   TaoComputeInequalityConstraints - Compute the variable bounds using the
   routine set by TaoSetInequalityConstraintsRoutine().

   Collective on Tao

   Input Parameters:
.  tao - the Tao context

   Level: developer

.seealso: `TaoSetInequalityConstraintsRoutine()`, `TaoComputeJacobianInequality()`
@*/

PetscErrorCode TaoComputeInequalityConstraints(Tao tao, Vec X, Vec CI)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao,TAO_CLASSID,1);
  PetscValidHeaderSpecific(X,VEC_CLASSID,2);
  PetscValidHeaderSpecific(CI,VEC_CLASSID,3);
  PetscCheckSameComm(tao,1,X,2);
  PetscCheckSameComm(tao,1,CI,3);
  PetscCheck(tao->ops->computeinequalityconstraints,PetscObjectComm((PetscObject)tao),PETSC_ERR_ARG_WRONGSTATE,"TaoSetInequalityConstraintsRoutine() has not been called");
  PetscCall(PetscLogEventBegin(TAO_ConstraintsEval,tao,X,CI,NULL));
  PetscStackPush("Tao inequality constraints evaluation routine");
  PetscCall((*tao->ops->computeinequalityconstraints)(tao,X,CI,tao->user_con_inequalityP));
  PetscStackPop;
  PetscCall(PetscLogEventEnd(TAO_ConstraintsEval,tao,X,CI,NULL));
  tao->nconstraints++;
  PetscFunctionReturn(0);
}
