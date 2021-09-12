#include <petsc/private/taoimpl.h> /*I "petsctao.h" I*/

/*@
  TaoSetVariableBounds - Sets the upper and lower bounds

  Logically collective on Tao

  Input Parameters:
+ tao - the Tao context
. XL  - vector of lower bounds
- XU  - vector of upper bounds

  Level: beginner

.seealso: TaoSetObjectiveRoutine(), TaoSetHessianRoutine() TaoSetObjectiveAndGradientRoutine()
@*/

PetscErrorCode TaoSetVariableBounds(Tao tao, Vec XL, Vec XU)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao,TAO_CLASSID,1);
  if (XL) {
    PetscValidHeaderSpecific(XL,VEC_CLASSID,2);
    PetscObjectReference((PetscObject)XL);
  }
  if (XU) {
    PetscValidHeaderSpecific(XU,VEC_CLASSID,3);
    PetscObjectReference((PetscObject)XU);
  }
  ierr = VecDestroy(&tao->XL);CHKERRQ(ierr);
  ierr = VecDestroy(&tao->XU);CHKERRQ(ierr);
  tao->XL = XL;
  tao->XU = XU;
  tao->bounded = PETSC_TRUE;
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

.seealso: TaoSetObjectiveRoutine(), TaoSetHessianRoutine() TaoSetObjectiveAndGradientRoutine(), TaoSetVariableBounds()

Note: The func passed in to TaoSetVariableBoundsRoutine() takes
precedence over any values set in TaoSetVariableBounds().

@*/
PetscErrorCode TaoSetVariableBoundsRoutine(Tao tao, PetscErrorCode (*func)(Tao, Vec, Vec, void*), void *ctx)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao,TAO_CLASSID,1);
  tao->user_boundsP = ctx;
  tao->ops->computebounds = func;
  tao->bounded = PETSC_TRUE;
  PetscFunctionReturn(0);
}

PetscErrorCode TaoGetVariableBounds(Tao tao, Vec *XL, Vec *XU)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao,TAO_CLASSID,1);
  if (XL) {
    *XL=tao->XL;
  }
  if (XU) {
    *XU=tao->XU;
  }
  PetscFunctionReturn(0);
}

/*@C
   TaoComputeVariableBounds - Compute the variable bounds using the
   routine set by TaoSetVariableBoundsRoutine().

   Collective on Tao

   Input Parameters:
.  tao - the Tao context

   Level: developer

.seealso: TaoSetVariableBoundsRoutine(), TaoSetVariableBounds()
@*/

PetscErrorCode TaoComputeVariableBounds(Tao tao)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao,TAO_CLASSID,1);
  PetscStackPush("Tao compute variable bounds");
  if (!tao->XL || !tao->XU) {
    if (!tao->solution) SETERRQ(PetscObjectComm((PetscObject)tao),PETSC_ERR_ORDER,"TaoSetInitialVector must be called before TaoComputeVariableBounds");
    ierr = VecDuplicate(tao->solution, &tao->XL);CHKERRQ(ierr);
    ierr = VecSet(tao->XL, PETSC_NINFINITY);CHKERRQ(ierr);
    ierr = VecDuplicate(tao->solution, &tao->XU);CHKERRQ(ierr);
    ierr = VecSet(tao->XU, PETSC_INFINITY);CHKERRQ(ierr);
  }
  if (tao->ops->computebounds) {
    ierr = (*tao->ops->computebounds)(tao,tao->XL,tao->XU,tao->user_boundsP);CHKERRQ(ierr);
  }
  PetscStackPop;
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

.seealso: TaoSetObjectiveRoutine(), TaoSetHessianRoutine() TaoSetObjectiveAndGradientRoutine()
@*/

PetscErrorCode TaoSetInequalityBounds(Tao tao, Vec IL, Vec IU)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao,TAO_CLASSID,1);
  if (IL) {
    PetscValidHeaderSpecific(IL,VEC_CLASSID,2);
    PetscObjectReference((PetscObject)IL);
  }
  if (IU) {
    PetscValidHeaderSpecific(IU,VEC_CLASSID,3);
    PetscObjectReference((PetscObject)IU);
  }
  ierr = VecDestroy(&tao->IL);CHKERRQ(ierr);
  ierr = VecDestroy(&tao->IU);CHKERRQ(ierr);
  tao->IL = IL;
  tao->IU = IU;
  tao->ineq_doublesided = PETSC_TRUE;
  PetscFunctionReturn(0);
}

PetscErrorCode TaoGetInequalityBounds(Tao tao, Vec *IL, Vec *IU)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao,TAO_CLASSID,1);
  if (IL) {
    *IL=tao->IL;
  }
  if (IU) {
    *IU=tao->IU;
  }
  PetscFunctionReturn(0);
}

/*@C
   TaoComputeConstraints - Compute the variable bounds using the
   routine set by TaoSetConstraintsRoutine().

   Collective on Tao

   Input Parameters:
.  tao - the Tao context

   Level: developer

.seealso: TaoSetConstraintsRoutine(), TaoComputeJacobian()
@*/

PetscErrorCode TaoComputeConstraints(Tao tao, Vec X, Vec C)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao,TAO_CLASSID,1);
  PetscValidHeaderSpecific(X,VEC_CLASSID,2);
  PetscValidHeaderSpecific(C,VEC_CLASSID,3);
  PetscCheckSameComm(tao,1,X,2);
  PetscCheckSameComm(tao,1,C,3);

  if (!tao->ops->computeconstraints) SETERRQ(PetscObjectComm((PetscObject)tao),PETSC_ERR_ARG_WRONGSTATE,"TaoSetConstraintsRoutine() has not been called");
  if (!tao->solution) SETERRQ(PetscObjectComm((PetscObject)tao),PETSC_ERR_ARG_WRONGSTATE,"TaoSetInitialVector must be called before TaoComputeConstraints");
  ierr = PetscLogEventBegin(TAO_ConstraintsEval,tao,X,C,NULL);CHKERRQ(ierr);
  PetscStackPush("Tao constraints evaluation routine");
  ierr = (*tao->ops->computeconstraints)(tao,X,C,tao->user_conP);CHKERRQ(ierr);
  PetscStackPop;
  ierr = PetscLogEventEnd(TAO_ConstraintsEval,tao,X,C,NULL);CHKERRQ(ierr);
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

.seealso: TaoSetObjectiveRoutine(), TaoSetHessianRoutine() TaoSetObjectiveAndGradientRoutine(), TaoSetVariablevBounds()

@*/
PetscErrorCode TaoSetConstraintsRoutine(Tao tao, Vec c, PetscErrorCode (*func)(Tao, Vec, Vec, void*), void *ctx)
{
    PetscFunctionBegin;
    PetscValidHeaderSpecific(tao,TAO_CLASSID,1);
    tao->constrained = PETSC_TRUE;
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

.seealso: TaoComputeObjective(), TaoSetVariableBounds()
@*/
PetscErrorCode TaoComputeDualVariables(Tao tao, Vec DL, Vec DU)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao,TAO_CLASSID,1);
  PetscValidHeaderSpecific(DL,VEC_CLASSID,2);
  PetscValidHeaderSpecific(DU,VEC_CLASSID,3);
  PetscCheckSameComm(tao,1,DL,2);
  PetscCheckSameComm(tao,1,DU,3);
  if (tao->ops->computedual) {
    ierr = (*tao->ops->computedual)(tao,DL,DU);CHKERRQ(ierr);
  }  else {
    ierr = VecSet(DL,0.0);CHKERRQ(ierr);
    ierr = VecSet(DU,0.0);CHKERRQ(ierr);
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

.seealso: TaoComputeDualVariables()
@*/
PetscErrorCode TaoGetDualVariables(Tao tao, Vec *DE, Vec *DI)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao,TAO_CLASSID,1);
  if (DE) {
    *DE = tao->DE;
  }
  if (DI) {
    *DI = tao->DI;
  }
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

.seealso: TaoSetObjectiveRoutine(), TaoSetHessianRoutine() TaoSetObjectiveAndGradientRoutine(), TaoSetVariableBounds()

@*/
PetscErrorCode TaoSetEqualityConstraintsRoutine(Tao tao, Vec ce, PetscErrorCode (*func)(Tao, Vec, Vec, void*), void *ctx)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao,TAO_CLASSID,1);
  if (ce) {
    PetscValidHeaderSpecific(ce,VEC_CLASSID,2);
    PetscObjectReference((PetscObject)ce);
  }
  ierr = VecDestroy(&tao->constraints_equality);CHKERRQ(ierr);
  tao->eq_constrained = PETSC_TRUE;
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

.seealso: TaoSetObjectiveRoutine(), TaoSetHessianRoutine() TaoSetObjectiveAndGradientRoutine(), TaoSetVariableBounds()

@*/
PetscErrorCode TaoSetInequalityConstraintsRoutine(Tao tao, Vec ci, PetscErrorCode (*func)(Tao, Vec, Vec, void*), void *ctx)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao,TAO_CLASSID,1);
  if (ci) {
    PetscValidHeaderSpecific(ci,VEC_CLASSID,2);
    PetscObjectReference((PetscObject)ci);
  }
  ierr = VecDestroy(&tao->constraints_inequality);CHKERRQ(ierr);
  tao->constraints_inequality = ci;
  tao->ineq_constrained = PETSC_TRUE;
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

.seealso: TaoSetEqualityConstraintsRoutine(), TaoComputeJacobianEquality()
@*/

PetscErrorCode TaoComputeEqualityConstraints(Tao tao, Vec X, Vec CE)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao,TAO_CLASSID,1);
  PetscValidHeaderSpecific(X,VEC_CLASSID,2);
  PetscValidHeaderSpecific(CE,VEC_CLASSID,3);
  PetscCheckSameComm(tao,1,X,2);
  PetscCheckSameComm(tao,1,CE,3);

  if (!tao->ops->computeequalityconstraints) SETERRQ(PetscObjectComm((PetscObject)tao),PETSC_ERR_ARG_WRONGSTATE,"TaoSetEqualityConstraintsRoutine() has not been called");
  if (!tao->solution) SETERRQ(PetscObjectComm((PetscObject)tao),PETSC_ERR_ARG_WRONGSTATE,"TaoSetInitialVector must be called before TaoComputeEqualityConstraints");
  ierr = PetscLogEventBegin(TAO_ConstraintsEval,tao,X,CE,NULL);CHKERRQ(ierr);
  PetscStackPush("Tao equality constraints evaluation routine");
  ierr = (*tao->ops->computeequalityconstraints)(tao,X,CE,tao->user_con_equalityP);CHKERRQ(ierr);
  PetscStackPop;
  ierr = PetscLogEventEnd(TAO_ConstraintsEval,tao,X,CE,NULL);CHKERRQ(ierr);
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

.seealso: TaoSetInequalityConstraintsRoutine(), TaoComputeJacobianInequality()
@*/

PetscErrorCode TaoComputeInequalityConstraints(Tao tao, Vec X, Vec CI)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao,TAO_CLASSID,1);
  PetscValidHeaderSpecific(X,VEC_CLASSID,2);
  PetscValidHeaderSpecific(CI,VEC_CLASSID,3);
  PetscCheckSameComm(tao,1,X,2);
  PetscCheckSameComm(tao,1,CI,3);

  if (!tao->ops->computeinequalityconstraints) SETERRQ(PetscObjectComm((PetscObject)tao),PETSC_ERR_ARG_WRONGSTATE,"TaoSetInequalityConstraintsRoutine() has not been called");
  if (!tao->solution) SETERRQ(PetscObjectComm((PetscObject)tao),PETSC_ERR_ARG_WRONGSTATE,"TaoSetInitialVector must be called before TaoComputeInequalityConstraints");
  ierr = PetscLogEventBegin(TAO_ConstraintsEval,tao,X,CI,NULL);CHKERRQ(ierr);
  PetscStackPush("Tao inequality constraints evaluation routine");
  ierr = (*tao->ops->computeinequalityconstraints)(tao,X,CI,tao->user_con_inequalityP);CHKERRQ(ierr);
  PetscStackPop;
  ierr = PetscLogEventEnd(TAO_ConstraintsEval,tao,X,CI,NULL);CHKERRQ(ierr);
  tao->nconstraints++;
  PetscFunctionReturn(0);
}
