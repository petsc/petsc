#include <petsc/private/taoimpl.h> /*I "petsctao.h" I*/

/*@C
   TaoSetHessianRoutine - Sets the function to compute the Hessian as well as the location to store the matrix.

   Logically collective on Tao

   Input Parameters:
+  tao - the Tao context
.  H - Matrix used for the hessian
.  Hpre - Matrix that will be used operated on by preconditioner, can be same as H
.  hess - Hessian evaluation routine
-  ctx - [optional] user-defined context for private data for the
         Hessian evaluation routine (may be NULL)

   Calling sequence of hess:
$    hess (Tao tao,Vec x,Mat H,Mat Hpre,void *ctx);

+  tao - the Tao  context
.  x - input vector
.  H - Hessian matrix
.  Hpre - preconditioner matrix, usually the same as H
-  ctx - [optional] user-defined Hessian context

   Level: beginner

@*/
PetscErrorCode TaoSetHessianRoutine(Tao tao, Mat H, Mat Hpre, PetscErrorCode (*func)(Tao, Vec, Mat, Mat, void*), void *ctx)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao,TAO_CLASSID,1);
  if (H) {
    PetscValidHeaderSpecific(H,MAT_CLASSID,2);
    PetscCheckSameComm(tao,1,H,2);
  }
  if (Hpre) {
    PetscValidHeaderSpecific(Hpre,MAT_CLASSID,3);
    PetscCheckSameComm(tao,1,Hpre,3);
  }
  if (ctx) {
    tao->user_hessP = ctx;
  }
  if (func) {
    tao->ops->computehessian = func;
  }
  if (H) {
    ierr = PetscObjectReference((PetscObject)H);CHKERRQ(ierr);
    ierr = MatDestroy(&tao->hessian);CHKERRQ(ierr);
    tao->hessian = H;
  }
  if (Hpre) {
    ierr = PetscObjectReference((PetscObject)Hpre);CHKERRQ(ierr);
    ierr = MatDestroy(&tao->hessian_pre);CHKERRQ(ierr);
    tao->hessian_pre = Hpre;
  }
  PetscFunctionReturn(0);
}

/*@C
   TaoComputeHessian - Computes the Hessian matrix that has been
   set with TaoSetHessianRoutine().

   Collective on Tao

   Input Parameters:
+  solver - the Tao solver context
-  xx - input vector

   Output Parameters:
+  H - Hessian matrix
-  Hpre - Preconditioning matrix

   Notes:
   Most users should not need to explicitly call this routine, as it
   is used internally within the minimization solvers.

   TaoComputeHessian() is typically used within minimization
   implementations, so most users would not generally call this routine
   themselves.

   Level: developer

.seealso:  TaoComputeObjective(), TaoComputeObjectiveAndGradient(), TaoSetHessian()

@*/
PetscErrorCode TaoComputeHessian(Tao tao, Vec X, Mat H, Mat Hpre)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao,TAO_CLASSID,1);
  PetscValidHeaderSpecific(X, VEC_CLASSID,2);
  PetscCheckSameComm(tao,1,X,2);

  if (!tao->ops->computehessian) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Must call TaoSetHessian() first");
  ++tao->nhess;
  ierr = PetscLogEventBegin(Tao_HessianEval,tao,X,H,Hpre);CHKERRQ(ierr);
  PetscStackPush("Tao user Hessian function");
  ierr = (*tao->ops->computehessian)(tao,X,H,Hpre,tao->user_hessP);CHKERRQ(ierr);
  PetscStackPop;
  ierr = PetscLogEventEnd(Tao_HessianEval,tao,X,H,Hpre);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
   TaoComputeJacobian - Computes the Jacobian matrix that has been
   set with TaoSetJacobianRoutine().

   Collective on Tao

   Input Parameters:
+  solver - the Tao solver context
-  xx - input vector

   Output Parameters:
+  H - Jacobian matrix
-  Hpre - Preconditioning matrix

   Notes:
   Most users should not need to explicitly call this routine, as it
   is used internally within the minimization solvers.

   TaoComputeJacobian() is typically used within minimization
   implementations, so most users would not generally call this routine
   themselves.

   Level: developer

.seealso:  TaoComputeObjective(), TaoComputeObjectiveAndGradient(), TaoSetJacobian()

@*/
PetscErrorCode TaoComputeJacobian(Tao tao, Vec X, Mat J, Mat Jpre)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao,TAO_CLASSID,1);
  PetscValidHeaderSpecific(X, VEC_CLASSID,2);
  PetscCheckSameComm(tao,1,X,2);

  if (!tao->ops->computejacobian) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Must call TaoSetJacobian() first");
  ++tao->njac;
  ierr = PetscLogEventBegin(Tao_JacobianEval,tao,X,J,Jpre);CHKERRQ(ierr);
  PetscStackPush("Tao user Jacobian function");
  ierr = (*tao->ops->computejacobian)(tao,X,J,Jpre,tao->user_jacP);CHKERRQ(ierr);
  PetscStackPop;
  ierr = PetscLogEventEnd(Tao_JacobianEval,tao,X,J,Jpre);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
   TaoComputeJacobianState - Computes the Jacobian matrix that has been
   set with TaoSetJacobianStateRoutine().

   Collective on Tao

   Input Parameters:
+  solver - the Tao solver context
-  xx - input vector

   Output Parameters:
+  H - Jacobian matrix
-  Hpre - Preconditioning matrix

   Notes:
   Most users should not need to explicitly call this routine, as it
   is used internally within the minimization solvers.

   TaoComputeJacobianState() is typically used within minimization
   implementations, so most users would not generally call this routine
   themselves.

   Level: developer

.seealso:  TaoComputeObjective(), TaoComputeObjectiveAndGradient(), TaoSetJacobianStateRoutine(), TaoComputeJacobianDesign(), TaoSetStateDesignIS()

@*/
PetscErrorCode TaoComputeJacobianState(Tao tao, Vec X, Mat J, Mat Jpre, Mat Jinv)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao,TAO_CLASSID,1);
  PetscValidHeaderSpecific(X, VEC_CLASSID,2);
  PetscCheckSameComm(tao,1,X,2);

  if (!tao->ops->computejacobianstate) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Must call TaoSetJacobianState() first");
  ++tao->njac_state;
  ierr = PetscLogEventBegin(Tao_JacobianEval,tao,X,J,Jpre);CHKERRQ(ierr);
  PetscStackPush("Tao user Jacobian(state) function");
  ierr = (*tao->ops->computejacobianstate)(tao,X,J,Jpre,Jinv,tao->user_jac_stateP);CHKERRQ(ierr);
  PetscStackPop;
  ierr = PetscLogEventEnd(Tao_JacobianEval,tao,X,J,Jpre);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
   TaoComputeJacobianDesign - Computes the Jacobian matrix that has been
   set with TaoSetJacobianDesignRoutine().

   Collective on Tao

   Input Parameters:
+  solver - the Tao solver context
-  xx - input vector

   Output Parameters:
.  H - Jacobian matrix

   Notes:
   Most users should not need to explicitly call this routine, as it
   is used internally within the minimization solvers.

   TaoComputeJacobianDesign() is typically used within minimization
   implementations, so most users would not generally call this routine
   themselves.

   Level: developer

.seealso:  TaoComputeObjective(), TaoComputeObjectiveAndGradient(), TaoSetJacobianDesignRoutine(), TaoComputeJacobianDesign(), TaoSetStateDesignIS()

@*/
PetscErrorCode TaoComputeJacobianDesign(Tao tao, Vec X, Mat J)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao,TAO_CLASSID,1);
  PetscValidHeaderSpecific(X, VEC_CLASSID,2);
  PetscCheckSameComm(tao,1,X,2);

  if (!tao->ops->computejacobiandesign) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Must call TaoSetJacobianDesign() first");
  ++tao->njac_design;
  ierr = PetscLogEventBegin(Tao_JacobianEval,tao,X,J,NULL);CHKERRQ(ierr);
  PetscStackPush("Tao user Jacobian(design) function");
  ierr = (*tao->ops->computejacobiandesign)(tao,X,J,tao->user_jac_designP);CHKERRQ(ierr);
  PetscStackPop;
  ierr = PetscLogEventEnd(Tao_JacobianEval,tao,X,J,NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
   TaoSetJacobianRoutine - Sets the function to compute the Jacobian as well as the location to store the matrix.

   Logically collective on Tao

   Input Parameters:
+  tao - the Tao context
.  J - Matrix used for the jacobian
.  Jpre - Matrix that will be used operated on by preconditioner, can be same as J
.  jac - Jacobian evaluation routine
-  ctx - [optional] user-defined context for private data for the
         Jacobian evaluation routine (may be NULL)

   Calling sequence of jac:
$    jac (Tao tao,Vec x,Mat *J,Mat *Jpre,void *ctx);

+  tao - the Tao  context
.  x - input vector
.  J - Jacobian matrix
.  Jpre - preconditioner matrix, usually the same as J
-  ctx - [optional] user-defined Jacobian context

   Level: intermediate

@*/
PetscErrorCode TaoSetJacobianRoutine(Tao tao, Mat J, Mat Jpre, PetscErrorCode (*func)(Tao, Vec, Mat, Mat, void*), void *ctx)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao,TAO_CLASSID,1);
  if (J) {
    PetscValidHeaderSpecific(J,MAT_CLASSID,2);
    PetscCheckSameComm(tao,1,J,2);
  }
  if (Jpre) {
    PetscValidHeaderSpecific(Jpre,MAT_CLASSID,3);
    PetscCheckSameComm(tao,1,Jpre,3);
  }
  if (ctx) {
    tao->user_jacP = ctx;
  }
  if (func) {
    tao->ops->computejacobian = func;
  }
  if (J) {
    ierr = PetscObjectReference((PetscObject)J);CHKERRQ(ierr);
    ierr = MatDestroy(&tao->jacobian);CHKERRQ(ierr);
    tao->jacobian = J;
  }
  if (Jpre) {
    ierr = PetscObjectReference((PetscObject)Jpre);CHKERRQ(ierr);
    ierr = MatDestroy(&tao->jacobian_pre);CHKERRQ(ierr);
    tao->jacobian_pre=Jpre;
  }
  PetscFunctionReturn(0);
}

/*@C
   TaoSetJacobianStateRoutine - Sets the function to compute the Jacobian
   (and its inverse) of the constraint function with respect to the state variables.
   Used only for pde-constrained optimization.

   Logically collective on Tao

   Input Parameters:
+  tao - the Tao context
.  J - Matrix used for the jacobian
.  Jpre - Matrix that will be used operated on by PETSc preconditioner, can be same as J.  Only used if Jinv is NULL
.  Jinv - [optional] Matrix used to apply the inverse of the state jacobian. Use NULL to default to PETSc KSP solvers to apply the inverse.
.  jac - Jacobian evaluation routine
-  ctx - [optional] user-defined context for private data for the
         Jacobian evaluation routine (may be NULL)

   Calling sequence of jac:
$    jac (Tao tao,Vec x,Mat *J,Mat *Jpre,void *ctx);

+  tao - the Tao  context
.  x - input vector
.  J - Jacobian matrix
.  Jpre - preconditioner matrix, usually the same as J
.  Jinv - inverse of J
-  ctx - [optional] user-defined Jacobian context

   Level: intermediate
.seealse: TaoComputeJacobianState(), TaoSetJacobianDesignRoutine(), TaoSetStateDesignIS()
@*/
PetscErrorCode TaoSetJacobianStateRoutine(Tao tao, Mat J, Mat Jpre, Mat Jinv, PetscErrorCode (*func)(Tao, Vec, Mat, Mat, Mat,void*), void *ctx)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao,TAO_CLASSID,1);
  if (J) {
    PetscValidHeaderSpecific(J,MAT_CLASSID,2);
    PetscCheckSameComm(tao,1,J,2);
  }
  if (Jpre) {
    PetscValidHeaderSpecific(Jpre,MAT_CLASSID,3);
    PetscCheckSameComm(tao,1,Jpre,3);
  }
  if (Jinv) {
    PetscValidHeaderSpecific(Jinv,MAT_CLASSID,4);
    PetscCheckSameComm(tao,1,Jinv,4);
  }
  if (ctx) {
    tao->user_jac_stateP = ctx;
  }
  if (func) {
    tao->ops->computejacobianstate = func;
  }
  if (J) {
    ierr = PetscObjectReference((PetscObject)J);CHKERRQ(ierr);
    ierr = MatDestroy(&tao->jacobian_state);CHKERRQ(ierr);
    tao->jacobian_state = J;
  }
  if (Jpre) {
    ierr = PetscObjectReference((PetscObject)Jpre);CHKERRQ(ierr);
    ierr = MatDestroy(&tao->jacobian_state_pre);CHKERRQ(ierr);
    tao->jacobian_state_pre=Jpre;
  }
  if (Jinv) {
    ierr = PetscObjectReference((PetscObject)Jinv);CHKERRQ(ierr);
    ierr = MatDestroy(&tao->jacobian_state_inv);CHKERRQ(ierr);
    tao->jacobian_state_inv=Jinv;
  }
  PetscFunctionReturn(0);
}

/*@C
   TaoSetJacobianDesignRoutine - Sets the function to compute the Jacobian of
   the constraint function with respect to the design variables.  Used only for
   pde-constrained optimization.

   Logically collective on Tao

   Input Parameters:
+  tao - the Tao context
.  J - Matrix used for the jacobian
.  jac - Jacobian evaluation routine
-  ctx - [optional] user-defined context for private data for the
         Jacobian evaluation routine (may be NULL)

   Calling sequence of jac:
$    jac (Tao tao,Vec x,Mat *J,void *ctx);

+  tao - the Tao  context
.  x - input vector
.  J - Jacobian matrix
-  ctx - [optional] user-defined Jacobian context


   Notes:

   The function jac() takes Mat * as the matrix arguments rather than Mat.
   This allows the Jacobian evaluation routine to replace A and/or B with a
   completely new new matrix structure (not just different matrix elements)
   when appropriate, for instance, if the nonzero structure is changing
   throughout the global iterations.

   Level: intermediate
.seealso: TaoComputeJacobianDesign(), TaoSetJacobianStateRoutine(), TaoSetStateDesignIS()
@*/
PetscErrorCode TaoSetJacobianDesignRoutine(Tao tao, Mat J, PetscErrorCode (*func)(Tao, Vec, Mat, void*), void *ctx)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao,TAO_CLASSID,1);
  if (J) {
    PetscValidHeaderSpecific(J,MAT_CLASSID,2);
    PetscCheckSameComm(tao,1,J,2);
  }
  if (ctx) {
    tao->user_jac_designP = ctx;
  }
  if (func) {
    tao->ops->computejacobiandesign = func;
  }
  if (J) {
    ierr = PetscObjectReference((PetscObject)J);CHKERRQ(ierr);
    ierr = MatDestroy(&tao->jacobian_design);CHKERRQ(ierr);
    tao->jacobian_design = J;
  }
  PetscFunctionReturn(0);
}

/*@
   TaoSetStateDesignIS - Indicate to the Tao which variables in the
   solution vector are state variables and which are design.  Only applies to
   pde-constrained optimization.

   Logically Collective on Tao

   Input Parameters:
+  tao - The Tao context
.  s_is - the index set corresponding to the state variables
-  d_is - the index set corresponding to the design variables

   Level: intermediate

.seealso: TaoSetJacobianStateRoutine(), TaoSetJacobianDesignRoutine()
@*/
PetscErrorCode TaoSetStateDesignIS(Tao tao, IS s_is, IS d_is)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscObjectReference((PetscObject)s_is);CHKERRQ(ierr);
  ierr = ISDestroy(&tao->state_is);CHKERRQ(ierr);
  tao->state_is = s_is;
  ierr = PetscObjectReference((PetscObject)(d_is));CHKERRQ(ierr);
  ierr = ISDestroy(&tao->design_is);CHKERRQ(ierr);
  tao->design_is = d_is;
  PetscFunctionReturn(0);
}

/*@C
   TaoComputeJacobianEquality - Computes the Jacobian matrix that has been
   set with TaoSetJacobianEqualityRoutine().

   Collective on Tao

   Input Parameters:
+  solver - the Tao solver context
-  xx - input vector

   Output Parameters:
+  H - Jacobian matrix
-  Hpre - Preconditioning matrix

   Notes:
   Most users should not need to explicitly call this routine, as it
   is used internally within the minimization solvers.

   Level: developer

.seealso:  TaoComputeObjective(), TaoComputeObjectiveAndGradient(), TaoSetJacobianStateRoutine(), TaoComputeJacobianDesign(), TaoSetStateDesignIS()

@*/
PetscErrorCode TaoComputeJacobianEquality(Tao tao, Vec X, Mat J, Mat Jpre)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao,TAO_CLASSID,1);
  PetscValidHeaderSpecific(X, VEC_CLASSID,2);
  PetscCheckSameComm(tao,1,X,2);

  if (!tao->ops->computejacobianequality) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Must call TaoSetJacobianEquality() first");
  ++tao->njac_equality;
  ierr = PetscLogEventBegin(Tao_JacobianEval,tao,X,J,Jpre);CHKERRQ(ierr);
  PetscStackPush("Tao user Jacobian(equality) function");
  ierr = (*tao->ops->computejacobianequality)(tao,X,J,Jpre,tao->user_jac_equalityP);CHKERRQ(ierr);
  PetscStackPop;
  ierr = PetscLogEventEnd(Tao_JacobianEval,tao,X,J,Jpre);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
   TaoComputeJacobianInequality - Computes the Jacobian matrix that has been
   set with TaoSetJacobianInequalityRoutine().

   Collective on Tao

   Input Parameters:
+  solver - the Tao solver context
-  xx - input vector

   Output Parameters:
+  H - Jacobian matrix
-  Hpre - Preconditioning matrix

   Notes:
   Most users should not need to explicitly call this routine, as it
   is used internally within the minimization solvers.

   Level: developer

.seealso:  TaoComputeObjective(), TaoComputeObjectiveAndGradient(), TaoSetJacobianStateRoutine(), TaoComputeJacobianDesign(), TaoSetStateDesignIS()

@*/
PetscErrorCode TaoComputeJacobianInequality(Tao tao, Vec X, Mat J, Mat Jpre)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao,TAO_CLASSID,1);
  PetscValidHeaderSpecific(X, VEC_CLASSID,2);
  PetscCheckSameComm(tao,1,X,2);

  if (!tao->ops->computejacobianinequality) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Must call TaoSetJacobianInequality() first");
  ++tao->njac_inequality;
  ierr = PetscLogEventBegin(Tao_JacobianEval,tao,X,J,Jpre);CHKERRQ(ierr);
  PetscStackPush("Tao user Jacobian(inequality) function");
  ierr = (*tao->ops->computejacobianinequality)(tao,X,J,Jpre,tao->user_jac_inequalityP);CHKERRQ(ierr);
  PetscStackPop;
  ierr = PetscLogEventEnd(Tao_JacobianEval,tao,X,J,Jpre);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
   TaoSetJacobianEqualityRoutine - Sets the function to compute the Jacobian
   (and its inverse) of the constraint function with respect to the equality variables.
   Used only for pde-constrained optimization.

   Logically collective on Tao

   Input Parameters:
+  tao - the Tao context
.  J - Matrix used for the jacobian
.  Jpre - Matrix that will be used operated on by PETSc preconditioner, can be same as J.
.  jac - Jacobian evaluation routine
-  ctx - [optional] user-defined context for private data for the
         Jacobian evaluation routine (may be NULL)

   Calling sequence of jac:
$    jac (Tao tao,Vec x,Mat *J,Mat *Jpre,void *ctx);

+  tao - the Tao  context
.  x - input vector
.  J - Jacobian matrix
.  Jpre - preconditioner matrix, usually the same as J
-  ctx - [optional] user-defined Jacobian context

   Level: intermediate
.seealse: TaoComputeJacobianEquality(), TaoSetJacobianDesignRoutine(), TaoSetEqualityDesignIS()
@*/
PetscErrorCode TaoSetJacobianEqualityRoutine(Tao tao, Mat J, Mat Jpre, PetscErrorCode (*func)(Tao, Vec, Mat, Mat,void*), void *ctx)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao,TAO_CLASSID,1);
  if (J) {
    PetscValidHeaderSpecific(J,MAT_CLASSID,2);
    PetscCheckSameComm(tao,1,J,2);
  }
  if (Jpre) {
    PetscValidHeaderSpecific(Jpre,MAT_CLASSID,3);
    PetscCheckSameComm(tao,1,Jpre,3);
  }
  if (ctx) {
    tao->user_jac_equalityP = ctx;
  }
  if (func) {
    tao->ops->computejacobianequality = func;
  }
  if (J) {
    ierr = PetscObjectReference((PetscObject)J);CHKERRQ(ierr);
    ierr = MatDestroy(&tao->jacobian_equality);CHKERRQ(ierr);
    tao->jacobian_equality = J;
  }
  if (Jpre) {
    ierr = PetscObjectReference((PetscObject)Jpre);CHKERRQ(ierr);
    ierr = MatDestroy(&tao->jacobian_equality_pre);CHKERRQ(ierr);
    tao->jacobian_equality_pre=Jpre;
  }
  PetscFunctionReturn(0);
}

/*@C
   TaoSetJacobianInequalityRoutine - Sets the function to compute the Jacobian
   (and its inverse) of the constraint function with respect to the inequality variables.
   Used only for pde-constrained optimization.

   Logically collective on Tao

   Input Parameters:
+  tao - the Tao context
.  J - Matrix used for the jacobian
.  Jpre - Matrix that will be used operated on by PETSc preconditioner, can be same as J.
.  jac - Jacobian evaluation routine
-  ctx - [optional] user-defined context for private data for the
         Jacobian evaluation routine (may be NULL)

   Calling sequence of jac:
$    jac (Tao tao,Vec x,Mat *J,Mat *Jpre,void *ctx);

+  tao - the Tao  context
.  x - input vector
.  J - Jacobian matrix
.  Jpre - preconditioner matrix, usually the same as J
-  ctx - [optional] user-defined Jacobian context

   Level: intermediate
.seealse: TaoComputeJacobianInequality(), TaoSetJacobianDesignRoutine(), TaoSetInequalityDesignIS()
@*/
PetscErrorCode TaoSetJacobianInequalityRoutine(Tao tao, Mat J, Mat Jpre, PetscErrorCode (*func)(Tao, Vec, Mat, Mat,void*), void *ctx)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao,TAO_CLASSID,1);
  if (J) {
    PetscValidHeaderSpecific(J,MAT_CLASSID,2);
    PetscCheckSameComm(tao,1,J,2);
  }
  if (Jpre) {
    PetscValidHeaderSpecific(Jpre,MAT_CLASSID,3);
    PetscCheckSameComm(tao,1,Jpre,3);
  }
  if (ctx) {
    tao->user_jac_inequalityP = ctx;
  }
  if (func) {
    tao->ops->computejacobianinequality = func;
  }
  if (J) {
    ierr = PetscObjectReference((PetscObject)J);CHKERRQ(ierr);
    ierr = MatDestroy(&tao->jacobian_inequality);CHKERRQ(ierr);
    tao->jacobian_inequality = J;
  }
  if (Jpre) {
    ierr = PetscObjectReference((PetscObject)Jpre);CHKERRQ(ierr);
    ierr = MatDestroy(&tao->jacobian_inequality_pre);CHKERRQ(ierr);
    tao->jacobian_inequality_pre=Jpre;
  }
  PetscFunctionReturn(0);
}
