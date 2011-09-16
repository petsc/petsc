#include "include/private/taosolver_impl.h"

#undef __FUNCT__ 
#define __FUNCT__ "TaoSolverSetHessianRoutine"
/*@C
   TaoSolverSetHessianRoutine - Sets the function to compute the Hessian as well as the location to store the matrix.

   Collective on TaoSolver

   Input Parameters:
+  tao - the TaoSolver context
.  H - Matrix used for the hessian
.  Hpre - Matrix that will be used operated on by preconditioner, can be same as H
.  hess - Hessian evaluation routine
-  ctx - [optional] user-defined context for private data for the 
         Hessian evaluation routine (may be TAO_NULL)

   Calling sequence of hess:
$    hess (TaoSolver tao,Vec x,Mat *H,Mat *Hpre,MatStructure *flag,void *ctx);

+  tao - the TaoSolver  context
.  x - input vector
.  H - Hessian matrix
.  Hpre - preconditioner matrix, usually the same as H
.  flag - flag indicating information about the preconditioner matrix
   structure (see below)
-  ctx - [optional] user-defined Hessian context


   Notes: 

   The function hess() takes Mat * as the matrix arguments rather than Mat.  
   This allows the Hessian evaluation routine to replace A and/or B with a 
   completely new new matrix structure (not just different matrix elements)
   when appropriate, for instance, if the nonzero structure is changing
   throughout the global iterations.

   The flag can be used to eliminate unnecessary work in the preconditioner 
   during the repeated solution of linear systems of the same size.  The
   available options are
$    SAME_PRECONDITIONER -
$      Hpre is identical during successive linear solves.
$      This option is intended for folks who are using
$      different Amat and Pmat matrices and want to reuse the
$      same preconditioner matrix.  For example, this option
$      saves work by not recomputing incomplete factorization
$      for ILU/ICC preconditioners.
$    SAME_NONZERO_PATTERN -
$      Hpre has the same nonzero structure during
$      successive linear solves. 
$    DIFFERENT_NONZERO_PATTERN -
$      Hpre does not have the same nonzero structure.

   Caution:
   If you specify SAME_NONZERO_PATTERN, the software believes your assertion
   and does not check the structure of the matrix.  If you erroneously
   claim that the structure is the same when it actually is not, the new
   preconditioner will not function correctly.  Thus, use this optimization
   feature carefully!

   If in doubt about whether your preconditioner matrix has changed
   structure or not, use the flag DIFFERENT_NONZERO_PATTERN.

   Level: beginner

*/
PetscErrorCode TaoSolverSetHessianRoutine(TaoSolver tao, Mat H, Mat Hpre, PetscErrorCode (*func)(TaoSolver, Vec, Mat*, Mat *, MatStructure *, void*), void *ctx)
{
    PetscErrorCode ierr;
    PetscFunctionBegin;
    PetscValidHeaderSpecific(tao,TAOSOLVER_CLASSID,1);
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
	ierr = PetscObjectReference((PetscObject)H); CHKERRQ(ierr);
	if (tao->hessian) {   ierr = MatDestroy(&tao->hessian); CHKERRQ(ierr);}
	tao->hessian = H;
    }
    if (Hpre) {
	ierr = PetscObjectReference((PetscObject)Hpre); CHKERRQ(ierr);
	if (tao->hessian_pre) { ierr = MatDestroy(&tao->hessian_pre); CHKERRQ(ierr);}
	tao->hessian_pre=Hpre;
    }
    PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TaoSolverComputeHessian"
/*@C
   TaoComputeHessian - Computes the Hessian matrix that has been
   set with TaoSetHessian().

   Collective on TAO_SOLVER and Mat

   Input Parameters:
+  solver - the TAO_SOLVER solver context
-  xx - input vector

   Output Parameters:
+  H - Hessian matrix
.  Hpre - Preconditioning matrix
-  flag - flag indicating matrix structure (SAME_NONZERO_PATTERN, DIFFERENT_NONZERO_PATTERN, or SAME_PRECONDITIONER)

   Notes: 
   Most users should not need to explicitly call this routine, as it
   is used internally within the minimization solvers. 

   TaoComputeHessian() is typically used within minimization
   implementations, so most users would not generally call this routine
   themselves.

   Level: developer

.keywords: TAO_SOLVER, compute, Hessian, matrix

.seealso:  TaoSolverComputeObjective(), TaoSolverComputeObjectiveAndGradient(), TaoSolverSetHessian()

@*/
PetscErrorCode TaoSolverComputeHessian(TaoSolver tao, Vec X, Mat *H, Mat *Hpre, MatStructure *flg)
{
    PetscErrorCode ierr;
    PetscFunctionBegin;
    PetscValidHeaderSpecific(tao,TAOSOLVER_CLASSID,1);
    PetscValidHeaderSpecific(X, VEC_CLASSID,2);
    PetscValidPointer(flg,5);
    PetscCheckSameComm(tao,1,X,2);
    
    if (!tao->ops->computehessian) {
	SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Must call TaoSolverSetHessian() first");
    }
    *flg = DIFFERENT_NONZERO_PATTERN;
    ++tao->nhess;
    ierr = PetscLogEventBegin(TaoSolver_HessianEval,tao,X,*H,*Hpre); CHKERRQ(ierr);
    PetscStackPush("TaoSolver user Hessian function");
    CHKMEMQ;
    ierr = (*tao->ops->computehessian)(tao,X,H,Hpre,flg,tao->user_hessP); CHKERRQ(ierr);
    CHKMEMQ;
    PetscStackPop;
    ierr = PetscLogEventEnd(TaoSolver_HessianEval,tao,X,*H,*Hpre); CHKERRQ(ierr);
    
    PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TaoSolverComputeJacobian"
/*@C
   TaoSolverComputeJacobian - Computes the Jacobian matrix that has been
   set with TaoSolverSetJacobianRoutine().

   Collective on TAO_SOLVER and Mat

   Input Parameters:
+  solver - the TAO_SOLVER solver context
-  xx - input vector

   Output Parameters:
+  H - Jacobian matrix
.  Hpre - Preconditioning matrix
-  flag - flag indicating matrix structure (SAME_NONZERO_PATTERN, DIFFERENT_NONZERO_PATTERN, or SAME_PRECONDITIONER)

   Notes: 
   Most users should not need to explicitly call this routine, as it
   is used internally within the minimization solvers. 

   TaoComputeJacobian() is typically used within minimization
   implementations, so most users would not generally call this routine
   themselves.

   Level: developer

.keywords: TAO_SOLVER, compute, Jacobian, matrix

.seealso:  TaoSolverComputeObjective(), TaoSolverComputeObjectiveAndGradient(), TaoSolverSetJacobian()

@*/
PetscErrorCode TaoSolverComputeJacobian(TaoSolver tao, Vec X, Mat *J, Mat *Jpre, MatStructure *flg)
{
    PetscErrorCode ierr;
    PetscFunctionBegin;
    PetscValidHeaderSpecific(tao,TAOSOLVER_CLASSID,1);
    PetscValidHeaderSpecific(X, VEC_CLASSID,2);
    PetscValidPointer(flg,5);
    PetscCheckSameComm(tao,1,X,2);
    
    if (!tao->ops->computejacobian) {
	SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Must call TaoSolverSetJacobian() first");
    }
    *flg = DIFFERENT_NONZERO_PATTERN;
    ++tao->njac;
    ierr = PetscLogEventBegin(TaoSolver_JacobianEval,tao,X,*J,*Jpre); CHKERRQ(ierr);
    PetscStackPush("TaoSolver user Jacobian function");
    CHKMEMQ;
    ierr = (*tao->ops->computejacobian)(tao,X,J,Jpre,flg,tao->user_jacP); CHKERRQ(ierr);
    CHKMEMQ;
    PetscStackPop;
    ierr = PetscLogEventEnd(TaoSolver_JacobianEval,tao,X,*J,*Jpre); CHKERRQ(ierr);
    
    PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TaoSolverComputeJacobianState"
/*@C
   TaoSolverComputeJacobianState - Computes the Jacobian matrix that has been
   set with TaoSolverSetJacobianStateRoutine().

   Collective on TAO_SOLVER and Mat

   Input Parameters:
+  solver - the TAO_SOLVER solver context
-  xx - input vector

   Output Parameters:
+  H - Jacobian matrix
.  Hpre - Preconditioning matrix
-  flag - flag indicating matrix structure (SAME_NONZERO_PATTERN, DIFFERENT_NONZERO_PATTERN, or SAME_PRECONDITIONER)

   Notes: 
   Most users should not need to explicitly call this routine, as it
   is used internally within the minimization solvers. 

   TaoSolverComputeJacobianState() is typically used within minimization
   implementations, so most users would not generally call this routine
   themselves.

   Level: developer

.keywords: TAO_SOLVER, compute, Jacobian, matrix

.seealso:  TaoSolverComputeObjective(), TaoSolverComputeObjectiveAndGradient(), TaoSolverSetJacobianStateRoutine(), TaoSolverComputeJacobianDesign(), TaoSolverSetStateDesignIS()

@*/
PetscErrorCode TaoSolverComputeJacobianState(TaoSolver tao, Vec X, Mat *J, Mat *Jpre, Mat *Jinv, MatStructure *flg)
{
    PetscErrorCode ierr;
    PetscFunctionBegin;
    PetscValidHeaderSpecific(tao,TAOSOLVER_CLASSID,1);
    PetscValidHeaderSpecific(X, VEC_CLASSID,2);
    PetscValidPointer(flg,5);
    PetscCheckSameComm(tao,1,X,2);
    
    if (!tao->ops->computejacobianstate) {
	SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Must call TaoSolverSetJacobianState() first");
    }
    *flg = DIFFERENT_NONZERO_PATTERN;
    ++tao->njac_state;
    ierr = PetscLogEventBegin(TaoSolver_JacobianEval,tao,X,*J,*Jpre); CHKERRQ(ierr);
    PetscStackPush("TaoSolver user Jacobian(state) function");
    CHKMEMQ;
    ierr = (*tao->ops->computejacobianstate)(tao,X,J,Jpre,Jinv,flg,tao->user_jac_stateP); CHKERRQ(ierr);
    CHKMEMQ;
    PetscStackPop;
    ierr = PetscLogEventEnd(TaoSolver_JacobianEval,tao,X,*J,*Jpre); CHKERRQ(ierr);
    
    PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TaoSolverComputeJacobianDesign"
/*@C
   TaoSolverComputeJacobianDesign - Computes the Jacobian matrix that has been
   set with TaoSolverSetJacobianDesignRoutine().

   Collective on TAO_SOLVER and Mat

   Input Parameters:
+  solver - the TAO_SOLVER solver context
-  xx - input vector

   Output Parameters:
.  H - Jacobian matrix

   Notes: 
   Most users should not need to explicitly call this routine, as it
   is used internally within the minimization solvers. 

   TaoSolverComputeJacobianDesign() is typically used within minimization
   implementations, so most users would not generally call this routine
   themselves.

   Level: developer

.keywords: TAO_SOLVER, compute, Jacobian, matrix

.seealso:  TaoSolverComputeObjective(), TaoSolverComputeObjectiveAndGradient(), TaoSolverSetJacobianDesignRoutine(), TaoSolverComputeJacobianDesign(), TaoSolverSetDesignDesignIS()

@*/
PetscErrorCode TaoSolverComputeJacobianDesign(TaoSolver tao, Vec X, Mat *J)
{
    PetscErrorCode ierr;
    PetscFunctionBegin;
    PetscValidHeaderSpecific(tao,TAOSOLVER_CLASSID,1);
    PetscValidHeaderSpecific(X, VEC_CLASSID,2);
    PetscCheckSameComm(tao,1,X,2);
    
    if (!tao->ops->computejacobiandesign) {
	SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Must call TaoSolverSetJacobianDesign() first");
    }
    ++tao->njac_design;
    ierr = PetscLogEventBegin(TaoSolver_JacobianEval,tao,X,*J,PETSC_NULL); CHKERRQ(ierr);
    PetscStackPush("TaoSolver user Jacobian(design) function");
    CHKMEMQ;
    ierr = (*tao->ops->computejacobiandesign)(tao,X,J,tao->user_jac_designP); CHKERRQ(ierr);
    CHKMEMQ;
    PetscStackPop;
    ierr = PetscLogEventEnd(TaoSolver_JacobianEval,tao,X,*J,PETSC_NULL); CHKERRQ(ierr);
    
    PetscFunctionReturn(0);
}



#undef __FUNCT__ 
#define __FUNCT__ "TaoSolverSetJacobianRoutine"
/*@C
   TaoSolverSetJacobianRoutine - Sets the function to compute the Jacobian as well as the location to store the matrix.

   Collective on TaoSolver

   Input Parameters:
+  tao - the TaoSolver context
.  J - Matrix used for the jacobian
.  Jpre - Matrix that will be used operated on by preconditioner, can be same as J
.  jac - Jacobian evaluation routine
-  ctx - [optional] user-defined context for private data for the 
         Jacobian evaluation routine (may be TAO_NULL)

   Calling sequence of jac:
$    jac (TaoSolver tao,Vec x,Mat *J,Mat *Jpre,MatStructure *flag,void *ctx);

+  tao - the TaoSolver  context
.  x - input vector
.  J - Jacobian matrix
.  Jpre - preconditioner matrix, usually the same as J
.  flag - flag indicating information about the preconditioner matrix
   structure (see below)
-  ctx - [optional] user-defined Jacobian context


   Notes: 

   The function jac() takes Mat * as the matrix arguments rather than Mat.  
   This allows the Jacobian evaluation routine to replace A and/or B with a 
   completely new new matrix structure (not just different matrix elements)
   when appropriate, for instance, if the nonzero structure is changing
   throughout the global iterations.

   The flag can be used to eliminate unnecessary work in the preconditioner 
   during the repeated solution of linear systems of the same size.  The
   available options are
$    SAME_PRECONDITIONER -
$      Jpre is identical during successive linear solves.
$      This option is intended for folks who are using
$      different Amat and Pmat matrices and want to reuse the
$      same preconditioner matrix.  For example, this option
$      saves work by not recomputing incomplete factorization
$      for ILU/ICC preconditioners.
$    SAME_NONZERO_PATTERN -
$      Jpre has the same nonzero structure during
$      successive linear solves. 
$    DIFFERENT_NONZERO_PATTERN -
$      Jpre does not have the same nonzero structure.

   Caution:
   If you specify SAME_NONZERO_PATTERN, the software believes your assertion
   and does not check the structure of the matrix.  If you erroneously
   claim that the structure is the same when it actually is not, the new
   preconditioner will not function correctly.  Thus, use this optimization
   feature carefully!

   If in doubt about whether your preconditioner matrix has changed
   structure or not, use the flag DIFFERENT_NONZERO_PATTERN.

   Level: beginner

*/
PetscErrorCode TaoSolverSetJacobianRoutine(TaoSolver tao, Mat J, Mat Jpre, PetscErrorCode (*func)(TaoSolver, Vec, Mat*, Mat *, MatStructure *, void*), void *ctx)
{
    PetscErrorCode ierr;
    PetscFunctionBegin;
    PetscValidHeaderSpecific(tao,TAOSOLVER_CLASSID,1);
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
	ierr = PetscObjectReference((PetscObject)J); CHKERRQ(ierr);
	if (tao->jacobian) {   ierr = MatDestroy(&tao->jacobian); CHKERRQ(ierr);}
	tao->jacobian = J;
    }
    if (Jpre) {
	ierr = PetscObjectReference((PetscObject)Jpre); CHKERRQ(ierr);
	if (tao->jacobian_pre) { ierr = MatDestroy(&tao->jacobian_pre); CHKERRQ(ierr);}
	tao->jacobian_pre=Jpre;
    }
    PetscFunctionReturn(0);
}


#undef __FUNCT__ 
#define __FUNCT__ "TaoSolverSetJacobianStateRoutine"
/*@C
   TaoSolverSetJacobianStateRoutine - Sets the function to compute the Jacobian 
   (and its inverse) of the constraint function with respect to the state variables.
   Used only for pde-constrained optimization.

   Collective on TaoSolver

   Input Parameters:
+  tao - the TaoSolver context
.  J - Matrix used for the jacobian
.  Jpre - Matrix that will be used operated on by PETSc preconditioner, can be same as J.  Only used if Jinv is PETSC_NULL
.  Jinv - [optional but recommended] Matrix used to apply the inverse of the state jacobian. Use PETSC_NULL to default to PETSc KSP solvers to apply the inverse.
.  jac - Jacobian evaluation routine
-  ctx - [optional] user-defined context for private data for the 
         Jacobian evaluation routine (may be PETSC_NULL)

   Calling sequence of jac:
$    jac (TaoSolver tao,Vec x,Mat *J,Mat *Jpre,MatStructure *flag,void *ctx);

+  tao - the TaoSolver  context
.  x - input vector
.  J - Jacobian matrix
.  Jpre - preconditioner matrix, usually the same as J
.  Jinv - inverse of J
.  flag - flag indicating information about the preconditioner matrix
   structure (see below)
-  ctx - [optional] user-defined Jacobian context


   Notes:
   Because of the structure of the jacobian matrix, 
   It is usually much more efficient for a pde-constrained application to provide 
   its own Jinv matrix.

   The function jac() takes Mat * as the matrix arguments rather than Mat.  
   This allows the Jacobian evaluation routine to replace A and/or B with a 
   completely new new maitrix structure (not just different matrix elements)
   when appropriate, for instance, if the nonzero structure is changing
   throughout the global iterations.

   The flag can be used to eliminate unnecessary work in the preconditioner 
   during the repeated solution of linear systems of the same size.  The
   available options are
$    SAME_PRECONDITIONER -
$      Jpre is identical during successive linear solves.
$      This option is intended for folks who are using
$      different Amat and Pmat matrices and want to reuse the
$      same preconditioner matrix.  For example, this option
$      saves work by not recomputing incomplete factorization
$      for ILU/ICC preconditioners.
$    SAME_NONZERO_PATTERN -
$      Jpre has the same nonzero structure during
$      successive linear solves. 
$    DIFFERENT_NONZERO_PATTERN -
$      Jpre does not have the same nonzero structure.

   Caution:
   If you specify SAME_NONZERO_PATTERN, the software believes your assertion
   and does not check the structure of the matrix.  If you erroneously
   claim that the structure is the same when it actually is not, the new
   preconditioner will not function correctly.  Thus, use this optimization
   feature carefully!

   If in doubt about whether your preconditioner matrix has changed
   structure or not, use the flag DIFFERENT_NONZERO_PATTERN.

   Level: beginner
.seealse: TaoSolverComputeJacobianState(), TaoSolverSetJacobianDesignRoutine(), TaoSolverSetStateDesignIS()
*/
PetscErrorCode TaoSolverSetJacobianStateRoutine(TaoSolver tao, Mat J, Mat Jpre, Mat Jinv, PetscErrorCode (*func)(TaoSolver, Vec, Mat*, Mat *, Mat *, MatStructure *, void*), void *ctx)
{
    PetscErrorCode ierr;
    PetscFunctionBegin;
    PetscValidHeaderSpecific(tao,TAOSOLVER_CLASSID,1);
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
	ierr = PetscObjectReference((PetscObject)J); CHKERRQ(ierr);
	if (tao->jacobian_state) {   ierr = MatDestroy(&tao->jacobian_state); CHKERRQ(ierr);}
	tao->jacobian_state = J;
    }
    if (Jpre) {
	ierr = PetscObjectReference((PetscObject)Jpre); CHKERRQ(ierr);
	if (tao->jacobian_state_pre) { ierr = MatDestroy(&tao->jacobian_state_pre); CHKERRQ(ierr);}
	tao->jacobian_state_pre=Jpre;
    }
    if (Jinv) {
      ierr = PetscObjectReference((PetscObject)Jinv); CHKERRQ(ierr);
      if (tao->jacobian_state_inv) {ierr = MatDestroy(&tao->jacobian_state_inv); CHKERRQ(ierr);}
      tao->jacobian_state_inv=Jinv;
    }
    PetscFunctionReturn(0);
}

#undef __FUNCT__ 
#define __FUNCT__ "TaoSolverSetJacobianDesignRoutine"
/*@C
   TaoSolverSetJacobianDesignRoutine - Sets the function to compute the Jacobian of 
   the constraint function with respect to the design variables.  Used only for 
   pde-constrained optimization.

   Collective on TaoSolver

   Input Parameters:
+  tao - the TaoSolver context
.  J - Matrix used for the jacobian
.  jac - Jacobian evaluation routine
-  ctx - [optional] user-defined context for private data for the 
         Jacobian evaluation routine (may be TAO_NULL)

   Calling sequence of jac:
$    jac (TaoSolver tao,Vec x,Mat *J,void *ctx);

+  tao - the TaoSolver  context
.  x - input vector
.  J - Jacobian matrix
-  ctx - [optional] user-defined Jacobian context


   Notes: 

   The function jac() takes Mat * as the matrix arguments rather than Mat.  
   This allows the Jacobian evaluation routine to replace A and/or B with a 
   completely new new matrix structure (not just different matrix elements)
   when appropriate, for instance, if the nonzero structure is changing
   throughout the global iterations.

   Level: beginner
.seealso: TaoSolverComputeJacobianDesign(), TaoSolverSetJacobianStateRoutine(), TaoSolverSetStateDesignIS()
*/
PetscErrorCode TaoSolverSetJacobianDesignRoutine(TaoSolver tao, Mat J, PetscErrorCode (*func)(TaoSolver, Vec, Mat*, void*), void *ctx)
{
    PetscErrorCode ierr;
    PetscFunctionBegin;
    PetscValidHeaderSpecific(tao,TAOSOLVER_CLASSID,1);
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
	ierr = PetscObjectReference((PetscObject)J); CHKERRQ(ierr);
	if (tao->jacobian_design) {   ierr = MatDestroy(&tao->jacobian_design); CHKERRQ(ierr);}
	tao->jacobian_design = J;
    }
    PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TaoSolverSetStateDesignIS"
/*@
   TaoSolverSetStateDesignIS - Indicate to the TaoSolver which variables in the 
   solution vector are state variables and which are design.  Only applies to
   pde-constrained optimization.

   Collective on TaoSolver

   Input Parameters:
+  tao - The TaoSolver context
.  s_is - the index set corresponding to the state variables
-  d_is - the index set corresponding to the design variables
  
   Level: intermediate

.seealso: TaoSolverSetJacobianStateRoutine(), TaoSolverSetJacobianDesignRoutine()
*/
PetscErrorCode TaoSolverSetStateDesignIS(TaoSolver tao, IS s_is, IS d_is)
{
  PetscErrorCode ierr;
  if (tao->state_is) {
    ierr = PetscObjectDereference((PetscObject)(tao->state_is)); CHKERRQ(ierr);
  }
  if (tao->design_is) {
    ierr = PetscObjectDereference((PetscObject)(tao->design_is)); CHKERRQ(ierr);
  }
  tao->state_is = s_is;
  tao->design_is = d_is;
  if (s_is) {
    ierr = PetscObjectReference((PetscObject)(tao->state_is)); CHKERRQ(ierr);
  }
  if (d_is) {
    ierr = PetscObjectReference((PetscObject)(tao->design_is)); CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}
