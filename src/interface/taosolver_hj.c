#include "include/private/taosolver_impl.h"

#undef __FUNCT__ 
#define __FUNCT__ "TaoSolverSetHessian"
PetscErrorCode TAOSOLVER_DLLEXPORT TaoSolverSetHessian(TaoSolver tao, Mat H, Mat Hpre, PetscErrorCode (*func)(TaoSolver, Vec, Mat*, Mat *, MatStructure *, void*), void *ctx)
{
    PetscErrorCode ierr;
    PetscFunctionBegin;
    PetscValidHeaderSpecific(tao,TAOSOLVER_COOKIE,1);
    if (H) {
	PetscValidHeaderSpecific(H,MAT_COOKIE,2);
	PetscCheckSameComm(tao,1,H,2);
    }
    if (Hpre) {
	PetscValidHeaderSpecific(Hpre,MAT_COOKIE,3);
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
	if (tao->hessian) {   ierr = MatDestroy(tao->hessian); CHKERRQ(ierr);}
	tao->hessian = H;
    }
    if (Hpre) {
	ierr = PetscObjectReference((PetscObject)Hpre); CHKERRQ(ierr);
	if (tao->hessian_pre) { ierr = MatDestroy(tao->hessian_pre); CHKERRQ(ierr);}
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
PetscErrorCode TAOSOLVER_DLLEXPORT TaoSolverComputeHessian(TaoSolver tao, Vec X, Mat *H, Mat *Hpre, MatStructure *flg)
{
    PetscErrorCode ierr;
    PetscTruth flag;
    PetscFunctionBegin;
    PetscValidHeaderSpecific(tao,TAOSOLVER_COOKIE,1);
    PetscValidHeaderSpecific(X, VEC_COOKIE,2);
    PetscValidPointer(flg,5);
    PetscCheckSameComm(tao,1,X,2);
    
    if (!tao->ops->computehessian) {
	SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Must call TaoSolverSetHessian() first");
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


