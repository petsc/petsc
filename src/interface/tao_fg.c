#include "taosolver_impl.h"


#undef __FUNCT__
#define __FUNCT__ "TaoSolverComputeGradient"
/*@
  TaoComputeGradient - Computes the gradient of the objective function

  Collective on TaoSOlver

  Input Parameters:
+ tao - the TaoSolver context
- X - input vector

  Output Parameter:
. G - gradient vector  

  Notes: TaoComputeGradient() is typically used within minimization implementations,
  so most users would not generally call this routine themselves.

  Level: advanced

.seealso: TaoSolverComputeObjective(), TaoSolverComputeObjectiveAndGradient(), TaoSolverSetGradient()
@*/
PetscErrorCode TAOSOLVER_DLLEXPORT TaoSolverComputeGradient(TaoSolver tao, Vec X, Vec G) 
{
    PetscErrorCode ierr;
    PetscReal dummy;
    PetscFunctionBegin;
    PetscValidHeaderSpecific(tao,TAOSOLVER_COOKIE,1);
    PetscValidHeaderSpecific(X,VEC_COOKIE,2);
    PetscValidHeaderSpecific(G,VEC_COOKIE,2);
    PetscCheckSameComm(tao,1,X,2);
    PetscCheckSameComm(tao,1,G,3);
    if (tao->ops->computegradient) {
	ierr = PetscLogEventBegin(TaoSolver_GradientEval,tao,X,G,PETSC_NULL); CHKERRQ(ierr);
	PetscStackPush("TaoSolver user gradient evaluation routine");
	CHKMEMQ;
	ierr = (*tao->ops->computegradient)(tao,X,G,tao->user_grad); CHKERRQ(ierr);
	CHKMEMQ;
	PetscStackPop;
	ierr = PetscLogEventEnd(TaoSolver_GradientEval,tao,X,G,PETSC_NULL); CHKERRQ(ierr);
	tao->ngrads++;
    } else if (tao->ops->computeobjectiveandgradient) {
	ierr = PetscLogEventBegin(TaoSolver_ObjGradientEval,tao,X,G,PETSC_NULL); CHKERRQ(ierr);
	PetscStackPush("TaoSolver user objective/gradient evaluation routine");
	CHKMEMQ;
	ierr = (*tao->ops->computeobjectiveandgradient)(tao,X,&dummy,G,tao->user_grad); CHKERRQ(ierr);
	CHKMEMQ;
	PetscStackPop;
	ierr = PetscLogEventEnd(TaoSolver_ObjGradientEval,tao,X,G,PETSC_NULL); CHKERRQ(ierr);
	tao->nfuncgrads++;
    }  else {
	SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"TaoSolverSetGradient() has not been called");
    }
    PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "TaoSolverComputeObjective"
/*@
  TaoComputeObjective - Computes the objective function value at a given point

  Collective on TaoSOlver

  Input Parameters:
+ tao - the TaoSolver context
- X - input vector

  Output Parameter:
. f - Objective value at X

  Notes: TaoComputeObjective() is typically used within minimization implementations,
  so most users would not generally call this routine themselves.

  Level: advanced

.seealso: TaoSolverComputeGradient(), TaoSolverComputeObjectiveAndGradient(), TaoSolverSetObjective()
@*/
PetscErrorCode TAOSOLVER_DLLEXPORT TaoSolverComputeGradient(TaoSolver tao, X, PetscReal *f) 
{
    PetscErrorCode ierr;
    PetscFunctionBegin;
    PetscValidHeaderSpecific(tao,TAOSOLVER_COOKIE,1);
    PetscValidHeaderSpecific(X,VEC_COOKIE,2);
    PetscCheckSameComm(tao,1,X,2);
    if (tao->ops->computeobjective) {
	ierr = PetscLogEventBegin(TaoSolver_GradientEval,tao,X,PETSC_NULL,PETSC_NULL); CHKERRQ(ierr);
	PetscStackPush("TaoSolver user objective evaluation routine");
	CHKMEMQ;
	ierr = (*tao->ops->computeobjective)(tao,X,G,tao->user_obj); CHKERRQ(ierr);
	CHKMEMQ;
	PetscStackPop;
	ierr = PetscLogEventEnd(TaoSolver_GradientEval,tao,X,PETSC_NULL,PETSC_NULL); CHKERRQ(ierr);
	tao->nfuncs++;
    } else if (tao->ops->computeobjectiveandgradient) {
	/*
	ierr = PetscLogEventBegin(TaoSolver_ObjGradientEval,tao,X,G,PETSC_NULL); CHKERRQ(ierr);
	PetscStackPush("TaoSolver user objective/gradient evaluation routine");
	CHKMEMQ;
	ierr = (*tao->ops->computeobjectiveandgradient)(tao,X,&dummy,G,tao->user_grad); CHKERRQ(ierr);
	CHKMEMQ;
	PetscStackPop;
	ierr = PetscLogEventEnd(TaoSolver_ObjGradientEval,tao,X,G,PETSC_NULL); CHKERRQ(ierr);
	tao->nfuncgrads++;
	*/
    }  else {
	SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"TaoSolverSetObjective() has not been called");
    }
    PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "TaoSolverSetObjective"
/*@
  TaoSolverSetObjective - Sets the function evaluation routine for minimization

  Collective on TaoSolver

  Input Parameter:
+ tao - the TaoSolver context
. func - the objective function
- ctx - [optional] user-defined context for private data for the function evaluation
        routine (may be PETSC_NULL)

  Calling sequence of func:
$      func (TaoSolver tao, Vec x, PetscReal *f, void *ctx);

+ x - input vector
. f - function value
- ctx - [optional] user-defined function context

  Level: beginner

.seealso: TaoSolverSetGradient(), TaoSolverSetHessian() TaoSolverSetObjectiveAndGradient()
@*/
PetscErrorCode TAOSOLVER_DLLEXPORT TaoSolverSetObjective(TaoSolver tao, PetscErrorCode (*func)(TaoSolver, Vec, PetscReal*,void*),void *ctx) 
{
    PetscFunctionBegin;
    PetscValidHeaderSpecific(tao,TAOSOLVER_COOKIE,1);
    tao->user_obj = ctx;
    tao->computeobjective = func;
    PetscFunctionReturn(0);
}



#undef __FUNCT__
#define __FUNCT__ "TaoSolverSetGradient"
/*@
  TaoSolverSetGradient - Sets the gradient evaluation routine for minimization

  Collective on TaoSolver

  Input Parameter:
+ tao - the TaoSolver context
. func - the gradient function
- ctx - [optional] user-defined context for private data for the gradient evaluation
        routine (may be PETSC_NULL)

  Calling sequence of func:
$      func (TaoSolver tao, Vec x, Vec g, void *ctx);

+ x - input vector
. g - gradient value (output)
- ctx - [optional] user-defined function context

  Level: beginner

.seealso: TaoSolverSetObjective(), TaoSolverSetHessian() TaoSolverSetObjectiveAndGradient()
@*/
PetscErrorCode TAOSOLVER_DLLEXPORT TaoSolverSetGradient(TaoSolver tao,  PetscErrorCode (*func)(TaoSolver, Vec, Vec, void*),void *ctx) 
{
    PetscFunctionBegin;
    PetscValidHeaderSpecific(tao,TAOSOLVER_COOKIE,1);
    tao->user_grad = ctx;
    tao->computegradient = func;
    PetscFunctionReturn(0);
}




  
