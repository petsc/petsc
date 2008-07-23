#include "include/private/taosolver_impl.h"

#undef __FUNCT__
#define __FUNCT__ "TaoSolverSetInitialVector"
PetscErrorCode TAOSOLVER_DLLEXPORT TaoSolverSetInitialVector(TaoSolver tao, Vec x0) {
    PetscErrorCode ierr;

    PetscFunctionBegin;
    PetscValidHeaderSpecific(tao,TAOSOLVER_COOKIE,1);
    if (x0) {
	PetscValidHeaderSpecific(x0,VEC_COOKIE,2);
	PetscObjectReference((PetscObject)x0);
    }
    if (tao->solution) {
	ierr = VecDestroy(tao->solution); CHKERRQ(ierr);
    }
    tao->solution = x0;
    PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TaoSolverComputeGradient"
/*@
  TaoComputeGradient - Computes the gradient of the objective function

  Collective on TaoSolver

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
	ierr = (*tao->ops->computegradient)(tao,X,G,tao->user_gradP); CHKERRQ(ierr);
	CHKMEMQ;
	PetscStackPop;
	ierr = PetscLogEventEnd(TaoSolver_GradientEval,tao,X,G,PETSC_NULL); CHKERRQ(ierr);
	tao->ngrads++;
    } else if (tao->ops->computeobjectiveandgradient) {
	ierr = PetscLogEventBegin(TaoSolver_ObjGradientEval,tao,X,G,PETSC_NULL); CHKERRQ(ierr);
	PetscStackPush("TaoSolver user objective/gradient evaluation routine");
	CHKMEMQ;
	ierr = (*tao->ops->computeobjectiveandgradient)(tao,X,&dummy,G,tao->user_objgradP); CHKERRQ(ierr);
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
PetscErrorCode TAOSOLVER_DLLEXPORT TaoSolverComputeObjective(TaoSolver tao, Vec X, PetscReal *f) 
{
    PetscErrorCode ierr;
    Vec temp;
    PetscFunctionBegin;
    PetscValidHeaderSpecific(tao,TAOSOLVER_COOKIE,1);
    PetscValidHeaderSpecific(X,VEC_COOKIE,2);
    PetscCheckSameComm(tao,1,X,2);
    if (tao->ops->computeobjective) {
	ierr = PetscLogEventBegin(TaoSolver_ObjectiveEval,tao,X,PETSC_NULL,PETSC_NULL); CHKERRQ(ierr);
	PetscStackPush("TaoSolver user objective evaluation routine");
	CHKMEMQ;
	ierr = (*tao->ops->computeobjective)(tao,X,f,tao->user_objP); CHKERRQ(ierr);
	CHKMEMQ;
	PetscStackPop;
	ierr = PetscLogEventEnd(TaoSolver_ObjectiveEval,tao,X,PETSC_NULL,PETSC_NULL); CHKERRQ(ierr);
	tao->nfuncs++;
    } else if (tao->ops->computeobjectiveandgradient) {
	ierr = PetscInfo(tao,"Duplicating variable vector in order to call func/grad routine"); CHKERRQ(ierr);
	ierr = VecDuplicate(X,&temp); CHKERRQ(ierr);
	ierr = PetscLogEventBegin(TaoSolver_ObjGradientEval,tao,X,PETSC_NULL,PETSC_NULL); CHKERRQ(ierr);
	PetscStackPush("TaoSolver user objective/gradient evaluation routine");
	CHKMEMQ;
	ierr = (*tao->ops->computeobjectiveandgradient)(tao,X,f,temp,tao->user_objgradP); CHKERRQ(ierr);
	CHKMEMQ;
	PetscStackPop;
	ierr = PetscLogEventEnd(TaoSolver_ObjGradientEval,tao,X,PETSC_NULL,PETSC_NULL); CHKERRQ(ierr);
	ierr = VecDestroy(temp); CHKERRQ(ierr);
	tao->nfuncgrads++;

    }  else {
	SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"TaoSolverSetObjective() has not been called");
    }
    PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TaoSolverComputeObjectiveAndGradient"
PetscErrorCode TAOSOLVER_DLLEXPORT TaoSolverComputeObjectiveAndGradient(TaoSolver tao, Vec X, PetscReal *f, Vec G)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao,TAOSOLVER_COOKIE,1);
  PetscValidHeaderSpecific(tao,VEC_COOKIE,2);
  PetscValidHeaderSpecific(tao,VEC_COOKIE,4);
  PetscCheckSameComm(tao,1,X,2);
  PetscCheckSameComm(tao,1,G,4);
  if (tao->ops->computeobjectiveandgradient) {
      ierr = PetscLogEventBegin(TaoSolver_ObjGradientEval,tao,X,G,PETSC_NULL); CHKERRQ(ierr);
      PetscStackPush("TaoSolver user objective/gradient evaluation routine");
      CHKMEMQ;
      ierr = (*tao->ops->computeobjectiveandgradient)(tao,X,f,G,tao->user_objgradP); CHKERRQ(ierr);
      CHKMEMQ;
      PetscStackPop;
      ierr = PetscLogEventEnd(TaoSolver_ObjGradientEval,tao,X,G,PETSC_NULL); CHKERRQ(ierr);
      tao->nfuncgrads++;
  } else if (tao->ops->computeobjective && tao->ops->computegradient) {
      ierr = PetscLogEventBegin(TaoSolver_ObjectiveEval,tao,X,PETSC_NULL,PETSC_NULL); CHKERRQ(ierr);
      PetscStackPush("TaoSolver user objective evaluation routine");
      CHKMEMQ;
      ierr = (*tao->ops->computeobjective)(tao,X,f,tao->user_objP); CHKERRQ(ierr);
      CHKMEMQ;
      PetscStackPop;
      ierr = PetscLogEventEnd(TaoSolver_ObjectiveEval,tao,X,PETSC_NULL,PETSC_NULL); CHKERRQ(ierr);
      tao->nfuncs++;
      
      ierr = PetscLogEventBegin(TaoSolver_GradientEval,tao,X,G,PETSC_NULL); CHKERRQ(ierr);
      PetscStackPush("TaoSolver user gradient evaluation routine");
      CHKMEMQ;
      ierr = (*tao->ops->computegradient)(tao,X,G,tao->user_gradP); CHKERRQ(ierr);
      CHKMEMQ;
      PetscStackPop;
      ierr = PetscLogEventEnd(TaoSolver_GradientEval,tao,X,G,PETSC_NULL); CHKERRQ(ierr);
      tao->ngrads++;
  } else {
      SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"TaoSolverSetObjective() or TaoSolverSetGradient() not set");
  }
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
    tao->user_objP = ctx;
    tao->ops->computeobjective = func;
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
    tao->user_gradP = ctx;
    tao->ops->computegradient = func;
    PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "TaoSolverSetObjectiveAndGradient"
PetscErrorCode TAOSOLVER_DLLEXPORT TaoSolverSetObjectiveAndGradient(TaoSolver tao, PetscErrorCode (*func)(TaoSolver, Vec, PetscReal *, Vec, void*), void *ctx)
{
    PetscFunctionBegin;
    PetscValidHeaderSpecific(tao,TAOSOLVER_COOKIE,1);
    tao->user_objgradP = ctx;
    tao->ops->computeobjectiveandgradient = func;
    PetscFunctionReturn(0);
}
  
