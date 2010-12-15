#include "include/private/taosolver_impl.h" /*I "taosolver.h" I*/

#undef __FUNCT__
#define __FUNCT__ "TaoSolverSetInitialVector"
/*@
  TaoSolverSetInitialVector - Sets the initial guess for the solve

  Collective on TaoSolver
  
  Input Parameters:
+ tao - the TaoSolver context
- x0  - the initial guess 
 
  Level: beginner
.seealso: TaoSolverCreate(), TaoSolverSolve()
@*/

PetscErrorCode TaoSolverSetInitialVector(TaoSolver tao, Vec x0) {
    PetscErrorCode ierr;

    PetscFunctionBegin;
    PetscValidHeaderSpecific(tao,TAOSOLVER_CLASSID,1);
    if (x0) {
	PetscValidHeaderSpecific(x0,VEC_CLASSID,2);
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
  TaoSolverComputeGradient - Computes the gradient of the objective function

  Collective on TaoSolver

  Input Parameters:
+ tao - the TaoSolver context
- X - input vector

  Output Parameter:
. G - gradient vector  

  Notes: TaoComputeGradient() is typically used within minimization implementations,
  so most users would not generally call this routine themselves.

  Level: advanced

.seealso: TaoSolverComputeObjective(), TaoSolverComputeObjectiveAndGradient(), TaoSolverSetGradientRoutine()
@*/
PetscErrorCode TaoSolverComputeGradient(TaoSolver tao, Vec X, Vec G) 
{
    PetscErrorCode ierr;
    PetscReal dummy;
    PetscFunctionBegin;
    PetscValidHeaderSpecific(tao,TAOSOLVER_CLASSID,1);
    PetscValidHeaderSpecific(X,VEC_CLASSID,2);
    PetscValidHeaderSpecific(G,VEC_CLASSID,2);
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
	SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"TaoSolverSetGradientRoutine() has not been called");
    }
    PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "TaoSolverComputeObjective"
/*@
  TaoSolverComputeObjective - Computes the objective function value at a given point

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
PetscErrorCode TaoSolverComputeObjective(TaoSolver tao, Vec X, PetscReal *f) 
{
    PetscErrorCode ierr;
    Vec temp;
    PetscFunctionBegin;
    PetscValidHeaderSpecific(tao,TAOSOLVER_CLASSID,1);
    PetscValidHeaderSpecific(X,VEC_CLASSID,2);
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
	SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"TaoSolverSetObjectiveRoutine() has not been called");
    }
    ierr = PetscInfo1(tao,"TAO Function evaluation: %14.12e\n",*f);CHKERRQ(ierr);    
    PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TaoSolverComputeObjectiveAndGradient"
/*@
  TaoSolverComputeObjectiveAndGradient - Computes the objective function value at a given point

  Collective on TaoSOlver

  Input Parameters:
+ tao - the TaoSolver context
- X - input vector

  Output Parameter:
+ f - Objective value at X
- g - Gradient vector at X

  Notes: TaoSolverComputeObjectiveAndGradient() is typically used within minimization implementations,
  so most users would not generally call this routine themselves.

  Level: advanced

.seealso: TaoSolverComputeGradient(), TaoSolverComputeObjectiveAndGradient(), TaoSolverSetObjectiveRoutine()
@*/
PetscErrorCode TaoSolverComputeObjectiveAndGradient(TaoSolver tao, Vec X, PetscReal *f, Vec G)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao,TAOSOLVER_CLASSID,1);
  PetscValidHeaderSpecific(X,VEC_CLASSID,2);
  PetscValidHeaderSpecific(G,VEC_CLASSID,4);
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
      SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"TaoSolverSetObjectiveRoutine() or TaoSolverSetGradientRoutine() not set");
  }
  ierr = PetscInfo1(tao,"TAO Function evaluation: %14.12e\n",*f);CHKERRQ(ierr);
  PetscFunctionReturn(0);
} 

#undef __FUNCT__
#define __FUNCT__ "TaoSolverSetObjectiveRoutine"
/*@C
  TaoSolverSetObjectiveRoutine - Sets the function evaluation routine for minimization

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

.seealso: TaoSolverSetGradientRoutine(), TaoSolverSetHessianRoutine() TaoSolverSetObjectiveAndGradientRoutine()
@*/
PetscErrorCode TaoSolverSetObjectiveRoutine(TaoSolver tao, PetscErrorCode (*func)(TaoSolver, Vec, PetscReal*,void*),void *ctx) 
{
    PetscFunctionBegin;
    PetscValidHeaderSpecific(tao,TAOSOLVER_CLASSID,1);
    tao->user_objP = ctx;
    tao->ops->computeobjective = func;
    PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TaoSolverSetSeparableObjectiveRoutine"
/*@C
  TaoSolverSetSeparableObjectiveRoutine - Sets the function evaluation routine for least-square applications

  Collective on TaoSolver

  Input Parameter:
+ tao - the TaoSolver context
. func - the objective function evaluation routine
- ctx - [optional] user-defined context for private data for the function evaluation
        routine (may be PETSC_NULL)

  Calling sequence of func:
$      func (TaoSolver tao, Vec x, Vec f, void *ctx);

+ x - input vector
. f - function value vector 
- ctx - [optional] user-defined function context

  Level: beginner

.seealso: TaoSolverSetObjectiveRoutine(), TaoSolverSetJacobianRoutine()
@*/
PetscErrorCode TaoSolverSetSeparableObjectiveRoutine(TaoSolver tao, Vec sepobj, PetscErrorCode (*func)(TaoSolver, Vec, Vec, void*),void *ctx)
{
    PetscFunctionBegin;
    PetscValidHeaderSpecific(tao,TAOSOLVER_CLASSID,1);
    PetscValidHeaderSpecific(sepobj, VEC_CLASSID,2);
    tao->user_sepobjP = ctx;
    tao->sep_objective = sepobj;
    tao->ops->computeseparableobjective = func;
    PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TaoSolverComputeSeparableObjective"
/*@
  TaoSolverComputeSeparableObjective - Computes an objective function vector at a given point

  Collective on TaoSolver

  Input Parameters:
+ tao - the TaoSolver context
- X - input vector

  Output Parameter:
. f - Objective vector at X

  Notes: TaoComputeSeparableObjective() is typically used within minimization implementations,
  so most users would not generally call this routine themselves.

  Level: advanced

.seealso: TaoSolverSetSeparableObjective()
@*/
PetscErrorCode TaoSolverComputeSeparableObjective(TaoSolver tao, Vec X, Vec F) 
{
    PetscErrorCode ierr;
    PetscFunctionBegin;
    PetscValidHeaderSpecific(tao,TAOSOLVER_CLASSID,1);
    PetscValidHeaderSpecific(X,VEC_CLASSID,2);
    PetscValidHeaderSpecific(F,VEC_CLASSID,3);
    PetscCheckSameComm(tao,1,X,2);
    PetscCheckSameComm(tao,1,F,3);
    if (tao->ops->computeseparableobjective) {
	ierr = PetscLogEventBegin(TaoSolver_ObjectiveEval,tao,X,PETSC_NULL,PETSC_NULL); CHKERRQ(ierr);
	PetscStackPush("TaoSolver user separable objective evaluation routine");
	CHKMEMQ;
	ierr = (*tao->ops->computeseparableobjective)(tao,X,F,tao->user_sepobjP); CHKERRQ(ierr);
	CHKMEMQ;
	PetscStackPop;
	ierr = PetscLogEventEnd(TaoSolver_ObjectiveEval,tao,X,PETSC_NULL,PETSC_NULL); CHKERRQ(ierr);
	tao->nfuncs++;
    } else {
	SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"TaoSolverSetSeparableObjectiveRoutine() has not been called");
    }
    ierr = PetscInfo(tao,"TAO separable function evaluation.\n"); CHKERRQ(ierr);
    PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TaoSolverSetGradientRoutine"
/*@C
  TaoSolverSetGradientRoutine - Sets the gradient evaluation routine for minimization

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

.seealso: TaoSolverSetObjectiveRoutine(), TaoSolverSetHessianRoutine() TaoSolverSetObjectiveAndGradientRoutine()
@*/
PetscErrorCode TaoSolverSetGradientRoutine(TaoSolver tao,  PetscErrorCode (*func)(TaoSolver, Vec, Vec, void*),void *ctx) 
{
    PetscFunctionBegin;
    PetscValidHeaderSpecific(tao,TAOSOLVER_CLASSID,1);
    tao->user_gradP = ctx;
    tao->ops->computegradient = func;
    PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "TaoSolverSetObjectiveAndGradientRoutine"
/*@C
  TaoSolverSetObjectiveAndGradientRoutine - Sets the gradient evaluation routine for minimization

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

.seealso: TaoSolverSetObjectiveRoutine(), TaoSolverSetHessianRoutine() TaoSolverSetObjectiveAndGradientRoutine()
@*/
PetscErrorCode TaoSolverSetObjectiveAndGradientRoutine(TaoSolver tao, PetscErrorCode (*func)(TaoSolver, Vec, PetscReal *, Vec, void*), void *ctx)
{
    PetscFunctionBegin;
    PetscValidHeaderSpecific(tao,TAOSOLVER_CLASSID,1);
    tao->user_objgradP = ctx;
    tao->ops->computeobjectiveandgradient = func;
    PetscFunctionReturn(0);
}
  
