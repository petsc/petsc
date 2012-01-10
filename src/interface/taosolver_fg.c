#include "include/private/taosolver_impl.h" /*I "taosolver.h" I*/

#undef __FUNCT__
#define __FUNCT__ "TaoSetInitialVector"
/*@
  TaoSetInitialVector - Sets the initial guess for the solve

  Logically collective on TaoSolver
  
  Input Parameters:
+ tao - the TaoSolver context
- x0  - the initial guess 
 
  Level: beginner
.seealso: TaoCreate(), TaoSolve()
@*/

PetscErrorCode TaoSetInitialVector(TaoSolver tao, Vec x0) {
    PetscErrorCode ierr;

    PetscFunctionBegin;
    PetscValidHeaderSpecific(tao,TAOSOLVER_CLASSID,1);
    if (x0) {
	PetscValidHeaderSpecific(x0,VEC_CLASSID,2);
	PetscObjectReference((PetscObject)x0);
    }
    if (tao->solution) {
	ierr = VecDestroy(&tao->solution); CHKERRQ(ierr);
    }
    tao->solution = x0;
    PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TaoComputeGradient"
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

.seealso: TaoComputeObjective(), TaoComputeObjectiveAndGradient(), TaoSetGradientRoutine()
@*/
PetscErrorCode TaoComputeGradient(TaoSolver tao, Vec X, Vec G) 
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
	PetscStackPush("Tao user objective/gradient evaluation routine");
	CHKMEMQ;
	ierr = (*tao->ops->computeobjectiveandgradient)(tao,X,&dummy,G,tao->user_objgradP); CHKERRQ(ierr);
	CHKMEMQ;
	PetscStackPop;
	ierr = PetscLogEventEnd(TaoSolver_ObjGradientEval,tao,X,G,PETSC_NULL); CHKERRQ(ierr);
	tao->nfuncgrads++;
    }  else {
	SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"TaoSetGradientRoutine() has not been called");
    }
    PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "TaoComputeObjective"
/*@
  TaoComputeObjective - Computes the objective function value at a given point

  Collective on TaoSolver

  Input Parameters:
+ tao - the TaoSolver context
- X - input vector

  Output Parameter:
. f - Objective value at X

  Notes: TaoComputeObjective() is typically used within minimization implementations,
  so most users would not generally call this routine themselves.

  Level: advanced

.seealso: TaoComputeGradient(), TaoComputeObjectiveAndGradient(), TaoSetObjectiveRoutine()
@*/
PetscErrorCode TaoComputeObjective(TaoSolver tao, Vec X, PetscReal *f) 
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
	ierr = VecDestroy(&temp); CHKERRQ(ierr);
	tao->nfuncgrads++;

    }  else {
	SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"TaoSetObjectiveRoutine() has not been called");
    }
    ierr = PetscInfo1(tao,"TAO Function evaluation: %14.12e\n",*f);CHKERRQ(ierr);    
    PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TaoComputeObjectiveAndGradient"
/*@
  TaoComputeObjectiveAndGradient - Computes the objective function value at a given point

  Collective on TaoSolver

  Input Parameters:
+ tao - the TaoSolver context
- X - input vector

  Output Parameter:
+ f - Objective value at X
- g - Gradient vector at X

  Notes: TaoComputeObjectiveAndGradient() is typically used within minimization implementations,
  so most users would not generally call this routine themselves.

  Level: advanced

.seealso: TaoComputeGradient(), TaoComputeObjectiveAndGradient(), TaoSetObjectiveRoutine()
@*/
PetscErrorCode TaoComputeObjectiveAndGradient(TaoSolver tao, Vec X, PetscReal *f, Vec G)
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
      if (tao->ops->computegradient == TaoDefaultComputeGradient) {
	/* Overwrite gradient with finite difference gradient */
	ierr = TaoDefaultComputeGradient(tao,X,G,tao->user_objgradP); CHKERRQ(ierr);
      }
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
      SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"TaoSetObjectiveRoutine() or TaoSetGradientRoutine() not set");
  }
  ierr = PetscInfo1(tao,"TAO Function evaluation: %14.12e\n",*f);CHKERRQ(ierr);
  PetscFunctionReturn(0);
} 

#undef __FUNCT__
#define __FUNCT__ "TaoSetObjectiveRoutine"
/*@C
  TaoSetObjectiveRoutine - Sets the function evaluation routine for minimization

  Logically collective on TaoSolver

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

.seealso: TaoSetGradientRoutine(), TaoSetHessianRoutine() TaoSetObjectiveAndGradientRoutine()
@*/
PetscErrorCode TaoSetObjectiveRoutine(TaoSolver tao, PetscErrorCode (*func)(TaoSolver, Vec, PetscReal*,void*),void *ctx) 
{
    PetscFunctionBegin;
    PetscValidHeaderSpecific(tao,TAOSOLVER_CLASSID,1);
    tao->user_objP = ctx;
    tao->ops->computeobjective = func;
    PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TaoSetSeparableObjectiveRoutine"
/*@C
  TaoSetSeparableObjectiveRoutine - Sets the function evaluation routine for least-square applications

  Logically collective on TaoSolver

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

.seealso: TaoSetObjectiveRoutine(), TaoSetJacobianRoutine()
@*/
PetscErrorCode TaoSetSeparableObjectiveRoutine(TaoSolver tao, Vec sepobj, PetscErrorCode (*func)(TaoSolver, Vec, Vec, void*),void *ctx)
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
#define __FUNCT__ "TaoComputeSeparableObjective"
/*@
  TaoComputeSeparableObjective - Computes a separable objective function vector at a given point (for least-square applications)

  Collective on TaoSolver

  Input Parameters:
+ tao - the TaoSolver context
- X - input vector

  Output Parameter:
. f - Objective vector at X

  Notes: TaoComputeSeparableObjective() is typically used within minimization implementations,
  so most users would not generally call this routine themselves.

  Level: advanced

.seealso: TaoSetSeparableObjectiveRoutine()
@*/
PetscErrorCode TaoComputeSeparableObjective(TaoSolver tao, Vec X, Vec F) 
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
	SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"TaoSetSeparableObjectiveRoutine() has not been called");
    }
    ierr = PetscInfo(tao,"TAO separable function evaluation.\n"); CHKERRQ(ierr);
    PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TaoSetGradientRoutine"
/*@C
  TaoSetGradientRoutine - Sets the gradient evaluation routine for minimization

  Logically collective on TaoSolver

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

.seealso: TaoSetObjectiveRoutine(), TaoSetHessianRoutine() TaoSetObjectiveAndGradientRoutine()
@*/
PetscErrorCode TaoSetGradientRoutine(TaoSolver tao,  PetscErrorCode (*func)(TaoSolver, Vec, Vec, void*),void *ctx) 
{
    PetscFunctionBegin;
    PetscValidHeaderSpecific(tao,TAOSOLVER_CLASSID,1);
    tao->user_gradP = ctx;
    tao->ops->computegradient = func;
    PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "TaoSetObjectiveAndGradientRoutine"
/*@C
  TaoSetObjectiveAndGradientRoutine - Sets a combined objective function and gradient evaluation routine for minimization

  Logically collective on TaoSolver

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

.seealso: TaoSetObjectiveRoutine(), TaoSetHessianRoutine() TaoSetObjectiveAndGradientRoutine()
@*/
PetscErrorCode TaoSetObjectiveAndGradientRoutine(TaoSolver tao, PetscErrorCode (*func)(TaoSolver, Vec, PetscReal *, Vec, void*), void *ctx)
{
    PetscFunctionBegin;
    PetscValidHeaderSpecific(tao,TAOSOLVER_CLASSID,1);
    tao->user_objgradP = ctx;
    tao->ops->computeobjectiveandgradient = func;
    PetscFunctionReturn(0);
}
  
#undef __FUNCT__
#define __FUNCT__ "TaoIsObjectiveDefined"
/*@
  TaoIsObjectiveDefined -- Checks to see if the user has
  declared an objective-only routine.  Useful for determining when
  it is appropriate to call TaoComputeObjective() or 
  TaoComputeObjectiveAndGradient()

  Collective on TaoSolver

  Input Parameter:
+ tao - the TaoSolver context
- ctx - PETSC_TRUE if objective function routine is set by user, 
        PETSC_FALSE otherwise
  Level: developer

.seealso: TaoSetObjectiveRoutine(), TaoIsGradientDefined(), TaoIsObjectiveAndGradientDefined()
@*/
PetscErrorCode TaoIsObjectiveDefined(TaoSolver tao, PetscBool *flg)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao,TAOSOLVER_CLASSID,1);
  if (tao->ops->computeobjective == 0) 
    *flg = PETSC_FALSE;
  else
    *flg = PETSC_TRUE;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TaoIsGradientDefined"
/*@
  TaoIsGradientDefined -- Checks to see if the user has
  declared an objective-only routine.  Useful for determining when
  it is appropriate to call TaoComputeGradient() or 
  TaoComputeGradientAndGradient()

  Not Collective

  Input Parameter:
+ tao - the TaoSolver context
- ctx - PETSC_TRUE if gradient routine is set by user, PETSC_FALSE otherwise
  Level: developer

.seealso: TaoSetGradientRoutine(), TaoIsObjectiveDefined(), TaoIsObjectiveAndGradientDefined()
@*/
PetscErrorCode TaoIsGradientDefined(TaoSolver tao, PetscBool *flg)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao,TAOSOLVER_CLASSID,1);
  if (tao->ops->computegradient == 0) 
    *flg = PETSC_FALSE;
  else
    *flg = PETSC_TRUE;
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "TaoIsObjectiveAndGradientDefined"
/*@
  TaoIsObjectiveAndGradientDefined -- Checks to see if the user has
  declared a joint objective/gradient routine.  Useful for determining when
  it is appropriate to call TaoComputeObjective() or 
  TaoComputeObjectiveAndGradient()

  Not Collective

  Input Parameter:
+ tao - the TaoSolver context
- ctx - PETSC_TRUE if objective/gradient routine is set by user, PETSC_FALSE otherwise
  Level: developer

.seealso: TaoSetObjectiveAndGradientRoutine(), TaoIsObjectiveDefined(), TaoIsGradientDefined()
@*/
PetscErrorCode TaoIsObjectiveAndGradientDefined(TaoSolver tao, PetscBool *flg)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao,TAOSOLVER_CLASSID,1);
  if (tao->ops->computeobjectiveandgradient == 0) 
    *flg = PETSC_FALSE;
  else
    *flg = PETSC_TRUE;
  PetscFunctionReturn(0);
}



