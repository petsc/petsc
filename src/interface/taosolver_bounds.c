#include "include/private/taosolver_impl.h" /*I "taosolver.h" I*/

#undef __FUNCT__
#define __FUNCT__ "TaoSolverSetVariableBounds"
/*@
  TaoSolverSetVariableBounds - Sets the upper and lower bounds

  Collective on TaoSolver

  Input Parameters:
+ tao - the TaoSolver context
. XL  - vector of lower bounds 
- XU  - vector of upper bounds

  Level: beginner

.seealso: TaoSolverSetObjectiveRoutine(), TaoSolverSetHessianRoutine() TaoSolverSetObjectiveAndGradientRoutine()
@*/

PetscErrorCode TaoSolverSetVariableBounds(TaoSolver tao, Vec XL, Vec XU)
{
    PetscErrorCode ierr;
    PetscFunctionBegin;
    PetscValidHeaderSpecific(tao,TAOSOLVER_CLASSID,1);
    if (XL) {
	PetscValidHeaderSpecific(XL,VEC_CLASSID,2);
	PetscObjectReference((PetscObject)XL);
    }
    if (XU) {
	PetscValidHeaderSpecific(XU,VEC_CLASSID,3);
	PetscObjectReference((PetscObject)XU);
    }
    if (tao->XL) {
	ierr = VecDestroy(&tao->XL); CHKERRQ(ierr);
    }
    if (tao->XU) {
	ierr = VecDestroy(&tao->XU); CHKERRQ(ierr);
    }	

    tao->XL = XL;
    tao->XU = XU;
	
    PetscFunctionReturn(0);
}
#undef __FUNCT__
#define __FUNCT__ "TaoSolverSetVariableBoundsRoutine"
/*@C
  TaoSolverSetVariableBoundsRoutine - Sets a function to be used to compute variable bounds

  Collective on TaoSolver

  Input Parameters:
+ tao - the TaoSolver context
. func - the bounds computation routine
- ctx - [optional] user-defined context for private data for the bounds computation (may be PETSC_NULL)
 
  Calling sequence of func:
$      func (TaoSolver tao, Vec xl, Vec xu);

+ tao - the TaoSolver
. xl  - vector of lower bounds 
. xu  - vector of upper bounds
- ctx - the (optional) user-defined function context

  Level: beginner

.seealso: TaoSolverSetObjectiveRoutine(), TaoSolverSetHessianRoutine() TaoSolverSetObjectiveAndGradientRoutine(), TaoSolverSetVariableBounds()

Note: The func passed in to TaoSolverSetVariableBoundsRoutine() takes 
precedence over any values set in TaoSolverSetVariableBounds().

@*/
PetscErrorCode TaoSolverSetVariableBoundsRoutine(TaoSolver tao, PetscErrorCode (*func)(TaoSolver, Vec, Vec, void*), void *ctx)
{
    PetscFunctionBegin;
    PetscValidHeaderSpecific(tao,TAOSOLVER_CLASSID,1);
    tao->user_boundsP = ctx;
    tao->ops->computebounds = func;
    PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TaoSolverGetVariableBounds"
PetscErrorCode TaoSolverGetVariableBounds(TaoSolver tao, Vec *XL, Vec *XU)
{
    PetscFunctionBegin;
    PetscValidHeaderSpecific(tao,TAOSOLVER_CLASSID,1);
    if (XL) {
	*XL=tao->XL;
    }
    if (XU) {
	*XU=tao->XU;
    }
    PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TaoSolverComputeVariableBounds"
/*@C
   TaoSolverComputeVariableBounds - Compute the variable bounds using the
   routine set by TaoSolverSetVariableBoundsRoutine(). 

   Collective on TaoSolver

   Input Parameters:
.  tao - the TaoSolver context

   Level: developer

.seealso: TaoSolverSetVariableBoundsRoutine(), TaoSolverSetVariableBounds()
@*/

PetscErrorCode TaoSolverComputeVariableBounds(TaoSolver tao)
{
    PetscErrorCode ierr;

    PetscFunctionBegin;
    PetscValidHeaderSpecific(tao,TAOSOLVER_CLASSID,1);
    if (tao->ops->computebounds == PETSC_NULL) {
	PetscFunctionReturn(0);
    }
    if (tao->XL == PETSC_NULL || tao->XU == PETSC_NULL) {
	if (tao->solution == PETSC_NULL) {
	    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"TaoSolverSetInitialVector must be called before TaoSolverComputeVariableBounds");
	}
	ierr = VecDuplicate(tao->solution, &tao->XL); CHKERRQ(ierr);
	ierr = VecSet(tao->XL, TAO_NINFINITY); CHKERRQ(ierr);
	ierr = VecDuplicate(tao->solution, &tao->XU); CHKERRQ(ierr);
	ierr = VecSet(tao->XU, TAO_INFINITY); CHKERRQ(ierr);
    }	
    CHKMEMQ;
    ierr = (*tao->ops->computebounds)(tao,tao->XL,tao->XU,tao->user_boundsP);
    CHKERRQ(ierr);
    CHKMEMQ;

    PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "TaoSolverComputeConstraints"
/*@C
   TaoSolverComputeConstraints - Compute the variable bounds using the
   routine set by TaoSolverSetConstraintsRoutine(). 

   Collective on TaoSolver

   Input Parameters:
.  tao - the TaoSolver context

   Level: developer

.seealso: TaoSolverSetConstraintsRoutine(), TaoSolverComputeJacobian()
@*/

PetscErrorCode TaoSolverComputeConstraints(TaoSolver tao, Vec X, Vec C)
{
    PetscErrorCode ierr;

    PetscFunctionBegin;
    PetscValidHeaderSpecific(tao,TAOSOLVER_CLASSID,1);
    PetscValidHeaderSpecific(X,VEC_CLASSID,2);
    PetscValidHeaderSpecific(C,VEC_CLASSID,2);
    PetscCheckSameComm(tao,1,X,2);
    PetscCheckSameComm(tao,1,C,3);

    if (tao->ops->computeconstraints == PETSC_NULL) {
      SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"TaoSolverSetConstraintsRoutine() has not been called");
    }
    if (tao->solution == PETSC_NULL) {
      SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"TaoSolverSetInitialVector must be called before TaoSolverComputeConstraints");
    }
    ierr = PetscLogEventBegin(TaoSolver_ConstraintsEval,tao,X,C,PETSC_NULL); CHKERRQ(ierr);
    PetscStackPush("TaoSolver constraints evaluation routine");
    CHKMEMQ;
    ierr = (*tao->ops->computeconstraints)(tao,X,C,tao->user_conP);
    CHKERRQ(ierr);
    CHKMEMQ;
    PetscStackPop;
    ierr = PetscLogEventEnd(TaoSolver_ConstraintsEval,tao,X,C,PETSC_NULL); CHKERRQ(ierr);
    tao->nconstraints++;
    PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "TaoSolverSetConstraintsRoutine"
/*@C
  TaoSolverSetConstraintsRoutine - Sets a function to be used to compute constraints.  TAO only handles constraints under certain conditions, see manual for details

  Collective on TaoSolver

  Input Parameters:
+ tao - the TaoSolver context
. c   - A vector that will be used to store constraint evaluation 
. func - the bounds computation routine
- ctx - [optional] user-defined context for private data for the constraints computation (may be PETSC_NULL)
 
  Calling sequence of func:
$      func (TaoSolver tao, Vec x, Vec c);

+ tao - the TaoSolver
. x   - point to evaluate constraints
. c   - vector constraints evaluated at x
- ctx - the (optional) user-defined function context

  Level: beginner

.seealso: TaoSolverSetObjectiveRoutine(), TaoSolverSetHessianRoutine() TaoSolverSetObjectiveAndGradientRoutine(), TaoSolverSetVariableBounds()

@*/
PetscErrorCode TaoSolverSetConstraintsRoutine(TaoSolver tao, Vec c, PetscErrorCode (*func)(TaoSolver, Vec, Vec, void*), void *ctx)
{
    PetscFunctionBegin;
    PetscValidHeaderSpecific(tao,TAOSOLVER_CLASSID,1);
    tao->constraints = c;
    tao->user_conP = ctx;
    tao->ops->computeconstraints = func;
    PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TaoSolverComputeDual"
/*@
  TaoSolverComputeDual - Computes the dual vectors corresponding to the bounds
  of the variables

  Collective on TaoSolver

  Input Parameters:
. tao - the TaoSolver context

  Output Parameter:
+ DL - dual variable vector for the lower bounds
- DU - dual variable vector for the upper bounds

  Level: advanced

  Notes: DL and DU should be created before calling this routine.  If calling
  this routine after using an unconstrained solver, DL and DU are set to all 
  zeros.

  Level: advanced

.seealso: TaoSolverComputeObjective(), TaoSolverSetVariableBounds()
@*/
PetscErrorCode TaoSolverComputeDual(TaoSolver tao, Vec DL, Vec DU) 
{
    PetscErrorCode ierr;
    PetscReal dummy;
    PetscFunctionBegin;
    PetscValidHeaderSpecific(tao,TAOSOLVER_CLASSID,1);
    PetscValidHeaderSpecific(DL,VEC_CLASSID,2);
    PetscValidHeaderSpecific(DU,VEC_CLASSID,2);
    PetscCheckSameComm(tao,1,DL,2);
    PetscCheckSameComm(tao,1,DU,3);
    if (tao->ops->computedual) {
      ierr = (*tao->ops->computedual)(tao,DL,DU); CHKERRQ(ierr);
    }  else {
      ierr = VecSet(DL,0.0); CHKERRQ(ierr);
      ierr = VecSet(DU,0.0); CHKERRQ(ierr);
    }
    PetscFunctionReturn(0);
}
