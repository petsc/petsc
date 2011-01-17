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
	ierr = VecDestroy(tao->XL); CHKERRQ(ierr);
    }
    if (tao->XU) {
	ierr = VecDestroy(tao->XU); CHKERRQ(ierr);
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

