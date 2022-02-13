#include <petsc/private/snesimpl.h>             /*I   "petscsnes.h"   I*/

typedef struct {
  PetscBool transpose_solve;
} SNES_KSPONLY;

static PetscErrorCode SNESSolve_KSPONLY(SNES snes)
{
  SNES_KSPONLY   *ksponly = (SNES_KSPONLY*)snes->data;
  PetscErrorCode ierr;
  PetscInt       lits;
  Vec            Y,X,F;

  PetscFunctionBegin;
  PetscCheckFalse(snes->xl || snes->xu || snes->ops->computevariablebounds,PetscObjectComm((PetscObject)snes),PETSC_ERR_ARG_WRONGSTATE, "SNES solver %s does not support bounds", ((PetscObject)snes)->type_name);

  snes->numFailures            = 0;
  snes->numLinearSolveFailures = 0;
  snes->reason                 = SNES_CONVERGED_ITERATING;
  snes->iter                   = 0;
  snes->norm                   = 0.0;

  X = snes->vec_sol;
  F = snes->vec_func;
  Y = snes->vec_sol_update;

  if (!snes->vec_func_init_set) {
    ierr = SNESComputeFunction(snes,X,F);CHKERRQ(ierr);
  } else snes->vec_func_init_set = PETSC_FALSE;

  if (snes->numbermonitors) {
    PetscReal fnorm;
    ierr = VecNorm(F,NORM_2,&fnorm);CHKERRQ(ierr);
    ierr = SNESMonitor(snes,0,fnorm);CHKERRQ(ierr);
  }

  /* Call general purpose update function */
  if (snes->ops->update) {
    ierr = (*snes->ops->update)(snes, 0);CHKERRQ(ierr);
  }

  /* Solve J Y = F, where J is Jacobian matrix */
  ierr = SNESComputeJacobian(snes,X,snes->jacobian,snes->jacobian_pre);CHKERRQ(ierr);

  SNESCheckJacobianDomainerror(snes);

  ierr = KSPSetOperators(snes->ksp,snes->jacobian,snes->jacobian_pre);CHKERRQ(ierr);
  if (ksponly->transpose_solve) {
    ierr = KSPSolveTranspose(snes->ksp,F,Y);CHKERRQ(ierr);
  } else {
    ierr = KSPSolve(snes->ksp,F,Y);CHKERRQ(ierr);
  }
  snes->reason = SNES_CONVERGED_ITS;
  SNESCheckKSPSolve(snes);

  ierr = KSPGetIterationNumber(snes->ksp,&lits);CHKERRQ(ierr);
  ierr = PetscInfo(snes,"iter=%D, linear solve iterations=%D\n",snes->iter,lits);CHKERRQ(ierr);
  snes->iter++;

  /* Take the computed step. */
  ierr = VecAXPY(X,-1.0,Y);CHKERRQ(ierr);
  if (snes->numbermonitors) {
    PetscReal fnorm;
    ierr = SNESComputeFunction(snes,X,F);CHKERRQ(ierr);
    ierr = VecNorm(F,NORM_2,&fnorm);CHKERRQ(ierr);
    ierr = SNESMonitor(snes,1,fnorm);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode SNESSetUp_KSPONLY(SNES snes)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = SNESSetUpMatrices(snes);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode SNESDestroy_KSPONLY(SNES snes)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFree(snes->data);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
/*MC
      SNESKSPONLY - Nonlinear solver that only performs one Newton step and does not compute any norms.
      The main purpose of this solver is to solve linear problems using the SNES interface, without
      any additional overhead in the form of vector operations.

   Level: beginner

.seealso:  SNESCreate(), SNES, SNESSetType(), SNESNEWTONLS, SNESNEWTONTR
M*/
PETSC_EXTERN PetscErrorCode SNESCreate_KSPONLY(SNES snes)
{
  SNES_KSPONLY   *ksponly;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  snes->ops->setup          = SNESSetUp_KSPONLY;
  snes->ops->solve          = SNESSolve_KSPONLY;
  snes->ops->destroy        = SNESDestroy_KSPONLY;
  snes->ops->setfromoptions = NULL;
  snes->ops->view           = NULL;
  snes->ops->reset          = NULL;

  snes->usesksp = PETSC_TRUE;
  snes->usesnpc = PETSC_FALSE;

  snes->alwayscomputesfinalresidual = PETSC_FALSE;

  ierr = PetscNewLog(snes,&ksponly);CHKERRQ(ierr);
  snes->data = (void*)ksponly;
  PetscFunctionReturn(0);
}

/*MC
      SNESKSPTRANSPOSEONLY - Nonlinear solver that only performs one Newton step and does not compute any norms.
      The main purpose of this solver is to solve transposed linear problems using the SNES interface, without
      any additional overhead in the form of vector operations within adjoint solvers.

   Level: beginner

.seealso:  SNESCreate(), SNES, SNESSetType(), SNESKSPTRANSPOSEONLY, SNESNEWTONLS, SNESNEWTONTR
M*/
PETSC_EXTERN PetscErrorCode SNESCreate_KSPTRANSPOSEONLY(SNES snes)
{
  SNES_KSPONLY   *kspo;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = SNESCreate_KSPONLY(snes);CHKERRQ(ierr);
  ierr = PetscObjectChangeTypeName((PetscObject)snes,SNESKSPTRANSPOSEONLY);CHKERRQ(ierr);
  kspo = (SNES_KSPONLY*)snes->data;
  kspo->transpose_solve = PETSC_TRUE;
  PetscFunctionReturn(0);
}
