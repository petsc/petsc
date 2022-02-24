#include <petsc/private/snesimpl.h>             /*I   "petscsnes.h"   I*/

typedef struct {
  PetscBool transpose_solve;
} SNES_KSPONLY;

static PetscErrorCode SNESSolve_KSPONLY(SNES snes)
{
  SNES_KSPONLY   *ksponly = (SNES_KSPONLY*)snes->data;
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
    CHKERRQ(SNESComputeFunction(snes,X,F));
  } else snes->vec_func_init_set = PETSC_FALSE;

  if (snes->numbermonitors) {
    PetscReal fnorm;
    CHKERRQ(VecNorm(F,NORM_2,&fnorm));
    CHKERRQ(SNESMonitor(snes,0,fnorm));
  }

  /* Call general purpose update function */
  if (snes->ops->update) {
    CHKERRQ((*snes->ops->update)(snes, 0));
  }

  /* Solve J Y = F, where J is Jacobian matrix */
  CHKERRQ(SNESComputeJacobian(snes,X,snes->jacobian,snes->jacobian_pre));

  SNESCheckJacobianDomainerror(snes);

  CHKERRQ(KSPSetOperators(snes->ksp,snes->jacobian,snes->jacobian_pre));
  if (ksponly->transpose_solve) {
    CHKERRQ(KSPSolveTranspose(snes->ksp,F,Y));
  } else {
    CHKERRQ(KSPSolve(snes->ksp,F,Y));
  }
  snes->reason = SNES_CONVERGED_ITS;
  SNESCheckKSPSolve(snes);

  CHKERRQ(KSPGetIterationNumber(snes->ksp,&lits));
  CHKERRQ(PetscInfo(snes,"iter=%D, linear solve iterations=%D\n",snes->iter,lits));
  snes->iter++;

  /* Take the computed step. */
  CHKERRQ(VecAXPY(X,-1.0,Y));
  if (snes->numbermonitors) {
    PetscReal fnorm;
    CHKERRQ(SNESComputeFunction(snes,X,F));
    CHKERRQ(VecNorm(F,NORM_2,&fnorm));
    CHKERRQ(SNESMonitor(snes,1,fnorm));
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode SNESSetUp_KSPONLY(SNES snes)
{
  PetscFunctionBegin;
  CHKERRQ(SNESSetUpMatrices(snes));
  PetscFunctionReturn(0);
}

static PetscErrorCode SNESDestroy_KSPONLY(SNES snes)
{
  PetscFunctionBegin;
  CHKERRQ(PetscFree(snes->data));
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

  CHKERRQ(PetscNewLog(snes,&ksponly));
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

  PetscFunctionBegin;
  CHKERRQ(SNESCreate_KSPONLY(snes));
  CHKERRQ(PetscObjectChangeTypeName((PetscObject)snes,SNESKSPTRANSPOSEONLY));
  kspo = (SNES_KSPONLY*)snes->data;
  kspo->transpose_solve = PETSC_TRUE;
  PetscFunctionReturn(0);
}
