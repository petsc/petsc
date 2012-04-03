#include <petsc-private/snesimpl.h>             /*I   "petscsnes.h"   I*/

typedef struct {
  PetscInt  sweeps;
} SNES_GS;

#undef __FUNCT__
#define __FUNCT__ "SNESReset_GS"
PetscErrorCode SNESReset_GS(SNES snes)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SNESDestroy_GS"
PetscErrorCode SNESDestroy_GS(SNES snes)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = SNESReset_GS(snes);CHKERRQ(ierr);
  ierr = PetscFree(snes->data);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SNESSetUp_GS"
PetscErrorCode SNESSetUp_GS(SNES snes)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SNESSetFromOptions_GS"
PetscErrorCode SNESSetFromOptions_GS(SNES snes)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscOptionsHead("SNES GS options");CHKERRQ(ierr);
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SNESView_GS"
PetscErrorCode SNESView_GS(SNES snes, PetscViewer viewer)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SNESSolve_GS"
PetscErrorCode SNESSolve_GS(SNES snes)
{
  Vec            F;
  Vec            X;
  Vec            B;
  PetscInt       i;
  PetscReal      fnorm;
  PetscErrorCode ierr;
  SNESNormType   normtype;

  PetscFunctionBegin;

  X = snes->vec_sol;
  F = snes->vec_func;
  B = snes->vec_rhs;

  ierr = PetscObjectTakeAccess(snes);CHKERRQ(ierr);
  snes->iter = 0;
  snes->norm = 0.;
  ierr = PetscObjectGrantAccess(snes);CHKERRQ(ierr);
  snes->reason = SNES_CONVERGED_ITERATING;

  ierr = SNESGetNormType(snes, &normtype);CHKERRQ(ierr);
  if (normtype == SNES_NORM_FUNCTION || normtype == SNES_NORM_INITIAL_ONLY || normtype == SNES_NORM_INITIAL_FINAL_ONLY) {
    /* compute the initial function and preconditioned update delX */
    if (!snes->vec_func_init_set) {
      ierr = SNESComputeFunction(snes,X,F);CHKERRQ(ierr);
      if (snes->domainerror) {
        snes->reason = SNES_DIVERGED_FUNCTION_DOMAIN;
        PetscFunctionReturn(0);
      }
    } else {
      snes->vec_func_init_set = PETSC_FALSE;
    }

    /* convergence test */
    if (!snes->norm_init_set) {
      ierr = VecNorm(F, NORM_2, &fnorm);CHKERRQ(ierr); /* fnorm <- ||F||  */
      if (PetscIsInfOrNanReal(fnorm)) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_FP,"Infinite or not-a-number generated in norm");
    } else {
      fnorm = snes->norm_init;
      snes->norm_init_set = PETSC_FALSE;
    }
    ierr = PetscObjectTakeAccess(snes);CHKERRQ(ierr);
    snes->iter = 0;
    snes->norm = fnorm;
    ierr = PetscObjectGrantAccess(snes);CHKERRQ(ierr);
    SNESLogConvHistory(snes,snes->norm,0);
    ierr = SNESMonitor(snes,0,snes->norm);CHKERRQ(ierr);

    /* set parameter for default relative tolerance convergence test */
    snes->ttol = fnorm*snes->rtol;

    /* test convergence */
    ierr = (*snes->ops->converged)(snes,0,0.0,0.0,fnorm,&snes->reason,snes->cnvP);CHKERRQ(ierr);
    if (snes->reason) PetscFunctionReturn(0);
  } else {
    ierr = PetscObjectGrantAccess(snes);CHKERRQ(ierr);
    SNESLogConvHistory(snes,snes->norm,0);
    ierr = SNESMonitor(snes,0,snes->norm);CHKERRQ(ierr);
  }


  /* Call general purpose update function */
  if (snes->ops->update) {
    ierr = (*snes->ops->update)(snes, snes->iter);CHKERRQ(ierr);
  }

  for (i = 0; i < snes->max_its; i++) {
    ierr = SNESComputeGS(snes, B, X);CHKERRQ(ierr);
    /* only compute norms if requested or about to exit due to maximum iterations */
    if (normtype == SNES_NORM_FUNCTION || ((i == snes->max_its - 1) && (normtype == SNES_NORM_INITIAL_FINAL_ONLY || normtype == SNES_NORM_FINAL_ONLY))) {
      ierr = SNESComputeFunction(snes,X,F);CHKERRQ(ierr);
      if (snes->domainerror) {
        snes->reason = SNES_DIVERGED_FUNCTION_DOMAIN;
        PetscFunctionReturn(0);
      }
      ierr = VecNorm(F, NORM_2, &fnorm);CHKERRQ(ierr); /* fnorm <- ||F||  */
      if (PetscIsInfOrNanReal(fnorm)) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_FP,"Infinite or not-a-number generated in norm");
    }
    /* Monitor convergence */
    ierr = PetscObjectTakeAccess(snes);CHKERRQ(ierr);
    snes->iter = i+1;
    snes->norm = fnorm;
    ierr = PetscObjectGrantAccess(snes);CHKERRQ(ierr);
    SNESLogConvHistory(snes,snes->norm,0);
    ierr = SNESMonitor(snes,snes->iter,snes->norm);CHKERRQ(ierr);
    /* Test for convergence */
    if (normtype == SNES_NORM_FUNCTION) ierr = (*snes->ops->converged)(snes,snes->iter,0.0,0.0,fnorm,&snes->reason,snes->cnvP);CHKERRQ(ierr);
    if (snes->reason) PetscFunctionReturn(0);
    /* Call general purpose update function */
    if (snes->ops->update) {
      ierr = (*snes->ops->update)(snes, snes->iter);CHKERRQ(ierr);
    }
  }
  if (normtype == SNES_NORM_FUNCTION) {
    if (i == snes->max_its) {
      ierr = PetscInfo1(snes,"Maximum number of iterations has been reached: %D\n",snes->max_its);CHKERRQ(ierr);
      if(!snes->reason) snes->reason = SNES_DIVERGED_MAX_IT;
    }
  } else {
    if (!snes->reason) snes->reason = SNES_CONVERGED_ITS; /* GS is meant to be used as a preconditioner */
  }
  PetscFunctionReturn(0);
}

/*MC
  SNESGS - Just calls the user-provided solution routine provided with SNESSetGS()

   Level: advanced

  Notes:
  the Gauss-Seidel smoother is inherited through composition.  If a solver has been created with SNESGetPC(), it will have
  its parent's Gauss-Seidel routine associated with it.

.seealso: SNESCreate(), SNES, SNESSetType(), SNESSetGS(), SNESType (for list of available types)
M*/

EXTERN_C_BEGIN
#undef __FUNCT__
#define __FUNCT__ "SNESCreate_GS"
PetscErrorCode SNESCreate_GS(SNES snes)
{
  SNES_GS        *gs;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  snes->ops->destroy        = SNESDestroy_GS;
  snes->ops->setup          = SNESSetUp_GS;
  snes->ops->setfromoptions = SNESSetFromOptions_GS;
  snes->ops->view           = SNESView_GS;
  snes->ops->solve          = SNESSolve_GS;
  snes->ops->reset          = SNESReset_GS;

  snes->usesksp             = PETSC_FALSE;
  snes->usespc              = PETSC_FALSE;

  if (!snes->tolerancesset) {
    snes->max_its             = 10000;
    snes->max_funcs           = 10000;
  }

  ierr = PetscNewLog(snes, SNES_GS, &gs);CHKERRQ(ierr);
  snes->data = (void*) gs;
  PetscFunctionReturn(0);
}
EXTERN_C_END
