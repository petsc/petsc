
#include <petsc/private/kspimpl.h>

static PetscErrorCode KSPSetUp_PREONLY(KSP ksp)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

static PetscErrorCode  KSPSolve_PREONLY(KSP ksp)
{
  PetscErrorCode ierr;
  PetscBool      diagonalscale;
  PCFailedReason pcreason;

  PetscFunctionBegin;
  ierr = PCGetDiagonalScale(ksp->pc,&diagonalscale);CHKERRQ(ierr);
  if (diagonalscale) SETERRQ1(PetscObjectComm((PetscObject)ksp),PETSC_ERR_SUP,"Krylov method %s does not support diagonal scaling",((PetscObject)ksp)->type_name);
  if (!ksp->guess_zero) SETERRQ(PetscObjectComm((PetscObject)ksp),PETSC_ERR_USER,"Running KSP of preonly doesn't make sense with nonzero initial guess\n\
               you probably want a KSP type of Richardson");
  ksp->its = 0;
  ierr     = KSP_PCApply(ksp,ksp->vec_rhs,ksp->vec_sol);CHKERRQ(ierr);
  ierr     = PCGetFailedReason(ksp->pc,&pcreason);CHKERRQ(ierr);
  if (pcreason) {
    ksp->reason = KSP_DIVERGED_PC_FAILED;
  } else {
    ksp->its    = 1;
    ksp->reason = KSP_CONVERGED_ITS;
    if (PetscDefined(USE_DEBUG)) {
      PetscReal norm;
      ierr = VecNorm(ksp->vec_sol,NORM_2,&norm);CHKERRQ(ierr);
      if (PetscIsInfOrNanReal(norm)) {
        ksp->reason = KSP_DIVERGED_NANORINF;
      }
    }
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode KSPMatSolve_PREONLY(KSP ksp, Mat B, Mat X)
{
  PetscErrorCode ierr;
  PetscBool      diagonalscale;
  PCFailedReason pcreason;

  PetscFunctionBegin;
  ierr = PCGetDiagonalScale(ksp->pc,&diagonalscale);CHKERRQ(ierr);
  if (diagonalscale) SETERRQ1(PetscObjectComm((PetscObject)ksp),PETSC_ERR_SUP,"Krylov method %s does not support diagonal scaling",((PetscObject)ksp)->type_name);
  if (!ksp->guess_zero) SETERRQ(PetscObjectComm((PetscObject)ksp),PETSC_ERR_USER,"Running KSP of preonly doesn't make sense with nonzero initial guess\n\
               you probably want a KSP type of Richardson");
  ksp->its = 0;
  ierr     = PCMatApply(ksp->pc,B,X);CHKERRQ(ierr);
  ierr     = PCGetFailedReason(ksp->pc,&pcreason);CHKERRQ(ierr);
  if (pcreason) {
    ksp->reason = KSP_DIVERGED_PC_FAILED;
  } else {
    ksp->its    = 1;
    ksp->reason = KSP_CONVERGED_ITS;
    if (PetscDefined(USE_DEBUG)) {
      PetscReal norm;
      ierr = MatNorm(X,NORM_FROBENIUS,&norm);CHKERRQ(ierr);
      if (PetscIsInfOrNanReal(norm)) {
        ksp->reason = KSP_DIVERGED_NANORINF;
      }
    }
  }
  PetscFunctionReturn(0);
}

/*MC
     KSPPREONLY - This implements a method that applies ONLY the preconditioner exactly once.
                  This may be used in inner iterations, where it is desired to
                  allow multiple iterations as well as the "0-iteration" case. It is
                  commonly used with the direct solver preconditioners like PCLU and PCCHOLESKY

   Options Database Keys:
.   -ksp_type preonly

   Level: beginner

   Notes:
    Since this does not involve an iteration the basic KSP parameters such as tolerances and iteration counts
          do not apply

    To apply multiple preconditioners in a simple iteration use KSPRICHARDSON

   Developer Notes:
    Even though this method does not use any norms, the user is allowed to set the KSPNormType to any value.
    This is so the users does not have to change KSPNormType options when they switch from other KSP methods to this one.

.seealso:  KSPCreate(), KSPSetType(), KSPType, KSP, KSPRICHARDSON, KSPCHEBYSHEV

M*/

PETSC_EXTERN PetscErrorCode KSPCreate_PREONLY(KSP ksp)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = KSPSetSupportedNorm(ksp,KSP_NORM_NONE,PC_LEFT,3);CHKERRQ(ierr);
  ierr = KSPSetSupportedNorm(ksp,KSP_NORM_NONE,PC_RIGHT,2);CHKERRQ(ierr);
  ierr = KSPSetSupportedNorm(ksp,KSP_NORM_PRECONDITIONED,PC_LEFT,2);CHKERRQ(ierr);
  ierr = KSPSetSupportedNorm(ksp,KSP_NORM_PRECONDITIONED,PC_RIGHT,2);CHKERRQ(ierr);
  ierr = KSPSetSupportedNorm(ksp,KSP_NORM_UNPRECONDITIONED,PC_LEFT,2);CHKERRQ(ierr);
  ierr = KSPSetSupportedNorm(ksp,KSP_NORM_UNPRECONDITIONED,PC_RIGHT,2);CHKERRQ(ierr);
  ierr = KSPSetSupportedNorm(ksp,KSP_NORM_NATURAL,PC_LEFT,2);CHKERRQ(ierr);

  ksp->data                = NULL;
  ksp->ops->setup          = KSPSetUp_PREONLY;
  ksp->ops->solve          = KSPSolve_PREONLY;
  ksp->ops->matsolve       = KSPMatSolve_PREONLY;
  ksp->ops->destroy        = KSPDestroyDefault;
  ksp->ops->buildsolution  = KSPBuildSolutionDefault;
  ksp->ops->buildresidual  = KSPBuildResidualDefault;
  ksp->ops->setfromoptions = NULL;
  ksp->ops->view           = NULL;
  PetscFunctionReturn(0);
}
