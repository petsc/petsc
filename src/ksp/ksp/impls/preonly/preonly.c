#define PETSCKSP_DLL

#include "include/private/kspimpl.h"

#undef __FUNCT__  
#define __FUNCT__ "KSPSetUp_PREONLY"
static PetscErrorCode KSPSetUp_PREONLY(KSP ksp)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "KSPSolve_PREONLY"
static PetscErrorCode  KSPSolve_PREONLY(KSP ksp)
{
  PetscErrorCode ierr;
  Vec            X,B;
  PetscTruth     diagonalscale;

  PetscFunctionBegin;
  ierr    = PCDiagonalScale(ksp->pc,&diagonalscale);CHKERRQ(ierr);
  if (diagonalscale) SETERRQ1(PETSC_ERR_SUP,"Krylov method %s does not support diagonal scaling",ksp->type_name);
  if (!ksp->guess_zero) {
    SETERRQ(PETSC_ERR_USER,"Running KSP of preonly doesn't make sense with nonzero initial guess\n\
               you probably want a KSP type of Richardson");
  }
  ksp->its    = 0;
  X           = ksp->vec_sol;
  B           = ksp->vec_rhs;
  ierr        = KSP_PCApply(ksp,B,X);CHKERRQ(ierr);
  ksp->its    = 1;
  ksp->reason = KSP_CONVERGED_ITS;
  PetscFunctionReturn(0);
}

/*MC
     KSPPREONLY - This implements a stub method that applies ONLY the preconditioner.
                  This may be used in inner iterations, where it is desired to 
                  allow multiple iterations as well as the "0-iteration" case. It is 
                  commonly used with the direct solver preconditioners like PCLU and PCCHOLESKY

   Options Database Keys:
.   see KSPSolve()

   Level: beginner

.seealso:  KSPCreate(), KSPSetType(), KSPType (for list of available types), KSP

M*/

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "KSPCreate_PREONLY"
PetscErrorCode PETSCKSP_DLLEXPORT KSPCreate_PREONLY(KSP ksp)
{
  PetscFunctionBegin;
  ksp->data                      = (void*)0;
  ksp->ops->setup                = KSPSetUp_PREONLY;
  ksp->ops->solve                = KSPSolve_PREONLY;
  ksp->ops->destroy              = KSPDefaultDestroy;
  ksp->ops->buildsolution        = KSPDefaultBuildSolution;
  ksp->ops->buildresidual        = KSPDefaultBuildResidual;
  ksp->ops->setfromoptions       = 0;
  ksp->ops->view                 = 0;
  PetscFunctionReturn(0);
}
EXTERN_C_END
