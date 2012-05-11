
/*
    This file implements improved flexible BiCGStab contributed by Jie Chen.
    Only right preconditioning is supported. 
*/
#include <../src/ksp/ksp/impls/bcgs/bcgsimpl.h>       /*I  "petscksp.h"  I*/

#undef __FUNCT__  
#define __FUNCT__ "KSPSetUp_IFBCGS"
PetscErrorCode KSPSetUp_IFBCGS(KSP ksp)
{
  PetscErrorCode ierr;
  
  PetscFunctionBegin;
  ierr = KSPDefaultGetWork(ksp,12);CHKERRQ(ierr); 
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "KSPSolve_IFBCGS"
PetscErrorCode  KSPSolve_IFBCGS(KSP ksp)
{
  PetscFunctionBegin;
  
  PetscFunctionReturn(0);
}

/*MC
     KSPIFBCGS - Implements the improved flexible BiCGStab (Stabilized version of BiConjugate Gradient Squared) method.

   Options Database Keys:
.   see KSPSolve()

   Level: beginner

   Notes: See KSPIFBCGSL for additional stabilization
          Only supports right preconditioning 

.seealso:  KSPCreate(), KSPSetType(), KSPType (for list of available types), KSP, KSPBICG, KSPFBCGSL, KSPSetPCSide()
M*/
EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "KSPCreate_IFBCGS"
PetscErrorCode  KSPCreate_IFBCGS(KSP ksp)
{
  PetscErrorCode ierr;
  KSP_BCGS       *bcgs;

  PetscFunctionBegin;
  ierr = PetscNewLog(ksp,KSP_BCGS,&bcgs);CHKERRQ(ierr);
  ksp->data                 = bcgs;
  ksp->ops->setup           = KSPSetUp_IFBCGS;
  ksp->ops->solve           = KSPSolve_IFBCGS;
  ksp->ops->destroy         = KSPDestroy_BCGS;
  ksp->ops->reset           = KSPReset_BCGS;
  ksp->ops->buildsolution   = KSPBuildSolution_BCGS;
  ksp->ops->buildresidual   = KSPDefaultBuildResidual;
  ksp->ops->setfromoptions  = KSPSetFromOptions_BCGS;
  ksp->ops->view            = KSPView_BCGS;
 
  ierr = KSPSetSupportedNorm(ksp,KSP_NORM_PRECONDITIONED,PC_LEFT,2);CHKERRQ(ierr);
  ierr = KSPSetSupportedNorm(ksp,KSP_NORM_UNPRECONDITIONED,PC_RIGHT,1);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
EXTERN_C_END
