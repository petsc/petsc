#define PETSCKSP_DLL

#include "src/ksp/ksp/impls/cg/cgctx.h"       /*I "petscksp.h" I*/

#undef __FUNCT__  
#define __FUNCT__ "KSPCGSetType" 
/*@
    KSPCGSetType - Sets the variant of the conjugate gradient method to
    use for solving a linear system with a complex coefficient matrix.
    This option is irrelevant when solving a real system.

    Collective on KSP

    Input Parameters:
+   ksp - the iterative context
-   type - the variant of CG to use, one of
.vb
      KSP_CG_HERMITIAN - complex, Hermitian matrix (default)
      KSP_CG_SYMMETRIC - complex, symmetric matrix
.ve

    Level: intermediate
    
    Options Database Keys:
+   -ksp_cg_Hermitian - Indicates Hermitian matrix
-   -ksp_cg_symmetric - Indicates symmetric matrix

    Note:
    By default, the matrix is assumed to be complex, Hermitian.

.keywords: CG, conjugate gradient, Hermitian, symmetric, set, type
@*/
PetscErrorCode PETSCKSP_DLLEXPORT KSPCGSetType(KSP ksp,KSPCGType type)
{
  PetscErrorCode ierr,(*f)(KSP,KSPCGType);

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_COOKIE,1);
  ierr = PetscObjectQueryFunction((PetscObject)ksp,"KSPCGSetType_C",(void (**)(void))&f);CHKERRQ(ierr);
  if (f) {
    ierr = (*f)(ksp,type);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}





