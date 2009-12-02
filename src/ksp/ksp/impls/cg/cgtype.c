#define PETSCKSP_DLL

#include "../src/ksp/ksp/impls/cg/cgimpl.h"       /*I "petscksp.h" I*/

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

#undef __FUNCT__  
#define __FUNCT__ "KSPCGUseSingleReduction" 
/*@
    KSPCGUseSingleReduction - Merge the two inner products needed in CG into a single 
     MPI_Allreduce() call.

    Collective on KSP

    Input Parameters:
+   ksp - the iterative context
-   flg - turn on or off the single reduction

    Options Database:
.   -ksp_cg_single_reduction 

    Level: intermediate

     The algorithm used in this case is described as Method 1 in Lapack Working Note 56, "Conjugate Gradient Algorithms with Reduced Synchronization Overhead 
     Distributed Memory Multiprocessors", by E. F. D'Azevedo, V. L. Eijkhout, and C. H. Romine, December 3, 1999. V. Eijkhout creates the algorithm 
     initially to Chronopoulos and Gear.

     It requires two extra work vectors than the conventional implementation in PETSc. 
    
.keywords: CG, conjugate gradient, Hermitian, symmetric, set, type
@*/
PetscErrorCode PETSCKSP_DLLEXPORT KSPCGUseSingleReduction(KSP ksp,PetscTruth flg)
{
  PetscErrorCode ierr,(*f)(KSP,PetscTruth);

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_COOKIE,1);
  ierr = PetscObjectQueryFunction((PetscObject)ksp,"KSPCGUseSingleReduction_C",(void (**)(void))&f);CHKERRQ(ierr);
  if (f) {
    ierr = (*f)(ksp,flg);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}





