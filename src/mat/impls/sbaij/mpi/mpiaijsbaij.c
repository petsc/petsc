#define PETSCMAT_DLL

#include "mpisbaij.h" /*I "petscmat.h" I*/

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "MatConvert_MPIAIJ_MPISBAIJ"
PetscErrorCode PETSCMAT_DLLEXPORT MatConvert_MPIAIJ_MPISBAIJ(Mat A, MatType newtype,MatReuse reuse,Mat *newmat) 
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /* Now, calls MatConvert_Basic(). Will implement later */
  ierr = MatConvert_Basic(A, newtype,reuse,newmat);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
EXTERN_C_END
