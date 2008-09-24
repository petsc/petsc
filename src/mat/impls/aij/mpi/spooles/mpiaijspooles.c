#define PETSCMAT_DLL

/* 
   Provides an interface to the Spooles parallel sparse solver (MPI SPOOLES)
*/


#include "src/mat/impls/aij/mpi/mpiaij.h"
#include "src/mat/impls/aij/seq/spooles/spooles.h"

/* Note the Petsc r and c permutations are ignored */
#undef __FUNCT__  
#define __FUNCT__ "MatLUFactorSymbolic_MPIAIJSpooles"
PetscErrorCode MatLUFactorSymbolic_MPIAIJSpooles(Mat F,Mat A,IS r,IS c,MatFactorInfo *info)
{
  Mat_Spooles    *lu;

  PetscFunctionBegin;	
  if (!info->dtcol) {
    lu = (Mat_Spooles*) F->spptr;
    lu->options.pivotingflag  = SPOOLES_NO_PIVOTING;
  }
  F->ops->lufactornumeric  = MatFactorNumeric_MPISpooles;
  F->ops->solve            = MatSolve_MPISpooles;
  PetscFunctionReturn(0); 
}

EXTERN_C_BEGIN  
#undef __FUNCT__  
#define __FUNCT__ "MatGetFactor_mpiaij_spooles"
PetscErrorCode MatGetFactor_mpiaij_spooles(Mat A,MatFactorType ftype,Mat *F)
{
  Mat_Spooles    *lu;
  Mat            B;
  PetscErrorCode ierr;

  PetscFunctionBegin;	

  /* Create the factorization matrix F */  
  ierr = MatCreate(((PetscObject)A)->comm,&B);CHKERRQ(ierr);
  ierr = MatSetSizes(B,A->rmap->n,A->cmap->n,A->rmap->N,A->cmap->N);CHKERRQ(ierr);
  ierr = MatSetType(B,((PetscObject)A)->type_name);CHKERRQ(ierr);
  ierr = MatMPIAIJSetPreallocation(B,0,PETSC_NULL,0,PETSC_NULL);CHKERRQ(ierr);

  ierr = PetscNewLog(B,Mat_Spooles,&lu);CHKERRQ(ierr);
  B->spptr          = lu;
  lu->flg           = DIFFERENT_NONZERO_PATTERN;
  lu->options.useQR = PETSC_FALSE;

  if (ftype == MAT_FACTOR_LU) {
    B->ops->lufactorsymbolic = MatLUFactorSymbolic_MPIAIJSpooles;
    B->ops->destroy          = MatDestroy_MPIAIJSpooles;  
    B->factor                = MAT_FACTOR_LU;  

    lu->options.symflag      = SPOOLES_NONSYMMETRIC;
    lu->options.pivotingflag = SPOOLES_PIVOTING; 
  } else SETERRQ(PETSC_ERR_SUP,"Only LU for AIJ matrices, use SBAIJ for Cholesky");
  B->factor = ftype;
  ierr = MPI_Comm_dup(((PetscObject)A)->comm,&(lu->comm_spooles));CHKERRQ(ierr);

  *F = B;
  PetscFunctionReturn(0); 
}
EXTERN_C_END
