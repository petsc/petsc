#define PETSCMAT_DLL

/* 
   Provides an interface to the Spooles serial sparse solver
*/

#include "src/mat/impls/aij/seq/spooles/spooles.h"
#include "src/mat/impls/aij/seq/aij.h"

#undef __FUNCT__
#define __FUNCT__ "MatView_Spooles"
PetscErrorCode MatView_Spooles(Mat A,PetscViewer viewer)
{
  PetscErrorCode    ierr;
  PetscTruth        iascii;
  PetscViewerFormat format;

  PetscFunctionBegin;
  ierr = MatView_SeqAIJ(A,viewer);CHKERRQ(ierr);

  ierr = PetscTypeCompare((PetscObject)viewer,PETSC_VIEWER_ASCII,&iascii);CHKERRQ(ierr);
  if (iascii) {
    ierr = PetscViewerGetFormat(viewer,&format);CHKERRQ(ierr);
    if (format == PETSC_VIEWER_ASCII_INFO) {
      ierr = MatFactorInfo_Spooles(A,viewer);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

/* Note the Petsc r and c permutations are ignored */
#undef __FUNCT__  
#define __FUNCT__ "MatLUFactorSymbolic_SeqAIJSpooles"
PetscErrorCode MatLUFactorSymbolic_SeqAIJSpooles(Mat A,IS r,IS c,MatFactorInfo *info,Mat *F)
{
  Mat_Spooles    *lu = (Mat_Spooles*)((*F)->spptr);;

  PetscFunctionBegin;	
  (*F)->ops->lufactornumeric =  MatFactorNumeric_SeqSpooles;
  if (!info->dtcol) {
    lu->options.pivotingflag  = SPOOLES_NO_PIVOTING;
  }
  PetscFunctionReturn(0); 
}

/* Note the Petsc r permutation is ignored */
#undef __FUNCT__  
#define __FUNCT__ "MatCholeskyFactorSymbolic_SeqSAIJSpooles"
PetscErrorCode MatCholeskyFactorSymbolic_SeqAIJSpooles(Mat A,IS r,MatFactorInfo *info,Mat *F)
{ 
  PetscFunctionBegin;	
  (*F)->ops->choleskyfactornumeric  = MatFactorNumeric_SeqSpooles;
#if !defined(PETSC_USE_COMPLEX)
  (*F)->ops->getinertia             = MatGetInertia_SeqSBAIJSpooles;
#endif
  PetscFunctionReturn(0); 
}

  
EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "MatGetFactor_seqaij_spooles"
PetscErrorCode MatGetFactor_seqaij_spooles(Mat A,MatFactorType ftype,Mat *F)
{ 
  Mat            B;
  Mat_Spooles    *lu;   
  PetscErrorCode ierr;
  int            m=A->rmap->n,n=A->cmap->n;

  PetscFunctionBegin;	
  /* Create the factorization matrix */
  ierr = MatCreate(((PetscObject)A)->comm,&B);CHKERRQ(ierr);
  ierr = MatSetSizes(B,m,n,PETSC_DETERMINE,PETSC_DETERMINE);CHKERRQ(ierr);
  ierr = MatSetType(B,((PetscObject)A)->type_name);CHKERRQ(ierr);
  ierr = MatSeqAIJSetPreallocation(B,PETSC_NULL,PETSC_NULL);CHKERRQ(ierr);
  
  ierr = PetscNewLog(B,Mat_Spooles,&lu);CHKERRQ(ierr);
  B->spptr = lu;
  lu->options.pivotingflag  = SPOOLES_NO_PIVOTING;
  lu->flg                   = DIFFERENT_NONZERO_PATTERN;
  lu->options.useQR         = PETSC_FALSE;

  if (ftype == MAT_FACTOR_LU) {
    B->ops->lufactorsymbolic = MatLUFactorSymbolic_SeqAIJSpooles;
    B->factor                = MAT_FACTOR_LU;

    lu->options.symflag      = SPOOLES_NONSYMMETRIC;
    lu->options.pivotingflag = SPOOLES_PIVOTING;
  } else if (ftype == MAT_FACTOR_CHOLESKY) {
    B->ops->choleskyfactorsymbolic = MatCholeskyFactorSymbolic_SeqAIJSpooles;
    B->factor                      = MAT_FACTOR_CHOLESKY;  
    lu->options.symflag            = SPOOLES_SYMMETRIC;   /* default */
  } else SETERRQ(PETSC_ERR_SUP,"Spooles only supports LU and Cholesky factorizations");
  B->ops->view    = MatView_Spooles;
  B->ops->destroy = MatDestroy_SeqAIJSpooles;  
  *F = B;
  PetscFunctionReturn(0); 
}
  
EXTERN_C_END
