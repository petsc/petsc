#define PETSCMAT_DLL

/* 
   Provides an interface to the Spooles serial sparse solver
*/

#include "src/mat/impls/aij/seq/spooles/spooles.h"

#undef __FUNCT__
#define __FUNCT__ "MatView_SeqAIJSpooles"
PetscErrorCode MatView_SeqAIJSpooles(Mat A,PetscViewer viewer)
{
  PetscErrorCode ierr;
  PetscTruth        iascii;
  PetscViewerFormat format;
  Mat_Spooles       *lu=(Mat_Spooles*)(A->spptr);

  PetscFunctionBegin;
  ierr = (*lu->MatView)(A,viewer);CHKERRQ(ierr);

  ierr = PetscTypeCompare((PetscObject)viewer,PETSC_VIEWER_ASCII,&iascii);CHKERRQ(ierr);
  if (iascii) {
    ierr = PetscViewerGetFormat(viewer,&format);CHKERRQ(ierr);
    if (format == PETSC_VIEWER_ASCII_FACTOR_INFO) {
      ierr = MatFactorInfo_Spooles(A,viewer);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatAssemblyEnd_SeqAIJSpooles"
PetscErrorCode MatAssemblyEnd_SeqAIJSpooles(Mat A,MatAssemblyType mode) {
  PetscErrorCode ierr;
  Mat_Spooles *lu=(Mat_Spooles *)(A->spptr);

  PetscFunctionBegin;
  ierr = (*lu->MatAssemblyEnd)(A,mode);CHKERRQ(ierr);

  lu->MatLUFactorSymbolic          = A->ops->lufactorsymbolic;
  lu->MatCholeskyFactorSymbolic    = A->ops->choleskyfactorsymbolic;
  if (lu->useQR){
    A->ops->lufactorsymbolic       = MatQRFactorSymbolic_SeqAIJSpooles;  
  } else {
    A->ops->choleskyfactorsymbolic = MatCholeskyFactorSymbolic_SeqAIJSpooles;
    A->ops->lufactorsymbolic       = MatLUFactorSymbolic_SeqAIJSpooles; 
  }
  PetscFunctionReturn(0);
}

/* Note the Petsc r and c permutations are ignored */
#undef __FUNCT__  
#define __FUNCT__ "MatLUFactorSymbolic_SeqAIJSpooles"
PetscErrorCode MatLUFactorSymbolic_SeqAIJSpooles(Mat A,IS r,IS c,MatFactorInfo *info,Mat *F)
{
  Mat          B;
  Mat_Spooles  *lu;
  PetscErrorCode ierr;
  int          m=A->m,n=A->n;

  PetscFunctionBegin;	
  /* Create the factorization matrix */  
  ierr = MatCreate(A->comm,&B);CHKERRQ(ierr);
  ierr = MatSetSizes(B,m,n,PETSC_DETERMINE,PETSC_DETERMINE);CHKERRQ(ierr);
  ierr = MatSetType(B,A->type_name);CHKERRQ(ierr);
  ierr = MatSeqAIJSetPreallocation(B,PETSC_NULL,PETSC_NULL);CHKERRQ(ierr);

  B->ops->lufactornumeric  = MatFactorNumeric_SeqAIJSpooles;
  B->factor                = FACTOR_LU;  

  lu                        = (Mat_Spooles*)(B->spptr);
  lu->options.symflag       = SPOOLES_NONSYMMETRIC;
  lu->options.pivotingflag  = SPOOLES_PIVOTING;
  lu->flg                   = DIFFERENT_NONZERO_PATTERN;
  lu->options.useQR         = PETSC_FALSE;

  if (!info->dtcol) {
    lu->options.pivotingflag  = SPOOLES_NO_PIVOTING;
  }
  *F = B;
  PetscFunctionReturn(0); 
}

/* Note the Petsc r and c permutations are ignored */
#undef __FUNCT__  
#define __FUNCT__ "MatQRFactorSymbolic_SeqAIJSpooles"
PetscErrorCode MatQRFactorSymbolic_SeqAIJSpooles(Mat A,IS r,IS c,MatFactorInfo *info,Mat *F)
{
  Mat          B;
  Mat_Spooles  *lu;   
  PetscErrorCode ierr;
  int          m=A->m,n=A->n;

  PetscFunctionBegin;
  SETERRQ(PETSC_ERR_SUP,"QR Factorization is unsupported as the Spooles implementation of QR is invalid.");
  /* Create the factorization matrix */  
  ierr = MatCreate(A->comm,&B);CHKERRQ(ierr);
  ierr = MatSetSizes(B,m,n,PETSC_DETERMINE,PETSC_DETERMINE);CHKERRQ(ierr);
  ierr = MatSetType(B,A->type_name);CHKERRQ(ierr);
  ierr = MatSeqAIJSetPreallocation(B,PETSC_NULL,PETSC_NULL);CHKERRQ(ierr);

  B->ops->lufactornumeric  = MatFactorNumeric_SeqAIJSpooles;
  B->factor                = FACTOR_LU;  

  lu                        = (Mat_Spooles*)(B->spptr);
  lu->options.symflag       = SPOOLES_NONSYMMETRIC;
  lu->options.pivotingflag  = SPOOLES_NO_PIVOTING;
  lu->flg                   = DIFFERENT_NONZERO_PATTERN;
  lu->options.useQR         = PETSC_TRUE;

  *F = B;
  PetscFunctionReturn(0); 
}

/* Note the Petsc r permutation is ignored */
#undef __FUNCT__  
#define __FUNCT__ "MatCholeskyFactorSymbolic_SeqSAIJSpooles"
PetscErrorCode MatCholeskyFactorSymbolic_SeqAIJSpooles(Mat A,IS r,MatFactorInfo *info,Mat *F)
{ 
  Mat         B;
  Mat_Spooles *lu;   
  PetscErrorCode ierr;
  int         m=A->m,n=A->n;

  PetscFunctionBegin;	
  /* Create the factorization matrix */
  ierr = MatCreate(A->comm,&B);CHKERRQ(ierr);
  ierr = MatSetSizes(B,m,n,PETSC_DETERMINE,PETSC_DETERMINE);CHKERRQ(ierr);
  ierr = MatSetType(B,A->type_name);CHKERRQ(ierr);
  ierr = MatSeqAIJSetPreallocation(B,PETSC_NULL,PETSC_NULL);CHKERRQ(ierr);

  B->ops->choleskyfactornumeric  = MatFactorNumeric_SeqAIJSpooles;
  B->ops->getinertia             = MatGetInertia_SeqSBAIJSpooles;
  B->factor                      = FACTOR_CHOLESKY;  

  lu                        = (Mat_Spooles*)(B->spptr);
  lu->options.pivotingflag  = SPOOLES_NO_PIVOTING;
  lu->options.symflag       = SPOOLES_SYMMETRIC;   /* default */
  lu->flg                   = DIFFERENT_NONZERO_PATTERN;
  lu->options.useQR         = PETSC_FALSE;

  *F = B;
  PetscFunctionReturn(0); 
}
