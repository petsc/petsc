/*$Id: aijspooles.c,v 1.10 2001/08/15 15:56:50 bsmith Exp $*/
/* 
   Provides an interface to the Spooles serial sparse solver
*/

#include "src/mat/impls/aij/seq/spooles/spooles.h"

#undef __FUNCT__
#define __FUNCT__ "MatView_SeqAIJSpooles"
int MatView_SeqAIJSpooles(Mat A,PetscViewer viewer)
{
  int               ierr;
  PetscTruth        isascii;
  PetscViewerFormat format;
  Mat_Spooles       *lu=(Mat_Spooles*)(A->spptr);

  PetscFunctionBegin;
  ierr = (*lu->MatView)(A,viewer);CHKERRQ(ierr);

  ierr = PetscTypeCompare((PetscObject)viewer,PETSC_VIEWER_ASCII,&isascii);CHKERRQ(ierr);
  if (isascii) {
    ierr = PetscViewerGetFormat(viewer,&format);CHKERRQ(ierr);
    if (format == PETSC_VIEWER_ASCII_FACTOR_INFO) {
      ierr = MatFactorInfo_Spooles(A,viewer);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatAssemblyEnd_SeqAIJSpooles"
int MatAssemblyEnd_SeqAIJSpooles(Mat A,MatAssemblyType mode) {
  int         ierr;
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
int MatLUFactorSymbolic_SeqAIJSpooles(Mat A,IS r,IS c,MatFactorInfo *info,Mat *F)
{
  Mat          B;
  Mat_Spooles  *lu;
  int          ierr,m=A->m,n=A->n;

  PetscFunctionBegin;	
  /* Create the factorization matrix */  
  ierr = MatCreate(A->comm,m,n,PETSC_NULL,PETSC_NULL,&B);CHKERRQ(ierr);
  ierr = MatSetType(B,MATSEQAIJSPOOLES);CHKERRQ(ierr);
  ierr = MatSeqAIJSetPreallocation(B,PETSC_NULL,PETSC_NULL);CHKERRQ(ierr);

  B->ops->lufactornumeric  = MatFactorNumeric_SeqAIJSpooles;
  B->factor                = FACTOR_LU;  

  lu                        = (Mat_Spooles*)(B->spptr);
  lu->options.symflag       = SPOOLES_NONSYMMETRIC;
  lu->options.pivotingflag  = SPOOLES_PIVOTING;
  lu->flg                   = DIFFERENT_NONZERO_PATTERN;
  lu->options.useQR         = PETSC_FALSE;

  if (info->dtcol == 0.0) {
    lu->options.pivotingflag  = SPOOLES_NO_PIVOTING;
  }
  *F = B;
  PetscFunctionReturn(0); 
}

/* Note the Petsc r and c permutations are ignored */
#undef __FUNCT__  
#define __FUNCT__ "MatQRFactorSymbolic_SeqAIJSpooles"
int MatQRFactorSymbolic_SeqAIJSpooles(Mat A,IS r,IS c,MatFactorInfo *info,Mat *F)
{
  Mat          B;
  Mat_Spooles  *lu;   
  int          ierr,m=A->m,n=A->n;

  PetscFunctionBegin;
  SETERRQ(PETSC_ERR_SUP,"QR Factorization is unsupported as the Spooles implementation of QR is invalid.");
  /* Create the factorization matrix */  
  ierr = MatCreate(A->comm,m,n,PETSC_NULL,PETSC_NULL,&B);CHKERRQ(ierr);
  ierr = MatSetType(B,MATSEQAIJSPOOLES);CHKERRQ(ierr);
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
int MatCholeskyFactorSymbolic_SeqAIJSpooles(Mat A,IS r,MatFactorInfo *info,Mat *F)
{ 
  Mat         B;
  Mat_Spooles *lu;   
  int         ierr,m=A->m,n=A->n;

  PetscFunctionBegin;	
  /* Create the factorization matrix */
  ierr = MatCreate(A->comm,m,n,PETSC_NULL,PETSC_NULL,&B);CHKERRQ(ierr);
  ierr = MatSetType(B,MATSEQAIJSPOOLES);CHKERRQ(ierr);
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
