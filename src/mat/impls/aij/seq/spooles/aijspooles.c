/*$Id: aijspooles.c,v 1.10 2001/08/15 15:56:50 bsmith Exp $*/
/* 
   Provides an interface to the Spooles serial sparse solver
*/

#include "src/mat/impls/aij/seq/aij.h"

#if defined(PETSC_HAVE_SPOOLES) && !defined(PETSC_USE_SINGLE) 
#include "src/mat/impls/aij/seq/spooles/spooles.h"

#undef __FUNCT__
#define __FUNCT__ "MatView_SeqAIJ_Spooles"
int MatView_SeqAIJ_Spooles(Mat A,PetscViewer viewer)
{
  int                   ierr;
  PetscTruth            isascii;
  PetscViewerFormat     format;
  PetscObjectContainer  container;
  Mat_Spooles           *lu=(Mat_Spooles*)(A->spptr);

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
#define __FUNCT__ "MatAssemblyEnd_SeqAIJ_Spooles"
int MatAssemblyEnd_SeqAIJ_Spooles(Mat A,MatAssemblyType mode) {
  int         ierr;
  Mat_Spooles *lu=(Mat_Spooles *)(A->spptr);

  PetscFunctionBegin;
  ierr = (*lu->MatAssemblyEnd)(A,mode);CHKERRQ(ierr);
  ierr = MatUseSpooles_SeqAIJ(A);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* Note the Petsc r and c permutations are ignored */
#undef __FUNCT__  
#define __FUNCT__ "MatLUFactorSymbolic_SeqAIJ_Spooles"
int MatLUFactorSymbolic_SeqAIJ_Spooles(Mat A,IS r,IS c,MatFactorInfo *info,Mat *F)
{
  Mat_Spooles   *lu;   
  int           ierr,m=A->m,n=A->n;

  PetscFunctionBegin;	
  /* Create the factorization matrix F */  
  ierr = MatCreate(A->comm,m,n,PETSC_NULL,PETSC_NULL,F);CHKERRQ(ierr);
  ierr = MatSetType(*F,MATSEQAIJSPOOLES);CHKERRQ(ierr);
  ierr = MatSeqAIJSetPreallocation(*F,PETSC_NULL,PETSC_NULL);CHKERRQ(ierr);

  (*F)->ops->lufactornumeric  = MatFactorNumeric_SeqAIJ_Spooles;
  (*F)->factor                = FACTOR_LU;  

  lu                        = (Mat_Spooles*)((*F)->spptr);
  lu->options.symflag       = SPOOLES_NONSYMMETRIC;
  lu->options.pivotingflag  = SPOOLES_PIVOTING;
  lu->flg                   = DIFFERENT_NONZERO_PATTERN;
  lu->options.useQR         = PETSC_FALSE;

  if (info->dtcol == 0.0) {
    lu->options.pivotingflag  = SPOOLES_NO_PIVOTING;
  }
 
  PetscFunctionReturn(0); 
}

/* Note the Petsc r and c permutations are ignored */
#undef __FUNCT__  
#define __FUNCT__ "MatQRFactorSymbolic_SeqAIJ_Spooles"
int MatQRFactorSymbolic_SeqAIJ_Spooles(Mat A,IS r,IS c,MatFactorInfo *info,Mat *F)
{
  Mat_Spooles   *lu;   
  int           ierr,m=A->m,n=A->n;

  PetscFunctionBegin;	
  /* Create the factorization matrix F */  
  ierr = MatCreate(A->comm,m,n,PETSC_NULL,PETSC_NULL,F);CHKERRQ(ierr);
  ierr = MatSetType(*F,MATSEQAIJSPOOLES);CHKERRQ(ierr);
  ierr = MatSeqAIJSetPreallocation(*F,PETSC_NULL,PETSC_NULL);CHKERRQ(ierr);

  (*F)->ops->lufactornumeric  = MatFactorNumeric_SeqAIJ_Spooles;
  (*F)->factor                = FACTOR_LU;  

  lu                        = (Mat_Spooles*)((*F)->spptr);
  lu->options.symflag       = SPOOLES_NONSYMMETRIC;
  lu->options.pivotingflag  = SPOOLES_NO_PIVOTING;
  lu->flg                   = DIFFERENT_NONZERO_PATTERN;
  lu->options.useQR         = PETSC_TRUE;

  PetscFunctionReturn(0); 
}

/* Note the Petsc r permutation is ignored */
#undef __FUNCT__  
#define __FUNCT__ "MatCholeskyFactorSymbolic_SeqSAIJ_Spooles"
int MatCholeskyFactorSymbolic_SeqAIJ_Spooles(Mat A,IS r,MatFactorInfo *info,Mat *F)
{ 
  Mat_Spooles          *lu;   
  int                  ierr,m=A->m,n=A->n;

  PetscFunctionBegin;	
  /* Create the factorization matrix F */
  ierr = MatCreate(A->comm,m,n,PETSC_NULL,PETSC_NULL,F);CHKERRQ(ierr);
  ierr = MatSetType(*F,MATSEQAIJSPOOLES);CHKERRQ(ierr);
  ierr = MatSeqAIJSetPreallocation(*F,PETSC_NULL,PETSC_NULL);CHKERRQ(ierr);

  (*F)->ops->choleskyfactornumeric  = MatFactorNumeric_SeqAIJ_Spooles;
/*   (*F)->ops->getinertia             = MatGetInertia_SeqSBAIJ_Spooles; */
  (*F)->factor                      = FACTOR_CHOLESKY;  

  lu                        = (Mat_Spooles*)((*F)->spptr);
  lu->options.pivotingflag  = SPOOLES_NO_PIVOTING;
  lu->options.symflag       = SPOOLES_SYMMETRIC;   /* default */
  lu->flg                   = DIFFERENT_NONZERO_PATTERN;
  lu->options.useQR         = PETSC_FALSE;

  PetscFunctionReturn(0); 
}

#undef __FUNCT__  
#define __FUNCT__ "MatUseSpooles_SeqAIJ"
int MatUseSpooles_SeqAIJ(Mat A)
{
  int          ierr;
  PetscTruth   useQR=PETSC_FALSE;
 
  PetscFunctionBegin;
  ierr = PetscOptionsHasName(A->prefix,"-mat_aij_spooles_qr",&useQR);CHKERRQ(ierr);
  if (useQR){
    A->ops->lufactorsymbolic = MatQRFactorSymbolic_SeqAIJ_Spooles;  
  } else {
    A->ops->choleskyfactorsymbolic = MatCholeskyFactorSymbolic_SeqAIJ_Spooles;
    A->ops->lufactorsymbolic       = MatLUFactorSymbolic_SeqAIJ_Spooles; 
  } 
  PetscFunctionReturn(0);
}

#endif


