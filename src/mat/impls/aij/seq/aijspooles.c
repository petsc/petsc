/*$Id: aijspooles.c,v 1.10 2001/08/15 15:56:50 bsmith Exp $*/
/* 
   Provides an interface to the Spooles serial sparse solver
*/

#include "src/mat/impls/aij/seq/aij.h"

#if defined(PETSC_HAVE_SPOOLES) && !defined(PETSC_USE_SINGLE) 
#include "src/mat/impls/aij/seq/spooles.h"

/* Note the Petsc r and c permutations are ignored */
#undef __FUNCT__  
#define __FUNCT__ "MatLUFactorSymbolic_SeqAIJ_Spooles"
int MatLUFactorSymbolic_SeqAIJ_Spooles(Mat A,IS r,IS c,MatLUInfo *info,Mat *F)
{
  Mat_Spooles   *lu;   
  int           ierr,m=A->m,n=A->n;

  PetscFunctionBegin;	
  /* Create the factorization matrix F */  
  ierr = MatCreateSeqAIJ(A->comm,m,n,PETSC_NULL,PETSC_NULL,F);CHKERRQ(ierr);

  (*F)->ops->lufactornumeric  = MatFactorNumeric_SeqAIJ_Spooles;
  (*F)->factor                = FACTOR_LU;  

  ierr                      = PetscNew(Mat_Spooles,&lu);CHKERRQ(ierr); 
  (*F)->spptr               = (void*)lu; 
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
int MatQRFactorSymbolic_SeqAIJ_Spooles(Mat A,IS r,IS c,MatLUInfo *info,Mat *F)
{
  Mat_Spooles   *lu;   
  int           ierr,m=A->m,n=A->n;

  PetscFunctionBegin;	
  /* Create the factorization matrix F */  
  ierr = MatCreateSeqAIJ(A->comm,m,n,PETSC_NULL,PETSC_NULL,F);CHKERRQ(ierr);

  (*F)->ops->lufactornumeric  = MatFactorNumeric_SeqAIJ_Spooles;
  (*F)->factor                = FACTOR_LU;  

  ierr                      = PetscNew(Mat_Spooles,&lu);CHKERRQ(ierr); 
  (*F)->spptr               = (void*)lu; 
  lu->options.symflag       = SPOOLES_NONSYMMETRIC;
  lu->options.pivotingflag  = SPOOLES_NO_PIVOTING;
  lu->flg                   = DIFFERENT_NONZERO_PATTERN;
  lu->options.useQR         = PETSC_TRUE;

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
    A->ops->lufactorsymbolic = MatLUFactorSymbolic_SeqAIJ_Spooles; 
  } 
  PetscFunctionReturn(0);
}

#else

#undef __FUNCT__  
#define __FUNCT__ "MatUseSpooles_SeqAIJ"
int MatUseSpooles_SeqAIJ(Mat A)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

#endif


