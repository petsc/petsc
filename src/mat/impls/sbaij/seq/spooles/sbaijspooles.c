/*$Id: sbaijspooles.c,v 1.10 2001/08/15 15:56:50 bsmith Exp $*/
/* 
   Provides an interface to the Spooles serial sparse solver
*/

#include "src/mat/impls/aij/seq/spooles/spooles.h"

#undef __FUNCT__  
#define __FUNCT__ "MatDestroy_SeqAIJ_Spooles"
int MatDestroy_SeqSBAIJ_Spooles(Mat A)
{
  Mat_Spooles *lu = (Mat_Spooles*)A->spptr; 
  int         ierr,(*destroy)(Mat);
  
  PetscFunctionBegin;
  /* SeqSBAIJ_Spooles isn't really the spooles type matrix, */
  /* so we don't have to clean up the stuff set by spooles */
  /* as in MatDestroy_SeqAIJ_Spooles */
  destroy = lu->MatDestroy;
  ierr    = PetscFree(lu);CHKERRQ(ierr); 
  ierr    = (*destroy)(A);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatAssemblyEnd_SeqSBAIJ_Spooles"
int MatAssemblyEnd_SeqSBAIJ_Spooles(Mat A,MatAssemblyType mode) {
  int         ierr;
  Mat_Spooles *lu=(Mat_Spooles *)(A->spptr);

  PetscFunctionBegin;
  ierr = (*lu->MatAssemblyEnd)(A,mode);CHKERRQ(ierr);
  ierr = MatUseSpooles_SeqSBAIJ(A);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* 
  input:
   F:                 numeric factor
  output:
   nneg, nzero, npos: matrix inertia 
*/

#undef __FUNCT__  
#define __FUNCT__ "MatGetInertia_SeqSBAIJ_Spooles"
int MatGetInertia_SeqSBAIJ_Spooles(Mat F,int *nneg,int *nzero,int *npos)
{ 
  Mat_Spooles          *lu = (Mat_Spooles*)F->spptr; 
  int                  ierr,neg,zero,pos;

  PetscFunctionBegin;
  FrontMtx_inertia(lu->frontmtx, &neg, &zero, &pos) ;
  if(nneg)  *nneg  = neg;
  if(nzero) *nzero = zero;
  if(npos)  *npos  = pos;
  PetscFunctionReturn(0);
}

/* Note the Petsc r permutation is ignored */
#undef __FUNCT__  
#define __FUNCT__ "MatCholeskyFactorSymbolic_SeqSBAIJ_Spooles"
int MatCholeskyFactorSymbolic_SeqSBAIJ_Spooles(Mat A,IS r,MatFactorInfo *info,Mat *F)
{ 
  Mat         B;
  Mat_Spooles *lu;   
  int         ierr,m=A->m,n=A->n;

  PetscFunctionBegin;	
  /* Create the factorization matrix */  
  ierr = MatCreate(A->comm,m,n,m,n,&B);
  ierr = MatSetType(B,MATSEQAIJSPOOLES);CHKERRQ(ierr);
  ierr = MatSeqAIJSetPreallocation(B,PETSC_NULL,PETSC_NULL);CHKERRQ(ierr);

  B->ops->choleskyfactornumeric  = MatFactorNumeric_SeqAIJ_Spooles;
  B->ops->getinertia             = MatGetInertia_SeqSBAIJ_Spooles;
  B->factor                      = FACTOR_CHOLESKY;  

  lu                        = (Mat_Spooles *)(B->spptr);
  lu->options.pivotingflag  = SPOOLES_NO_PIVOTING;
  lu->options.symflag       = SPOOLES_SYMMETRIC;   /* default */
  lu->flg                   = DIFFERENT_NONZERO_PATTERN;
  lu->options.useQR         = PETSC_FALSE;

  PetscFunctionReturn(0); 
}

#undef __FUNCT__  
#define __FUNCT__ "MatUseSpooles_SeqSBAIJ"
int MatUseSpooles_SeqSBAIJ(Mat A)
{
  int ierr,bs;

  PetscFunctionBegin;
  ierr = MatGetBlockSize(A,&bs);CHKERRQ(ierr);
  if (bs > 1) SETERRQ1(1,"Block size %d not supported by Spooles",bs);
  A->ops->choleskyfactorsymbolic = MatCholeskyFactorSymbolic_SeqSBAIJ_Spooles;  
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
#undef __FUNCT__
#define __FUNCT__ "MatCreate_SeqSBAIJ_Spooles"
int MatCreate_SeqSBAIJ_Spooles(Mat A) {
  int ierr;
  Mat_Spooles *lu;

  PetscFunctionBegin;
  ierr = MatSetType(A,MATSEQSBAIJ);CHKERRQ(ierr);
  ierr = MatUseSpooles_SeqSBAIJ(A);CHKERRQ(ierr);

  ierr                = PetscNew(Mat_Spooles,&lu);CHKERRQ(ierr); 
  lu->MatAssemblyEnd  = A->ops->assemblyend;
  lu->MatDestroy      = A->ops->destroy;
  A->spptr            = (void*)lu;
  A->ops->assemblyend = MatAssemblyEnd_SeqSBAIJ_Spooles;
  A->ops->destroy     = MatDestroy_SeqSBAIJ_Spooles;
  PetscFunctionReturn(0);
}
EXTERN_C_END
