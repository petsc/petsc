/*$Id: sbaijspooles.c,v 1.10 2001/08/15 15:56:50 bsmith Exp $*/
/* 
   Provides an interface to the Spooles serial sparse solver
*/

#include "src/mat/impls/sbaij/seq/sbaij.h"

#if defined(PETSC_HAVE_SPOOLES) && !defined(PETSC_USE_SINGLE) 
#include "src/mat/impls/aij/seq/spooles/spooles.h"

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
  Mat_Spooles          *lu;   
  int                  ierr,m=A->m,n=A->n;

  PetscFunctionBegin;	
  /* Create the factorization matrix F */  
  ierr = MatCreateSeqAIJ(A->comm,m,n,PETSC_NULL,PETSC_NULL,F);CHKERRQ(ierr);

  (*F)->ops->choleskyfactornumeric  = MatFactorNumeric_SeqAIJ_Spooles;
  (*F)->ops->getinertia             = MatGetInertia_SeqSBAIJ_Spooles;
  (*F)->factor                      = FACTOR_CHOLESKY;  

  ierr                      = PetscNew(Mat_Spooles,&lu);CHKERRQ(ierr); 
  (*F)->spptr               = (void*)lu;
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
  Mat_SeqSBAIJ *sbaij = (Mat_SeqSBAIJ*)A->data;
  int          bs = sbaij->bs;

  PetscFunctionBegin;
  if (bs > 1) SETERRQ1(1,"Block size %d not supported by Spooles",bs);
  A->ops->choleskyfactorsymbolic = MatCholeskyFactorSymbolic_SeqSBAIJ_Spooles;  
  PetscFunctionReturn(0);
}

#else

#undef __FUNCT__  
#define __FUNCT__ "MatUseSpooles_SeqSBAIJ"
int MatUseSpooles_SeqSBAIJ(Mat A)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

#endif


