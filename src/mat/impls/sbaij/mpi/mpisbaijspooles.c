/*$Id: mpisbaijspooles.c,v 1.10 2001/08/15 15:56:50 bsmith Exp $*/
/* 
   Provides an interface to the Spooles parallel sparse solver (MPI SPOOLES)
*/


#include "src/mat/impls/sbaij/mpi/mpisbaij.h"

#if defined(PETSC_HAVE_SPOOLES) && !defined(PETSC_USE_SINGLE)
#include "src/mat/impls/aij/seq/spooles.h"

/* Note the Petsc r permutation is ignored */
#undef __FUNCT__  
#define __FUNCT__ "MatCholeskyFactorSymbolic_MPISBAIJ_Spooles"
int MatCholeskyFactorSymbolic_MPISBAIJ_Spooles(Mat A,IS r,PetscReal f,Mat *F)
{
  Mat_MPISBAIJ  *mat = (Mat_MPISBAIJ*)A->data;
  Mat_Spooles   *lu;   
  int           ierr;
  
  PetscFunctionBegin;	
  A->ops->lufactornumeric  = MatFactorNumeric_MPIAIJ_Spooles;  

  /* Create the factorization matrix F */  
  ierr = MatCreateMPIAIJ(A->comm,A->m,A->n,A->M,A->N,0,PETSC_NULL,0,PETSC_NULL,F);CHKERRQ(ierr);
  
  (*F)->ops->choleskyfactornumeric = MatFactorNumeric_MPIAIJ_Spooles;
  (*F)->factor                     = FACTOR_CHOLESKY;  

  ierr                     = PetscNew(Mat_Spooles,&lu);CHKERRQ(ierr); 
  (*F)->spptr              = (void*)lu;  
  lu->options.pivotingflag = SPOOLES_NO_PIVOTING; 
  lu->flg                  = DIFFERENT_NONZERO_PATTERN;
  lu->options.useQR        = PETSC_FALSE;
#if defined(PETSC_USE_COMPLEX)
  lu->options.symflag      = SPOOLES_HERMITIAN;
  lu->options.typeflag     = SPOOLES_COMPLEX;
#else
  lu->options.symflag      = SPOOLES_SYMMETRIC;
  lu->options.typeflag     = SPOOLES_REAL;
#endif

  PetscFunctionReturn(0); 
}

#undef __FUNCT__  
#define __FUNCT__ "MatUseSpooles_MPISBAIJ"
int MatUseSpooles_MPISBAIJ(Mat A)
{
  Mat_MPISBAIJ *sbaij = (Mat_MPISBAIJ*)A->data;
  int          bs = sbaij->bs;

  PetscFunctionBegin;
  if (bs > 1) SETERRQ1(1,"Block size %d not supported by Spooles",bs);
  A->ops->choleskyfactorsymbolic = MatCholeskyFactorSymbolic_MPISBAIJ_Spooles;  
  PetscFunctionReturn(0);
}

#else

#undef __FUNCT__  
#define __FUNCT__ "MatUseSpooles_MPISBAIJ"
int MatUseSpooles_MPISBAIJ(Mat A)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

#endif
