/*$Id: mpisbaijspooles.c,v 1.10 2001/08/15 15:56:50 bsmith Exp $*/
/* 
   Provides an interface to the Spooles parallel sparse solver (MPI SPOOLES)
*/


#include "src/mat/impls/sbaij/mpi/mpisbaij.h"

#if defined(PETSC_HAVE_SPOOLES) && !defined(PETSC_USE_SINGLE) && !defined(PETSC_USE_COMPLEX)
#include "src/mat/impls/aij/mpi/mpispooles.h"

/* Note the Petsc r permutation is ignored */
#undef __FUNCT__  
#define __FUNCT__ "MatCholeskyFactorSymbolic_MPISBAIJ_Spooles"
int MatCholeskyFactorSymbolic_MPISBAIJ_Spooles(Mat A,IS r,PetscReal f,Mat *F)
{
  Mat_MPISBAIJ     *mat = (Mat_MPISBAIJ*)A->data;
  Mat_MPISpooles   *lu;   
  int              ierr,M=A->M,N=A->N;
  
  PetscFunctionBegin;	
  A->ops->lufactornumeric  = MatFactorNumeric_MPIAIJ_Spooles;  

  /* Create the factorization matrix F */  
  ierr = MatCreateMPIAIJ(A->comm,PETSC_DECIDE,PETSC_DECIDE,M,N,0,PETSC_NULL,0,PETSC_NULL,F);CHKERRQ(ierr);
  
  (*F)->ops->choleskyfactornumeric = MatFactorNumeric_MPIAIJ_Spooles;
  (*F)->factor                     = FACTOR_CHOLESKY;  

  ierr = PetscNew(Mat_MPISpooles,&lu);CHKERRQ(ierr); 
  (*F)->spptr      = (void*)lu;
  lu->symflag      = SPOOLES_SYMMETRIC;
  lu->pivotingflag = SPOOLES_NO_PIVOTING; 
  lu->flg          = DIFFERENT_NONZERO_PATTERN;

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
