/*$Id: mpiaijspooles.c,v 1.10 2001/08/15 15:56:50 bsmith Exp $*/
/* 
   Provides an interface to the Spooles parallel sparse solver (MPI SPOOLES)
*/


#include "src/mat/impls/aij/mpi/mpiaij.h"

#if defined(PETSC_HAVE_SPOOLES) && !defined(PETSC_USE_SINGLE)
#include "src/mat/impls/aij/seq/spooles/spooles.h"

/* Note the Petsc r and c permutations are ignored */
#undef __FUNCT__  
#define __FUNCT__ "MatLUFactorSymbolic_MPIAIJ_Spooles"
int MatLUFactorSymbolic_MPIAIJ_Spooles(Mat A,IS r,IS c,MatFactorInfo *info,Mat *F)
{
  Mat_MPIAIJ    *mat = (Mat_MPIAIJ*)A->data;
  Mat_Spooles   *lu;   
  int           ierr;

  PetscFunctionBegin;	
  A->ops->lufactornumeric = MatFactorNumeric_MPIAIJ_Spooles; 

  /* Create the factorization matrix F */  
  ierr = MatCreateMPIAIJ(A->comm,A->m,A->n,A->M,A->N,0,PETSC_NULL,0,PETSC_NULL,F);CHKERRQ(ierr);
  
  (*F)->ops->lufactornumeric = MatFactorNumeric_MPIAIJ_Spooles;
  (*F)->factor               = FACTOR_LU;  

  ierr                     = PetscNew(Mat_Spooles,&lu);CHKERRQ(ierr); 
  (*F)->spptr              = (void*)lu;
  lu->options.symflag      = SPOOLES_NONSYMMETRIC;
  lu->options.pivotingflag = SPOOLES_PIVOTING; 
  lu->flg                  = DIFFERENT_NONZERO_PATTERN;
  lu->options.useQR        = PETSC_FALSE;

  ierr = MPI_Comm_dup(A->comm,&(lu->comm_spooles));CHKERRQ(ierr);

  if (info->dtcol == 0.0) {
    lu->options.pivotingflag  = SPOOLES_NO_PIVOTING;
  }

  PetscFunctionReturn(0); 
}

#undef __FUNCT__  
#define __FUNCT__ "MatUseSpooles_MPIAIJ"
int MatUseSpooles_MPIAIJ(Mat A)
{
  PetscFunctionBegin;
  A->ops->lufactorsymbolic = MatLUFactorSymbolic_MPIAIJ_Spooles;  
  PetscFunctionReturn(0);
}

#else

#undef __FUNCT__  
#define __FUNCT__ "MatUseSpooles_MPIAIJ"
int MatUseSpooles_MPIAIJ(Mat A)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

#endif
