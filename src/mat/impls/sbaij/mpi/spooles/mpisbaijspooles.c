/*$Id: mpisbaijspooles.c,v 1.10 2001/08/15 15:56:50 bsmith Exp $*/
/* 
   Provides an interface to the Spooles parallel sparse solver (MPI SPOOLES)
*/


#include "src/mat/impls/sbaij/mpi/mpisbaij.h"
#include "src/mat/impls/aij/seq/spooles/spooles.h"

#undef __FUNCT__
#define __FUNCT__ "MatAssemblyEnd_MPISBAIJ_Spooles"
int MatAssemblyEnd_MPISBAIJ_Spooles(Mat A,MatAssemblyType mode) {
  int         ierr;
  Mat_Spooles *lu=(Mat_Spooles *)(A->spptr);

  PetscFunctionBegin;
  ierr = (*lu->MatAssemblyEnd)(A,mode);CHKERRQ(ierr);
  ierr = MatUseSpooles_MPISBAIJ(A);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* Note the Petsc r permutation is ignored */
#undef __FUNCT__  
#define __FUNCT__ "MatCholeskyFactorSymbolic_MPISBAIJ_Spooles"
int MatCholeskyFactorSymbolic_MPISBAIJ_Spooles(Mat A,IS r,MatFactorInfo *info,Mat *F)
{
  Mat           B;
  Mat_Spooles   *lu;   
  int           ierr;
  
  PetscFunctionBegin;	
  A->ops->lufactornumeric  = MatFactorNumeric_MPIAIJ_Spooles;  

  /* Create the factorization matrix */  
  ierr = MatCreate(A->comm,A->m,A->n,A->M,A->N,&B);
  ierr = MatSetType(B,MATMPIAIJSPOOLES);CHKERRQ(ierr);
  ierr = MatMPIAIJSetPreallocation(B,0,PETSC_NULL,0,PETSC_NULL);CHKERRQ(ierr);
  
  B->ops->choleskyfactornumeric = MatFactorNumeric_MPIAIJ_Spooles;
  B->factor                     = FACTOR_CHOLESKY;  

  lu                       = (Mat_Spooles*)(B->spptr);
  lu->options.pivotingflag = SPOOLES_NO_PIVOTING; 
  lu->flg                  = DIFFERENT_NONZERO_PATTERN;
  lu->options.useQR        = PETSC_FALSE;
  lu->options.symflag      = SPOOLES_SYMMETRIC;  /* default */

  ierr = MPI_Comm_dup(A->comm,&(lu->comm_spooles));CHKERRQ(ierr);
  *F = B;
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

EXTERN_C_BEGIN
#undef __FUNCT__
#define __FUNCT__ "MatCreate_MPISBAIJ_Spooles"
int MatCreate_MPISBAIJ_Spooles(Mat A) {
  int ierr;
  Mat_Spooles *lu;

  PetscFunctionBegin;
  ierr = MatSetType(A,MATMPISBAIJ);CHKERRQ(ierr);
  ierr = MatUseSpooles_MPISBAIJ(A);CHKERRQ(ierr);

  ierr                = PetscNew(Mat_Spooles,&lu);CHKERRQ(ierr); 
  lu->MatAssemblyEnd  = A->ops->assemblyend;
  lu->MatDestroy      = A->ops->destroy;
  A->spptr            = (void*)lu;
  A->ops->assemblyend = MatAssemblyEnd_MPISBAIJ_Spooles;
  A->ops->destroy     = MatDestroy_SeqSBAIJ_Spooles;
  PetscFunctionReturn(0);
}
EXTERN_C_END
