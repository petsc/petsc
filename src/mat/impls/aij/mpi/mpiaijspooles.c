/*$Id: mpiaijspooles.c,v 1.10 2001/08/15 15:56:50 bsmith Exp $*/
/* 
   Provides an interface to the Spooles parallel sparse solver (MPI SPOOLES)
*/


#include "src/mat/impls/aij/mpi/mpiaij.h"
#include "src/mat/impls/aij/seq/spooles.h"

#if defined(PETSC_HAVE_SPOOLES) && !defined(PETSC_USE_SINGLE) && !defined(PETSC_USE_COMPLEX)
/* Note the Petsc r and c permutations are ignored */
#undef __FUNCT__  
#define __FUNCT__ "MatLUFactorSymbolic_MPIAIJ_Spooles"
int MatLUFactorSymbolic_MPIAIJ_Spooles(Mat A,IS r,IS c,MatLUInfo *info,Mat *F)
{
  Mat_MPIAIJ           *mat = (Mat_MPIAIJ*)A->data;
  Mat_MPISpooles   *lu;   
  char                 buff[32],*ordertype[] = {"BestOfNDandMS","MMD","MS","ND"}; 
  PetscTruth           flg;
  double               *opcounts,  minops, cutoff, *val;
  Graph                *graph ;
  IVL                  *adjIVL;
  DV                   *cumopsDV ;
  InpMtx               *newA ; 
  Mat_MPIAIJ           *aij =  (Mat_MPIAIJ*)A->data;  
  Mat_SeqAIJ           *aa=(Mat_SeqAIJ*)(aij->A)->data,*bb=(Mat_SeqAIJ*)(aij->B)->data;
  PetscScalar          *av=aa->a, *bv=bb->a; 
  int                  *ai=aa->i, *aj=aa->j, *bi=bb->i,*bj=bb->j, nz,
                       i,j,irow,jcol,countA,countB,jB,*row,*col,colA_start,jj;
  int                  ierr,size,rank,M=A->M,N=A->N,m=A->m,root,nedges;

  PetscFunctionBegin;	
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRQ(ierr);

  A->ops->lufactornumeric  = MatLUFactorNumeric_MPIAIJ_Spooles; 

  /* Create the factorization matrix F */  
  ierr = MatCreateMPIAIJ(A->comm,PETSC_DECIDE,PETSC_DECIDE,M,N,0,PETSC_NULL,0,PETSC_NULL,F);CHKERRQ(ierr);
  
  (*F)->ops->lufactornumeric = MatLUFactorNumeric_MPIAIJ_Spooles;
  (*F)->factor               = FACTOR_LU;  

  ierr = PetscNew(Mat_MPISpooles,&lu);CHKERRQ(ierr); 
  (*F)->spptr      = (void*)lu;
  lu->symflag      = SPOOLES_NONSYMMETRIC;
  lu->pivotingflag = SPOOLES_PIVOTING; 
  lu->flg          = DIFFERENT_NONZERO_PATTERN;

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
