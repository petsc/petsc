/*$Id: dsc.c,v 1.1 2000/12/15 18:21:57 balay Exp balay $*/
/* 
        Provides an interface to the DSCPACK-S
*/

#include "src/mat/impls/aij/seq/aij.h" 
#if defined(PETSC_HAVE_DSCPACK) && !defined(PETSC_USE_COMPLEX) 
#include "dscmain.h"


/* golbal data for DSCPACK  communcation between reordering and factorization */
int dsc_s_nz = 0;      /* num of nonzeros in lower/upper half of the matrix */
int dsc_pass = 0;      /* num of numeric factorizations for a single symbolic factorization */

#undef __FUNC__  
#define __FUNC__ "MatDestroy_SeqAIJ_DSC_Fac"
int MatDestroy_SeqAIJ_DSC_Fac(Mat A)
{
  int ierr;

  PetscFunctionBegin;  
  ierr = MatDestroy_SeqAIJ(A);CHKERRQ(ierr);
  DSC_Final_Free_All();  /* free Cholesky factor and other relevant data structures */
  /* DSC_Do_Stats(); */
  DSC_Clean_Up();        /* terminate DSC solver */

  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__  "MatSolve_SeqAIJ_DSC"
int MatSolve_SeqAIJ_DSC(Mat A,Vec b,Vec x)
{
  double *rhs_vec, *solution_vec;
  int    ierr;
  
  PetscFunctionBegin;  
  ierr = VecGetArray(x, &solution_vec);CHKERRQ(ierr);
  ierr = VecGetArray(b, &rhs_vec);CHKERRQ(ierr);
      
  DSC_Input_Rhs(rhs_vec, A->m);
  DSC_N_Solve();
  if (DSC_STATUS.cont_or_stop == DSC_STOP_TYPE) goto ERROR_HANDLE;

  DSC_Get_Solution(solution_vec);
  if (DSC_STATUS.cont_or_stop == DSC_STOP_TYPE) goto ERROR_HANDLE;  
         
  ierr = VecRestoreArray(x, &solution_vec);CHKERRQ(ierr);
  ierr = VecRestoreArray(b, &rhs_vec);CHKERRQ(ierr);

ERROR_HANDLE:  
  if (DSC_STATUS.error_code != DSC_NO_ERROR) {
    DSC_Error_Display();
    SETERRQ(PETSC_ERR_ARG_SIZ, "DSC ERROR");
  }

  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MatCholeskyFactorNumeric_SeqAIJ_DSC"
int MatCholeskyFactorNumeric_SeqAIJ_DSC(Mat A, Mat *F)
{
  Mat_SeqAIJ       *a=(Mat_SeqAIJ*)A->data, *fac=(Mat_SeqAIJ*)(*F)->data;
  IS               iscol = fac->col,isicol = fac->icol;
  PetscTruth       flg;
  int              m,ierr;  
  int              *ai = a->i, *aj = a->j;
  int              *perm, *iperm;
  real_number_type *a_nonz = a->a, *s_a_nonz;

  PetscFunctionBegin;
  m = A->m; 
  if (dsc_pass == 0){ /* check the arguments */
    if (m != A->n) SETERRQ(PETSC_ERR_ARG_SIZ, "matrix must be square"); 
    ierr = PetscTypeCompare((PetscObject)A,MATSEQAIJ,&flg);CHKERRQ(ierr);
    if (!flg) SETERRQ(PETSC_ERR_ARG_SIZ, "matrix must be Seq_AIJ");
    if (m != (*F)->m) SETERRQ(PETSC_ERR_ARG_SIZ, "factorization struct inconsistent"); 

  } else { /* frees up Cholesky factor used by previous numeric factorization */
    DSC_N_Fact_Free();
    DSC_Re_Init();
  }

  ierr  = ISGetIndices(iscol,&perm);CHKERRQ(ierr);
  ierr  = ISGetIndices(isicol,&iperm);CHKERRQ(ierr);
  ierr =Initialize_A_Nonz(m,ai,aj,a_nonz,dsc_s_nz,perm,iperm, &s_a_nonz);
  if (ierr <0) SETERRQ(PETSC_ERR_ARG_SIZ, "Error setting up permuted nonzero vector");
              
  DSC_N_Fact(s_a_nonz); 

  free ((char *) s_a_nonz);
  dsc_pass++;
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MatCholeskyFactorSymbolic_SeqAIJ_DSC"
int MatCholeskyFactorSymbolic_SeqAIJ_DSC(Mat A,IS perm,PetscReal f,Mat *F)
{
  /************************************************************************/
  /* Input                                                                */
  /*     A     - matrix to factor                                         */
  /*     perm  - row/col permutation (ignored)                            */
  /*     f     - fill (ignored)                                           */
  /*                                                                      */
  /* Output                                                               */
  /*     F  - matrix storing partial information for factorization        */
  /************************************************************************/

  int             ierr,m;
  int             max_mem_estimate, max_single_malloc_blk,MAX_MEM_ALLOWED=800;
  PetscTruth      flg;
  IS              iperm;
  Mat_SeqAIJ      *b;
 
  PetscFunctionBegin;
  m = A->m;
  if (m != A->n) SETERRQ(PETSC_ERR_ARG_SIZ, "matrix must be square"); 
  ierr = PetscTypeCompare((PetscObject)A,MATSEQAIJ,&flg);CHKERRQ(ierr);
  if (!flg) SETERRQ(PETSC_ERR_ARG_SIZ, "matrix must be Seq_AIJ");

  /* Create the factorization */     
  ierr = MatCreateSeqAIJ(A->comm, m, m, 0, PETSC_NULL, F);CHKERRQ(ierr); 
  
  (*F)->ops->destroy               = MatDestroy_SeqAIJ_DSC_Fac;
  (*F)->ops->choleskyfactornumeric = MatCholeskyFactorNumeric_SeqAIJ_DSC;
  (*F)->ops->solve                 = MatSolve_SeqAIJ_DSC;
  (*F)->factor                     = FACTOR_CHOLESKY;

  b = (Mat_SeqAIJ*)(*F)->data;
  ierr = ISInvertPermutation(perm,PETSC_DECIDE,&iperm);CHKERRQ(ierr);
  (b)->col  = perm;
  (b)->icol = iperm;

  /* Symbolic factorization  */
  DSC_S_Fact (&max_mem_estimate, &max_single_malloc_blk, MAX_MEM_ALLOWED);
  if (DSC_STATUS.cont_or_stop == DSC_STOP_TYPE)  goto ERROR_HANDLE;

ERROR_HANDLE:  
  if (DSC_STATUS.error_code != DSC_NO_ERROR) {
    DSC_Error_Display();
    SETERRQ(PETSC_ERR_ARG_SIZ, "DSC_ERROR");
  }

  PetscFunctionReturn(0);
}

EXTERN_C_END

#undef __FUNC__  
#define __FUNC__ "MatSeqAIJUseDSC"
int MatSeqAIJUseDSC(Mat A)
{
  int        ierr; 
  PetscTruth flg;
   
  PetscFunctionBegin;
  PetscValidHeaderSpecific(A,MAT_COOKIE);
  if (A->m != A->n) SETERRQ(PETSC_ERR_ARG_SIZ, "matrix must be square");

  ierr = PetscTypeCompare((PetscObject)A,MATSEQAIJ,&flg);CHKERRQ(ierr);
  if (!flg) SETERRQ(PETSC_ERR_ARG_SIZ, "matrix must be SeqAIJ"); 
    
  A->ops->choleskyfactorsymbolic = MatCholeskyFactorSymbolic_SeqAIJ_DSC; 
  PLogInfo(0,"Using DSC for SeqAIJ Cholesky factorization and solve.");
  PetscFunctionReturn(0);
}

#else

#undef __FUNC__  
#define __FUNC__ "MatSeqAIJUseDSC"
int MatSeqAIJUseDSC(Mat A)
{
     PetscFunctionBegin;
     PetscValidHeaderSpecific(A,MAT_COOKIE);
     PLogInfo(0,"DSCPACK not istalled. Not using DSC.");
     PetscFunctionReturn(0);
}

#endif

