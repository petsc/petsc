/*$Id: dsc.c,v 1.5 2000/10/24 20:25:32 bsmith Exp $*/
/* 
        Provides an interface to the DSCPACK-S
*/

#include "src/mat/impls/aij/seq/aij.h" 
#include "/home/petsc/software/DSCPACK-S/DSC_LIB/dscmain.h"

#define PETSC_HAVE_DSC 
#if defined(PETSC_HAVE_DSC) && !defined(PETSC_USE_COMPLEX) 

EXTERN_C_BEGIN

typedef struct 
{
  int    s_nz;           /* num of nonzeros in lower/upper half of the matrix */
  int    *perm, *iperm;  /* permutation */
  int    pass;           /* num of numeric factorizations for a single symbolic factorization */
} Mat_SeqAIJ_DSC;

#undef __FUNC__  
#define __FUNC__ "MatDestroy_SeqAIJ_DSC_A"
int MatDestroy_SeqAIJ_DSC_A(Mat A)
{
  Mat_SeqAIJ     *a = (Mat_SeqAIJ *)A->data;
  Mat_SeqAIJ_DSC *dsc = (Mat_SeqAIJ_DSC *)a->spptr; 
  int            ierr;

  PetscFunctionBegin;
  
  ierr = PetscFree(dsc);CHKERRQ(ierr);
  ierr = MatDestroy_SeqAIJ(A);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

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
  Mat_SeqAIJ_DSC   *dsc = (Mat_SeqAIJ_DSC*)a->spptr;
  PetscTruth       flg;
  int              m,ierr;  
  int              *ai = a->i, *aj = a->j,s_nz=dsc->s_nz;
  int              *perm=dsc->perm, *iperm= dsc->iperm;
  real_number_type *a_nonz = a->a, *s_a_nonz;

  PetscFunctionBegin;
  m = A->m; 
  if (dsc->pass == 0){ /* check the arguments */
    if (m != A->n) SETERRQ(PETSC_ERR_ARG_SIZ, "matrix must be square"); 
    ierr = PetscTypeCompare((PetscObject)A,MATSEQAIJ,&flg);CHKERRQ(ierr);
    if (!flg) SETERRQ(PETSC_ERR_ARG_SIZ, "matrix must be Seq_AIJ");
    if (m != (*F)->m) SETERRQ(PETSC_ERR_ARG_SIZ, "factorization struct inconsistent"); 

  } else { /* frees up Cholesky factor used by previous numeric factorization */
    DSC_N_Fact_Free();
    DSC_Re_Init();
  }

  ierr =Initialize_A_Nonz(m,ai,aj,a_nonz,s_nz,perm,iperm, &s_a_nonz);
  if (ierr <0) SETERRQ(PETSC_ERR_ARG_SIZ, "Error setting up permuted nonzero vector");
              
  DSC_N_Fact(s_a_nonz); 

  free ((char *) s_a_nonz);
  dsc->pass++;
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

#undef __FUNC__
#define __FUNC__ /*<a name=""></a>*/"MatOrdering_DSC"
int MatOrdering_DSC(Mat mat,MatOrderingType type,IS *row,IS *col)
{
  Mat_SeqAIJ      *a = (Mat_SeqAIJ*)mat->data;
  Mat_SeqAIJ_DSC  *dsc;
  int             ierr,order_code,m,*ai,*aj;
  int             s_nz, *perm, *iperm;
  PetscTruth      flg;

  PetscFunctionBegin;

  ierr = PetscStrcmp(type,MATORDERING_DSC_ND,&flg); 
  if (flg) { 
    order_code = 1; 
  } else {
    ierr = PetscStrcmp(type,MATORDERING_DSC_MMD,&flg);
    if (flg) {
      order_code = 2;
    } else {
      ierr = PetscStrcmp(type,MATORDERING_DSC_MDF,&flg);
      if (flg) {
        order_code = 3;
      } else {
        printf(" default ordering: MATORDERING_DSC_ND is used \n");
        order_code = 1;
      }
    }
  }
  
  ierr = MatGetRowIJ(mat,0,PETSC_TRUE,&m,&ai,&aj,&flg);CHKERRQ(ierr);
  if (!flg) SETERRQ1(PETSC_ERR_SUP,"Cannot get rows for matrix type %s",((PetscObject)mat)->type_name);
  
  dsc = PetscNew(Mat_SeqAIJ_DSC);CHKPTRQ(dsc);
  a->spptr = (void *)dsc;
  
  DSC_Open0();

  DSC_Order(order_code, m, ai, aj, &s_nz, &perm, &iperm);        
  if (DSC_STATUS.cont_or_stop == DSC_STOP_TYPE) goto ERROR_HANDLE;

  dsc->s_nz  = s_nz;
  dsc->perm  = perm;
  dsc->iperm = iperm;
  dsc->pass  = 0;

  ierr = ISCreateGeneral(PETSC_COMM_SELF,m,perm,row);CHKERRA(ierr);
  ierr = ISCreateGeneral(PETSC_COMM_SELF,m,perm,col);CHKERRA(ierr);                                          
ERROR_HANDLE:  
  if (DSC_STATUS.error_code != DSC_NO_ERROR) {
    DSC_Error_Display();
    SETERRQ(PETSC_ERR_ARG_SIZ, "DSC_ERROR");
  }

  ierr = MatRestoreRowIJ(mat,0,PETSC_TRUE,&m,&ai,&aj,&flg);CHKERRQ(ierr);
  mat->ops->destroy = MatDestroy_SeqAIJ_DSC_A;

  PetscFunctionReturn(0);
}

EXTERN_C_END

#undef __FUNC__  
#define __FUNC__ "MatUseDSC_SeqAIJ"
int MatUseDSC_SeqAIJ(Mat A)
{
  int        ierr; 
  PetscTruth flg;
   
  PetscFunctionBegin;
  if (A->m != A->n) SETERRQ(PETSC_ERR_ARG_SIZ, "matrix must be square");

  ierr = PetscTypeCompare((PetscObject)A,MATSEQAIJ,&flg);CHKERRQ(ierr);
  if (!flg) SETERRQ(PETSC_ERR_ARG_SIZ, "matrix must be SeqAIJ"); 
    
  A->ops->choleskyfactorsymbolic = MatCholeskyFactorSymbolic_SeqAIJ_DSC; 
  PLogInfo(0,"Using DSC for SeqAIJ Cholesky factorization and solve.");
  PetscFunctionReturn(0);
}

#else

#undef __FUNC__  
#define __FUNC__ "MatUseDSC_SeqAIJ"
int MatUseDSC_SeqAIJ(Mat A)
{
     PetscFunctionBegin;
     PetscFunctionReturn(0);
}

#endif

