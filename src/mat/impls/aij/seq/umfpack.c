/*$Id: umfpack.c,v 1.10 2001/08/15 15:56:50 bsmith Exp $*/

/* 
        Provides an interface to the UMFPACK sparse solver
*/

#include "src/mat/impls/aij/seq/aij.h"

#if defined(PETSC_HAVE_UMFPACK) && !defined(PETSC_USE_SINGLE) && !defined(PETSC_USE_COMPLEX)
EXTERN_C_BEGIN
#include "umfpack.h"
EXTERN_C_END

typedef struct {
  void         *Symbolic, *Numeric;
  double       Info[UMFPACK_INFO], Control[UMFPACK_CONTROL],*W;
  int          *Wi;
  MatStructure flg;
} Mat_SeqAIJ_UMFPACK;


extern int MatDestroy_SeqAIJ(Mat);

#undef __FUNCT__  
#define __FUNCT__ "MatDestroy_SeqAIJ_UMFPACK"
int MatDestroy_SeqAIJ_UMFPACK(Mat A)
{
  Mat_SeqAIJ_UMFPACK *lu = (Mat_SeqAIJ_UMFPACK*)A->spptr;
  int                ierr;

  PetscFunctionBegin;
  printf("MatDestroy_SeqAIJ_UMFPACK is called ...\n");
  umfpack_di_free_symbolic(&lu->Symbolic) ;
  umfpack_di_free_numeric(&lu->Numeric) ;
  ierr = PetscFree(lu->Wi);CHKERRQ(ierr);
  ierr = PetscFree(lu->W);CHKERRQ(ierr);

  ierr = PetscFree(lu);CHKERRQ(ierr);
  ierr = MatDestroy_SeqAIJ(A);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatSolve_SeqAIJ_UMFPACK"
int MatSolve_SeqAIJ_UMFPACK(Mat A,Vec b,Vec x)
{
  Mat_SeqAIJ         *mat=(Mat_SeqAIJ*)A->data;
  Mat_SeqAIJ_UMFPACK *lu = (Mat_SeqAIJ_UMFPACK*)A->spptr;
  PetscScalar        *av=mat->a,*b_array,*x_array;
  int                n=A->n,*Wi,ierr,
                     *ai=mat->i,*aj=mat->j,status;
  double             *W;
  
  PetscFunctionBegin;
  printf(" Solve _SeqAIJ_UMFPACK is called ...\n");

  /* solve A'x=b by umfpack_di_wsolve */
  /* ----------------------------------*/
  ierr = VecGetArray(b,&b_array);
  ierr = VecGetArray(x,&x_array);

  status = umfpack_di_wsolve (UMFPACK_At, ai, aj, av, x_array, b_array,
	lu->Numeric, lu->Control, lu->Info, lu->Wi, lu->W) ;
  /* umfpack_di_report_info (Control, Info) ; */
  if (status < 0){
    umfpack_di_report_status(lu->Control, status) ;
    SETERRQ(1,"umfpack_di_wsolve failed") ;
  }
  /*
    printf ("\nx (solution of C'x=b): ") ;
    (void) umfpack_di_report_vector (n, x, Control) ;
  */
    
  ierr = VecRestoreArray(b,&b_array);
  ierr = VecRestoreArray(x,&x_array);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatLUFactorNumeric_SeqAIJ_UMFPACK"
int MatLUFactorNumeric_SeqAIJ_UMFPACK(Mat A,Mat *F)
{
  Mat_SeqAIJ         *mat=(Mat_SeqAIJ*)(A)->data;
  Mat_SeqAIJ_UMFPACK *lu = (Mat_SeqAIJ_UMFPACK*)(*F)->spptr;
  int                *ai=mat->i,*aj=mat->j,n=A->n,status,ierr;
  PetscScalar        *av=mat->a;

  PetscFunctionBegin;
  printf(" Numeric_SeqAIJ_UMFPACK is called ...\n");
  /* numeric factorization of A' */
  /* ----------------------------*/
  status = umfpack_di_numeric (ai,aj,av,lu->Symbolic,&lu->Numeric,lu->Control,lu->Info) ;
  if (status < 0) SETERRQ(1,"umfpack_di_numeric failed");

  /*
    printf ("\nNumeric factorization of C: ") ;
    (void) umfpack_di_report_numeric (lu->Numeric, lu->Control) ;
  */
  /* working arrays to be used by Solve */
  if (lu->flg == DIFFERENT_NONZERO_PATTERN){  /* first numeric factorization */
    ierr = PetscMalloc(n * sizeof(int), &lu->Wi);CHKERRQ(ierr);
    ierr = PetscMalloc(5*n * sizeof(double), &lu->W);CHKERRQ(ierr);
  }

  lu->flg = SAME_NONZERO_PATTERN;
  PetscFunctionReturn(0);
}

/*
   Note the r permutation is ignored
*/
#undef __FUNCT__  
#define __FUNCT__ "MatLUFactorSymbolic_SeqAIJ_UMFPACK"
int MatLUFactorSymbolic_SeqAIJ_UMFPACK(Mat A,IS r,IS c,MatLUInfo *info,Mat *F)
{
  Mat_SeqAIJ_UMFPACK  *lu;
  int                 ierr,m=A->m,n=A->n;
  Mat_SeqAIJ          *mat=(Mat_SeqAIJ*)A->data;
  int                 *ai=mat->i,*aj=mat->j,status;
  PetscScalar         *av=mat->a;
  
  PetscFunctionBegin;
  printf(" Symbolic_SeqAIJ_UMFPACK is called ...\n");
  /* Create the factorization matrix F */  
  ierr            = MatCreateSeqAIJ(A->comm,m,n,PETSC_NULL,PETSC_NULL,F);CHKERRQ(ierr);
  
  (*F)->ops->lufactornumeric = MatLUFactorNumeric_SeqAIJ_UMFPACK;
  (*F)->ops->solve           = MatSolve_SeqAIJ_UMFPACK;
  (*F)->ops->destroy         = MatDestroy_SeqAIJ_UMFPACK;
  (*F)->factor               = FACTOR_LU;
  
  ierr            = PetscNew(Mat_SeqAIJ_UMFPACK,&lu);CHKERRQ(ierr);
  (*F)->spptr     = (void*)lu;
  
  /* initializations */
  /* ---------------------------------------------------------------------- */
  /* get the default control parameters */
  umfpack_di_defaults (lu->Control) ;

  /* change the default print level for this demo */
  /* (otherwise, nothing will print) */
  lu->Control[UMFPACK_PRL] = 6 ;

  /* print the control parameters */
  /* umfpack_di_report_control (Control) ;*/

  /* symbolic factorization of A' */
  /* ---------------------------------------------------------------------- */
  int *Qinit = PETSC_NULL;  /* set defaul col permutation: colamd */
  status = umfpack_di_qsymbolic (m,n,ai,aj,Qinit,&lu->Symbolic,lu->Control,lu->Info) ;
  if (status < 0){
    umfpack_di_report_info(lu->Control, lu->Info) ;
    umfpack_di_report_status(lu->Control, status) ;
    SETERRQ(1,"umfpack_di_symbolic failed");
  }
  /*
    printf ("\nSymbolic factorization of C: ") ;
    (void) umfpack_di_report_symbolic (lu->Symbolic, lu->Control) ;
  */

  lu->flg = DIFFERENT_NONZERO_PATTERN;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatUseUMFPACK_SeqAIJ"
int MatUseUMFPACK_SeqAIJ(Mat A)
{
  PetscFunctionBegin;
  printf(" MatUseUMFPACK_ is called ...\n");
  A->ops->lufactorsymbolic = MatLUFactorSymbolic_SeqAIJ_UMFPACK;  
  PetscFunctionReturn(0);
}

#else

#undef __FUNCT__  
#define __FUNCT__ "MatUseUMFPACK_SeqAIJ"
int MatUseUMFPACK_SeqAIJ(Mat A)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

#endif


