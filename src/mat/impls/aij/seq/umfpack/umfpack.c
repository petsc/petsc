/*$Id: umfpack.c,v 1.10 2001/08/15 15:56:50 bsmith Exp $*/

/* 
        Provides an interface to the UMFPACK sparse solver
*/

#include "src/mat/impls/aij/seq/aij.h"

EXTERN_C_BEGIN
#include "umfpack.h"
EXTERN_C_END

typedef struct {
  void         *Symbolic, *Numeric;
  double       Info[UMFPACK_INFO], Control[UMFPACK_CONTROL],*W;
  int          *Wi,*ai,*aj,*perm_c;
  PetscScalar  *av;
  MatStructure flg;
  PetscTruth   PetscMatOdering;

  /* A few function pointers for inheritance */
  int (*MatDuplicate)(Mat,MatDuplicateOption,Mat*);
  int (*MatView)(Mat,PetscViewer);
  int (*MatAssemblyEnd)(Mat,MatAssemblyType);
  int (*MatLUFactorSymbolic)(Mat,IS,IS,MatFactorInfo*,Mat*);
  int (*MatDestroy)(Mat);
  
  /* Flag to clean up UMFPACK objects during Destroy */
  PetscTruth CleanUpUMFPACK;
} Mat_UMFPACK;

EXTERN int MatDuplicate_UMFPACK(Mat,MatDuplicateOption,Mat*);

EXTERN_C_BEGIN
#undef __FUNCT__
#define __FUNCT__ "MatConvert_UMFPACK_SeqAIJ"
int MatConvert_UMFPACK_SeqAIJ(Mat A,MatType type,Mat *newmat) {
  /* This routine is only called to convert an unfactored PETSc-UMFPACK matrix */
  /* to its base PETSc type, so we will ignore 'MatType type'. */
  int         ierr;
  Mat         B=*newmat;
  Mat_UMFPACK *lu=(Mat_UMFPACK*)A->spptr;

  PetscFunctionBegin;
  if (B != A) {
    /* This routine was inherited from SeqAIJ. */
    ierr = MatDuplicate(A,MAT_COPY_VALUES,&B);CHKERRQ(ierr);
  }
  /* Reset the original function pointers */
  A->ops->duplicate        = lu->MatDuplicate;
  A->ops->view             = lu->MatView;
  A->ops->assemblyend      = lu->MatAssemblyEnd;
  A->ops->lufactorsymbolic = lu->MatLUFactorSymbolic;
  A->ops->destroy          = lu->MatDestroy;
    
  ierr = PetscFree(lu);CHKERRQ(ierr);
  ierr = PetscObjectChangeTypeName((PetscObject)B,MATSEQAIJ);CHKERRQ(ierr);
  *newmat = B;
  PetscFunctionReturn(0);
}
EXTERN_C_END

#undef __FUNCT__  
#define __FUNCT__ "MatDestroy_UMFPACK"
int MatDestroy_UMFPACK(Mat A) {
  int         ierr;
  Mat_UMFPACK *lu=(Mat_UMFPACK*)A->spptr;

  PetscFunctionBegin;
  if (lu->CleanUpUMFPACK) {
    umfpack_di_free_symbolic(&lu->Symbolic) ;
    umfpack_di_free_numeric(&lu->Numeric) ;
    ierr = PetscFree(lu->Wi);CHKERRQ(ierr);
    ierr = PetscFree(lu->W);CHKERRQ(ierr);

    if (lu->PetscMatOdering) {
      ierr = PetscFree(lu->perm_c);CHKERRQ(ierr);
    }
  }
  ierr = MatConvert_UMFPACK_SeqAIJ(A,MATSEQAIJ,&A);CHKERRQ(ierr);
  ierr = (*A->ops->destroy)(A);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatSolve_UMFPACK"
int MatSolve_UMFPACK(Mat A,Vec b,Vec x) {
  Mat_UMFPACK *lu = (Mat_UMFPACK*)A->spptr;
  PetscScalar *av=lu->av,*ba,*xa;
  int         ierr,*ai=lu->ai,*aj=lu->aj,status;
  
  PetscFunctionBegin;
  /* solve Ax = b by umfpack_di_wsolve */
  /* ----------------------------------*/
  ierr = VecGetArray(b,&ba);
  ierr = VecGetArray(x,&xa);

  status = umfpack_di_wsolve(UMFPACK_At,ai,aj,av,xa,ba,lu->Numeric,lu->Control,lu->Info,lu->Wi,lu->W);  
  umfpack_di_report_info(lu->Control, lu->Info); 
  if (status < 0){
    umfpack_di_report_status(lu->Control, status) ;
    SETERRQ(1,"umfpack_di_wsolve failed") ;
  }
    
  ierr = VecRestoreArray(b,&ba);
  ierr = VecRestoreArray(x,&xa);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatLUFactorNumeric_UMFPACK"
int MatLUFactorNumeric_UMFPACK(Mat A,Mat *F) {
  Mat_UMFPACK *lu=(Mat_UMFPACK*)(*F)->spptr;
  int         *ai=lu->ai,*aj=lu->aj,m=A->m,status,ierr;
  PetscScalar *av=lu->av;

  PetscFunctionBegin;
  /* numeric factorization of A' */
  /* ----------------------------*/
  status = umfpack_di_numeric (ai,aj,av,lu->Symbolic,&lu->Numeric,lu->Control,lu->Info) ;
  if (status < 0) SETERRQ(1,"umfpack_di_numeric failed");
  /* report numeric factorization of A' when Control[PRL] > 3 */
  (void) umfpack_di_report_numeric (lu->Numeric, lu->Control) ;

  if (lu->flg == DIFFERENT_NONZERO_PATTERN){  /* first numeric factorization */
    /* allocate working space to be used by Solve */
    ierr = PetscMalloc(m * sizeof(int), &lu->Wi);CHKERRQ(ierr);
    ierr = PetscMalloc(5*m * sizeof(double), &lu->W);CHKERRQ(ierr);

    lu->flg = SAME_NONZERO_PATTERN;
  }

  PetscFunctionReturn(0);
}

/*
   Note the r permutation is ignored
*/
#undef __FUNCT__  
#define __FUNCT__ "MatLUFactorSymbolic_UMFPACK"
int MatLUFactorSymbolic_UMFPACK(Mat A,IS r,IS c,MatFactorInfo *info,Mat *F) {
  Mat         B;
  Mat_SeqAIJ  *mat=(Mat_SeqAIJ*)A->data;
  Mat_UMFPACK *lu;
  int         ierr,m=A->m,n=A->n,*ai=mat->i,*aj=mat->j,status,*ca;
  PetscScalar *av=mat->a;
  
  PetscFunctionBegin;
  /* Create the factorization matrix F */  
  ierr = MatCreate(A->comm,PETSC_DECIDE,PETSC_DECIDE,m,n,&B);CHKERRQ(ierr);
  ierr = MatSetType(B,MATUMFPACK);CHKERRQ(ierr);
  ierr = MatSeqAIJSetPreallocation(B,0,PETSC_NULL);CHKERRQ(ierr);
  
  B->ops->lufactornumeric = MatLUFactorNumeric_UMFPACK;
  B->ops->solve           = MatSolve_UMFPACK;
  B->factor               = FACTOR_LU;
  B->assembled            = PETSC_TRUE;  /* required by -ksp_view */

  lu = (Mat_UMFPACK*)(B->spptr);
  
  /* initializations */
  /* ------------------------------------------------*/
  /* get the default control parameters */
  umfpack_di_defaults (lu->Control) ;
  lu->perm_c = PETSC_NULL;  /* use defaul UMFPACK col permutation */

  ierr = PetscOptionsBegin(A->comm,A->prefix,"UMFPACK Options","Mat");CHKERRQ(ierr);
  /* Control parameters used by reporting routiones */
  ierr = PetscOptionsReal("-mat_umfpack_prl","Control[UMFPACK_PRL]","None",lu->Control[UMFPACK_PRL],&lu->Control[UMFPACK_PRL],PETSC_NULL);CHKERRQ(ierr);

  /* Control parameters for symbolic factorization */
  ierr = PetscOptionsReal("-mat_umfpack_dense_col","Control[UMFPACK_DENSE_COL]","None",lu->Control[UMFPACK_DENSE_COL],&lu->Control[UMFPACK_DENSE_COL],PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-mat_umfpack_dense_row","Control[UMFPACK_DENSE_ROW]","None",lu->Control[UMFPACK_DENSE_ROW],&lu->Control[UMFPACK_DENSE_ROW],PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-mat_umfpack_block_size","Control[UMFPACK_BLOCK_SIZE]","None",lu->Control[UMFPACK_BLOCK_SIZE],&lu->Control[UMFPACK_BLOCK_SIZE],PETSC_NULL);CHKERRQ(ierr);

  /* Control parameters used by numeric factorization */
  ierr = PetscOptionsReal("-mat_umfpack_pivot_tolerance","Control[UMFPACK_PIVOT_TOLERANCE]","None",lu->Control[UMFPACK_PIVOT_TOLERANCE],&lu->Control[UMFPACK_PIVOT_TOLERANCE],PETSC_NULL);CHKERRQ(ierr);
#if !defined(PETSC_HAVE_UMFPACK_41_OR_NEWER)
  ierr = PetscOptionsReal("-mat_umfpack_relaxed_amalgamation","Control[UMFPACK_RELAXED_AMALGAMATION]","None",lu->Control[UMFPACK_RELAXED_AMALGAMATION],&lu->Control[UMFPACK_RELAXED_AMALGAMATION],PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-mat_umfpack_relaxed2_amalgamation","Control[UMFPACK_RELAXED2_AMALGAMATION]","None",lu->Control[UMFPACK_RELAXED2_AMALGAMATION],&lu->Control[UMFPACK_RELAXED2_AMALGAMATION],PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-mat_umfpack_relaxed3_amalgamation","Control[UMFPACK_RELAXED3_AMALGAMATION]","None",lu->Control[UMFPACK_RELAXED3_AMALGAMATION],&lu->Control[UMFPACK_RELAXED3_AMALGAMATION],PETSC_NULL);CHKERRQ(ierr);
#endif
  ierr = PetscOptionsReal("-mat_umfpack_alloc_init","Control[UMFPACK_ALLOC_INIT]","None",lu->Control[UMFPACK_ALLOC_INIT],&lu->Control[UMFPACK_ALLOC_INIT],PETSC_NULL);CHKERRQ(ierr);

  /* Control parameters used by solve */
  ierr = PetscOptionsReal("-mat_umfpack_irstep","Control[UMFPACK_IRSTEP]","None",lu->Control[UMFPACK_IRSTEP],&lu->Control[UMFPACK_IRSTEP],PETSC_NULL);CHKERRQ(ierr);
  
  /* use Petsc mat ordering (notice size is for the transpose) */
  ierr = PetscOptionsHasName(PETSC_NULL,"-pc_lu_mat_ordering_type",&lu->PetscMatOdering);CHKERRQ(ierr);  
  if (lu->PetscMatOdering) {
    ierr = ISGetIndices(c,&ca);CHKERRQ(ierr);
    ierr = PetscMalloc(A->m*sizeof(int),&lu->perm_c);CHKERRQ(ierr);  
    ierr = PetscMemcpy(lu->perm_c,ca,A->m*sizeof(int));CHKERRQ(ierr);
    ierr = ISRestoreIndices(c,&ca);CHKERRQ(ierr);
  }
  PetscOptionsEnd();

  /* print the control parameters */
  if( lu->Control[UMFPACK_PRL] > 1 ) umfpack_di_report_control (lu->Control);

  /* symbolic factorization of A' */
  /* ---------------------------------------------------------------------- */
#if defined(PETSC_HAVE_UMFPACK_41_OR_NEWER)
  status = umfpack_di_qsymbolic(n,m,ai,aj,PETSC_NULL,lu->perm_c,&lu->Symbolic,lu->Control,lu->Info) ;
#else
  status = umfpack_di_qsymbolic(n,m,ai,aj,lu->perm_c,&lu->Symbolic,lu->Control,lu->Info) ;
#endif
  if (status < 0){
    umfpack_di_report_info(lu->Control, lu->Info) ;
    umfpack_di_report_status(lu->Control, status) ;
    SETERRQ(1,"umfpack_di_symbolic failed");
  }
  /* report sumbolic factorization of A' when Control[PRL] > 3 */
  (void) umfpack_di_report_symbolic(lu->Symbolic, lu->Control) ;

  lu->flg = DIFFERENT_NONZERO_PATTERN;
  lu->ai  = ai; 
  lu->aj  = aj;
  lu->av  = av;
  lu->CleanUpUMFPACK = PETSC_TRUE;
  *F = B;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatAssemblyEnd_UMFPACK"
int MatAssemblyEnd_UMFPACK(Mat A,MatAssemblyType mode) {
  int         ierr;
  Mat_UMFPACK *lu=(Mat_UMFPACK*)(A->spptr);

  PetscFunctionBegin;
  ierr = (*lu->MatAssemblyEnd)(A,mode);CHKERRQ(ierr);
  lu->MatLUFactorSymbolic  = A->ops->lufactorsymbolic;
  A->ops->lufactorsymbolic = MatLUFactorSymbolic_UMFPACK;  
  PetscFunctionReturn(0);
}

/* used by -ksp_view */
#undef __FUNCT__  
#define __FUNCT__ "MatFactorInfo_UMFPACK"
int MatFactorInfo_UMFPACK(Mat A,PetscViewer viewer) {
  Mat_UMFPACK *lu= (Mat_UMFPACK*)A->spptr;
  int         ierr;

  PetscFunctionBegin;
  /* check if matrix is UMFPACK type */
  if (A->ops->solve != MatSolve_UMFPACK) PetscFunctionReturn(0);

  ierr = PetscViewerASCIIPrintf(viewer,"UMFPACK run parameters:\n");CHKERRQ(ierr);
  /* Control parameters used by reporting routiones */
  ierr = PetscViewerASCIIPrintf(viewer,"  Control[UMFPACK_PRL]: %g\n",lu->Control[UMFPACK_PRL]);CHKERRQ(ierr);

  /* Control parameters used by symbolic factorization */
  ierr = PetscViewerASCIIPrintf(viewer,"  Control[UMFPACK_DENSE_COL]: %g\n",lu->Control[UMFPACK_DENSE_COL]);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"  Control[UMFPACK_DENSE_ROW]: %g\n",lu->Control[UMFPACK_DENSE_ROW]);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"  Control[UMFPACK_BLOCK_SIZE]: %g\n",lu->Control[UMFPACK_BLOCK_SIZE]);CHKERRQ(ierr);

  /* Control parameters used by numeric factorization */
  ierr = PetscViewerASCIIPrintf(viewer,"  Control[UMFPACK_PIVOT_TOLERANCE]: %g\n",lu->Control[UMFPACK_PIVOT_TOLERANCE]);CHKERRQ(ierr);
#if !defined(PETSC_HAVE_UMFPACK_41_OR_NEWER)
  ierr = PetscViewerASCIIPrintf(viewer,"  Control[UMFPACK_RELAXED_AMALGAMATION]: %g\n",lu->Control[UMFPACK_RELAXED_AMALGAMATION]);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"  Control[UMFPACK_RELAXED2_AMALGAMATION]: %g\n",lu->Control[UMFPACK_RELAXED2_AMALGAMATION]);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"  Control[UMFPACK_RELAXED3_AMALGAMATION]: %g\n",lu->Control[UMFPACK_RELAXED3_AMALGAMATION]);CHKERRQ(ierr);
#endif
  ierr = PetscViewerASCIIPrintf(viewer,"  Control[UMFPACK_ALLOC_INIT]: %g\n",lu->Control[UMFPACK_ALLOC_INIT]);CHKERRQ(ierr);

  /* Control parameters used by solve */
  ierr = PetscViewerASCIIPrintf(viewer,"  Control[UMFPACK_IRSTEP]: %g\n",lu->Control[UMFPACK_IRSTEP]);CHKERRQ(ierr);

  /* mat ordering */
  if(!lu->PetscMatOdering) ierr = PetscViewerASCIIPrintf(viewer,"  UMFPACK default matrix ordering is used (not the PETSc matrix ordering) \n");CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatView_UMFPACK"
int MatView_UMFPACK(Mat A,PetscViewer viewer) {
  int               ierr;
  PetscTruth        isascii;
  PetscViewerFormat format;
  Mat_UMFPACK       *lu=(Mat_UMFPACK*)(A->spptr);

  PetscFunctionBegin;
  ierr = (*lu->MatView)(A,viewer);CHKERRQ(ierr);

  ierr = PetscTypeCompare((PetscObject)viewer,PETSC_VIEWER_ASCII,&isascii);CHKERRQ(ierr);
  if (isascii) {
    ierr = PetscViewerGetFormat(viewer,&format);CHKERRQ(ierr);
    if (format == PETSC_VIEWER_ASCII_FACTOR_INFO) {
      ierr = MatFactorInfo_UMFPACK(A,viewer);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
#undef __FUNCT__
#define __FUNCT__ "MatConvert_SeqAIJ_UMFPACK"
int MatConvert_SeqAIJ_UMFPACK(Mat A,MatType type,Mat *newmat) {
  /* This routine is only called to convert to MATUMFPACK */
  /* from MATSEQAIJ, so we will ignore 'MatType type'. */
  int         ierr;
  Mat         B=*newmat;
  Mat_UMFPACK *lu;

  PetscFunctionBegin;
  if (B != A) {
    ierr = MatDuplicate(A,MAT_COPY_VALUES,&B);CHKERRQ(ierr);
  }

  ierr = PetscNew(Mat_UMFPACK,&lu);CHKERRQ(ierr);
  lu->MatDuplicate         = A->ops->duplicate;
  lu->MatView              = A->ops->view;
  lu->MatAssemblyEnd       = A->ops->assemblyend;
  lu->MatLUFactorSymbolic  = A->ops->lufactorsymbolic;
  lu->MatDestroy           = A->ops->destroy;
  lu->CleanUpUMFPACK       = PETSC_FALSE;

  B->spptr                 = (void*)lu;
  B->ops->duplicate        = MatDuplicate_UMFPACK;
  B->ops->view             = MatView_UMFPACK;
  B->ops->assemblyend      = MatAssemblyEnd_UMFPACK;
  B->ops->lufactorsymbolic = MatLUFactorSymbolic_UMFPACK;
  B->ops->destroy          = MatDestroy_UMFPACK;

  ierr = PetscObjectComposeFunctionDynamic((PetscObject)B,"MatConvert_seqaij_umfpack_C",
                                           "MatConvert_SeqAIJ_UMFPACK",MatConvert_SeqAIJ_UMFPACK);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)B,"MatConvert_umfpack_seqaij_C",
                                           "MatConvert_UMFPACK_SeqAIJ",MatConvert_UMFPACK_SeqAIJ);CHKERRQ(ierr);
  PetscLogInfo(0,"Using UMFPACK for SeqAIJ LU factorization and solves.");
  ierr = PetscObjectChangeTypeName((PetscObject)B,MATUMFPACK);CHKERRQ(ierr);
  *newmat = B;
  PetscFunctionReturn(0);
}
EXTERN_C_END

#undef __FUNCT__
#define __FUNCT__ "MatDuplicate_UMFPACK"
int MatDuplicate_UMFPACK(Mat A, MatDuplicateOption op, Mat *M) {
  int         ierr;
  Mat_UMFPACK *lu=(Mat_UMFPACK*)A->spptr;

  PetscFunctionBegin;
  ierr = (*lu->MatDuplicate)(A,op,M);CHKERRQ(ierr);
  ierr = MatConvert_SeqAIJ_UMFPACK(*M,MATUMFPACK,M);CHKERRQ(ierr);
  ierr = PetscMemcpy((*M)->spptr,lu,sizeof(Mat_UMFPACK));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*MC
  MATUMFPACK - MATUMFPACK = "umfpack" - A matrix type providing direct solvers (LU) for sequential matrices 
  via the external package UMFPACK.

  If UMFPACK is installed (see the manual for
  instructions on how to declare the existence of external packages),
  a matrix type can be constructed which invokes UMFPACK solvers.
  After calling MatCreate(...,A), simply call MatSetType(A,UMFPACK).
  This matrix type is only supported for double precision real.

  This matrix inherits from MATSEQAIJ.  As a result, MatSeqAIJSetPreallocation is 
  supported for this matrix type.  One can also call MatConvert for an inplace conversion to or from 
  the MATSEQAIJ type without data copy.

  Consult UMFPACK documentation for more information about the Control parameters
  which correspond to the options database keys below.

  Options Database Keys:
+ -mat_type umfpack - sets the matrix type to "umfpack" during a call to MatSetFromOptions()
. -mat_umfpack_prl - UMFPACK print level: Control[UMFPACK_PRL]
. -mat_umfpack_dense_col <alpha_c> - UMFPACK dense column threshold: Control[UMFPACK_DENSE_COL]
. -mat_umfpack_block_size <bs> - UMFPACK block size for BLAS-Level 3 calls: Control[UMFPACK_BLOCK_SIZE]
. -mat_umfpack_pivot_tolerance <delta> - UMFPACK partial pivot tolerance: Control[UMFPACK_PIVOT_TOLERANCE]
. -mat_umfpack_alloc_init <delta> - UMFPACK factorized matrix allocation modifier: Control[UMFPACK_ALLOC_INIT]
- -mat_umfpack_irstep <maxit> - UMFPACK maximum number of iterative refinement steps: Control[UMFPACK_IRSTEP]

   Level: beginner

.seealso: PCLU
M*/

EXTERN_C_BEGIN
#undef __FUNCT__
#define __FUNCT__ "MatCreate_UMFPACK"
int MatCreate_UMFPACK(Mat A) {
  int                ierr;

  PetscFunctionBegin;
  /* Change type name before calling MatSetType to force proper construction of SeqAIJ and UMFPACK types */
  ierr = PetscObjectChangeTypeName((PetscObject)A,MATUMFPACK);CHKERRQ(ierr);
  ierr = MatSetType(A,MATSEQAIJ);CHKERRQ(ierr);
  ierr = MatConvert_SeqAIJ_UMFPACK(A,MATUMFPACK,&A);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
EXTERN_C_END
