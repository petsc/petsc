#define PETSCMAT_DLL

/* 
        Provides an interface to the UMFPACKv4.3 sparse solver
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
  PetscErrorCode (*MatDuplicate)(Mat,MatDuplicateOption,Mat*);
  PetscErrorCode (*MatView)(Mat,PetscViewer);
  PetscErrorCode (*MatAssemblyEnd)(Mat,MatAssemblyType);
  PetscErrorCode (*MatLUFactorSymbolic)(Mat,IS,IS,MatFactorInfo*,Mat*);
  PetscErrorCode (*MatDestroy)(Mat);
  
  /* Flag to clean up UMFPACK objects during Destroy */
  PetscTruth CleanUpUMFPACK;
} Mat_UMFPACK;

EXTERN PetscErrorCode MatDuplicate_UMFPACK(Mat,MatDuplicateOption,Mat*);

EXTERN_C_BEGIN
#undef __FUNCT__
#define __FUNCT__ "MatConvert_UMFPACK_SeqAIJ"
PetscErrorCode PETSCMAT_DLLEXPORT MatConvert_UMFPACK_SeqAIJ(Mat A,MatType type,MatReuse reuse,Mat *newmat)
{
  /* This routine is only called to convert an unfactored PETSc-UMFPACK matrix */
  /* to its base PETSc type, so we will ignore 'MatType type'. */
  PetscErrorCode ierr;
  Mat         B=*newmat;
  Mat_UMFPACK *lu=(Mat_UMFPACK*)A->spptr;

  PetscFunctionBegin;
  if (reuse == MAT_INITIAL_MATRIX) {
    ierr = MatDuplicate(A,MAT_COPY_VALUES,&B);CHKERRQ(ierr);
  }
  /* Reset the original function pointers */
  B->ops->duplicate        = lu->MatDuplicate;
  B->ops->view             = lu->MatView;
  B->ops->assemblyend      = lu->MatAssemblyEnd;
  B->ops->lufactorsymbolic = lu->MatLUFactorSymbolic;
  B->ops->destroy          = lu->MatDestroy;
    
  ierr = PetscFree(lu);CHKERRQ(ierr);

  ierr = PetscObjectComposeFunction((PetscObject)B,"MatConvert_seqaij_umfpack_C","",PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)B,"MatConvert_umfpack_seqaij_C","",PETSC_NULL);CHKERRQ(ierr);

  ierr = PetscObjectChangeTypeName((PetscObject)B,MATSEQAIJ);CHKERRQ(ierr);
  *newmat = B;
  PetscFunctionReturn(0);
}
EXTERN_C_END

#undef __FUNCT__  
#define __FUNCT__ "MatDestroy_UMFPACK"
PetscErrorCode MatDestroy_UMFPACK(Mat A) 
{
  PetscErrorCode ierr;
  Mat_UMFPACK *lu=(Mat_UMFPACK*)A->spptr;

  PetscFunctionBegin;
  if (lu->CleanUpUMFPACK) {
    umfpack_di_free_symbolic(&lu->Symbolic);
    umfpack_di_free_numeric(&lu->Numeric);
    ierr = PetscFree(lu->Wi);CHKERRQ(ierr);
    ierr = PetscFree(lu->W);CHKERRQ(ierr);
    if (lu->PetscMatOdering) {
      ierr = PetscFree(lu->perm_c);CHKERRQ(ierr);
    }
  }
  ierr = MatConvert_UMFPACK_SeqAIJ(A,MATSEQAIJ,MAT_REUSE_MATRIX,&A);CHKERRQ(ierr);
  ierr = (*A->ops->destroy)(A);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatSolve_UMFPACK"
PetscErrorCode MatSolve_UMFPACK(Mat A,Vec b,Vec x)
{
  Mat_UMFPACK *lu = (Mat_UMFPACK*)A->spptr;
  PetscScalar *av=lu->av,*ba,*xa;
  PetscErrorCode ierr;
  int          *ai=lu->ai,*aj=lu->aj,status;
  
  PetscFunctionBegin;
  /* solve Ax = b by umfpack_di_wsolve */
  /* ----------------------------------*/
  ierr = VecGetArray(b,&ba);
  ierr = VecGetArray(x,&xa);

  status = umfpack_di_wsolve(UMFPACK_At,ai,aj,av,xa,ba,lu->Numeric,lu->Control,lu->Info,lu->Wi,lu->W);  
  umfpack_di_report_info(lu->Control, lu->Info); 
  if (status < 0){
    umfpack_di_report_status(lu->Control, status);
    SETERRQ(PETSC_ERR_LIB,"umfpack_di_wsolve failed");
  }
    
  ierr = VecRestoreArray(b,&ba);
  ierr = VecRestoreArray(x,&xa);

  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatLUFactorNumeric_UMFPACK"
PetscErrorCode MatLUFactorNumeric_UMFPACK(Mat A,MatFactorInfo *info,Mat *F) 
{
  Mat_UMFPACK *lu=(Mat_UMFPACK*)(*F)->spptr;
  PetscErrorCode ierr;
  int         *ai=lu->ai,*aj=lu->aj,m=A->rmap.n,status;
  PetscScalar *av=lu->av;

  PetscFunctionBegin;
  /* numeric factorization of A' */
  /* ----------------------------*/
  if (lu->flg == SAME_NONZERO_PATTERN && lu->Numeric){
      umfpack_di_free_numeric(&lu->Numeric);
  }
  status = umfpack_di_numeric (ai,aj,av,lu->Symbolic,&lu->Numeric,lu->Control,lu->Info);
  if (status < 0) SETERRQ(PETSC_ERR_LIB,"umfpack_di_numeric failed");
  /* report numeric factorization of A' when Control[PRL] > 3 */
  (void) umfpack_di_report_numeric (lu->Numeric, lu->Control);

  if (lu->flg == DIFFERENT_NONZERO_PATTERN){  /* first numeric factorization */
    /* allocate working space to be used by Solve */
    ierr = PetscMalloc(m * sizeof(int), &lu->Wi);CHKERRQ(ierr);
    ierr = PetscMalloc(5*m * sizeof(double), &lu->W);CHKERRQ(ierr);
  }
  lu->flg = SAME_NONZERO_PATTERN;
  lu->CleanUpUMFPACK = PETSC_TRUE;
  PetscFunctionReturn(0);
}

/*
   Note the r permutation is ignored
*/
#undef __FUNCT__  
#define __FUNCT__ "MatLUFactorSymbolic_UMFPACK"
PetscErrorCode MatLUFactorSymbolic_UMFPACK(Mat A,IS r,IS c,MatFactorInfo *info,Mat *F) 
{
  Mat         B;
  Mat_SeqAIJ  *mat=(Mat_SeqAIJ*)A->data;
  Mat_UMFPACK *lu;
  PetscErrorCode ierr;
  int          m=A->rmap.n,n=A->cmap.n,*ai=mat->i,*aj=mat->j,status,*ra,idx;
  PetscScalar *av=mat->a;
  const char  *strategy[]={"AUTO","UNSYMMETRIC","SYMMETRIC","2BY2"},
              *scale[]={"NONE","SUM","MAX"}; 
  PetscTruth  flg;
  
  PetscFunctionBegin;
  /* Create the factorization matrix F */  
  ierr = MatCreate(A->comm,&B);CHKERRQ(ierr);
  ierr = MatSetSizes(B,PETSC_DECIDE,PETSC_DECIDE,m,n);CHKERRQ(ierr);
  ierr = MatSetType(B,A->type_name);CHKERRQ(ierr);
  ierr = MatSeqAIJSetPreallocation(B,0,PETSC_NULL);CHKERRQ(ierr);
  
  B->ops->lufactornumeric = MatLUFactorNumeric_UMFPACK;
  B->ops->solve           = MatSolve_UMFPACK;
  B->factor               = FACTOR_LU;
  B->assembled            = PETSC_TRUE;  /* required by -ksp_view */

  lu = (Mat_UMFPACK*)(B->spptr);
  
  /* initializations */
  /* ------------------------------------------------*/
  /* get the default control parameters */
  umfpack_di_defaults (lu->Control);
  lu->perm_c = PETSC_NULL;  /* use defaul UMFPACK col permutation */
  lu->Control[UMFPACK_IRSTEP] = 0; /* max num of iterative refinement steps to attempt */

  ierr = PetscOptionsBegin(A->comm,A->prefix,"UMFPACK Options","Mat");CHKERRQ(ierr);
  /* Control parameters used by reporting routiones */
  ierr = PetscOptionsReal("-mat_umfpack_prl","Control[UMFPACK_PRL]","None",lu->Control[UMFPACK_PRL],&lu->Control[UMFPACK_PRL],PETSC_NULL);CHKERRQ(ierr);

  /* Control parameters for symbolic factorization */
  ierr = PetscOptionsEList("-mat_umfpack_strategy","ordering and pivoting strategy","None",strategy,4,strategy[0],&idx,&flg);CHKERRQ(ierr);
  if (flg) {
    switch (idx){
    case 0: lu->Control[UMFPACK_STRATEGY] = UMFPACK_STRATEGY_AUTO; break;
    case 1: lu->Control[UMFPACK_STRATEGY] = UMFPACK_STRATEGY_UNSYMMETRIC; break;
    case 2: lu->Control[UMFPACK_STRATEGY] = UMFPACK_STRATEGY_SYMMETRIC; break;
    case 3: lu->Control[UMFPACK_STRATEGY] = UMFPACK_STRATEGY_2BY2; break;
    }
  }
  ierr = PetscOptionsReal("-mat_umfpack_dense_col","Control[UMFPACK_DENSE_COL]","None",lu->Control[UMFPACK_DENSE_COL],&lu->Control[UMFPACK_DENSE_COL],PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-mat_umfpack_dense_row","Control[UMFPACK_DENSE_ROW]","None",lu->Control[UMFPACK_DENSE_ROW],&lu->Control[UMFPACK_DENSE_ROW],PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-mat_umfpack_amd_dense","Control[UMFPACK_AMD_DENSE]","None",lu->Control[UMFPACK_AMD_DENSE],&lu->Control[UMFPACK_AMD_DENSE],PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-mat_umfpack_block_size","Control[UMFPACK_BLOCK_SIZE]","None",lu->Control[UMFPACK_BLOCK_SIZE],&lu->Control[UMFPACK_BLOCK_SIZE],PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-mat_umfpack_2by2_tolerance","Control[UMFPACK_2BY2_TOLERANCE]","None",lu->Control[UMFPACK_2BY2_TOLERANCE],&lu->Control[UMFPACK_2BY2_TOLERANCE],PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-mat_umfpack_fixq","Control[UMFPACK_FIXQ]","None",lu->Control[UMFPACK_FIXQ],&lu->Control[UMFPACK_FIXQ],PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-mat_umfpack_aggressive","Control[UMFPACK_AGGRESSIVE]","None",lu->Control[UMFPACK_AGGRESSIVE],&lu->Control[UMFPACK_AGGRESSIVE],PETSC_NULL);CHKERRQ(ierr);

  /* Control parameters used by numeric factorization */
  ierr = PetscOptionsReal("-mat_umfpack_pivot_tolerance","Control[UMFPACK_PIVOT_TOLERANCE]","None",lu->Control[UMFPACK_PIVOT_TOLERANCE],&lu->Control[UMFPACK_PIVOT_TOLERANCE],PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-mat_umfpack_sym_pivot_tolerance","Control[UMFPACK_SYM_PIVOT_TOLERANCE]","None",lu->Control[UMFPACK_SYM_PIVOT_TOLERANCE],&lu->Control[UMFPACK_SYM_PIVOT_TOLERANCE],PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEList("-mat_umfpack_scale","Control[UMFPACK_SCALE]","None",scale,3,scale[0],&idx,&flg);CHKERRQ(ierr);
  if (flg) {
    switch (idx){
    case 0: lu->Control[UMFPACK_SCALE] = UMFPACK_SCALE_NONE; break;
    case 1: lu->Control[UMFPACK_SCALE] = UMFPACK_SCALE_SUM; break;
    case 2: lu->Control[UMFPACK_SCALE] = UMFPACK_SCALE_MAX; break;
    }
  }
  ierr = PetscOptionsReal("-mat_umfpack_alloc_init","Control[UMFPACK_ALLOC_INIT]","None",lu->Control[UMFPACK_ALLOC_INIT],&lu->Control[UMFPACK_ALLOC_INIT],PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-mat_umfpack_front_alloc_init","Control[UMFPACK_FRONT_ALLOC_INIT]","None",lu->Control[UMFPACK_FRONT_ALLOC_INIT],&lu->Control[UMFPACK_ALLOC_INIT],PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-mat_umfpack_droptol","Control[UMFPACK_DROPTOL]","None",lu->Control[UMFPACK_DROPTOL],&lu->Control[UMFPACK_DROPTOL],PETSC_NULL);CHKERRQ(ierr);

  /* Control parameters used by solve */
  ierr = PetscOptionsReal("-mat_umfpack_irstep","Control[UMFPACK_IRSTEP]","None",lu->Control[UMFPACK_IRSTEP],&lu->Control[UMFPACK_IRSTEP],PETSC_NULL);CHKERRQ(ierr);
  
  /* use Petsc mat ordering (note: size is for the transpose, and PETSc r = Umfpack perm_c) */
  ierr = PetscOptionsHasName(PETSC_NULL,"-pc_factor_mat_ordering_type",&lu->PetscMatOdering);CHKERRQ(ierr);  
  if (lu->PetscMatOdering) {
    ierr = ISGetIndices(r,&ra);CHKERRQ(ierr);
    ierr = PetscMalloc(A->rmap.n*sizeof(int),&lu->perm_c);CHKERRQ(ierr);  
    ierr = PetscMemcpy(lu->perm_c,ra,A->rmap.n*sizeof(int));CHKERRQ(ierr);
    ierr = ISRestoreIndices(r,&ra);CHKERRQ(ierr);
  }
  PetscOptionsEnd();

  /* print the control parameters */
  if( lu->Control[UMFPACK_PRL] > 1 ) umfpack_di_report_control (lu->Control);

  /* symbolic factorization of A' */
  /* ---------------------------------------------------------------------- */
  if (lu->PetscMatOdering) { /* use Petsc row ordering */
    status = umfpack_di_qsymbolic(n,m,ai,aj,av,lu->perm_c,&lu->Symbolic,lu->Control,lu->Info);
  } else { /* use Umfpack col ordering */
    status = umfpack_di_symbolic(n,m,ai,aj,av,&lu->Symbolic,lu->Control,lu->Info);
  }
  if (status < 0){
    umfpack_di_report_info(lu->Control, lu->Info);
    umfpack_di_report_status(lu->Control, status);
    SETERRQ(PETSC_ERR_LIB,"umfpack_di_symbolic failed");
  }
  /* report sumbolic factorization of A' when Control[PRL] > 3 */
  (void) umfpack_di_report_symbolic(lu->Symbolic, lu->Control);

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
PetscErrorCode MatAssemblyEnd_UMFPACK(Mat A,MatAssemblyType mode)
{
  PetscErrorCode ierr;
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
PetscErrorCode MatFactorInfo_UMFPACK(Mat A,PetscViewer viewer)
{
  Mat_UMFPACK *lu= (Mat_UMFPACK*)A->spptr;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /* check if matrix is UMFPACK type */
  if (A->ops->solve != MatSolve_UMFPACK) PetscFunctionReturn(0);

  ierr = PetscViewerASCIIPrintf(viewer,"UMFPACK run parameters:\n");CHKERRQ(ierr);
  /* Control parameters used by reporting routiones */
  ierr = PetscViewerASCIIPrintf(viewer,"  Control[UMFPACK_PRL]: %g\n",lu->Control[UMFPACK_PRL]);CHKERRQ(ierr);

  /* Control parameters used by symbolic factorization */
  ierr = PetscViewerASCIIPrintf(viewer,"  Control[UMFPACK_STRATEGY]: %g\n",lu->Control[UMFPACK_STRATEGY]);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"  Control[UMFPACK_DENSE_COL]: %g\n",lu->Control[UMFPACK_DENSE_COL]);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"  Control[UMFPACK_DENSE_ROW]: %g\n",lu->Control[UMFPACK_DENSE_ROW]);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"  Control[UMFPACK_AMD_DENSE]: %g\n",lu->Control[UMFPACK_AMD_DENSE]);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"  Control[UMFPACK_BLOCK_SIZE]: %g\n",lu->Control[UMFPACK_BLOCK_SIZE]);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"  Control[UMFPACK_2BY2_TOLERANCE]: %g\n",lu->Control[UMFPACK_2BY2_TOLERANCE]);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"  Control[UMFPACK_FIXQ]: %g\n",lu->Control[UMFPACK_FIXQ]);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"  Control[UMFPACK_AGGRESSIVE]: %g\n",lu->Control[UMFPACK_AGGRESSIVE]);CHKERRQ(ierr);

  /* Control parameters used by numeric factorization */
  ierr = PetscViewerASCIIPrintf(viewer,"  Control[UMFPACK_PIVOT_TOLERANCE]: %g\n",lu->Control[UMFPACK_PIVOT_TOLERANCE]);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"  Control[UMFPACK_SYM_PIVOT_TOLERANCE]: %g\n",lu->Control[UMFPACK_SYM_PIVOT_TOLERANCE]);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"  Control[UMFPACK_SCALE]: %g\n",lu->Control[UMFPACK_SCALE]);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"  Control[UMFPACK_ALLOC_INIT]: %g\n",lu->Control[UMFPACK_ALLOC_INIT]);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"  Control[UMFPACK_DROPTOL]: %g\n",lu->Control[UMFPACK_DROPTOL]);CHKERRQ(ierr);

  /* Control parameters used by solve */
  ierr = PetscViewerASCIIPrintf(viewer,"  Control[UMFPACK_IRSTEP]: %g\n",lu->Control[UMFPACK_IRSTEP]);CHKERRQ(ierr);

  /* mat ordering */
  if(!lu->PetscMatOdering) ierr = PetscViewerASCIIPrintf(viewer,"  UMFPACK default matrix ordering is used (not the PETSc matrix ordering) \n");CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatView_UMFPACK"
PetscErrorCode MatView_UMFPACK(Mat A,PetscViewer viewer) 
{
  PetscErrorCode ierr;
  PetscTruth        iascii;
  PetscViewerFormat format;
  Mat_UMFPACK       *lu=(Mat_UMFPACK*)(A->spptr);

  PetscFunctionBegin;
  ierr = (*lu->MatView)(A,viewer);CHKERRQ(ierr);

  ierr = PetscTypeCompare((PetscObject)viewer,PETSC_VIEWER_ASCII,&iascii);CHKERRQ(ierr);
  if (iascii) {
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
PetscErrorCode PETSCMAT_DLLEXPORT MatConvert_SeqAIJ_UMFPACK(Mat A,MatType type,MatReuse reuse,Mat *newmat) 
{
  /* This routine is only called to convert to MATUMFPACK */
  /* from MATSEQAIJ, so we will ignore 'MatType type'. */
  PetscErrorCode ierr;
  Mat         B=*newmat;
  Mat_UMFPACK *lu;

  PetscFunctionBegin;
  if (reuse == MAT_INITIAL_MATRIX) {
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
  ierr = PetscInfo(0,"Using UMFPACK for SeqAIJ LU factorization and solves.\n");CHKERRQ(ierr);
  ierr = PetscObjectChangeTypeName((PetscObject)B,MATUMFPACK);CHKERRQ(ierr);
  *newmat = B;
  PetscFunctionReturn(0);
}
EXTERN_C_END

#undef __FUNCT__
#define __FUNCT__ "MatDuplicate_UMFPACK"
PetscErrorCode MatDuplicate_UMFPACK(Mat A, MatDuplicateOption op, Mat *M) 
{
  PetscErrorCode ierr;
  Mat_UMFPACK *lu=(Mat_UMFPACK*)A->spptr;

  PetscFunctionBegin;
  ierr = (*lu->MatDuplicate)(A,op,M);CHKERRQ(ierr);
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
PetscErrorCode PETSCMAT_DLLEXPORT MatCreate_UMFPACK(Mat A) 
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /* Change type name before calling MatSetType to force proper construction of SeqAIJ and UMFPACK types */
  ierr = PetscObjectChangeTypeName((PetscObject)A,MATUMFPACK);CHKERRQ(ierr);
  ierr = MatSetType(A,MATSEQAIJ);CHKERRQ(ierr);
  ierr = MatConvert_SeqAIJ_UMFPACK(A,MATUMFPACK,MAT_REUSE_MATRIX,&A);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
EXTERN_C_END
