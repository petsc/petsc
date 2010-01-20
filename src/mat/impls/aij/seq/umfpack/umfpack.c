#define PETSCMAT_DLL

/* 
   Provides an interface to the UMFPACKv5.1 sparse solver

   When build with PETSC_USE_64BIT_INDICES this will use UF_Long as the 
   integer type in UMFPACK, otherwise it will use int. This means
   all integers in this file as simply declared as PetscInt. Also it means
   that UMFPACK UL_Long version MUST be built with 64 bit integers when used.

*/
#include "../src/mat/impls/aij/seq/aij.h"

#if defined(PETSC_USE_64BIT_INDICES)
#if defined(PETSC_USE_COMPLEX)
#define umfpack_UMF_free_symbolic   umfpack_zl_free_symbolic
#define umfpack_UMF_free_numeric    umfpack_zl_free_numeric
#define umfpack_UMF_wsolve          umfpack_zl_wsolve
#define umfpack_UMF_numeric         umfpack_zl_numeric
#define umfpack_UMF_report_numeric  umfpack_zl_report_numeric
#define umfpack_UMF_report_control  umfpack_zl_report_control
#define umfpack_UMF_report_status   umfpack_zl_report_status
#define umfpack_UMF_report_info     umfpack_zl_report_info
#define umfpack_UMF_report_symbolic umfpack_zl_report_symbolic
#define umfpack_UMF_qsymbolic       umfpack_zl_qsymbolic
#define umfpack_UMF_symbolic        umfpack_zl_symbolic
#define umfpack_UMF_defaults        umfpack_zl_defaults

#else
#define umfpack_UMF_free_symbolic   umfpack_dl_free_symbolic
#define umfpack_UMF_free_numeric    umfpack_dl_free_numeric
#define umfpack_UMF_wsolve          umfpack_dl_wsolve
#define umfpack_UMF_numeric         umfpack_dl_numeric
#define umfpack_UMF_report_numeric  umfpack_dl_report_numeric
#define umfpack_UMF_report_control  umfpack_dl_report_control
#define umfpack_UMF_report_status   umfpack_dl_report_status
#define umfpack_UMF_report_info     umfpack_dl_report_info
#define umfpack_UMF_report_symbolic umfpack_dl_report_symbolic
#define umfpack_UMF_qsymbolic       umfpack_dl_qsymbolic
#define umfpack_UMF_symbolic        umfpack_dl_symbolic
#define umfpack_UMF_defaults        umfpack_dl_defaults
#endif

#else
#if defined(PETSC_USE_COMPLEX)
#define umfpack_UMF_free_symbolic   umfpack_zi_free_symbolic
#define umfpack_UMF_free_numeric    umfpack_zi_free_numeric
#define umfpack_UMF_wsolve          umfpack_zi_wsolve
#define umfpack_UMF_numeric         umfpack_zi_numeric
#define umfpack_UMF_report_numeric  umfpack_zi_report_numeric
#define umfpack_UMF_report_control  umfpack_zi_report_control
#define umfpack_UMF_report_status   umfpack_zi_report_status
#define umfpack_UMF_report_info     umfpack_zi_report_info
#define umfpack_UMF_report_symbolic umfpack_zi_report_symbolic
#define umfpack_UMF_qsymbolic       umfpack_zi_qsymbolic
#define umfpack_UMF_symbolic        umfpack_zi_symbolic
#define umfpack_UMF_defaults        umfpack_zi_defaults

#else
#define umfpack_UMF_free_symbolic   umfpack_di_free_symbolic
#define umfpack_UMF_free_numeric    umfpack_di_free_numeric
#define umfpack_UMF_wsolve          umfpack_di_wsolve
#define umfpack_UMF_numeric         umfpack_di_numeric
#define umfpack_UMF_report_numeric  umfpack_di_report_numeric
#define umfpack_UMF_report_control  umfpack_di_report_control
#define umfpack_UMF_report_status   umfpack_di_report_status
#define umfpack_UMF_report_info     umfpack_di_report_info
#define umfpack_UMF_report_symbolic umfpack_di_report_symbolic
#define umfpack_UMF_qsymbolic       umfpack_di_qsymbolic
#define umfpack_UMF_symbolic        umfpack_di_symbolic
#define umfpack_UMF_defaults        umfpack_di_defaults
#endif
#endif


#define UF_long long long
#define UF_long_max LONG_LONG_MAX
#define UF_long_id "%lld"

EXTERN_C_BEGIN
#include "umfpack.h"
EXTERN_C_END

typedef struct {
  void         *Symbolic, *Numeric;
  double       Info[UMFPACK_INFO], Control[UMFPACK_CONTROL],*W;
  PetscInt      *Wi,*ai,*aj,*perm_c;
  PetscScalar  *av;
  MatStructure flg;
  PetscTruth   PetscMatOdering;

  /* Flag to clean up UMFPACK objects during Destroy */
  PetscTruth CleanUpUMFPACK;
} Mat_UMFPACK;

#undef __FUNCT__  
#define __FUNCT__ "MatDestroy_UMFPACK"
static PetscErrorCode MatDestroy_UMFPACK(Mat A)
{
  PetscErrorCode ierr;
  Mat_UMFPACK    *lu=(Mat_UMFPACK*)A->spptr;

  PetscFunctionBegin;
  if (lu->CleanUpUMFPACK) {
    umfpack_UMF_free_symbolic(&lu->Symbolic);
    umfpack_UMF_free_numeric(&lu->Numeric);
    ierr = PetscFree(lu->Wi);CHKERRQ(ierr);
    ierr = PetscFree(lu->W);CHKERRQ(ierr);
    if (lu->PetscMatOdering) {
      ierr = PetscFree(lu->perm_c);CHKERRQ(ierr);
    }
  }
  ierr = MatDestroy_SeqAIJ(A);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatSolve_UMFPACK_Private"
static PetscErrorCode MatSolve_UMFPACK_Private(Mat A,Vec b,Vec x,int uflag)
{
  Mat_UMFPACK    *lu = (Mat_UMFPACK*)A->spptr;
  PetscScalar    *av=lu->av,*ba,*xa;
  PetscErrorCode ierr;
  PetscInt       *ai=lu->ai,*aj=lu->aj,status;
  
  PetscFunctionBegin;
  /* solve Ax = b by umfpack_*_wsolve */
  /* ----------------------------------*/

  ierr = VecGetArray(b,&ba);
  ierr = VecGetArray(x,&xa);
#if defined(PETSC_USE_COMPLEX)
  status = umfpack_UMF_wsolve(uflag,ai,aj,(PetscReal*)av,NULL,(PetscReal*)xa,NULL,(PetscReal*)ba,NULL,
                              lu->Numeric,lu->Control,lu->Info,lu->Wi,lu->W);
#else  
  status = umfpack_UMF_wsolve(uflag,ai,aj,av,xa,ba,lu->Numeric,lu->Control,lu->Info,lu->Wi,lu->W);
#endif
  umfpack_UMF_report_info(lu->Control, lu->Info); 
  if (status < 0){
    umfpack_UMF_report_status(lu->Control, status);
    SETERRQ(PETSC_ERR_LIB,"umfpack_UMF_wsolve failed");
  }

  ierr = VecRestoreArray(b,&ba);
  ierr = VecRestoreArray(x,&xa);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatSolve_UMFPACK"
static PetscErrorCode MatSolve_UMFPACK(Mat A,Vec b,Vec x)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /* We gave UMFPACK the algebraic transpose (because it assumes column alignment) */
  ierr = MatSolve_UMFPACK_Private(A,b,x,UMFPACK_Aat);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatSolveTranspose_UMFPACK"
static PetscErrorCode MatSolveTranspose_UMFPACK(Mat A,Vec b,Vec x)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /* We gave UMFPACK the algebraic transpose (because it assumes column alignment) */
  ierr = MatSolve_UMFPACK_Private(A,b,x,UMFPACK_A);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatLUFactorNumeric_UMFPACK"
static PetscErrorCode MatLUFactorNumeric_UMFPACK(Mat F,Mat A,const MatFactorInfo *info)
{
  Mat_UMFPACK *lu=(Mat_UMFPACK*)(F)->spptr;
  PetscErrorCode ierr;
  PetscInt     *ai=lu->ai,*aj=lu->aj,m=A->rmap->n,status;
  PetscScalar *av=lu->av;

  PetscFunctionBegin;
  /* numeric factorization of A' */
  /* ----------------------------*/

  if (lu->flg == SAME_NONZERO_PATTERN && lu->Numeric){
    umfpack_UMF_free_numeric(&lu->Numeric);
  }
#if defined(PETSC_USE_COMPLEX)
  status = umfpack_UMF_numeric(ai,aj,(double*)av,NULL,lu->Symbolic,&lu->Numeric,lu->Control,lu->Info);
#else
  status = umfpack_UMF_numeric(ai,aj,av,lu->Symbolic,&lu->Numeric,lu->Control,lu->Info);
#endif
  if (status < 0) {
    umfpack_UMF_report_status(lu->Control, status);
    SETERRQ(PETSC_ERR_LIB,"umfpack_UMF_numeric failed");
  }
  /* report numeric factorization of A' when Control[PRL] > 3 */
  (void) umfpack_UMF_report_numeric(lu->Numeric, lu->Control);

  if (lu->flg == DIFFERENT_NONZERO_PATTERN){  /* first numeric factorization */
    /* allocate working space to be used by Solve */
    ierr = PetscMalloc(m * sizeof(PetscInt), &lu->Wi);CHKERRQ(ierr);
    ierr = PetscMalloc(5*m * sizeof(PetscScalar), &lu->W);CHKERRQ(ierr);
  }

  lu->flg = SAME_NONZERO_PATTERN;
  lu->CleanUpUMFPACK = PETSC_TRUE;
  F->ops->solve          = MatSolve_UMFPACK;
  F->ops->solvetranspose = MatSolveTranspose_UMFPACK;
  PetscFunctionReturn(0);
}

/*
   Note the r permutation is ignored
*/
#undef __FUNCT__  
#define __FUNCT__ "MatLUFactorSymbolic_UMFPACK"
static PetscErrorCode MatLUFactorSymbolic_UMFPACK(Mat F,Mat A,IS r,IS c,const MatFactorInfo *info)
{
  Mat_SeqAIJ     *mat=(Mat_SeqAIJ*)A->data;
  Mat_UMFPACK    *lu = (Mat_UMFPACK*)(F->spptr);
  PetscErrorCode ierr;
  PetscInt       i,m=A->rmap->n,n=A->cmap->n;
  const PetscInt *ra;
  PetscInt        status;
  PetscScalar    *av=mat->a;
  
  PetscFunctionBegin;
  if (lu->PetscMatOdering) {
    ierr = ISGetIndices(r,&ra);CHKERRQ(ierr);
    ierr = PetscMalloc(m*sizeof(PetscInt),&lu->perm_c);CHKERRQ(ierr);  
    /* we cannot simply memcpy on 64 bit archs */
    for(i = 0; i < m; i++) lu->perm_c[i] = ra[i];
    ierr = ISRestoreIndices(r,&ra);CHKERRQ(ierr);
  }

  lu->ai = mat->i;
  lu->aj = mat->j;

  /* print the control parameters */
  if(lu->Control[UMFPACK_PRL] > 1) umfpack_UMF_report_control(lu->Control);

  /* symbolic factorization of A' */
  /* ---------------------------------------------------------------------- */
  if (lu->PetscMatOdering) { /* use Petsc row ordering */
#if !defined(PETSC_USE_COMPLEX)
    status = umfpack_UMF_qsymbolic(n,m,lu->ai,lu->aj,av,lu->perm_c,&lu->Symbolic,lu->Control,lu->Info);
#else
    status = umfpack_UMF_qsymbolic(n,m,lu->ai,lu->aj,NULL,NULL,
                                   lu->perm_c,&lu->Symbolic,lu->Control,lu->Info);
#endif
  } else { /* use Umfpack col ordering */
#if !defined(PETSC_USE_COMPLEX)
    status = umfpack_UMF_symbolic(n,m,lu->ai,lu->aj,av,&lu->Symbolic,lu->Control,lu->Info);
#else
    status = umfpack_UMF_symbolic(n,m,lu->ai,lu->aj,NULL,NULL,&lu->Symbolic,lu->Control,lu->Info);
#endif
  }
  if (status < 0){
    umfpack_UMF_report_info(lu->Control, lu->Info);
    umfpack_UMF_report_status(lu->Control, status);
    SETERRQ(PETSC_ERR_LIB,"umfpack_UMF_symbolic failed");
  }
  /* report sumbolic factorization of A' when Control[PRL] > 3 */
  (void) umfpack_UMF_report_symbolic(lu->Symbolic, lu->Control);

  lu->flg = DIFFERENT_NONZERO_PATTERN;
  lu->av  = av;
  lu->CleanUpUMFPACK = PETSC_TRUE;
  (F)->ops->lufactornumeric  = MatLUFactorNumeric_UMFPACK;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatFactorInfo_UMFPACK"
static PetscErrorCode MatFactorInfo_UMFPACK(Mat A,PetscViewer viewer)
{
  Mat_UMFPACK    *lu= (Mat_UMFPACK*)A->spptr;
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
static PetscErrorCode MatView_UMFPACK(Mat A,PetscViewer viewer)
{
  PetscErrorCode    ierr;
  PetscTruth        iascii;
  PetscViewerFormat format;

  PetscFunctionBegin;
  ierr = MatView_SeqAIJ(A,viewer);CHKERRQ(ierr);

  ierr = PetscTypeCompare((PetscObject)viewer,PETSC_VIEWER_ASCII,&iascii);CHKERRQ(ierr);
  if (iascii) {
    ierr = PetscViewerGetFormat(viewer,&format);CHKERRQ(ierr);
    if (format == PETSC_VIEWER_ASCII_INFO) {
      ierr = MatFactorInfo_UMFPACK(A,viewer);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN 
#undef __FUNCT__  
#define __FUNCT__ "MatFactorGetSolverPackage_seqaij_umfpack"
PetscErrorCode MatFactorGetSolverPackage_seqaij_umfpack(Mat A,const MatSolverPackage *type)
{
  PetscFunctionBegin;
  *type = MAT_SOLVER_UMFPACK;
  PetscFunctionReturn(0);
}
EXTERN_C_END
  

/*MC
  MAT_SOLVER_UMFPACK = "umfpack" - A matrix type providing direct solvers (LU) for sequential matrices 
  via the external package UMFPACK.

  config/configure.py --download-umfpack to install PETSc to use UMFPACK

  Consult UMFPACK documentation for more information about the Control parameters
  which correspond to the options database keys below.

  Options Database Keys:
+ -mat_umfpack_prl - UMFPACK print level: Control[UMFPACK_PRL]
. -mat_umfpack_strategy <AUTO> (choose one of) AUTO UNSYMMETRIC SYMMETRIC 2BY2
. -mat_umfpack_dense_col <alpha_c> - UMFPACK dense column threshold: Control[UMFPACK_DENSE_COL]
. -mat_umfpack_dense_row <0.2>: Control[UMFPACK_DENSE_ROW] 
. -mat_umfpack_amd_dense <10>: Control[UMFPACK_AMD_DENSE] 
. -mat_umfpack_block_size <bs> - UMFPACK block size for BLAS-Level 3 calls: Control[UMFPACK_BLOCK_SIZE]
. -mat_umfpack_2by2_tolerance <0.01>: Control[UMFPACK_2BY2_TOLERANCE] 
. -mat_umfpack_fixq <0>: Control[UMFPACK_FIXQ] 
. -mat_umfpack_aggressive <1>: Control[UMFPACK_AGGRESSIVE] 
. -mat_umfpack_pivot_tolerance <delta> - UMFPACK partial pivot tolerance: Control[UMFPACK_PIVOT_TOLERANCE]
.  -mat_umfpack_sym_pivot_tolerance <0.001>: Control[UMFPACK_SYM_PIVOT_TOLERANCE] 
.  -mat_umfpack_scale <NONE> (choose one of) NONE SUM MAX
. -mat_umfpack_alloc_init <delta> - UMFPACK factorized matrix allocation modifier: Control[UMFPACK_ALLOC_INIT]
.  -mat_umfpack_droptol <0>: Control[UMFPACK_DROPTOL] 
- -mat_umfpack_irstep <maxit> - UMFPACK maximum number of iterative refinement steps: Control[UMFPACK_IRSTEP]

   Level: beginner

.seealso: PCLU, MAT_SOLVER_SUPERLU, MAT_SOLVER_MUMPS, MAT_SOLVER_SPOOLES, PCFactorSetMatSolverPackage(), MatSolverPackage
M*/
EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "MatGetFactor_seqaij_umfpack"
PetscErrorCode MatGetFactor_seqaij_umfpack(Mat A,MatFactorType ftype,Mat *F) 
{
  Mat            B;
  Mat_UMFPACK    *lu;
  PetscErrorCode ierr;
  PetscInt       m=A->rmap->n,n=A->cmap->n,idx;

  const char     *strategy[]={"AUTO","UNSYMMETRIC","SYMMETRIC","2BY2"},
                 *scale[]={"NONE","SUM","MAX"}; 
  PetscTruth     flg;
  
  PetscFunctionBegin;
  /* Create the factorization matrix F */  
  ierr = MatCreate(((PetscObject)A)->comm,&B);CHKERRQ(ierr);
  ierr = MatSetSizes(B,PETSC_DECIDE,PETSC_DECIDE,m,n);CHKERRQ(ierr);
  ierr = MatSetType(B,((PetscObject)A)->type_name);CHKERRQ(ierr);
  ierr = MatSeqAIJSetPreallocation(B,0,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscNewLog(B,Mat_UMFPACK,&lu);CHKERRQ(ierr);
  B->spptr                 = lu;
  B->ops->lufactorsymbolic = MatLUFactorSymbolic_UMFPACK;
  B->ops->destroy          = MatDestroy_UMFPACK;
  B->ops->view             = MatView_UMFPACK;
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)B,"MatFactorGetSolverPackage_C","MatFactorGetSolverPackage_seqaij_umfpack",MatFactorGetSolverPackage_seqaij_umfpack);CHKERRQ(ierr);
  B->factor                = MAT_FACTOR_LU;
  B->assembled             = PETSC_TRUE;  /* required by -ksp_view */
  B->preallocated          = PETSC_TRUE;
  
  /* initializations */
  /* ------------------------------------------------*/
  /* get the default control parameters */
  umfpack_UMF_defaults(lu->Control);
  lu->perm_c = PETSC_NULL;  /* use defaul UMFPACK col permutation */
  lu->Control[UMFPACK_IRSTEP] = 0; /* max num of iterative refinement steps to attempt */

  ierr = PetscOptionsBegin(((PetscObject)A)->comm,((PetscObject)A)->prefix,"UMFPACK Options","Mat");CHKERRQ(ierr);
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
  PetscOptionsEnd();
  *F = B;
  PetscFunctionReturn(0);
}
EXTERN_C_END

