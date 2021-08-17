
/*
   Provides an interface to the UMFPACK sparse solver available through SuiteSparse version 4.2.1

   When build with PETSC_USE_64BIT_INDICES this will use Suitesparse_long as the
   integer type in UMFPACK, otherwise it will use int. This means
   all integers in this file as simply declared as PetscInt. Also it means
   that one cannot use 64BIT_INDICES on 32bit machines [as Suitesparse_long is 32bit only]

*/
#include <../src/mat/impls/aij/seq/aij.h>

#if defined(PETSC_USE_64BIT_INDICES)
#if defined(PETSC_USE_COMPLEX)
#define umfpack_UMF_free_symbolic                      umfpack_zl_free_symbolic
#define umfpack_UMF_free_numeric                       umfpack_zl_free_numeric
/* the type casts are needed because PetscInt is long long while SuiteSparse_long is long and compilers warn even when they are identical */
#define umfpack_UMF_wsolve(a,b,c,d,e,f,g,h,i,j,k,l,m,n) umfpack_zl_wsolve(a,(SuiteSparse_long*)b,(SuiteSparse_long*)c,d,e,f,g,h,i,(SuiteSparse_long*)j,k,l,(SuiteSparse_long*)m,n)
#define umfpack_UMF_numeric(a,b,c,d,e,f,g,h)          umfpack_zl_numeric((SuiteSparse_long*)a,(SuiteSparse_long*)b,c,d,e,f,g,h)
#define umfpack_UMF_report_numeric                    umfpack_zl_report_numeric
#define umfpack_UMF_report_control                    umfpack_zl_report_control
#define umfpack_UMF_report_status                     umfpack_zl_report_status
#define umfpack_UMF_report_info                       umfpack_zl_report_info
#define umfpack_UMF_report_symbolic                   umfpack_zl_report_symbolic
#define umfpack_UMF_qsymbolic(a,b,c,d,e,f,g,h,i,j)    umfpack_zl_qsymbolic(a,b,(SuiteSparse_long*)c,(SuiteSparse_long*)d,e,f,(SuiteSparse_long*)g,h,i,j)
#define umfpack_UMF_symbolic(a,b,c,d,e,f,g,h,i)       umfpack_zl_symbolic(a,b,(SuiteSparse_long*)c,(SuiteSparse_long*)d,e,f,g,h,i)
#define umfpack_UMF_defaults                          umfpack_zl_defaults

#else
#define umfpack_UMF_free_symbolic                  umfpack_dl_free_symbolic
#define umfpack_UMF_free_numeric                   umfpack_dl_free_numeric
#define umfpack_UMF_wsolve(a,b,c,d,e,f,g,h,i,j,k)  umfpack_dl_wsolve(a,(SuiteSparse_long*)b,(SuiteSparse_long*)c,d,e,f,g,h,i,(SuiteSparse_long*)j,k)
#define umfpack_UMF_numeric(a,b,c,d,e,f,g)         umfpack_dl_numeric((SuiteSparse_long*)a,(SuiteSparse_long*)b,c,d,e,f,g)
#define umfpack_UMF_report_numeric                 umfpack_dl_report_numeric
#define umfpack_UMF_report_control                 umfpack_dl_report_control
#define umfpack_UMF_report_status                  umfpack_dl_report_status
#define umfpack_UMF_report_info                    umfpack_dl_report_info
#define umfpack_UMF_report_symbolic                umfpack_dl_report_symbolic
#define umfpack_UMF_qsymbolic(a,b,c,d,e,f,g,h,i)   umfpack_dl_qsymbolic(a,b,(SuiteSparse_long*)c,(SuiteSparse_long*)d,e,(SuiteSparse_long*)f,g,h,i)
#define umfpack_UMF_symbolic(a,b,c,d,e,f,g,h)      umfpack_dl_symbolic(a,b,(SuiteSparse_long*)c,(SuiteSparse_long*)d,e,f,g,h)
#define umfpack_UMF_defaults                       umfpack_dl_defaults
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

EXTERN_C_BEGIN
#include <umfpack.h>
EXTERN_C_END

static const char *const UmfpackOrderingTypes[] = {"CHOLMOD","AMD","GIVEN","METIS","BEST","NONE","USER","UmfpackOrderingTypes","UMFPACK_ORDERING_",0};

typedef struct {
  void         *Symbolic, *Numeric;
  double       Info[UMFPACK_INFO], Control[UMFPACK_CONTROL],*W;
  PetscInt     *Wi,*perm_c;
  Mat          A;               /* Matrix used for factorization */
  MatStructure flg;

  /* Flag to clean up UMFPACK objects during Destroy */
  PetscBool CleanUpUMFPACK;
} Mat_UMFPACK;

static PetscErrorCode MatDestroy_UMFPACK(Mat A)
{
  PetscErrorCode ierr;
  Mat_UMFPACK    *lu=(Mat_UMFPACK*)A->data;

  PetscFunctionBegin;
  if (lu->CleanUpUMFPACK) {
    umfpack_UMF_free_symbolic(&lu->Symbolic);
    umfpack_UMF_free_numeric(&lu->Numeric);
    ierr = PetscFree(lu->Wi);CHKERRQ(ierr);
    ierr = PetscFree(lu->W);CHKERRQ(ierr);
    ierr = PetscFree(lu->perm_c);CHKERRQ(ierr);
  }
  ierr = MatDestroy(&lu->A);CHKERRQ(ierr);
  ierr = PetscFree(A->data);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatSolve_UMFPACK_Private(Mat A,Vec b,Vec x,int uflag)
{
  Mat_UMFPACK       *lu = (Mat_UMFPACK*)A->data;
  Mat_SeqAIJ        *a  = (Mat_SeqAIJ*)lu->A->data;
  PetscScalar       *av = a->a,*xa;
  const PetscScalar *ba;
  PetscErrorCode    ierr;
  PetscInt          *ai = a->i,*aj = a->j,status;
  static PetscBool  cite = PETSC_FALSE;

  PetscFunctionBegin;
  if (!A->rmap->n) PetscFunctionReturn(0);
  ierr = PetscCitationsRegister("@article{davis2004algorithm,\n  title={Algorithm 832: {UMFPACK} V4.3---An Unsymmetric-Pattern Multifrontal Method},\n  author={Davis, Timothy A},\n  journal={ACM Transactions on Mathematical Software (TOMS)},\n  volume={30},\n  number={2},\n  pages={196--199},\n  year={2004},\n  publisher={ACM}\n}\n",&cite);CHKERRQ(ierr);
  /* solve Ax = b by umfpack_*_wsolve */
  /* ----------------------------------*/

  if (!lu->Wi) {  /* first time, allocate working space for wsolve */
    ierr = PetscMalloc1(A->rmap->n,&lu->Wi);CHKERRQ(ierr);
    ierr = PetscMalloc1(5*A->rmap->n,&lu->W);CHKERRQ(ierr);
  }

  ierr = VecGetArrayRead(b,&ba);CHKERRQ(ierr);
  ierr = VecGetArray(x,&xa);CHKERRQ(ierr);
#if defined(PETSC_USE_COMPLEX)
  status = umfpack_UMF_wsolve(uflag,ai,aj,(PetscReal*)av,NULL,(PetscReal*)xa,NULL,(PetscReal*)ba,NULL,lu->Numeric,lu->Control,lu->Info,lu->Wi,lu->W);
#else
  status = umfpack_UMF_wsolve(uflag,ai,aj,av,xa,ba,lu->Numeric,lu->Control,lu->Info,lu->Wi,lu->W);
#endif
  umfpack_UMF_report_info(lu->Control, lu->Info);
  if (status < 0) {
    umfpack_UMF_report_status(lu->Control, status);
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_LIB,"umfpack_UMF_wsolve failed");
  }

  ierr = VecRestoreArrayRead(b,&ba);CHKERRQ(ierr);
  ierr = VecRestoreArray(x,&xa);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatSolve_UMFPACK(Mat A,Vec b,Vec x)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /* We gave UMFPACK the algebraic transpose (because it assumes column alignment) */
  ierr = MatSolve_UMFPACK_Private(A,b,x,UMFPACK_Aat);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatSolveTranspose_UMFPACK(Mat A,Vec b,Vec x)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /* We gave UMFPACK the algebraic transpose (because it assumes column alignment) */
  ierr = MatSolve_UMFPACK_Private(A,b,x,UMFPACK_A);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatLUFactorNumeric_UMFPACK(Mat F,Mat A,const MatFactorInfo *info)
{
  Mat_UMFPACK    *lu = (Mat_UMFPACK*)(F)->data;
  Mat_SeqAIJ     *a  = (Mat_SeqAIJ*)A->data;
  PetscInt       *ai = a->i,*aj=a->j,status;
  PetscScalar    *av = a->a;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!A->rmap->n) PetscFunctionReturn(0);
  /* numeric factorization of A' */
  /* ----------------------------*/

  if (lu->flg == SAME_NONZERO_PATTERN && lu->Numeric) {
    umfpack_UMF_free_numeric(&lu->Numeric);
  }
#if defined(PETSC_USE_COMPLEX)
  status = umfpack_UMF_numeric(ai,aj,(double*)av,NULL,lu->Symbolic,&lu->Numeric,lu->Control,lu->Info);
#else
  status = umfpack_UMF_numeric(ai,aj,av,lu->Symbolic,&lu->Numeric,lu->Control,lu->Info);
#endif
  if (status < 0) {
    umfpack_UMF_report_status(lu->Control, status);
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_LIB,"umfpack_UMF_numeric failed");
  }
  /* report numeric factorization of A' when Control[PRL] > 3 */
  (void) umfpack_UMF_report_numeric(lu->Numeric, lu->Control);

  ierr = PetscObjectReference((PetscObject)A);CHKERRQ(ierr);
  ierr = MatDestroy(&lu->A);CHKERRQ(ierr);

  lu->A                  = A;
  lu->flg                = SAME_NONZERO_PATTERN;
  lu->CleanUpUMFPACK     = PETSC_TRUE;
  F->ops->solve          = MatSolve_UMFPACK;
  F->ops->solvetranspose = MatSolveTranspose_UMFPACK;
  PetscFunctionReturn(0);
}

static PetscErrorCode MatLUFactorSymbolic_UMFPACK(Mat F,Mat A,IS r,IS c,const MatFactorInfo *info)
{
  Mat_SeqAIJ     *a  = (Mat_SeqAIJ*)A->data;
  Mat_UMFPACK    *lu = (Mat_UMFPACK*)(F->data);
  PetscErrorCode ierr;
  PetscInt       i,*ai = a->i,*aj = a->j,m=A->rmap->n,n=A->cmap->n;
#if !defined(PETSC_USE_COMPLEX)
  PetscScalar    *av = a->a;
#endif
  const PetscInt *ra;
  PetscInt       status;

  PetscFunctionBegin;
  (F)->ops->lufactornumeric = MatLUFactorNumeric_UMFPACK;
  if (!n) PetscFunctionReturn(0);
  if (r) {
    ierr = ISGetIndices(r,&ra);CHKERRQ(ierr);
    ierr = PetscMalloc1(m,&lu->perm_c);CHKERRQ(ierr);
    /* we cannot simply memcpy on 64 bit archs */
    for (i = 0; i < m; i++) lu->perm_c[i] = ra[i];
    ierr = ISRestoreIndices(r,&ra);CHKERRQ(ierr);
  }

  /* print the control parameters */
  if (lu->Control[UMFPACK_PRL] > 1) umfpack_UMF_report_control(lu->Control);

  /* symbolic factorization of A' */
  /* ---------------------------------------------------------------------- */
  if (r) { /* use Petsc row ordering */
#if !defined(PETSC_USE_COMPLEX)
    status = umfpack_UMF_qsymbolic(n,m,ai,aj,av,lu->perm_c,&lu->Symbolic,lu->Control,lu->Info);
#else
    status = umfpack_UMF_qsymbolic(n,m,ai,aj,NULL,NULL,lu->perm_c,&lu->Symbolic,lu->Control,lu->Info);
#endif
  } else { /* use Umfpack col ordering */
#if !defined(PETSC_USE_COMPLEX)
    status = umfpack_UMF_symbolic(n,m,ai,aj,av,&lu->Symbolic,lu->Control,lu->Info);
#else
    status = umfpack_UMF_symbolic(n,m,ai,aj,NULL,NULL,&lu->Symbolic,lu->Control,lu->Info);
#endif
  }
  if (status < 0) {
    umfpack_UMF_report_info(lu->Control, lu->Info);
    umfpack_UMF_report_status(lu->Control, status);
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_LIB,"umfpack_UMF_symbolic failed");
  }
  /* report sumbolic factorization of A' when Control[PRL] > 3 */
  (void) umfpack_UMF_report_symbolic(lu->Symbolic, lu->Control);

  lu->flg                   = DIFFERENT_NONZERO_PATTERN;
  lu->CleanUpUMFPACK        = PETSC_TRUE;
  PetscFunctionReturn(0);
}

static PetscErrorCode MatView_Info_UMFPACK(Mat A,PetscViewer viewer)
{
  Mat_UMFPACK    *lu= (Mat_UMFPACK*)A->data;
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
  if (!lu->perm_c) {
    ierr = PetscViewerASCIIPrintf(viewer,"  Control[UMFPACK_ORDERING]: AMD (not using the PETSc ordering)\n",UmfpackOrderingTypes[(int)lu->Control[UMFPACK_ORDERING]]);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode MatView_UMFPACK(Mat A,PetscViewer viewer)
{
  PetscErrorCode    ierr;
  PetscBool         iascii;
  PetscViewerFormat format;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii);CHKERRQ(ierr);
  if (iascii) {
    ierr = PetscViewerGetFormat(viewer,&format);CHKERRQ(ierr);
    if (format == PETSC_VIEWER_ASCII_INFO) {
      ierr = MatView_Info_UMFPACK(A,viewer);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

PetscErrorCode MatFactorGetSolverType_seqaij_umfpack(Mat A,MatSolverType *type)
{
  PetscFunctionBegin;
  *type = MATSOLVERUMFPACK;
  PetscFunctionReturn(0);
}

/*MC
  MATSOLVERUMFPACK = "umfpack" - A matrix type providing direct solvers (LU) for sequential matrices
  via the external package UMFPACK.

  Use ./configure --download-suitesparse to install PETSc to use UMFPACK

  Use -pc_type lu -pc_factor_mat_solver_type umfpack to use this direct solver

  Consult UMFPACK documentation for more information about the Control parameters
  which correspond to the options database keys below.

  Options Database Keys:
+ -mat_umfpack_ordering                - CHOLMOD, AMD, GIVEN, METIS, BEST, NONE
. -mat_umfpack_prl                     - UMFPACK print level: Control[UMFPACK_PRL]
. -mat_umfpack_strategy <AUTO>         - (choose one of) AUTO UNSYMMETRIC SYMMETRIC 2BY2
. -mat_umfpack_dense_col <alpha_c>     - UMFPACK dense column threshold: Control[UMFPACK_DENSE_COL]
. -mat_umfpack_dense_row <0.2>         - Control[UMFPACK_DENSE_ROW]
. -mat_umfpack_amd_dense <10>          - Control[UMFPACK_AMD_DENSE]
. -mat_umfpack_block_size <bs>         - UMFPACK block size for BLAS-Level 3 calls: Control[UMFPACK_BLOCK_SIZE]
. -mat_umfpack_2by2_tolerance <0.01>   - Control[UMFPACK_2BY2_TOLERANCE]
. -mat_umfpack_fixq <0>                - Control[UMFPACK_FIXQ]
. -mat_umfpack_aggressive <1>          - Control[UMFPACK_AGGRESSIVE]
. -mat_umfpack_pivot_tolerance <delta> - UMFPACK partial pivot tolerance: Control[UMFPACK_PIVOT_TOLERANCE]
. -mat_umfpack_sym_pivot_tolerance <0.001> - Control[UMFPACK_SYM_PIVOT_TOLERANCE]
. -mat_umfpack_scale <NONE>           - (choose one of) NONE SUM MAX
. -mat_umfpack_alloc_init <delta>      - UMFPACK factorized matrix allocation modifier: Control[UMFPACK_ALLOC_INIT]
. -mat_umfpack_droptol <0>            - Control[UMFPACK_DROPTOL]
- -mat_umfpack_irstep <maxit>          - UMFPACK maximum number of iterative refinement steps: Control[UMFPACK_IRSTEP]

   Level: beginner

   Note: UMFPACK is part of SuiteSparse http://faculty.cse.tamu.edu/davis/suitesparse.html

.seealso: PCLU, MATSOLVERSUPERLU, MATSOLVERMUMPS, PCFactorSetMatSolverType(), MatSolverType
M*/

PETSC_EXTERN PetscErrorCode MatGetFactor_seqaij_umfpack(Mat A,MatFactorType ftype,Mat *F)
{
  Mat            B;
  Mat_UMFPACK    *lu;
  PetscErrorCode ierr;
  PetscInt       m=A->rmap->n,n=A->cmap->n,idx;
  const char     *strategy[]={"AUTO","UNSYMMETRIC","SYMMETRIC"};
  const char     *scale[]   ={"NONE","SUM","MAX"};
  PetscBool      flg;

  PetscFunctionBegin;
  /* Create the factorization matrix F */
  ierr = MatCreate(PetscObjectComm((PetscObject)A),&B);CHKERRQ(ierr);
  ierr = MatSetSizes(B,PETSC_DECIDE,PETSC_DECIDE,m,n);CHKERRQ(ierr);
  ierr = PetscStrallocpy("umfpack",&((PetscObject)B)->type_name);CHKERRQ(ierr);
  ierr = MatSetUp(B);CHKERRQ(ierr);

  ierr = PetscNewLog(B,&lu);CHKERRQ(ierr);

  B->data                   = lu;
  B->ops->getinfo          = MatGetInfo_External;
  B->ops->lufactorsymbolic = MatLUFactorSymbolic_UMFPACK;
  B->ops->destroy          = MatDestroy_UMFPACK;
  B->ops->view             = MatView_UMFPACK;
  B->ops->matsolve         = NULL;

  ierr = PetscObjectComposeFunction((PetscObject)B,"MatFactorGetSolverType_C",MatFactorGetSolverType_seqaij_umfpack);CHKERRQ(ierr);

  B->factortype   = MAT_FACTOR_LU;
  B->assembled    = PETSC_TRUE;           /* required by -ksp_view */
  B->preallocated = PETSC_TRUE;

  ierr = PetscFree(B->solvertype);CHKERRQ(ierr);
  ierr = PetscStrallocpy(MATSOLVERUMFPACK,&B->solvertype);CHKERRQ(ierr);
  B->canuseordering = PETSC_TRUE;
  ierr = PetscStrallocpy(MATORDERINGEXTERNAL,(char**)&B->preferredordering[MAT_FACTOR_LU]);CHKERRQ(ierr);

  /* initializations */
  /* ------------------------------------------------*/
  /* get the default control parameters */
  umfpack_UMF_defaults(lu->Control);
  lu->perm_c                  = NULL; /* use defaul UMFPACK col permutation */
  lu->Control[UMFPACK_IRSTEP] = 0;          /* max num of iterative refinement steps to attempt */

  ierr = PetscOptionsBegin(PetscObjectComm((PetscObject)A),((PetscObject)A)->prefix,"UMFPACK Options","Mat");CHKERRQ(ierr);
  /* Control parameters used by reporting routiones */
  ierr = PetscOptionsReal("-mat_umfpack_prl","Control[UMFPACK_PRL]","None",lu->Control[UMFPACK_PRL],&lu->Control[UMFPACK_PRL],NULL);CHKERRQ(ierr);

  /* Control parameters for symbolic factorization */
  ierr = PetscOptionsEList("-mat_umfpack_strategy","ordering and pivoting strategy","None",strategy,3,strategy[0],&idx,&flg);CHKERRQ(ierr);
  if (flg) {
    switch (idx) {
    case 0: lu->Control[UMFPACK_STRATEGY] = UMFPACK_STRATEGY_AUTO; break;
    case 1: lu->Control[UMFPACK_STRATEGY] = UMFPACK_STRATEGY_UNSYMMETRIC; break;
    case 2: lu->Control[UMFPACK_STRATEGY] = UMFPACK_STRATEGY_SYMMETRIC; break;
    }
  }
  ierr = PetscOptionsEList("-mat_umfpack_ordering","Internal ordering method","None",UmfpackOrderingTypes,sizeof(UmfpackOrderingTypes)/sizeof(UmfpackOrderingTypes[0]),UmfpackOrderingTypes[(int)lu->Control[UMFPACK_ORDERING]],&idx,&flg);CHKERRQ(ierr);
  if (flg) lu->Control[UMFPACK_ORDERING] = (int)idx;
  ierr = PetscOptionsReal("-mat_umfpack_dense_col","Control[UMFPACK_DENSE_COL]","None",lu->Control[UMFPACK_DENSE_COL],&lu->Control[UMFPACK_DENSE_COL],NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-mat_umfpack_dense_row","Control[UMFPACK_DENSE_ROW]","None",lu->Control[UMFPACK_DENSE_ROW],&lu->Control[UMFPACK_DENSE_ROW],NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-mat_umfpack_amd_dense","Control[UMFPACK_AMD_DENSE]","None",lu->Control[UMFPACK_AMD_DENSE],&lu->Control[UMFPACK_AMD_DENSE],NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-mat_umfpack_block_size","Control[UMFPACK_BLOCK_SIZE]","None",lu->Control[UMFPACK_BLOCK_SIZE],&lu->Control[UMFPACK_BLOCK_SIZE],NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-mat_umfpack_fixq","Control[UMFPACK_FIXQ]","None",lu->Control[UMFPACK_FIXQ],&lu->Control[UMFPACK_FIXQ],NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-mat_umfpack_aggressive","Control[UMFPACK_AGGRESSIVE]","None",lu->Control[UMFPACK_AGGRESSIVE],&lu->Control[UMFPACK_AGGRESSIVE],NULL);CHKERRQ(ierr);

  /* Control parameters used by numeric factorization */
  ierr = PetscOptionsReal("-mat_umfpack_pivot_tolerance","Control[UMFPACK_PIVOT_TOLERANCE]","None",lu->Control[UMFPACK_PIVOT_TOLERANCE],&lu->Control[UMFPACK_PIVOT_TOLERANCE],NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-mat_umfpack_sym_pivot_tolerance","Control[UMFPACK_SYM_PIVOT_TOLERANCE]","None",lu->Control[UMFPACK_SYM_PIVOT_TOLERANCE],&lu->Control[UMFPACK_SYM_PIVOT_TOLERANCE],NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEList("-mat_umfpack_scale","Control[UMFPACK_SCALE]","None",scale,3,scale[0],&idx,&flg);CHKERRQ(ierr);
  if (flg) {
    switch (idx) {
    case 0: lu->Control[UMFPACK_SCALE] = UMFPACK_SCALE_NONE; break;
    case 1: lu->Control[UMFPACK_SCALE] = UMFPACK_SCALE_SUM; break;
    case 2: lu->Control[UMFPACK_SCALE] = UMFPACK_SCALE_MAX; break;
    }
  }
  ierr = PetscOptionsReal("-mat_umfpack_alloc_init","Control[UMFPACK_ALLOC_INIT]","None",lu->Control[UMFPACK_ALLOC_INIT],&lu->Control[UMFPACK_ALLOC_INIT],NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-mat_umfpack_front_alloc_init","Control[UMFPACK_FRONT_ALLOC_INIT]","None",lu->Control[UMFPACK_FRONT_ALLOC_INIT],&lu->Control[UMFPACK_ALLOC_INIT],NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-mat_umfpack_droptol","Control[UMFPACK_DROPTOL]","None",lu->Control[UMFPACK_DROPTOL],&lu->Control[UMFPACK_DROPTOL],NULL);CHKERRQ(ierr);

  /* Control parameters used by solve */
  ierr = PetscOptionsReal("-mat_umfpack_irstep","Control[UMFPACK_IRSTEP]","None",lu->Control[UMFPACK_IRSTEP],&lu->Control[UMFPACK_IRSTEP],NULL);CHKERRQ(ierr);
  PetscOptionsEnd();
  *F = B;
  PetscFunctionReturn(0);
}

PETSC_INTERN PetscErrorCode MatGetFactor_seqaij_cholmod(Mat,MatFactorType,Mat*);
PETSC_INTERN PetscErrorCode MatGetFactor_seqsbaij_cholmod(Mat,MatFactorType,Mat*);
PETSC_INTERN PetscErrorCode MatGetFactor_seqaij_klu(Mat,MatFactorType,Mat*);
PETSC_INTERN PetscErrorCode MatGetFactor_seqaij_spqr(Mat,MatFactorType,Mat*);

PETSC_EXTERN PetscErrorCode MatSolverTypeRegister_SuiteSparse(void)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatSolverTypeRegister(MATSOLVERUMFPACK,MATSEQAIJ,  MAT_FACTOR_LU,MatGetFactor_seqaij_umfpack);CHKERRQ(ierr);
  ierr = MatSolverTypeRegister(MATSOLVERCHOLMOD,MATSEQAIJ,  MAT_FACTOR_CHOLESKY,MatGetFactor_seqaij_cholmod);CHKERRQ(ierr);
  ierr = MatSolverTypeRegister(MATSOLVERCHOLMOD,MATSEQSBAIJ,MAT_FACTOR_CHOLESKY,MatGetFactor_seqsbaij_cholmod);CHKERRQ(ierr);
  ierr = MatSolverTypeRegister(MATSOLVERKLU,MATSEQAIJ,      MAT_FACTOR_LU,MatGetFactor_seqaij_klu);CHKERRQ(ierr);
  ierr = MatSolverTypeRegister(MATSOLVERSPQR,MATSEQAIJ,     MAT_FACTOR_QR,MatGetFactor_seqaij_spqr);CHKERRQ(ierr);
  if (!PetscDefined(USE_COMPLEX)) {
    ierr = MatSolverTypeRegister(MATSOLVERSPQR,MATNORMAL,   MAT_FACTOR_QR,MatGetFactor_seqaij_spqr);CHKERRQ(ierr);
  }
  ierr = MatSolverTypeRegister(MATSOLVERSPQR,MATNORMALHERMITIAN, MAT_FACTOR_QR,MatGetFactor_seqaij_spqr);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
