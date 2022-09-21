
/*
   Provides an interface to the KLUv1.2 sparse solver

   When build with PETSC_USE_64BIT_INDICES this will use SuiteSparse_long as the
   integer type in KLU, otherwise it will use int. This means
   all integers in this file are simply declared as PetscInt. Also it means
   that KLU SuiteSparse_long version MUST be built with 64 bit integers when used.

*/
#include <../src/mat/impls/aij/seq/aij.h>

#if defined(PETSC_USE_64BIT_INDICES)
  #define klu_K_defaults                        klu_l_defaults
  #define klu_K_analyze(a, b, c, d)             klu_l_analyze((SuiteSparse_long)a, (SuiteSparse_long *)b, (SuiteSparse_long *)c, d)
  #define klu_K_analyze_given(a, b, c, d, e, f) klu_l_analyze_given((SuiteSparse_long)a, (SuiteSparse_long *)b, (SuiteSparse_long *)c, (SuiteSparse_long *)d, (SuiteSparse_long *)e, f)
  #define klu_K_free_symbolic                   klu_l_free_symbolic
  #define klu_K_free_numeric                    klu_l_free_numeric
  #define klu_K_common                          klu_l_common
  #define klu_K_symbolic                        klu_l_symbolic
  #define klu_K_numeric                         klu_l_numeric
  #if defined(PETSC_USE_COMPLEX)
    #define klu_K_factor(a, b, c, d, e) klu_zl_factor((SuiteSparse_long *)a, (SuiteSparse_long *)b, c, d, e);
    #define klu_K_solve                 klu_zl_solve
    #define klu_K_tsolve                klu_zl_tsolve
    #define klu_K_refactor              klu_zl_refactor
    #define klu_K_sort                  klu_zl_sort
    #define klu_K_flops                 klu_zl_flops
    #define klu_K_rgrowth               klu_zl_rgrowth
    #define klu_K_condest               klu_zl_condest
    #define klu_K_rcond                 klu_zl_rcond
    #define klu_K_scale                 klu_zl_scale
  #else
    #define klu_K_factor(a, b, c, d, e) klu_l_factor((SuiteSparse_long *)a, (SuiteSparse_long *)b, c, d, e);
    #define klu_K_solve                 klu_l_solve
    #define klu_K_tsolve                klu_l_tsolve
    #define klu_K_refactor              klu_l_refactor
    #define klu_K_sort                  klu_l_sort
    #define klu_K_flops                 klu_l_flops
    #define klu_K_rgrowth               klu_l_rgrowth
    #define klu_K_condest               klu_l_condest
    #define klu_K_rcond                 klu_l_rcond
    #define klu_K_scale                 klu_l_scale
  #endif
#else
  #define klu_K_defaults      klu_defaults
  #define klu_K_analyze       klu_analyze
  #define klu_K_analyze_given klu_analyze_given
  #define klu_K_free_symbolic klu_free_symbolic
  #define klu_K_free_numeric  klu_free_numeric
  #define klu_K_common        klu_common
  #define klu_K_symbolic      klu_symbolic
  #define klu_K_numeric       klu_numeric
  #if defined(PETSC_USE_COMPLEX)
    #define klu_K_factor   klu_z_factor
    #define klu_K_solve    klu_z_solve
    #define klu_K_tsolve   klu_z_tsolve
    #define klu_K_refactor klu_z_refactor
    #define klu_K_sort     klu_z_sort
    #define klu_K_flops    klu_z_flops
    #define klu_K_rgrowth  klu_z_rgrowth
    #define klu_K_condest  klu_z_condest
    #define klu_K_rcond    klu_z_rcond
    #define klu_K_scale    klu_z_scale
  #else
    #define klu_K_factor   klu_factor
    #define klu_K_solve    klu_solve
    #define klu_K_tsolve   klu_tsolve
    #define klu_K_refactor klu_refactor
    #define klu_K_sort     klu_sort
    #define klu_K_flops    klu_flops
    #define klu_K_rgrowth  klu_rgrowth
    #define klu_K_condest  klu_condest
    #define klu_K_rcond    klu_rcond
    #define klu_K_scale    klu_scale
  #endif
#endif

EXTERN_C_BEGIN
#include <klu.h>
EXTERN_C_END

static const char *KluOrderingTypes[] = {"AMD", "COLAMD"};
static const char *scale[]            = {"NONE", "SUM", "MAX"};

typedef struct {
  klu_K_common    Common;
  klu_K_symbolic *Symbolic;
  klu_K_numeric  *Numeric;
  PetscInt       *perm_c, *perm_r;
  MatStructure    flg;
  PetscBool       PetscMatOrdering;
  PetscBool       CleanUpKLU;
} Mat_KLU;

static PetscErrorCode MatDestroy_KLU(Mat A)
{
  Mat_KLU *lu = (Mat_KLU *)A->data;

  PetscFunctionBegin;
  if (lu->CleanUpKLU) {
    klu_K_free_symbolic(&lu->Symbolic, &lu->Common);
    klu_K_free_numeric(&lu->Numeric, &lu->Common);
    PetscCall(PetscFree2(lu->perm_r, lu->perm_c));
  }
  PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatFactorGetSolverType_C", NULL));
  PetscCall(PetscFree(A->data));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatSolveTranspose_KLU(Mat A, Vec b, Vec x)
{
  Mat_KLU     *lu = (Mat_KLU *)A->data;
  PetscScalar *xa;
  PetscInt     status;

  PetscFunctionBegin;
  /* KLU uses a column major format, solve Ax = b by klu_*_solve */
  /* ----------------------------------*/
  PetscCall(VecCopy(b, x)); /* klu_solve stores the solution in rhs */
  PetscCall(VecGetArray(x, &xa));
  status = klu_K_solve(lu->Symbolic, lu->Numeric, A->rmap->n, 1, (PetscReal *)xa, &lu->Common);
  PetscCheck(status == 1, PETSC_COMM_SELF, PETSC_ERR_LIB, "KLU Solve failed");
  PetscCall(VecRestoreArray(x, &xa));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatSolve_KLU(Mat A, Vec b, Vec x)
{
  Mat_KLU     *lu = (Mat_KLU *)A->data;
  PetscScalar *xa;
  PetscInt     status;

  PetscFunctionBegin;
  /* KLU uses a column major format, solve A^Tx = b by klu_*_tsolve */
  /* ----------------------------------*/
  PetscCall(VecCopy(b, x)); /* klu_solve stores the solution in rhs */
  PetscCall(VecGetArray(x, &xa));
#if defined(PETSC_USE_COMPLEX)
  PetscInt conj_solve = 1;
  status              = klu_K_tsolve(lu->Symbolic, lu->Numeric, A->rmap->n, 1, (PetscReal *)xa, conj_solve, &lu->Common); /* conjugate solve */
#else
  status = klu_K_tsolve(lu->Symbolic, lu->Numeric, A->rmap->n, 1, xa, &lu->Common);
#endif
  PetscCheck(status == 1, PETSC_COMM_SELF, PETSC_ERR_LIB, "KLU Solve failed");
  PetscCall(VecRestoreArray(x, &xa));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatLUFactorNumeric_KLU(Mat F, Mat A, const MatFactorInfo *info)
{
  Mat_KLU     *lu = (Mat_KLU *)(F)->data;
  Mat_SeqAIJ  *a  = (Mat_SeqAIJ *)A->data;
  PetscInt    *ai = a->i, *aj = a->j;
  PetscScalar *av = a->a;

  PetscFunctionBegin;
  /* numeric factorization of A' */
  /* ----------------------------*/

  if (lu->flg == SAME_NONZERO_PATTERN && lu->Numeric) klu_K_free_numeric(&lu->Numeric, &lu->Common);
  lu->Numeric = klu_K_factor(ai, aj, (PetscReal *)av, lu->Symbolic, &lu->Common);
  PetscCheck(lu->Numeric, PETSC_COMM_SELF, PETSC_ERR_LIB, "KLU Numeric factorization failed");

  lu->flg                = SAME_NONZERO_PATTERN;
  lu->CleanUpKLU         = PETSC_TRUE;
  F->ops->solve          = MatSolve_KLU;
  F->ops->solvetranspose = MatSolveTranspose_KLU;
  PetscFunctionReturn(0);
}

static PetscErrorCode MatLUFactorSymbolic_KLU(Mat F, Mat A, IS r, IS c, const MatFactorInfo *info)
{
  Mat_SeqAIJ     *a  = (Mat_SeqAIJ *)A->data;
  Mat_KLU        *lu = (Mat_KLU *)(F->data);
  PetscInt        i, *ai = a->i, *aj = a->j, m = A->rmap->n, n = A->cmap->n;
  const PetscInt *ra, *ca;

  PetscFunctionBegin;
  if (lu->PetscMatOrdering) {
    PetscCall(ISGetIndices(r, &ra));
    PetscCall(ISGetIndices(c, &ca));
    PetscCall(PetscMalloc2(m, &lu->perm_r, n, &lu->perm_c));
    /* we cannot simply memcpy on 64 bit archs */
    for (i = 0; i < m; i++) lu->perm_r[i] = ra[i];
    for (i = 0; i < n; i++) lu->perm_c[i] = ca[i];
    PetscCall(ISRestoreIndices(r, &ra));
    PetscCall(ISRestoreIndices(c, &ca));
  }

  /* symbolic factorization of A' */
  /* ---------------------------------------------------------------------- */
  if (r) {
    lu->PetscMatOrdering = PETSC_TRUE;
    lu->Symbolic         = klu_K_analyze_given(n, ai, aj, lu->perm_c, lu->perm_r, &lu->Common);
  } else { /* use klu internal ordering */
    lu->Symbolic = klu_K_analyze(n, ai, aj, &lu->Common);
  }
  PetscCheck(lu->Symbolic, PETSC_COMM_SELF, PETSC_ERR_LIB, "KLU Symbolic Factorization failed");

  lu->flg                   = DIFFERENT_NONZERO_PATTERN;
  lu->CleanUpKLU            = PETSC_TRUE;
  (F)->ops->lufactornumeric = MatLUFactorNumeric_KLU;
  PetscFunctionReturn(0);
}

static PetscErrorCode MatView_Info_KLU(Mat A, PetscViewer viewer)
{
  Mat_KLU       *lu      = (Mat_KLU *)A->data;
  klu_K_numeric *Numeric = (klu_K_numeric *)lu->Numeric;

  PetscFunctionBegin;
  PetscCall(PetscViewerASCIIPrintf(viewer, "KLU stats:\n"));
  PetscCall(PetscViewerASCIIPrintf(viewer, "  Number of diagonal blocks: %" PetscInt_FMT "\n", (PetscInt)(Numeric->nblocks)));
  PetscCall(PetscViewerASCIIPrintf(viewer, "  Total nonzeros=%" PetscInt_FMT "\n", (PetscInt)(Numeric->lnz + Numeric->unz)));
  PetscCall(PetscViewerASCIIPrintf(viewer, "KLU runtime parameters:\n"));
  /* Control parameters used by numeric factorization */
  PetscCall(PetscViewerASCIIPrintf(viewer, "  Partial pivoting tolerance: %g\n", lu->Common.tol));
  /* BTF preordering */
  PetscCall(PetscViewerASCIIPrintf(viewer, "  BTF preordering enabled: %" PetscInt_FMT "\n", (PetscInt)(lu->Common.btf)));
  /* mat ordering */
  if (!lu->PetscMatOrdering) PetscCall(PetscViewerASCIIPrintf(viewer, "  Ordering: %s (not using the PETSc ordering)\n", KluOrderingTypes[(int)lu->Common.ordering]));
  /* matrix row scaling */
  PetscCall(PetscViewerASCIIPrintf(viewer, "  Matrix row scaling: %s\n", scale[(int)lu->Common.scale]));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatView_KLU(Mat A, PetscViewer viewer)
{
  PetscBool         iascii;
  PetscViewerFormat format;

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERASCII, &iascii));
  if (iascii) {
    PetscCall(PetscViewerGetFormat(viewer, &format));
    if (format == PETSC_VIEWER_ASCII_INFO) PetscCall(MatView_Info_KLU(A, viewer));
  }
  PetscFunctionReturn(0);
}

PetscErrorCode MatFactorGetSolverType_seqaij_klu(Mat A, MatSolverType *type)
{
  PetscFunctionBegin;
  *type = MATSOLVERKLU;
  PetscFunctionReturn(0);
}

/*MC
  MATSOLVERKLU = "klu" - A matrix type providing direct solvers, LU, for sequential matrices
  via the external package KLU.

  ./configure --download-suitesparse to install PETSc to use KLU

  Use -pc_type lu -pc_factor_mat_solver_type klu to use this direct solver

  Consult KLU documentation for more information on the options database keys below.

  Options Database Keys:
+ -mat_klu_pivot_tol <0.001>                  - Partial pivoting tolerance
. -mat_klu_use_btf <1>                        - Use BTF preordering
. -mat_klu_ordering <AMD>                     - KLU reordering scheme to reduce fill-in (choose one of) AMD COLAMD PETSC
- -mat_klu_row_scale <NONE>                   - Matrix row scaling (choose one of) NONE SUM MAX

   Note: KLU is part of SuiteSparse http://faculty.cse.tamu.edu/davis/suitesparse.html

   Level: beginner

.seealso: `PCLU`, `MATSOLVERUMFPACK`, `MATSOLVERCHOLMOD`, `PCFactorSetMatSolverType()`, `MatSolverType`
M*/

PETSC_INTERN PetscErrorCode MatGetFactor_seqaij_klu(Mat A, MatFactorType ftype, Mat *F)
{
  Mat       B;
  Mat_KLU  *lu;
  PetscInt  m = A->rmap->n, n = A->cmap->n, idx = 0, status;
  PetscBool flg;

  PetscFunctionBegin;
  /* Create the factorization matrix F */
  PetscCall(MatCreate(PetscObjectComm((PetscObject)A), &B));
  PetscCall(MatSetSizes(B, PETSC_DECIDE, PETSC_DECIDE, m, n));
  PetscCall(PetscStrallocpy("klu", &((PetscObject)B)->type_name));
  PetscCall(MatSetUp(B));

  PetscCall(PetscNew(&lu));

  B->data                  = lu;
  B->ops->getinfo          = MatGetInfo_External;
  B->ops->lufactorsymbolic = MatLUFactorSymbolic_KLU;
  B->ops->destroy          = MatDestroy_KLU;
  B->ops->view             = MatView_KLU;

  PetscCall(PetscObjectComposeFunction((PetscObject)B, "MatFactorGetSolverType_C", MatFactorGetSolverType_seqaij_klu));

  B->factortype   = MAT_FACTOR_LU;
  B->assembled    = PETSC_TRUE; /* required by -ksp_view */
  B->preallocated = PETSC_TRUE;

  PetscCall(PetscFree(B->solvertype));
  PetscCall(PetscStrallocpy(MATSOLVERKLU, &B->solvertype));
  B->canuseordering = PETSC_TRUE;
  PetscCall(PetscStrallocpy(MATORDERINGEXTERNAL, (char **)&B->preferredordering[MAT_FACTOR_LU]));

  /* initializations */
  /* ------------------------------------------------*/
  /* get the default control parameters */
  status = klu_K_defaults(&lu->Common);
  PetscCheck(status > 0, PETSC_COMM_SELF, PETSC_ERR_LIB, "KLU Initialization failed");

  lu->Common.scale = 0; /* No row scaling */

  PetscOptionsBegin(PetscObjectComm((PetscObject)B), ((PetscObject)B)->prefix, "KLU Options", "Mat");
  /* Partial pivoting tolerance */
  PetscCall(PetscOptionsReal("-mat_klu_pivot_tol", "Partial pivoting tolerance", "None", lu->Common.tol, &lu->Common.tol, NULL));
  /* BTF pre-ordering */
  PetscCall(PetscOptionsInt("-mat_klu_use_btf", "Enable BTF preordering", "None", (PetscInt)lu->Common.btf, (PetscInt *)&lu->Common.btf, NULL));
  /* Matrix reordering */
  PetscCall(PetscOptionsEList("-mat_klu_ordering", "Internal ordering method", "None", KluOrderingTypes, PETSC_STATIC_ARRAY_LENGTH(KluOrderingTypes), KluOrderingTypes[0], &idx, &flg));
  lu->Common.ordering = (int)idx;
  /* Matrix row scaling */
  PetscCall(PetscOptionsEList("-mat_klu_row_scale", "Matrix row scaling", "None", scale, 3, scale[0], &idx, &flg));
  PetscOptionsEnd();
  *F = B;
  PetscFunctionReturn(0);
}
