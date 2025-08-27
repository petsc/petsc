/*
 Provides an interface to the PaStiX sparse solver
 */
#include <../src/mat/impls/aij/seq/aij.h>
#include <../src/mat/impls/aij/mpi/mpiaij.h>
#include <../src/mat/impls/sbaij/seq/sbaij.h>
#include <../src/mat/impls/sbaij/mpi/mpisbaij.h>

#if defined(PETSC_USE_COMPLEX)
  #define _H_COMPLEX
#endif

#include <pastix.h>

#if defined(PETSC_USE_COMPLEX)
  #if defined(PETSC_USE_REAL_SINGLE)
    #define SPM_FLTTYPE SpmComplex32
  #else
    #define SPM_FLTTYPE SpmComplex64
  #endif
#else /* PETSC_USE_COMPLEX */

  #if defined(PETSC_USE_REAL_SINGLE)
    #define SPM_FLTTYPE SpmFloat
  #else
    #define SPM_FLTTYPE SpmDouble
  #endif

#endif /* PETSC_USE_COMPLEX */

typedef PetscScalar PastixScalar;

typedef struct Mat_Pastix_ {
  pastix_data_t *pastix_data;       /* Pastix data storage structure                             */
  MPI_Comm       comm;              /* MPI Communicator used to initialize pastix                */
  spmatrix_t    *spm;               /* SPM matrix structure                                      */
  MatStructure   matstruc;          /* DIFFERENT_NONZERO_PATTERN if uninitialized, SAME otherwise */
  PetscScalar   *rhs;               /* Right-hand-side member                                    */
  PetscInt       rhsnbr;            /* Right-hand-side number                                    */
  pastix_int_t   iparm[IPARM_SIZE]; /* Integer parameters                                        */
  double         dparm[DPARM_SIZE]; /* Floating point parameters                                 */
} Mat_Pastix;

/*
   convert PETSc matrix to SPM structure

  input:
    A       - matrix in aij, baij or sbaij format
    reuse   - MAT_INITIAL_MATRIX: spaces are allocated and values are set for the triple
              MAT_REUSE_MATRIX:   only the values in v array are updated
  output:
    spm     - The SPM built from A
 */
static PetscErrorCode MatConvertToSPM(Mat A, MatReuse reuse, Mat_Pastix *pastix)
{
  Mat                A_loc, A_aij;
  const PetscInt    *row, *col;
  PetscInt           n, i;
  const PetscScalar *val;
  PetscBool          ismpiaij, isseqaij, ismpisbaij, isseqsbaij;
  PetscBool          flag;
  spmatrix_t         spm2, *spm = NULL;
  int                spm_err;

  PetscFunctionBegin;
  /* Get A datas */
  PetscCall(PetscObjectBaseTypeCompare((PetscObject)A, MATSEQAIJ, &isseqaij));
  PetscCall(PetscObjectBaseTypeCompare((PetscObject)A, MATMPIAIJ, &ismpiaij));
  /* TODO: Block Aij should be handled with dof in spm */
  PetscCall(PetscObjectBaseTypeCompare((PetscObject)A, MATSEQSBAIJ, &isseqsbaij));
  PetscCall(PetscObjectBaseTypeCompare((PetscObject)A, MATMPISBAIJ, &ismpisbaij));

  if (isseqsbaij || ismpisbaij) PetscCall(MatConvert(A, MATAIJ, reuse, &A_aij));
  else A_aij = A;

  if (ismpiaij || ismpisbaij) PetscCall(MatMPIAIJGetLocalMat(A_aij, MAT_INITIAL_MATRIX, &A_loc));
  else if (isseqaij || isseqsbaij) A_loc = A_aij;
  else SETERRQ(PetscObjectComm((PetscObject)A_aij), PETSC_ERR_SUP, "Not for type %s", ((PetscObject)A)->type_name);

  /* Use getRowIJ and the trick CSC/CSR instead of GetColumnIJ for performance */
  PetscCall(MatGetRowIJ(A_loc, 0, PETSC_FALSE, PETSC_FALSE, &n, &row, &col, &flag));
  PetscCheck(flag, PETSC_COMM_SELF, PETSC_ERR_SUP, "GetRowIJ failed");
  PetscCall(MatSeqAIJGetArrayRead(A_loc, &val));

  PetscCall(PetscMalloc1(1, &spm));
  PetscStackCallExternalVoid("spmInitDist", spmInitDist(spm, pastix->comm));

  spm->n          = n;
  spm->nnz        = row[n];
  spm->fmttype    = SpmCSR;
  spm->flttype    = SPM_FLTTYPE;
  spm->replicated = !(A->rmap->n != A->rmap->N);

  PetscStackCallExternalVoid("spmUpdateComputedFields", spmUpdateComputedFields(spm));
  PetscStackCallExternalVoid("spmAlloc", spmAlloc(spm));

  /* Get data distribution */
  if (!spm->replicated) {
    for (i = A->rmap->rstart; i < A->rmap->rend; i++) spm->loc2glob[i - A->rmap->rstart] = i;
  }

  /* Copy  arrays */
  PetscCall(PetscArraycpy(spm->colptr, col, spm->nnz));
  PetscCall(PetscArraycpy(spm->rowptr, row, spm->n + 1));
  PetscCall(PetscArraycpy((PetscScalar *)spm->values, val, spm->nnzexp));
  PetscCall(MatRestoreRowIJ(A_loc, 0, PETSC_FALSE, PETSC_FALSE, &n, &row, &col, &flag));
  PetscCheck(flag, PETSC_COMM_SELF, PETSC_ERR_SUP, "RestoreRowIJ failed");
  PetscCall(MatSeqAIJRestoreArrayRead(A_loc, &val));
  if (ismpiaij || ismpisbaij) PetscCall(MatDestroy(&A_loc));

  /*
    PaStiX works only with CSC matrices for now, so let's lure him by telling him
    that the PETSc CSR matrix is a CSC matrix.
    Note that this is not available yet for Hermitian matrices and LL^h/LDL^h factorizations.
   */
  {
    spm_int_t *tmp;
    spm->fmttype                         = SpmCSC;
    tmp                                  = spm->colptr;
    spm->colptr                          = spm->rowptr;
    spm->rowptr                          = tmp;
    pastix->iparm[IPARM_TRANSPOSE_SOLVE] = PastixTrans;
  }

  /* Update matrix to be in PaStiX format */
  PetscStackCallExternalVoid("spmCheckAndCorrect", spm_err = spmCheckAndCorrect(spm, &spm2));
  if (spm_err != 0) {
    PetscStackCallExternalVoid("spmExit", spmExit(spm));
    *spm = spm2;
  }

  if (isseqsbaij || ismpisbaij) PetscCall(MatDestroy(&A_aij));

  pastix->spm = spm;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  Call clean step of PaStiX if initialized
  Free the SpM matrix and the PaStiX instance.
 */
static PetscErrorCode MatDestroy_PaStiX(Mat A)
{
  Mat_Pastix *pastix = (Mat_Pastix *)A->data;

  PetscFunctionBegin;
  /* Finalize SPM (matrix handler of PaStiX) */
  if (pastix->spm) {
    PetscStackCallExternalVoid("spmExit", spmExit(pastix->spm));
    PetscCall(PetscFree(pastix->spm));
  }

  /* clear composed functions */
  PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatFactorGetSolverType_C", NULL));

  /* Finalize PaStiX */
  if (pastix->pastix_data) pastixFinalize(&pastix->pastix_data);

  /* Deallocate PaStiX structure */
  PetscCall(PetscFree(A->data));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  Gather right-hand side.
  Call for Solve step.
  Scatter solution.
 */
static PetscErrorCode MatSolve_PaStiX(Mat A, Vec b, Vec x)
{
  Mat_Pastix        *pastix = (Mat_Pastix *)A->data;
  const PetscScalar *bptr;
  PetscInt           ldrhs;

  PetscFunctionBegin;
  pastix->rhsnbr = 1;
  ldrhs          = pastix->spm->n;

  PetscCall(VecCopy(b, x));
  PetscCall(VecGetArray(x, &pastix->rhs));
  PetscCall(VecGetArrayRead(b, &bptr));

  /* solve phase */
  /*-------------*/
  PetscCheck(pastix->pastix_data, PETSC_COMM_SELF, PETSC_ERR_SUP, "PaStiX hasn't been initialized");
  PetscCallExternal(pastix_task_solve, pastix->pastix_data, ldrhs, pastix->rhsnbr, pastix->rhs, ldrhs);
  PetscCallExternal(pastix_task_refine, pastix->pastix_data, ldrhs, pastix->rhsnbr, (PetscScalar *)bptr, ldrhs, pastix->rhs, ldrhs);

  PetscCall(VecRestoreArray(x, &pastix->rhs));
  PetscCall(VecRestoreArrayRead(b, &bptr));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  Numeric factorisation using PaStiX solver.

  input:
    F       - PETSc matrix that contains PaStiX interface.
    A       - PETSc matrix in aij, bail or sbaij format
 */
static PetscErrorCode MatFactorNumeric_PaStiX(Mat F, Mat A, const MatFactorInfo *info)
{
  Mat_Pastix *pastix = (Mat_Pastix *)F->data;

  PetscFunctionBegin;
  F->ops->solve = MatSolve_PaStiX;

  /* Perform Numerical Factorization */
  PetscCheck(pastix->pastix_data, PETSC_COMM_SELF, PETSC_ERR_SUP, "PaStiX hasn't been initialized");
  PetscCallExternal(pastix_task_numfact, pastix->pastix_data, pastix->spm);

  F->assembled = PETSC_TRUE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatLUFactorNumeric_PaStiX(Mat F, Mat A, const MatFactorInfo *info)
{
  Mat_Pastix *pastix = (Mat_Pastix *)F->data;

  PetscFunctionBegin;
  PetscCheck(pastix->iparm[IPARM_FACTORIZATION] == PastixFactGETRF, PetscObjectComm((PetscObject)F), PETSC_ERR_SUP, "Incorrect factorization type for symbolic and numerical factorization by PaStiX");
  pastix->iparm[IPARM_FACTORIZATION] = PastixFactGETRF;
  PetscCall(MatFactorNumeric_PaStiX(F, A, info));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatCholeskyFactorNumeric_PaStiX(Mat F, Mat A, const MatFactorInfo *info)
{
  Mat_Pastix *pastix = (Mat_Pastix *)F->data;

  PetscFunctionBegin;
  PetscCheck(pastix->iparm[IPARM_FACTORIZATION] == PastixFactSYTRF, PetscObjectComm((PetscObject)F), PETSC_ERR_SUP, "Incorrect factorization type for symbolic and numerical factorization by PaStiX");
  pastix->iparm[IPARM_FACTORIZATION] = PastixFactSYTRF;
  PetscCall(MatFactorNumeric_PaStiX(F, A, info));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  Perform Ordering step and Symbolic Factorization step

  Note the PETSc r and c permutations are ignored
  input:
    F       - PETSc matrix that contains PaStiX interface.
    A       - matrix in aij, bail or sbaij format
    r       - permutation ?
    c       - TODO
    info    - Information about the factorization to perform.
  output:
    pastix_data - This instance will be updated with the SolverMatrix allocated.
 */
static PetscErrorCode MatFactorSymbolic_PaStiX(Mat F, Mat A, IS r, IS c, const MatFactorInfo *info)
{
  Mat_Pastix *pastix = (Mat_Pastix *)F->data;

  PetscFunctionBegin;
  pastix->matstruc = DIFFERENT_NONZERO_PATTERN;

  /* Initialise SPM structure */
  PetscCall(MatConvertToSPM(A, MAT_INITIAL_MATRIX, pastix));

  /* Ordering - Symbolic factorization - Build SolverMatrix  */
  PetscCheck(pastix->pastix_data, PETSC_COMM_SELF, PETSC_ERR_SUP, "PaStiX hasn't been initialized");
  PetscCallExternal(pastix_task_analyze, pastix->pastix_data, pastix->spm);
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatLUFactorSymbolic_PaStiX(Mat F, Mat A, IS r, IS c, const MatFactorInfo *info)
{
  Mat_Pastix *pastix = (Mat_Pastix *)F->data;

  PetscFunctionBegin;
  pastix->iparm[IPARM_FACTORIZATION] = PastixFactGETRF;
  PetscCall(MatFactorSymbolic_PaStiX(F, A, r, c, info));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Note the PETSc r permutation is ignored */
static PetscErrorCode MatCholeskyFactorSymbolic_PaStiX(Mat F, Mat A, IS r, const MatFactorInfo *info)
{
  Mat_Pastix *pastix = (Mat_Pastix *)F->data;

  PetscFunctionBegin;
  /* Warning: Cholesky in PETSc wrapper does not handle (complex) Hermitian matrices.
     The factorization type can be forced using the parameter
     mat_pastix_factorization (see enum pastix_factotype_e in
     https://solverstack.gitlabpages.inria.fr/pastix/group__pastix__api.html). */
  pastix->iparm[IPARM_FACTORIZATION] = PastixFactSYTRF;
  PetscCall(MatFactorSymbolic_PaStiX(F, A, r, NULL, info));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatView_PaStiX(Mat A, PetscViewer viewer)
{
  PetscBool         isascii;
  PetscViewerFormat format;

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERASCII, &isascii));
  if (isascii) {
    PetscCall(PetscViewerGetFormat(viewer, &format));
    if (format == PETSC_VIEWER_ASCII_INFO) {
      Mat_Pastix *pastix = (Mat_Pastix *)A->data;
      spmatrix_t *spm    = pastix->spm;
      PetscCheck(!spm, PETSC_COMM_SELF, PETSC_ERR_SUP, "Sparse matrix isn't initialized");

      PetscCall(PetscViewerASCIIPrintf(viewer, "PaStiX run parameters:\n"));
      PetscCall(PetscViewerASCIIPrintf(viewer, "  Matrix type :                      %s \n", ((spm->mtxtype == SpmSymmetric) ? "Symmetric" : "Unsymmetric")));
      PetscCall(PetscViewerASCIIPrintf(viewer, "  Level of printing (0,1,2):         %ld \n", (long)pastix->iparm[IPARM_VERBOSE]));
      PetscCall(PetscViewerASCIIPrintf(viewer, "  Number of refinements iterations : %ld \n", (long)pastix->iparm[IPARM_NBITER]));
      PetscCall(PetscPrintf(PETSC_COMM_SELF, "  Error :                            %e \n", pastix->dparm[DPARM_RELATIVE_ERROR]));
      if (pastix->iparm[IPARM_VERBOSE] > 0) spmPrintInfo(spm, stdout);
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*MC
     MATSOLVERPASTIX  - A solver package providing direct solvers (LU) for distributed
  and sequential matrices via the external package PaStiX.

  Use `./configure --download-hwloc --download-metis --download-ptscotch --download-pastix --download-netlib-lapack [or MKL for ex. --with-blaslapack-dir=${MKLROOT}]`
  to have PETSc installed with PasTiX.

  Use `-pc_type lu` `-pc_factor_mat_solver_type pastix` to use this direct solver.

  Options Database Keys:
  -mat_pastix_verbose <0,1,2>             - print level of information messages from PaStiX
  -mat_pastix_factorization <0,1,2,3,4>   - Factorization algorithm (Cholesky, LDL^t, LU, LL^t, LDL^h)
  -mat_pastix_itermax <integer>           - Maximum number of iterations during refinement
  -mat_pastix_epsilon_refinement <double> - Epsilon for refinement
  -mat_pastix_epsilon_magn_ctrl <double>  - Epsilon for magnitude control
  -mat_pastix_ordering <0,1>              - Ordering (Scotch or Metis)
  -mat_pastix_thread_nbr <integer>        - Set the numbers of threads for each MPI process
  -mat_pastix_scheduler <0,1,2,3,4>       - Scheduler (sequential, static, parsec, starpu, dynamic)
  -mat_pastix_compress_when <0,1,2,3>     - When to compress (never, minimal-theory, just-in-time, supernodes)
  -mat_pastix_compress_tolerance <double> - Tolerance for low-rank kernels.

  Notes:
    This only works for matrices with symmetric nonzero structure, if you pass it a matrix with
   nonsymmetric structure PasTiX, and hence, PETSc return with an error.

  Level: beginner

.seealso: [](ch_matrices), `Mat`, `PCFactorSetMatSolverType()`, `MatSolverType`, `MatGetFactor()`
M*/

static PetscErrorCode MatGetInfo_PaStiX(Mat A, MatInfoType flag, MatInfo *info)
{
  Mat_Pastix *pastix = (Mat_Pastix *)A->data;

  PetscFunctionBegin;
  info->block_size        = 1.0;
  info->nz_allocated      = pastix->iparm[IPARM_ALLOCATED_TERMS];
  info->nz_used           = pastix->iparm[IPARM_NNZEROS];
  info->nz_unneeded       = 0.0;
  info->assemblies        = 0.0;
  info->mallocs           = 0.0;
  info->memory            = 0.0;
  info->fill_ratio_given  = 0;
  info->fill_ratio_needed = 0;
  info->factor_mallocs    = 0;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatFactorGetSolverType_PaStiX(Mat A, MatSolverType *type)
{
  PetscFunctionBegin;
  *type = MATSOLVERPASTIX;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Sets PaStiX options from the options database */
static PetscErrorCode MatSetFromOptions_PaStiX(Mat A)
{
  Mat_Pastix   *pastix = (Mat_Pastix *)A->data;
  pastix_int_t *iparm  = pastix->iparm;
  double       *dparm  = pastix->dparm;
  PetscInt      icntl;
  PetscReal     dcntl;
  PetscBool     set;

  PetscFunctionBegin;
  PetscOptionsBegin(PetscObjectComm((PetscObject)A), ((PetscObject)A)->prefix, "PaStiX Options", "Mat");

  iparm[IPARM_VERBOSE] = 0;
  iparm[IPARM_ITERMAX] = 20;

  PetscCall(PetscOptionsRangeInt("-mat_pastix_verbose", "iparm[IPARM_VERBOSE] : level of printing (0 to 2)", "None", iparm[IPARM_VERBOSE], &icntl, &set, 0, 2));
  if (set) iparm[IPARM_VERBOSE] = (pastix_int_t)icntl;

  PetscCall(PetscOptionsRangeInt("-mat_pastix_factorization", "iparm[IPARM_FACTORIZATION]: Factorization algorithm", "None", iparm[IPARM_FACTORIZATION], &icntl, &set, 0, 4));
  if (set) iparm[IPARM_FACTORIZATION] = (pastix_int_t)icntl;

  PetscCall(PetscOptionsBoundedInt("-mat_pastix_itermax", "iparm[IPARM_ITERMAX]: Max iterations", "None", iparm[IPARM_ITERMAX], &icntl, &set, 1));
  if (set) iparm[IPARM_ITERMAX] = (pastix_int_t)icntl;

  PetscCall(PetscOptionsBoundedReal("-mat_pastix_epsilon_refinement", "dparm[DPARM_EPSILON_REFINEMENT]: Epsilon refinement", "None", dparm[DPARM_EPSILON_REFINEMENT], &dcntl, &set, -1.));
  if (set) dparm[DPARM_EPSILON_REFINEMENT] = (double)dcntl;

  PetscCall(PetscOptionsReal("-mat_pastix_epsilon_magn_ctrl", "dparm[DPARM_EPSILON_MAGN_CTRL]: Epsilon magnitude control", "None", dparm[DPARM_EPSILON_MAGN_CTRL], &dcntl, &set));
  if (set) dparm[DPARM_EPSILON_MAGN_CTRL] = (double)dcntl;

  PetscCall(PetscOptionsRangeInt("-mat_pastix_ordering", "iparm[IPARM_ORDERING]: Ordering algorithm", "None", iparm[IPARM_ORDERING], &icntl, &set, 0, 2));
  if (set) iparm[IPARM_ORDERING] = (pastix_int_t)icntl;

  PetscCall(PetscOptionsBoundedInt("-mat_pastix_thread_nbr", "iparm[IPARM_THREAD_NBR]: Number of thread by MPI node", "None", iparm[IPARM_THREAD_NBR], &icntl, &set, -1));
  if (set) iparm[IPARM_THREAD_NBR] = (pastix_int_t)icntl;

  PetscCall(PetscOptionsRangeInt("-mat_pastix_scheduler", "iparm[IPARM_SCHEDULER]: Scheduler", "None", iparm[IPARM_SCHEDULER], &icntl, &set, 0, 4));
  if (set) iparm[IPARM_SCHEDULER] = (pastix_int_t)icntl;

  PetscCall(PetscOptionsRangeInt("-mat_pastix_compress_when", "iparm[IPARM_COMPRESS_WHEN]: When to compress", "None", iparm[IPARM_COMPRESS_WHEN], &icntl, &set, 0, 3));
  if (set) iparm[IPARM_COMPRESS_WHEN] = (pastix_int_t)icntl;

  PetscCall(PetscOptionsBoundedReal("-mat_pastix_compress_tolerance", "dparm[DPARM_COMPRESS_TOLERANCE]: Tolerance for low-rank kernels", "None", dparm[DPARM_COMPRESS_TOLERANCE], &dcntl, &set, 0.));
  if (set) dparm[DPARM_COMPRESS_TOLERANCE] = (double)dcntl;

  PetscOptionsEnd();
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatGetFactor_pastix(Mat A, MatFactorType ftype, Mat *F, const char *mattype)
{
  Mat         B;
  Mat_Pastix *pastix;

  PetscFunctionBegin;
  /* Create the factorization matrix */
  PetscCall(MatCreate(PetscObjectComm((PetscObject)A), &B));
  PetscCall(MatSetSizes(B, A->rmap->n, A->cmap->n, A->rmap->N, A->cmap->N));
  PetscCall(PetscStrallocpy(MATSOLVERPASTIX, &((PetscObject)B)->type_name));
  PetscCall(MatSetUp(B));

  PetscCheck(ftype == MAT_FACTOR_LU || ftype == MAT_FACTOR_CHOLESKY, PETSC_COMM_SELF, PETSC_ERR_SUP, "Factor type not supported by PaStiX");

  /* set solvertype */
  PetscCall(PetscFree(B->solvertype));
  PetscCall(PetscStrallocpy(MATSOLVERPASTIX, &B->solvertype));

  B->ops->lufactorsymbolic       = MatLUFactorSymbolic_PaStiX;
  B->ops->lufactornumeric        = MatLUFactorNumeric_PaStiX;
  B->ops->choleskyfactorsymbolic = MatCholeskyFactorSymbolic_PaStiX;
  B->ops->choleskyfactornumeric  = MatCholeskyFactorNumeric_PaStiX;
  B->ops->view                   = MatView_PaStiX;
  B->ops->getinfo                = MatGetInfo_PaStiX;
  B->ops->destroy                = MatDestroy_PaStiX;

  PetscCall(PetscObjectComposeFunction((PetscObject)B, "MatFactorGetSolverType_C", MatFactorGetSolverType_PaStiX));

  B->factortype = ftype;

  /* Create the pastix structure */
  PetscCall(PetscNew(&pastix));
  B->data = (void *)pastix;

  /* Call to set default pastix options */
  PetscStackCallExternalVoid("pastixInitParam", pastixInitParam(pastix->iparm, pastix->dparm));
  PetscCall(MatSetFromOptions_PaStiX(B));

  /* Get PETSc communicator */
  PetscCall(PetscObjectGetComm((PetscObject)A, &pastix->comm));

  /* Initialise PaStiX structure */
  pastix->iparm[IPARM_SCOTCH_MT] = 0;
  PetscStackCallExternalVoid("pastixInit", pastixInit(&pastix->pastix_data, pastix->comm, pastix->iparm, pastix->dparm));

  /* Warning: Cholesky in PETSc wrapper does not handle (complex) Hermitian matrices.
     The factorization type can be forced using the parameter
     mat_pastix_factorization (see enum pastix_factotype_e in
     https://solverstack.gitlabpages.inria.fr/pastix/group__pastix__api.html). */
  pastix->iparm[IPARM_FACTORIZATION] = ftype == MAT_FACTOR_CHOLESKY ? PastixFactSYTRF : PastixFactGETRF;

  *F = B;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatGetFactor_mpiaij_pastix(Mat A, MatFactorType ftype, Mat *F)
{
  PetscFunctionBegin;
  PetscCheck(ftype == MAT_FACTOR_LU, PETSC_COMM_SELF, PETSC_ERR_SUP, "Cannot use PETSc AIJ matrices with PaStiX Cholesky, use SBAIJ matrix");
  PetscCall(MatGetFactor_pastix(A, ftype, F, MATMPIAIJ));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatGetFactor_seqaij_pastix(Mat A, MatFactorType ftype, Mat *F)
{
  PetscFunctionBegin;
  PetscCheck(ftype == MAT_FACTOR_LU, PETSC_COMM_SELF, PETSC_ERR_SUP, "Cannot use PETSc AIJ matrices with PaStiX Cholesky, use SBAIJ matrix");
  PetscCall(MatGetFactor_pastix(A, ftype, F, MATSEQAIJ));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatGetFactor_mpisbaij_pastix(Mat A, MatFactorType ftype, Mat *F)
{
  PetscFunctionBegin;
  PetscCheck(ftype == MAT_FACTOR_CHOLESKY, PETSC_COMM_SELF, PETSC_ERR_SUP, "Cannot use PETSc SBAIJ matrices with PaStiX LU, use AIJ matrix");
  PetscCall(MatGetFactor_pastix(A, ftype, F, MATMPISBAIJ));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatGetFactor_seqsbaij_pastix(Mat A, MatFactorType ftype, Mat *F)
{
  PetscFunctionBegin;
  PetscCheck(ftype == MAT_FACTOR_CHOLESKY, PETSC_COMM_SELF, PETSC_ERR_SUP, "Cannot use PETSc SBAIJ matrices with PaStiX LU, use AIJ matrix");
  PetscCall(MatGetFactor_pastix(A, ftype, F, MATSEQSBAIJ));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode MatSolverTypeRegister_Pastix(void)
{
  PetscFunctionBegin;
  PetscCall(MatSolverTypeRegister(MATSOLVERPASTIX, MATMPIAIJ, MAT_FACTOR_LU, MatGetFactor_mpiaij_pastix));
  PetscCall(MatSolverTypeRegister(MATSOLVERPASTIX, MATSEQAIJ, MAT_FACTOR_LU, MatGetFactor_seqaij_pastix));
  PetscCall(MatSolverTypeRegister(MATSOLVERPASTIX, MATMPISBAIJ, MAT_FACTOR_CHOLESKY, MatGetFactor_mpisbaij_pastix));
  PetscCall(MatSolverTypeRegister(MATSOLVERPASTIX, MATSEQSBAIJ, MAT_FACTOR_CHOLESKY, MatGetFactor_seqsbaij_pastix));
  PetscFunctionReturn(PETSC_SUCCESS);
}
