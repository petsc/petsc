#if defined(PETSC_HAVE_LIBMKL_INTEL_ILP64)
#define MKL_ILP64
#endif

#include <../src/mat/impls/aij/seq/aij.h>    /*I "petscmat.h" I*/
#include <../src/mat/impls/dense/seq/dense.h>

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mkl.h>

/*
 *  Possible mkl_pardiso phases that controls the execution of the solver.
 *  For more information check mkl_pardiso manual.
 */
#define JOB_ANALYSIS 11
#define JOB_ANALYSIS_NUMERICAL_FACTORIZATION 12
#define JOB_ANALYSIS_NUMERICAL_FACTORIZATION_SOLVE_ITERATIVE_REFINEMENT 13
#define JOB_NUMERICAL_FACTORIZATION 22
#define JOB_NUMERICAL_FACTORIZATION_SOLVE_ITERATIVE_REFINEMENT 23
#define JOB_SOLVE_ITERATIVE_REFINEMENT 33
#define JOB_SOLVE_FORWARD_SUBSTITUTION 331
#define JOB_SOLVE_DIAGONAL_SUBSTITUTION 332
#define JOB_SOLVE_BACKWARD_SUBSTITUTION 333
#define JOB_RELEASE_OF_LU_MEMORY 0
#define JOB_RELEASE_OF_ALL_MEMORY -1

#define IPARM_SIZE 64

#if defined(PETSC_USE_64BIT_INDICES)
 #if defined(PETSC_HAVE_LIBMKL_INTEL_ILP64)
  /* sizeof(MKL_INT) == sizeof(long long int) if ilp64*/
  #define INT_TYPE long long int
  #define MKL_PARDISO pardiso
  #define MKL_PARDISO_INIT pardisoinit
 #else
  #define INT_TYPE long long int
  #define MKL_PARDISO pardiso_64
  #define MKL_PARDISO_INIT pardiso_64init
 #endif
#else
 #define INT_TYPE int
 #define MKL_PARDISO pardiso
 #define MKL_PARDISO_INIT pardisoinit
#endif


/*
 *  Internal data structure.
 *  For more information check mkl_pardiso manual.
 */
typedef struct {

  /* Configuration vector*/
  INT_TYPE     iparm[IPARM_SIZE];

  /*
   * Internal mkl_pardiso memory location.
   * After the first call to mkl_pardiso do not modify pt, as that could cause a serious memory leak.
   */
  void         *pt[IPARM_SIZE];

  /* Basic mkl_pardiso info*/
  INT_TYPE     phase, maxfct, mnum, mtype, n, nrhs, msglvl, err;

  /* Matrix structure*/
  void         *a;
  INT_TYPE     *ia, *ja;

  /* Number of non-zero elements*/
  INT_TYPE     nz;

  /* Row permutaton vector*/
  INT_TYPE     *perm;

  /* Define if matrix preserves sparse structure.*/
  MatStructure matstruc;

  /* True if mkl_pardiso function have been used.*/
  PetscBool CleanUp;
} Mat_MKL_PARDISO;


void pardiso_64init(void *pt, INT_TYPE *mtype, INT_TYPE iparm [])
{
  int iparm_copy[IPARM_SIZE], mtype_copy, i;
  
  mtype_copy = *mtype;
  pardisoinit(pt, &mtype_copy, iparm_copy);
  for(i = 0; i < IPARM_SIZE; i++){
    iparm[i] = iparm_copy[i];
  }
}


/*
 * Copy the elements of matrix A.
 * Input:
 *   - Mat A: MATSEQAIJ matrix
 *   - int shift: matrix index.
 *     - 0 for c representation
 *     - 1 for fortran representation
 *   - MatReuse reuse:
 *     - MAT_INITIAL_MATRIX: Create a new aij representation
 *     - MAT_REUSE_MATRIX: Reuse all aij representation and just change values
 * Output:
 *   - int *nnz: Number of nonzero-elements.
 *   - int **r pointer to i index
 *   - int **c pointer to j elements
 *   - MATRIXTYPE **v: Non-zero elements
 */
#undef __FUNCT__
#define __FUNCT__ "MatCopy_MKL_PARDISO"
PetscErrorCode MatCopy_MKL_PARDISO(Mat A, MatReuse reuse, INT_TYPE *nnz, INT_TYPE **r, INT_TYPE **c, void **v)
{
  Mat_SeqAIJ *aa=(Mat_SeqAIJ*)A->data;

  PetscFunctionBegin;
  *v=aa->a;
  if (reuse == MAT_INITIAL_MATRIX) {
    *r   = (INT_TYPE*)aa->i;
    *c   = (INT_TYPE*)aa->j;
    *nnz = aa->nz;
  }
  PetscFunctionReturn(0);
}


/*
 * Free memory for Mat_MKL_PARDISO structure and pointers to objects.
 */
#undef __FUNCT__
#define __FUNCT__ "MatDestroy_MKL_PARDISO"
PetscErrorCode MatDestroy_MKL_PARDISO(Mat A)
{
  Mat_MKL_PARDISO *mat_mkl_pardiso=(Mat_MKL_PARDISO*)A->spptr;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  /* Terminate instance, deallocate memories */
  if (mat_mkl_pardiso->CleanUp) {
    mat_mkl_pardiso->phase = JOB_RELEASE_OF_ALL_MEMORY;

    MKL_PARDISO (mat_mkl_pardiso->pt,
      &mat_mkl_pardiso->maxfct,
      &mat_mkl_pardiso->mnum,
      &mat_mkl_pardiso->mtype,
      &mat_mkl_pardiso->phase,
      &mat_mkl_pardiso->n,
      NULL,
      NULL,
      NULL,
      mat_mkl_pardiso->perm,
      &mat_mkl_pardiso->nrhs,
      mat_mkl_pardiso->iparm,
      &mat_mkl_pardiso->msglvl,
      NULL,
      NULL,
      &mat_mkl_pardiso->err);
  }
  ierr = PetscFree(mat_mkl_pardiso->perm);CHKERRQ(ierr);
  ierr = PetscFree(A->spptr);CHKERRQ(ierr);

  /* clear composed functions */
  ierr = PetscObjectComposeFunction((PetscObject)A,"MatFactorGetSolverPackage_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)A,"MatMkl_PardisoSetCntl_C",NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
 * Computes Ax = b
 */
#undef __FUNCT__
#define __FUNCT__ "MatSolve_MKL_PARDISO"
PetscErrorCode MatSolve_MKL_PARDISO(Mat A,Vec b,Vec x)
{
  Mat_MKL_PARDISO   *mat_mkl_pardiso=(Mat_MKL_PARDISO*)(A)->spptr;
  PetscErrorCode    ierr;
  PetscScalar       *xarray;
  const PetscScalar *barray;

  PetscFunctionBegin;
  mat_mkl_pardiso->nrhs = 1;
  ierr = VecGetArray(x,&xarray);CHKERRQ(ierr);
  ierr = VecGetArrayRead(b,&barray);CHKERRQ(ierr);

  /* solve phase */
  /*-------------*/
  mat_mkl_pardiso->phase = JOB_SOLVE_ITERATIVE_REFINEMENT;
  MKL_PARDISO (mat_mkl_pardiso->pt,
    &mat_mkl_pardiso->maxfct,
    &mat_mkl_pardiso->mnum,
    &mat_mkl_pardiso->mtype,
    &mat_mkl_pardiso->phase,
    &mat_mkl_pardiso->n,
    mat_mkl_pardiso->a,
    mat_mkl_pardiso->ia,
    mat_mkl_pardiso->ja,
    mat_mkl_pardiso->perm,
    &mat_mkl_pardiso->nrhs,
    mat_mkl_pardiso->iparm,
    &mat_mkl_pardiso->msglvl,
    (void*)barray,
    (void*)xarray,
    &mat_mkl_pardiso->err);

  if (mat_mkl_pardiso->err < 0) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Error reported by MKL_PARDISO: err=%d. Please check manual\n",mat_mkl_pardiso->err);
  ierr = VecRestoreArray(x,&xarray);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(b,&barray);CHKERRQ(ierr);
  mat_mkl_pardiso->CleanUp = PETSC_TRUE;
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "MatSolveTranspose_MKL_PARDISO"
PetscErrorCode MatSolveTranspose_MKL_PARDISO(Mat A,Vec b,Vec x)
{
  Mat_MKL_PARDISO *mat_mkl_pardiso=(Mat_MKL_PARDISO*)A->spptr;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
#if defined(PETSC_USE_COMPLEX)
  mat_mkl_pardiso->iparm[12 - 1] = 1;
#else
  mat_mkl_pardiso->iparm[12 - 1] = 2;
#endif
  ierr = MatSolve_MKL_PARDISO(A,b,x);CHKERRQ(ierr);
  mat_mkl_pardiso->iparm[12 - 1] = 0;
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "MatMatSolve_MKL_PARDISO"
PetscErrorCode MatMatSolve_MKL_PARDISO(Mat A,Mat B,Mat X)
{
  Mat_MKL_PARDISO   *mat_mkl_pardiso=(Mat_MKL_PARDISO*)(A)->spptr;
  PetscErrorCode    ierr;
  PetscScalar       *barray, *xarray;
  PetscBool         flg;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject)B,MATSEQDENSE,&flg);CHKERRQ(ierr);
  if (!flg) SETERRQ(PetscObjectComm((PetscObject)A),PETSC_ERR_ARG_WRONG,"Matrix B must be MATSEQDENSE matrix");
  ierr = PetscObjectTypeCompare((PetscObject)X,MATSEQDENSE,&flg);CHKERRQ(ierr);
  if (!flg) SETERRQ(PetscObjectComm((PetscObject)A),PETSC_ERR_ARG_WRONG,"Matrix X must be MATSEQDENSE matrix");

  ierr = MatGetSize(B,NULL,(PetscInt*)&mat_mkl_pardiso->nrhs);CHKERRQ(ierr);

  if(mat_mkl_pardiso->nrhs > 0){
    ierr = MatDenseGetArray(B,&barray);
    ierr = MatDenseGetArray(X,&xarray);

    /* solve phase */
    /*-------------*/
    mat_mkl_pardiso->phase = JOB_SOLVE_ITERATIVE_REFINEMENT;
    MKL_PARDISO (mat_mkl_pardiso->pt,
      &mat_mkl_pardiso->maxfct,
      &mat_mkl_pardiso->mnum,
      &mat_mkl_pardiso->mtype,
      &mat_mkl_pardiso->phase,
      &mat_mkl_pardiso->n,
      mat_mkl_pardiso->a,
      mat_mkl_pardiso->ia,
      mat_mkl_pardiso->ja,
      mat_mkl_pardiso->perm,
      &mat_mkl_pardiso->nrhs,
      mat_mkl_pardiso->iparm,
      &mat_mkl_pardiso->msglvl,
      (void*)barray,
      (void*)xarray,
      &mat_mkl_pardiso->err);
    if (mat_mkl_pardiso->err < 0) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Error reported by MKL_PARDISO: err=%d. Please check manual\n",mat_mkl_pardiso->err);
  }
  mat_mkl_pardiso->CleanUp = PETSC_TRUE;
  PetscFunctionReturn(0);
}

/*
 * LU Decomposition
 */
#undef __FUNCT__
#define __FUNCT__ "MatFactorNumeric_MKL_PARDISO"
PetscErrorCode MatFactorNumeric_MKL_PARDISO(Mat F,Mat A,const MatFactorInfo *info)
{
  Mat_MKL_PARDISO *mat_mkl_pardiso=(Mat_MKL_PARDISO*)(F)->spptr;
  PetscErrorCode  ierr;

  /* numerical factorization phase */
  /*-------------------------------*/
  PetscFunctionBegin;
  mat_mkl_pardiso->matstruc = SAME_NONZERO_PATTERN;
  ierr = MatCopy_MKL_PARDISO(A, MAT_REUSE_MATRIX, &mat_mkl_pardiso->nz, &mat_mkl_pardiso->ia, &mat_mkl_pardiso->ja, &mat_mkl_pardiso->a);CHKERRQ(ierr);

  /* numerical factorization phase */
  /*-------------------------------*/
  mat_mkl_pardiso->phase = JOB_NUMERICAL_FACTORIZATION;
  MKL_PARDISO (mat_mkl_pardiso->pt,
    &mat_mkl_pardiso->maxfct,
    &mat_mkl_pardiso->mnum,
    &mat_mkl_pardiso->mtype,
    &mat_mkl_pardiso->phase,
    &mat_mkl_pardiso->n,
    mat_mkl_pardiso->a,
    mat_mkl_pardiso->ia,
    mat_mkl_pardiso->ja,
    mat_mkl_pardiso->perm,
    &mat_mkl_pardiso->nrhs,
    mat_mkl_pardiso->iparm,
    &mat_mkl_pardiso->msglvl,
    NULL,
    NULL,
    &mat_mkl_pardiso->err);
  if (mat_mkl_pardiso->err < 0) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Error reported by MKL_PARDISO: err=%d. Please check manual\n",mat_mkl_pardiso->err);

  mat_mkl_pardiso->matstruc = SAME_NONZERO_PATTERN;
  mat_mkl_pardiso->CleanUp  = PETSC_TRUE;
  PetscFunctionReturn(0);
}

/* Sets mkl_pardiso options from the options database */
#undef __FUNCT__
#define __FUNCT__ "PetscSetMKL_PARDISOFromOptions"
PetscErrorCode PetscSetMKL_PARDISOFromOptions(Mat F, Mat A)
{
  Mat_MKL_PARDISO     *mat_mkl_pardiso = (Mat_MKL_PARDISO*)F->spptr;
  PetscErrorCode      ierr;
  PetscInt            icntl;
  PetscBool           flg;
  int                 pt[IPARM_SIZE], threads = 1;

  PetscFunctionBegin;
  ierr = PetscOptionsBegin(PetscObjectComm((PetscObject)A),((PetscObject)A)->prefix,"MKL_PARDISO Options","Mat");CHKERRQ(ierr);
  ierr = PetscOptionsInt("-mat_mkl_pardiso_65","Number of threads to use","None",threads,&threads,&flg);CHKERRQ(ierr);
  if (flg) mkl_set_num_threads(threads);

  ierr = PetscOptionsInt("-mat_mkl_pardiso_66","Maximum number of factors with identical sparsity structure that must be kept in memory at the same time","None",mat_mkl_pardiso->maxfct,&icntl,&flg);CHKERRQ(ierr);
  if (flg) mat_mkl_pardiso->maxfct = icntl;

  ierr = PetscOptionsInt("-mat_mkl_pardiso_67","Indicates the actual matrix for the solution phase","None",mat_mkl_pardiso->mnum,&icntl,&flg);CHKERRQ(ierr);
  if (flg) mat_mkl_pardiso->mnum = icntl;
 
  ierr = PetscOptionsInt("-mat_mkl_pardiso_68","Message level information","None",mat_mkl_pardiso->msglvl,&icntl,&flg);CHKERRQ(ierr);
  if (flg) mat_mkl_pardiso->msglvl = icntl;

  ierr = PetscOptionsInt("-mat_mkl_pardiso_69","Defines the matrix type","None",mat_mkl_pardiso->mtype,&icntl,&flg);CHKERRQ(ierr);
  if(flg){
   mat_mkl_pardiso->mtype = icntl;
   MKL_PARDISO_INIT(&pt, &mat_mkl_pardiso->mtype, mat_mkl_pardiso->iparm);
#if defined(PETSC_USE_REAL_SINGLE)
    mat_mkl_pardiso->iparm[27] = 1;
#else
    mat_mkl_pardiso->iparm[27] = 0;
#endif
    mat_mkl_pardiso->iparm[34] = 1;
  }
  ierr = PetscOptionsInt("-mat_mkl_pardiso_1","Use default values","None",mat_mkl_pardiso->iparm[0],&icntl,&flg);CHKERRQ(ierr);

  if(flg && icntl != 0){
    ierr = PetscOptionsInt("-mat_mkl_pardiso_2","Fill-in reducing ordering for the input matrix","None",mat_mkl_pardiso->iparm[1],&icntl,&flg);CHKERRQ(ierr);
    if (flg) mat_mkl_pardiso->iparm[1] = icntl;

    ierr = PetscOptionsInt("-mat_mkl_pardiso_4","Preconditioned CGS/CG","None",mat_mkl_pardiso->iparm[3],&icntl,&flg);CHKERRQ(ierr);
    if (flg) mat_mkl_pardiso->iparm[3] = icntl;

    ierr = PetscOptionsInt("-mat_mkl_pardiso_5","User permutation","None",mat_mkl_pardiso->iparm[4],&icntl,&flg);CHKERRQ(ierr);
    if (flg) mat_mkl_pardiso->iparm[4] = icntl;

    ierr = PetscOptionsInt("-mat_mkl_pardiso_6","Write solution on x","None",mat_mkl_pardiso->iparm[5],&icntl,&flg);CHKERRQ(ierr);
    if (flg) mat_mkl_pardiso->iparm[5] = icntl;

    ierr = PetscOptionsInt("-mat_mkl_pardiso_8","Iterative refinement step","None",mat_mkl_pardiso->iparm[7],&icntl,&flg);CHKERRQ(ierr);
    if (flg) mat_mkl_pardiso->iparm[7] = icntl;

    ierr = PetscOptionsInt("-mat_mkl_pardiso_10","Pivoting perturbation","None",mat_mkl_pardiso->iparm[9],&icntl,&flg);CHKERRQ(ierr);
    if (flg) mat_mkl_pardiso->iparm[9] = icntl;

    ierr = PetscOptionsInt("-mat_mkl_pardiso_11","Scaling vectors","None",mat_mkl_pardiso->iparm[10],&icntl,&flg);CHKERRQ(ierr);
    if (flg) mat_mkl_pardiso->iparm[10] = icntl;

    ierr = PetscOptionsInt("-mat_mkl_pardiso_12","Solve with transposed or conjugate transposed matrix A","None",mat_mkl_pardiso->iparm[11],&icntl,&flg);CHKERRQ(ierr);
    if (flg) mat_mkl_pardiso->iparm[11] = icntl;

    ierr = PetscOptionsInt("-mat_mkl_pardiso_13","Improved accuracy using (non-) symmetric weighted matching","None",mat_mkl_pardiso->iparm[12],&icntl,&flg);CHKERRQ(ierr);
    if (flg) mat_mkl_pardiso->iparm[12] = icntl;

    ierr = PetscOptionsInt("-mat_mkl_pardiso_18","Numbers of non-zero elements","None",mat_mkl_pardiso->iparm[17],&icntl,&flg);CHKERRQ(ierr);
    if (flg) mat_mkl_pardiso->iparm[17] = icntl;

    ierr = PetscOptionsInt("-mat_mkl_pardiso_19","Report number of floating point operations","None",mat_mkl_pardiso->iparm[18],&icntl,&flg);CHKERRQ(ierr);
    if (flg) mat_mkl_pardiso->iparm[18] = icntl;

    ierr = PetscOptionsInt("-mat_mkl_pardiso_21","Pivoting for symmetric indefinite matrices","None",mat_mkl_pardiso->iparm[20],&icntl,&flg);CHKERRQ(ierr);
    if (flg) mat_mkl_pardiso->iparm[20] = icntl;

    ierr = PetscOptionsInt("-mat_mkl_pardiso_24","Parallel factorization control","None",mat_mkl_pardiso->iparm[23],&icntl,&flg);CHKERRQ(ierr);
    if (flg) mat_mkl_pardiso->iparm[23] = icntl;

    ierr = PetscOptionsInt("-mat_mkl_pardiso_25","Parallel forward/backward solve control","None",mat_mkl_pardiso->iparm[24],&icntl,&flg);CHKERRQ(ierr);
    if (flg) mat_mkl_pardiso->iparm[24] = icntl;

    ierr = PetscOptionsInt("-mat_mkl_pardiso_27","Matrix checker","None",mat_mkl_pardiso->iparm[26],&icntl,&flg);CHKERRQ(ierr);
    if (flg) mat_mkl_pardiso->iparm[26] = icntl;

    ierr = PetscOptionsInt("-mat_mkl_pardiso_31","Partial solve and computing selected components of the solution vectors","None",mat_mkl_pardiso->iparm[30],&icntl,&flg);CHKERRQ(ierr);
    if (flg) mat_mkl_pardiso->iparm[30] = icntl;

    ierr = PetscOptionsInt("-mat_mkl_pardiso_34","Optimal number of threads for conditional numerical reproducibility (CNR) mode","None",mat_mkl_pardiso->iparm[33],&icntl,&flg);CHKERRQ(ierr);
    if (flg) mat_mkl_pardiso->iparm[33] = icntl;

    ierr = PetscOptionsInt("-mat_mkl_pardiso_60","Intel MKL_PARDISO mode","None",mat_mkl_pardiso->iparm[59],&icntl,&flg);CHKERRQ(ierr);
    if (flg) mat_mkl_pardiso->iparm[59] = icntl;
  }
  PetscOptionsEnd();
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatFactorMKL_PARDISOInitialize_Private"
PetscErrorCode MatFactorMKL_PARDISOInitialize_Private(Mat A, MatFactorType ftype, Mat_MKL_PARDISO *mat_mkl_pardiso)
{
  PetscErrorCode ierr;
  PetscInt       i;

  PetscFunctionBegin;
  for ( i = 0; i < IPARM_SIZE; i++ ){
    mat_mkl_pardiso->iparm[i] = 0;
  }

  for ( i = 0; i < IPARM_SIZE; i++ ){
    mat_mkl_pardiso->pt[i] = 0;
  }
  
  /*Default options for both sym and unsym */
  mat_mkl_pardiso->iparm[ 0] =  1; /* Solver default parameters overriden with provided by iparm */
  mat_mkl_pardiso->iparm[ 1] =  2; /* Metis reordering */
  mat_mkl_pardiso->iparm[ 5] =  0; /* Write solution into x */
  mat_mkl_pardiso->iparm[ 7] =  2; /* Max number of iterative refinement steps */
  mat_mkl_pardiso->iparm[17] = -1; /* Output: Number of nonzeros in the factor LU */
  mat_mkl_pardiso->iparm[18] = -1; /* Output: Mflops for LU factorization */
#if 0
  mat_mkl_pardiso->iparm[23] =  1; /* Parallel factorization control*/
#endif
  mat_mkl_pardiso->iparm[34] =  1; /* Cluster Sparse Solver use C-style indexing for ia and ja arrays */
  mat_mkl_pardiso->iparm[39] =  0; /* Input: matrix/rhs/solution stored on master */
  
  mat_mkl_pardiso->CleanUp   = PETSC_FALSE;
  mat_mkl_pardiso->maxfct    = 1; /* Maximum number of numerical factorizations. */
  mat_mkl_pardiso->mnum      = 1; /* Which factorization to use. */
  mat_mkl_pardiso->msglvl    = 0; /* 0: do not print 1: Print statistical information in file */
  mat_mkl_pardiso->phase     = -1;
  mat_mkl_pardiso->err       = 0;
  
  mat_mkl_pardiso->n         = A->rmap->N;
  mat_mkl_pardiso->nrhs      = 1;
  mat_mkl_pardiso->err       = 0;
  mat_mkl_pardiso->phase     = -1;
  
  if(ftype == MAT_FACTOR_LU){
    /*Default type for non-sym*/
#if defined(PETSC_USE_COMPLEX)
    mat_mkl_pardiso->mtype     = 13;
#else
    mat_mkl_pardiso->mtype     = 11;
#endif

    mat_mkl_pardiso->iparm[ 9] = 13; /* Perturb the pivot elements with 1E-13 */
    mat_mkl_pardiso->iparm[10] =  1; /* Use nonsymmetric permutation and scaling MPS */
    mat_mkl_pardiso->iparm[12] =  1; /* Switch on Maximum Weighted Matching algorithm (default for non-symmetric) */

  } else {
    /*Default type for sym*/
#if defined(PETSC_USE_COMPLEX)
    mat_mkl_pardiso ->mtype    = 3;
#else
    mat_mkl_pardiso ->mtype    = -2;
#endif
    mat_mkl_pardiso->iparm[ 9] = 13; /* Perturb the pivot elements with 1E-13 */
    mat_mkl_pardiso->iparm[10] = 0; /* Use nonsymmetric permutation and scaling MPS */
    mat_mkl_pardiso->iparm[12] = 1; /* Switch on Maximum Weighted Matching algorithm (default for non-symmetric) */
/*    mat_mkl_pardiso->iparm[20] =  1; */ /* Apply 1x1 and 2x2 Bunch-Kaufman pivoting during the factorization process */
#if defined(PETSC_USE_DEBUG)
    mat_mkl_pardiso->iparm[26] = 1; /* Matrix checker */
#endif
  }
  ierr = PetscMalloc1(A->rmap->N*sizeof(INT_TYPE), &mat_mkl_pardiso->perm);CHKERRQ(ierr);
  for(i = 0; i < A->rmap->N; i++){
    mat_mkl_pardiso->perm[i] = 0;
  }
  PetscFunctionReturn(0);
}

/*
 * Symbolic decomposition. Mkl_Pardiso analysis phase.
 */
#undef __FUNCT__
#define __FUNCT__ "MatFactorSymbolic_AIJMKL_PARDISO_Private"
PetscErrorCode MatFactorSymbolic_AIJMKL_PARDISO_Private(Mat F,Mat A,const MatFactorInfo *info)
{
  Mat_MKL_PARDISO *mat_mkl_pardiso = (Mat_MKL_PARDISO*)F->spptr;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  mat_mkl_pardiso->matstruc = DIFFERENT_NONZERO_PATTERN;

  /* Set MKL_PARDISO options from the options database */
  ierr = PetscSetMKL_PARDISOFromOptions(F,A);CHKERRQ(ierr);

  ierr = MatCopy_MKL_PARDISO(A, MAT_INITIAL_MATRIX, &mat_mkl_pardiso->nz, &mat_mkl_pardiso->ia, &mat_mkl_pardiso->ja, &mat_mkl_pardiso->a);CHKERRQ(ierr);
  mat_mkl_pardiso->n = A->rmap->N;

  /* analysis phase */
  /*----------------*/
  mat_mkl_pardiso->phase = JOB_ANALYSIS;

  MKL_PARDISO (mat_mkl_pardiso->pt,
    &mat_mkl_pardiso->maxfct,
    &mat_mkl_pardiso->mnum,
    &mat_mkl_pardiso->mtype,
    &mat_mkl_pardiso->phase,
    &mat_mkl_pardiso->n,
    mat_mkl_pardiso->a,
    mat_mkl_pardiso->ia,
    mat_mkl_pardiso->ja,
    mat_mkl_pardiso->perm,
    &mat_mkl_pardiso->nrhs,
    mat_mkl_pardiso->iparm,
    &mat_mkl_pardiso->msglvl,
    NULL,
    NULL,
    &mat_mkl_pardiso->err);
  if (mat_mkl_pardiso->err < 0) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Error reported by MKL_PARDISO: err=%d\n. Please check manual",mat_mkl_pardiso->err);

  mat_mkl_pardiso->CleanUp = PETSC_TRUE;

  if(F->factortype == MAT_FACTOR_LU){
    F->ops->lufactornumeric = MatFactorNumeric_MKL_PARDISO;
  } else {
    F->ops->choleskyfactornumeric = MatFactorNumeric_MKL_PARDISO;
  }
  F->ops->solve           = MatSolve_MKL_PARDISO;
  F->ops->solvetranspose  = MatSolveTranspose_MKL_PARDISO;
  F->ops->matsolve        = MatMatSolve_MKL_PARDISO;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatLUFactorSymbolic_AIJMKL_PARDISO"
PetscErrorCode MatLUFactorSymbolic_AIJMKL_PARDISO(Mat F,Mat A,IS r,IS c,const MatFactorInfo *info)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatFactorSymbolic_AIJMKL_PARDISO_Private(F, A, info);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatCholeskyFactorSymbolic_AIJMKL_PARDISO"
PetscErrorCode MatCholeskyFactorSymbolic_AIJMKL_PARDISO(Mat F,Mat A,IS r,const MatFactorInfo *info)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatFactorSymbolic_AIJMKL_PARDISO_Private(F, A, info);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatView_MKL_PARDISO"
PetscErrorCode MatView_MKL_PARDISO(Mat A, PetscViewer viewer)
{
  PetscErrorCode    ierr;
  PetscBool         iascii;
  PetscViewerFormat format;
  Mat_MKL_PARDISO   *mat_mkl_pardiso=(Mat_MKL_PARDISO*)A->spptr;
  PetscInt          i;

  PetscFunctionBegin;
  /* check if matrix is mkl_pardiso type */
  if (A->ops->solve != MatSolve_MKL_PARDISO) PetscFunctionReturn(0);

  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii);CHKERRQ(ierr);
  if (iascii) {
    ierr = PetscViewerGetFormat(viewer,&format);CHKERRQ(ierr);
    if (format == PETSC_VIEWER_ASCII_INFO) {
      ierr = PetscViewerASCIIPrintf(viewer,"MKL_PARDISO run parameters:\n");CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer,"MKL_PARDISO phase:             %d \n",mat_mkl_pardiso->phase);CHKERRQ(ierr);
      for(i = 1; i <= 64; i++){
        ierr = PetscViewerASCIIPrintf(viewer,"MKL_PARDISO iparm[%d]:     %d \n",i, mat_mkl_pardiso->iparm[i - 1]);CHKERRQ(ierr);
      }
      ierr = PetscViewerASCIIPrintf(viewer,"MKL_PARDISO maxfct:     %d \n", mat_mkl_pardiso->maxfct);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer,"MKL_PARDISO mnum:     %d \n", mat_mkl_pardiso->mnum);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer,"MKL_PARDISO mtype:     %d \n", mat_mkl_pardiso->mtype);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer,"MKL_PARDISO n:     %d \n", mat_mkl_pardiso->n);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer,"MKL_PARDISO nrhs:     %d \n", mat_mkl_pardiso->nrhs);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer,"MKL_PARDISO msglvl:     %d \n", mat_mkl_pardiso->msglvl);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "MatGetInfo_MKL_PARDISO"
PetscErrorCode MatGetInfo_MKL_PARDISO(Mat A, MatInfoType flag, MatInfo *info)
{
  Mat_MKL_PARDISO *mat_mkl_pardiso =(Mat_MKL_PARDISO*)A->spptr;

  PetscFunctionBegin;
  info->block_size        = 1.0;
  info->nz_allocated      = mat_mkl_pardiso->nz + 0.0;
  info->nz_unneeded       = 0.0;
  info->assemblies        = 0.0;
  info->mallocs           = 0.0;
  info->memory            = 0.0;
  info->fill_ratio_given  = 0;
  info->fill_ratio_needed = 0;
  info->factor_mallocs    = 0;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatMkl_PardisoSetCntl_MKL_PARDISO"
PetscErrorCode MatMkl_PardisoSetCntl_MKL_PARDISO(Mat F,PetscInt icntl,PetscInt ival)
{
  Mat_MKL_PARDISO *mat_mkl_pardiso =(Mat_MKL_PARDISO*)F->spptr;

  PetscFunctionBegin;
  if(icntl <= 64){
    mat_mkl_pardiso->iparm[icntl - 1] = ival;
  } else {
    if(icntl == 65)
      mkl_set_num_threads((int)ival);
    else if(icntl == 66)
      mat_mkl_pardiso->maxfct = ival;
    else if(icntl == 67)
      mat_mkl_pardiso->mnum = ival;
    else if(icntl == 68)
      mat_mkl_pardiso->msglvl = ival;
    else if(icntl == 69){
      int pt[IPARM_SIZE];
      mat_mkl_pardiso->mtype = ival;
      MKL_PARDISO_INIT(&pt, &mat_mkl_pardiso->mtype, mat_mkl_pardiso->iparm);
#if defined(PETSC_USE_REAL_SINGLE)
      mat_mkl_pardiso->iparm[27] = 1;
#else
      mat_mkl_pardiso->iparm[27] = 0;
#endif
      mat_mkl_pardiso->iparm[34] = 1;
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatMkl_PardisoSetCntl"
/*@
  MatMkl_PardisoSetCntl - Set Mkl_Pardiso parameters

   Logically Collective on Mat

   Input Parameters:
+  F - the factored matrix obtained by calling MatGetFactor()
.  icntl - index of Mkl_Pardiso parameter
-  ival - value of Mkl_Pardiso parameter

  Options Database:
.   -mat_mkl_pardiso_<icntl> <ival>

   Level: beginner

   References: Mkl_Pardiso Users' Guide

.seealso: MatGetFactor()
@*/
PetscErrorCode MatMkl_PardisoSetCntl(Mat F,PetscInt icntl,PetscInt ival)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscTryMethod(F,"MatMkl_PardisoSetCntl_C",(Mat,PetscInt,PetscInt),(F,icntl,ival));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


/*MC
  MATSOLVERMKL_PARDISO -  A matrix type providing direct solvers (LU) for
  sequential matrices via the external package MKL_PARDISO.

  Works with MATSEQAIJ matrices

  Options Database Keys:
+ -mat_mkl_pardiso_65 - Number of thrads to use
. -mat_mkl_pardiso_66 - Maximum number of factors with identical sparsity structure that must be kept in memory at the same time
. -mat_mkl_pardiso_67 - Indicates the actual matrix for the solution phase
. -mat_mkl_pardiso_68 - Message level information
. -mat_mkl_pardiso_69 - Defines the matrix type. IMPORTANT: When you set this flag, iparm parameters are going to be set to the default ones for the matrix type
. -mat_mkl_pardiso_1 - Use default values
. -mat_mkl_pardiso_2 - Fill-in reducing ordering for the input matrix
. -mat_mkl_pardiso_4 - Preconditioned CGS/CG
. -mat_mkl_pardiso_5 - User permutation
. -mat_mkl_pardiso_6 - Write solution on x
. -mat_mkl_pardiso_8 - Iterative refinement step
. -mat_mkl_pardiso_10 - Pivoting perturbation
. -mat_mkl_pardiso_11 - Scaling vectors
. -mat_mkl_pardiso_12 - Solve with transposed or conjugate transposed matrix A
. -mat_mkl_pardiso_13 - Improved accuracy using (non-) symmetric weighted matching
. -mat_mkl_pardiso_18 - Numbers of non-zero elements
. -mat_mkl_pardiso_19 - Report number of floating point operations
. -mat_mkl_pardiso_21 - Pivoting for symmetric indefinite matrices
. -mat_mkl_pardiso_24 - Parallel factorization control
. -mat_mkl_pardiso_25 - Parallel forward/backward solve control
. -mat_mkl_pardiso_27 - Matrix checker
. -mat_mkl_pardiso_31 - Partial solve and computing selected components of the solution vectors
. -mat_mkl_pardiso_34 - Optimal number of threads for conditional numerical reproducibility (CNR) mode
- -mat_mkl_pardiso_60 - Intel MKL_PARDISO mode

  Level: beginner

  For more information please check  mkl_pardiso manual

.seealso: PCFactorSetMatSolverPackage(), MatSolverPackage

M*/
#undef __FUNCT__
#define __FUNCT__ "MatFactorGetSolverPackage_mkl_pardiso"
static PetscErrorCode MatFactorGetSolverPackage_mkl_pardiso(Mat A, const MatSolverPackage *type)
{
  PetscFunctionBegin;
  *type = MATSOLVERMKL_PARDISO;
  PetscFunctionReturn(0);
}

/* MatGetFactor for Seq sbAIJ matrices */
#undef __FUNCT__
#define __FUNCT__ "MatGetFactor_sbaij_mkl_pardiso"
PETSC_EXTERN PetscErrorCode MatGetFactor_sbaij_mkl_pardiso(Mat A,MatFactorType ftype,Mat *F)
{
  Mat             B;
  PetscErrorCode  ierr;
  Mat_MKL_PARDISO *mat_mkl_pardiso;
  PetscBool       isSeqSBAIJ;
  PetscInt        bs;

  PetscFunctionBegin;
  /* Create the factorization matrix */
  ierr = PetscObjectTypeCompare((PetscObject)A,MATSEQSBAIJ,&isSeqSBAIJ);CHKERRQ(ierr);
  ierr = MatCreate(PetscObjectComm((PetscObject)A),&B);CHKERRQ(ierr);
  ierr = MatSetSizes(B,A->rmap->n,A->cmap->n,A->rmap->N,A->cmap->N);CHKERRQ(ierr);
  ierr = MatSetType(B,((PetscObject)A)->type_name);CHKERRQ(ierr);
  if (isSeqSBAIJ) {
    ierr = MatSeqSBAIJSetPreallocation(B,1,0,NULL);CHKERRQ(ierr);
  } else SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Is not allowed other types of matrices apart from MATSEQSBAIJ.");

  ierr = MatGetBlockSize(A,&bs); CHKERRQ(ierr);

  if(bs != 1) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Matrice MATSEQSBAIJ with block size other than 1 is not supported by Pardiso");

  if(ftype != MAT_FACTOR_CHOLESKY) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Matrice MATSEQAIJ should be used only with MAT_FACTOR_CHOLESKY.");
  
  B->ops->choleskyfactorsymbolic = MatCholeskyFactorSymbolic_AIJMKL_PARDISO;
  B->factortype                  = MAT_FACTOR_CHOLESKY;
  B->ops->destroy                = MatDestroy_MKL_PARDISO;
  B->ops->view                   = MatView_MKL_PARDISO;
  B->factortype                  = ftype;
  B->ops->getinfo                = MatGetInfo_MKL_PARDISO;
  B->assembled                   = PETSC_TRUE;           /* required by -ksp_view */

  ierr = PetscNewLog(B,&mat_mkl_pardiso);CHKERRQ(ierr);
  B->spptr = mat_mkl_pardiso;
  ierr = PetscObjectComposeFunction((PetscObject)B,"MatFactorGetSolverPackage_C",MatFactorGetSolverPackage_mkl_pardiso);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)B,"MatMkl_PardisoSetCntl_C",MatMkl_PardisoSetCntl_MKL_PARDISO);CHKERRQ(ierr);
  ierr = MatFactorMKL_PARDISOInitialize_Private(A, ftype, mat_mkl_pardiso);CHKERRQ(ierr);
  *F = B;
  PetscFunctionReturn(0);
}

/* MatGetFactor for Seq AIJ matrices */
#undef __FUNCT__
#define __FUNCT__ "MatGetFactor_aij_mkl_pardiso"
PETSC_EXTERN PetscErrorCode MatGetFactor_aij_mkl_pardiso(Mat A,MatFactorType ftype,Mat *F)
{
  Mat             B;
  PetscErrorCode  ierr;
  Mat_MKL_PARDISO *mat_mkl_pardiso;
  PetscBool       isSeqAIJ;

  PetscFunctionBegin;
  /* Create the factorization matrix */
  ierr = PetscObjectTypeCompare((PetscObject)A,MATSEQAIJ,&isSeqAIJ);CHKERRQ(ierr);
  ierr = MatCreate(PetscObjectComm((PetscObject)A),&B);CHKERRQ(ierr);
  ierr = MatSetSizes(B,A->rmap->n,A->cmap->n,A->rmap->N,A->cmap->N);CHKERRQ(ierr);
  ierr = MatSetType(B,((PetscObject)A)->type_name);CHKERRQ(ierr);
  if (isSeqAIJ) {
    ierr = MatSeqAIJSetPreallocation(B,0,NULL);CHKERRQ(ierr);
  } else SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Is not allowed other types of matrices apart from MATSEQAIJ.");

  if(ftype != MAT_FACTOR_LU) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Matrice MATSEQAIJ should be used only with MAT_FACTOR_LU.");

  B->ops->lufactorsymbolic = MatLUFactorSymbolic_AIJMKL_PARDISO;
  B->factortype            = MAT_FACTOR_LU;
  B->ops->destroy          = MatDestroy_MKL_PARDISO;
  B->ops->view             = MatView_MKL_PARDISO;
  B->factortype            = ftype;
  B->ops->getinfo          = MatGetInfo_MKL_PARDISO;
  B->assembled             = PETSC_TRUE;           /* required by -ksp_view */

  ierr = PetscNewLog(B,&mat_mkl_pardiso);CHKERRQ(ierr);
  B->spptr = mat_mkl_pardiso;
  ierr = PetscObjectComposeFunction((PetscObject)B,"MatFactorGetSolverPackage_C",MatFactorGetSolverPackage_mkl_pardiso);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)B,"MatMkl_PardisoSetCntl_C",MatMkl_PardisoSetCntl_MKL_PARDISO);CHKERRQ(ierr);
  ierr = MatFactorMKL_PARDISOInitialize_Private(A, ftype, mat_mkl_pardiso);CHKERRQ(ierr);

  *F = B;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatSolverPackageRegister_MKL_Pardiso"
PETSC_EXTERN PetscErrorCode MatSolverPackageRegister_MKL_Pardiso(void)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatSolverPackageRegister(MATSOLVERMKL_PARDISO,MATSEQAIJ,   MAT_FACTOR_LU,      MatGetFactor_aij_mkl_pardiso  );CHKERRQ(ierr);
  ierr = MatSolverPackageRegister(MATSOLVERMKL_PARDISO,MATSEQSBAIJ, MAT_FACTOR_CHOLESKY,MatGetFactor_sbaij_mkl_pardiso);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

