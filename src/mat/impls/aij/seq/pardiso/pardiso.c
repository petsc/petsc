#include <../src/mat/impls/aij/seq/aij.h>    /*I "petscmat.h" I*/
#include <../src/mat/impls/dense/seq/dense.h>

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mkl.h>

/*
 *  Possible pardiso phases that controls the execution of the solver.
 *  For more information check pardiso manual.
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
#define INT_TYPE long long int
#define MKL_PARDISO pardiso_64
#define PARDISO_INIT pardiso_64init
#else
#define INT_TYPE int
#define MKL_PARDISO pardiso
#define PARDISO_INIT pardisoinit
#endif


/*
 *  Internal data structure.
 *  For more information check pardiso manual.
 */
typedef struct {

  /*Configuration vector*/
  INT_TYPE     iparm[IPARM_SIZE];

  /*
   * Internal pardiso memory location.
   * After the first call to pardiso do not modify pt, as that could cause a serious memory leak.
   */
  void         *pt[IPARM_SIZE];

  /*Basic pardiso info*/
  INT_TYPE     phase, maxfct, mnum, mtype, n, nrhs, msglvl, err;

  /*Matrix structure*/
  void         *a;
  INT_TYPE     *ia, *ja;

  /*Number of non-zero elements*/
  INT_TYPE     nz;

  /*Row permutaton vector*/
  INT_TYPE     *perm;

  /*Deffine is matrix preserve sparce structure.*/
  MatStructure matstruc;

  /*True if pardiso function have been used.*/
  PetscBool CleanUpPardiso;
} Mat_PARDISO;


void pardiso_64init(void *pt, INT_TYPE *mtype, INT_TYPE iparm []){
  int     iparm_copy[IPARM_SIZE], mtype_copy, i;
  mtype_copy = *mtype;
  pardisoinit(pt, &mtype_copy, iparm_copy);
  for(i = 0; i < IPARM_SIZE; i++)
    iparm[i] = iparm_copy[i];
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
#define __FUNCT__ "MatCopy_PARDISO"
PetscErrorCode MatCopy_PARDISO(Mat A, MatReuse reuse, INT_TYPE *nnz, INT_TYPE **r, INT_TYPE **c, void **v){

  Mat_SeqAIJ     *aa=(Mat_SeqAIJ*)A->data;

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
 * Free memory for Mat_PARDISO structure and pointers to objects.
 */
#undef __FUNCT__
#define __FUNCT__ "MatDestroy_PARDISO"
PetscErrorCode MatDestroy_PARDISO(Mat A){
  Mat_PARDISO      *mat_pardiso=(Mat_PARDISO*)A->spptr;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /* Terminate instance, deallocate memories */
  if (mat_pardiso->CleanUpPardiso) {
    mat_pardiso->phase = JOB_RELEASE_OF_ALL_MEMORY;


    MKL_PARDISO (mat_pardiso->pt,
      &mat_pardiso->maxfct,
      &mat_pardiso->mnum,
      &mat_pardiso->mtype,
      &mat_pardiso->phase,
      &mat_pardiso->n,
      NULL,
      NULL,
      NULL,
      mat_pardiso->perm,
      &mat_pardiso->nrhs,
      mat_pardiso->iparm,
      &mat_pardiso->msglvl,
      NULL,
      NULL,
      &mat_pardiso->err);
  }
  ierr = PetscFree(mat_pardiso->perm);CHKERRQ(ierr);
  ierr = PetscFree(A->spptr);CHKERRQ(ierr);

  /* clear composed functions */
  ierr = PetscObjectComposeFunction((PetscObject)A,"MatFactorGetSolverPackage_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)A,"MatPardisoSetCntl_C",NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


/*
 * Computes Ax = b
 */
#undef __FUNCT__
#define __FUNCT__ "MatSolve_PARDISO"
PetscErrorCode MatSolve_PARDISO(Mat A,Vec b,Vec x){
  Mat_PARDISO       *mat_pardiso=(Mat_PARDISO*)(A)->spptr;
  PetscErrorCode    ierr;
  PetscScalar       *barray, *xarray;

  PetscFunctionBegin;


  mat_pardiso->nrhs = 1;
  ierr = VecGetArray(x,&xarray);CHKERRQ(ierr);
  ierr = VecGetArray(b,&barray);CHKERRQ(ierr);

  /* solve phase */
  /*-------------*/
  mat_pardiso->phase = JOB_SOLVE_ITERATIVE_REFINEMENT;
  MKL_PARDISO (mat_pardiso->pt,
    &mat_pardiso->maxfct,
    &mat_pardiso->mnum,
    &mat_pardiso->mtype,
    &mat_pardiso->phase,
    &mat_pardiso->n,
    mat_pardiso->a,
    mat_pardiso->ia,
    mat_pardiso->ja,
    mat_pardiso->perm,
    &mat_pardiso->nrhs,
    mat_pardiso->iparm,
    &mat_pardiso->msglvl,
    (void*)barray,
    (void*)xarray,
    &mat_pardiso->err);


  if (mat_pardiso->err < 0) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Error reported by PARDISO: err=%d. Please check manual\n",mat_pardiso->err);

  mat_pardiso->CleanUpPardiso = PETSC_TRUE;
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "MatSolveTranspose_PARDISO"
PetscErrorCode MatSolveTranspose_PARDISO(Mat A,Vec b,Vec x){
  Mat_PARDISO      *mat_pardiso=(Mat_PARDISO*)A->spptr;
  PetscErrorCode ierr;

  PetscFunctionBegin;
#if defined(PETSC_USE_COMPLEX)
  mat_pardiso->iparm[12 - 1] = 1;
#else
  mat_pardiso->iparm[12 - 1] = 2;
#endif
  ierr = MatSolve_PARDISO(A,b,x);CHKERRQ(ierr);
  mat_pardiso->iparm[12 - 1] = 0;
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "MatMatSolve_PARDISO"
PetscErrorCode MatMatSolve_PARDISO(Mat A,Mat B,Mat X){
  Mat_PARDISO      *mat_pardiso=(Mat_PARDISO*)(A)->spptr;
  PetscErrorCode    ierr;
  PetscScalar       *barray, *xarray;
  PetscBool      flg;

  PetscFunctionBegin;

  ierr = PetscObjectTypeCompare((PetscObject)B,MATSEQDENSE,&flg);CHKERRQ(ierr);
  if (!flg) SETERRQ(PetscObjectComm((PetscObject)A),PETSC_ERR_ARG_WRONG,"Matrix B must be MATSEQDENSE matrix");
  ierr = PetscObjectTypeCompare((PetscObject)X,MATSEQDENSE,&flg);CHKERRQ(ierr);
  if (!flg) SETERRQ(PetscObjectComm((PetscObject)A),PETSC_ERR_ARG_WRONG,"Matrix X must be MATSEQDENSE matrix");

  ierr = MatGetSize(B,NULL,(PetscInt*)&mat_pardiso->nrhs);CHKERRQ(ierr);

  if(mat_pardiso->nrhs > 0){
    ierr = MatDenseGetArray(B,&barray);
    ierr = MatDenseGetArray(X,&xarray);

    /* solve phase */
    /*-------------*/
    mat_pardiso->phase = JOB_SOLVE_ITERATIVE_REFINEMENT;
    MKL_PARDISO (mat_pardiso->pt,
      &mat_pardiso->maxfct,
      &mat_pardiso->mnum,
      &mat_pardiso->mtype,
      &mat_pardiso->phase,
      &mat_pardiso->n,
      mat_pardiso->a,
      mat_pardiso->ia,
      mat_pardiso->ja,
      mat_pardiso->perm,
      &mat_pardiso->nrhs,
      mat_pardiso->iparm,
      &mat_pardiso->msglvl,
      (void*)barray,
      (void*)xarray,
      &mat_pardiso->err);
    if (mat_pardiso->err < 0) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Error reported by PARDISO: err=%d. Please check manual\n",mat_pardiso->err);
  }
  mat_pardiso->CleanUpPardiso = PETSC_TRUE;
  PetscFunctionReturn(0);

}

/*
 * LU Decomposition
 */
#undef __FUNCT__
#define __FUNCT__ "MatFactorNumeric_PARDISO"
PetscErrorCode MatFactorNumeric_PARDISO(Mat F,Mat A,const MatFactorInfo *info){
  Mat_PARDISO      *mat_pardiso=(Mat_PARDISO*)(F)->spptr;
  PetscErrorCode ierr;

  /* numerical factorization phase */
  /*-------------------------------*/

  PetscFunctionBegin;

  mat_pardiso->matstruc = SAME_NONZERO_PATTERN;
  ierr = MatCopy_PARDISO(A, MAT_REUSE_MATRIX, &mat_pardiso->nz, &mat_pardiso->ia, &mat_pardiso->ja, &mat_pardiso->a);CHKERRQ(ierr);

  /* numerical factorization phase */
  /*-------------------------------*/
  mat_pardiso->phase = JOB_NUMERICAL_FACTORIZATION;
  MKL_PARDISO (mat_pardiso->pt,
    &mat_pardiso->maxfct,
    &mat_pardiso->mnum,
    &mat_pardiso->mtype,
    &mat_pardiso->phase,
    &mat_pardiso->n,
    mat_pardiso->a,
    mat_pardiso->ia,
    mat_pardiso->ja,
    mat_pardiso->perm,
    &mat_pardiso->nrhs,
    mat_pardiso->iparm,
    &mat_pardiso->msglvl,
    NULL,
    NULL,
    &mat_pardiso->err);
  if (mat_pardiso->err < 0) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Error reported by PARDISO: err=%d. Please check manual\n",mat_pardiso->err);

  mat_pardiso->matstruc     = SAME_NONZERO_PATTERN;
  mat_pardiso->CleanUpPardiso = PETSC_TRUE;
  PetscFunctionReturn(0);
}

/* Sets pardiso options from the options database */
#undef __FUNCT__
#define __FUNCT__ "PetscSetPARDISOFromOptions"
PetscErrorCode PetscSetPARDISOFromOptions(Mat F, Mat A){
  Mat_PARDISO         *mat_pardiso = (Mat_PARDISO*)F->spptr;
  PetscErrorCode      ierr;
  PetscInt            icntl;
  PetscBool           flg;
  int                 pt[IPARM_SIZE], threads;

  PetscFunctionBegin;
  ierr = PetscOptionsBegin(PetscObjectComm((PetscObject)A),((PetscObject)A)->prefix,"PARDISO Options","Mat");CHKERRQ(ierr);
  ierr = PetscOptionsInt("-mat_pardiso_65",
    "Number of thrads to use",
    "None",
    threads,
    &threads,
    &flg);CHKERRQ(ierr);
  if (flg) mkl_set_num_threads(threads);

  ierr = PetscOptionsInt("-mat_pardiso_66",
    "Maximum number of factors with identical sparsity structure that must be kept in memory at the same time",
    "None",
     mat_pardiso->maxfct,
    &icntl,
    &flg);CHKERRQ(ierr);
  if (flg) mat_pardiso->maxfct = icntl;

  ierr = PetscOptionsInt("-mat_pardiso_67",
    "Indicates the actual matrix for the solution phase",
    "None",
    mat_pardiso->mnum,
    &icntl,
    &flg);CHKERRQ(ierr);
  if (flg) mat_pardiso->mnum = icntl;
 
  ierr = PetscOptionsInt("-mat_pardiso_68",
    "Message level information",
    "None",
    mat_pardiso->msglvl,
    &icntl,
    &flg);CHKERRQ(ierr);
  if (flg) mat_pardiso->msglvl = icntl;

  ierr = PetscOptionsInt("-mat_pardiso_69",
    "Defines the matrix type",
    "None",
    mat_pardiso->mtype,
    &icntl,
    &flg);CHKERRQ(ierr);
  if(flg){
   mat_pardiso->mtype = icntl;
   PARDISO_INIT(&pt, &mat_pardiso->mtype, mat_pardiso->iparm);
#if defined(PETSC_USE_REAL_SINGLE)
    mat_pardiso->iparm[27] = 1;
#else
    mat_pardiso->iparm[27] = 0;
#endif
    mat_pardiso->iparm[34] = 1;
  }
  ierr = PetscOptionsInt("-mat_pardiso_1",
    "Use default values",
    "None",
    mat_pardiso->iparm[0],
    &icntl,
    &flg);CHKERRQ(ierr);

  if(flg && icntl != 0){
    ierr = PetscOptionsInt("-mat_pardiso_2",
      "Fill-in reducing ordering for the input matrix",
      "None",
      mat_pardiso->iparm[1],
      &icntl,
      &flg);CHKERRQ(ierr);
    if (flg) mat_pardiso->iparm[1] = icntl;

    ierr = PetscOptionsInt("-mat_pardiso_4",
      "Preconditioned CGS/CG",
      "None",
      mat_pardiso->iparm[3],
      &icntl,
      &flg);CHKERRQ(ierr);
    if (flg) mat_pardiso->iparm[3] = icntl;

    ierr = PetscOptionsInt("-mat_pardiso_5",
      "User permutation",
      "None",
      mat_pardiso->iparm[4],
      &icntl,
      &flg);CHKERRQ(ierr);
    if (flg) mat_pardiso->iparm[4] = icntl;

    ierr = PetscOptionsInt("-mat_pardiso_6",
      "Write solution on x",
      "None",
      mat_pardiso->iparm[5],
      &icntl,
      &flg);CHKERRQ(ierr);
    if (flg) mat_pardiso->iparm[5] = icntl;

    ierr = PetscOptionsInt("-mat_pardiso_8",
      "Iterative refinement step",
      "None",
      mat_pardiso->iparm[7],
      &icntl,
      &flg);CHKERRQ(ierr);
    if (flg) mat_pardiso->iparm[7] = icntl;

    ierr = PetscOptionsInt("-mat_pardiso_10",
      "Pivoting perturbation",
      "None",
      mat_pardiso->iparm[9],
      &icntl,
      &flg);CHKERRQ(ierr);
    if (flg) mat_pardiso->iparm[9] = icntl;

    ierr = PetscOptionsInt("-mat_pardiso_11",
      "Scaling vectors",
      "None",
      mat_pardiso->iparm[10],
      &icntl,
      &flg);CHKERRQ(ierr);
    if (flg) mat_pardiso->iparm[10] = icntl;

    ierr = PetscOptionsInt("-mat_pardiso_12",
      "Solve with transposed or conjugate transposed matrix A",
      "None",
      mat_pardiso->iparm[11],
      &icntl,
      &flg);CHKERRQ(ierr);
    if (flg) mat_pardiso->iparm[11] = icntl;

    ierr = PetscOptionsInt("-mat_pardiso_13",
      "Improved accuracy using (non-) symmetric weighted matching",
      "None",
      mat_pardiso->iparm[12],
      &icntl,
      &flg);CHKERRQ(ierr);
    if (flg) mat_pardiso->iparm[12] = icntl;

    ierr = PetscOptionsInt("-mat_pardiso_18",
      "Numbers of non-zero elements",
      "None",
      mat_pardiso->iparm[17],
      &icntl,
      &flg);CHKERRQ(ierr);
    if (flg) mat_pardiso->iparm[17] = icntl;

    ierr = PetscOptionsInt("-mat_pardiso_19",
      "Report number of floating point operations",
      "None",
      mat_pardiso->iparm[18],
      &icntl,
      &flg);CHKERRQ(ierr);
    if (flg) mat_pardiso->iparm[18] = icntl;

    ierr = PetscOptionsInt("-mat_pardiso_21",
      "Pivoting for symmetric indefinite matrices",
      "None",
      mat_pardiso->iparm[20],
      &icntl,
      &flg);CHKERRQ(ierr);
    if (flg) mat_pardiso->iparm[20] = icntl;

    ierr = PetscOptionsInt("-mat_pardiso_24",
      "Parallel factorization control",
      "None",
      mat_pardiso->iparm[23],
      &icntl,
      &flg);CHKERRQ(ierr);
    if (flg) mat_pardiso->iparm[23] = icntl;

    ierr = PetscOptionsInt("-mat_pardiso_25",
      "Parallel forward/backward solve control",
      "None",
      mat_pardiso->iparm[24],
      &icntl,
      &flg);CHKERRQ(ierr);
    if (flg) mat_pardiso->iparm[24] = icntl;

    ierr = PetscOptionsInt("-mat_pardiso_27",
      "Matrix checker",
      "None",
      mat_pardiso->iparm[26],
      &icntl,
      &flg);CHKERRQ(ierr);
    if (flg) mat_pardiso->iparm[26] = icntl;

    ierr = PetscOptionsInt("-mat_pardiso_31",
      "Partial solve and computing selected components of the solution vectors",
      "None",
      mat_pardiso->iparm[30],
      &icntl,
      &flg);CHKERRQ(ierr);
    if (flg) mat_pardiso->iparm[30] = icntl;

    ierr = PetscOptionsInt("-mat_pardiso_34",
      "Optimal number of threads for conditional numerical reproducibility (CNR) mode",
      "None",
      mat_pardiso->iparm[33],
      &icntl,
      &flg);CHKERRQ(ierr);
    if (flg) mat_pardiso->iparm[33] = icntl;

    ierr = PetscOptionsInt("-mat_pardiso_60",
      "Intel MKL PARDISO mode",
      "None",
      mat_pardiso->iparm[59],
      &icntl,
      &flg);CHKERRQ(ierr);
    if (flg) mat_pardiso->iparm[59] = icntl;
  }

  PetscOptionsEnd();
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "PetscInitializePARDISO"
PetscErrorCode PetscInitializePARDISO(Mat A, Mat_PARDISO *mat_pardiso){
  PetscErrorCode ierr;
  PetscInt       i;

  PetscFunctionBegin;
  mat_pardiso->CleanUpPardiso = PETSC_FALSE;
  mat_pardiso->maxfct = 1;
  mat_pardiso->mnum = 1;
  mat_pardiso->n = A->rmap->N;
  mat_pardiso->msglvl = 0;
  mat_pardiso->nrhs = 1;
  mat_pardiso->err = 0;
  mat_pardiso->phase = -1;
#if defined(PETSC_USE_COMPLEX)
  mat_pardiso->mtype = 13;
#else
  mat_pardiso->mtype = 11;
#endif

  PARDISO_INIT(mat_pardiso->pt, &mat_pardiso->mtype, mat_pardiso->iparm);

#if defined(PETSC_USE_REAL_SINGLE)
  mat_pardiso->iparm[27] = 1;
#else
  mat_pardiso->iparm[27] = 0;
#endif

  mat_pardiso->iparm[34] = 1;

  ierr = PetscMalloc(A->rmap->N*sizeof(INT_TYPE), &mat_pardiso->perm);CHKERRQ(ierr);
  for(i = 0; i < A->rmap->N; i++)
    mat_pardiso->perm[i] = 0;
  PetscFunctionReturn(0);
}


/*
 * Symbolic decomposition. Pardiso analysis phase.
 */
#undef __FUNCT__
#define __FUNCT__ "MatLUFactorSymbolic_AIJPARDISO"
PetscErrorCode MatLUFactorSymbolic_AIJPARDISO(Mat F,Mat A,IS r,IS c,const MatFactorInfo *info){

  Mat_PARDISO      *mat_pardiso = (Mat_PARDISO*)F->spptr;
  PetscErrorCode ierr;

  PetscFunctionBegin;

  mat_pardiso->matstruc = DIFFERENT_NONZERO_PATTERN;

  /* Set PARDISO options from the options database */
  ierr = PetscSetPARDISOFromOptions(F,A);CHKERRQ(ierr);

  ierr = MatCopy_PARDISO(A, MAT_INITIAL_MATRIX, &mat_pardiso->nz, &mat_pardiso->ia, &mat_pardiso->ja, &mat_pardiso->a);CHKERRQ(ierr);
  mat_pardiso->n = A->rmap->N;

  /* analysis phase */
  /*----------------*/

  mat_pardiso->phase = JOB_ANALYSIS;  

  MKL_PARDISO (mat_pardiso->pt,
    &mat_pardiso->maxfct,
    &mat_pardiso->mnum,
    &mat_pardiso->mtype,
    &mat_pardiso->phase,
    &mat_pardiso->n,
    mat_pardiso->a,
    mat_pardiso->ia,
    mat_pardiso->ja,
    mat_pardiso->perm,
    &mat_pardiso->nrhs,
    mat_pardiso->iparm,
    &mat_pardiso->msglvl,
    NULL,
    NULL,
    &mat_pardiso->err);

  if (mat_pardiso->err < 0) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Error reported by PARDISO: err=%d\n. Please check manual",mat_pardiso->err);

  mat_pardiso->CleanUpPardiso = PETSC_TRUE;
  F->ops->lufactornumeric = MatFactorNumeric_PARDISO;
  F->ops->solve           = MatSolve_PARDISO;
  F->ops->solvetranspose  = MatSolveTranspose_PARDISO;
  F->ops->matsolve        = MatMatSolve_PARDISO;
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "MatView_PARDISO"
PetscErrorCode MatView_PARDISO(Mat A, PetscViewer viewer){
  PetscErrorCode    ierr;
  PetscBool         iascii;
  PetscViewerFormat format;
  Mat_PARDISO       *mat_pardiso=(Mat_PARDISO*)A->spptr;
  PetscInt          i;

  PetscFunctionBegin;
  /* check if matrix is pardiso type */
  if (A->ops->solve != MatSolve_PARDISO) PetscFunctionReturn(0);

  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii);CHKERRQ(ierr);
  if (iascii) {
    ierr = PetscViewerGetFormat(viewer,&format);CHKERRQ(ierr);
    if (format == PETSC_VIEWER_ASCII_INFO) {
      ierr = PetscViewerASCIIPrintf(viewer,"PARDISO run parameters:\n");CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer,"PARDISO phase:             %d \n",mat_pardiso->phase);CHKERRQ(ierr);
      for(i = 1; i <= 64; i++){
        ierr = PetscViewerASCIIPrintf(viewer,"PARDISO iparm[%d]:     %d \n",i, mat_pardiso->iparm[i - 1]);CHKERRQ(ierr);
      }
      ierr = PetscViewerASCIIPrintf(viewer,"PARDISO maxfct:     %d \n", mat_pardiso->maxfct);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer,"PARDISO mnum:     %d \n", mat_pardiso->mnum);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer,"PARDISO mtype:     %d \n", mat_pardiso->mtype);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer,"PARDISO n:     %d \n", mat_pardiso->n);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer,"PARDISO nrhs:     %d \n", mat_pardiso->nrhs);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer,"PARDISO msglvl:     %d \n", mat_pardiso->msglvl);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "MatGetInfo_PARDISO"
PetscErrorCode MatGetInfo_PARDISO(Mat A, MatInfoType flag, MatInfo *info){
  Mat_PARDISO *mat_pardiso =(Mat_PARDISO*)A->spptr;

  PetscFunctionBegin;
  info->block_size        = 1.0;
  info->nz_allocated      = mat_pardiso->nz + 0.0;
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
#define __FUNCT__ "MatPardisoSetCntl_PARDISO"
PetscErrorCode MatPardisoSetCntl_PARDISO(Mat F,PetscInt icntl,PetscInt ival){
  Mat_PARDISO *mat_pardiso =(Mat_PARDISO*)F->spptr;
  PetscFunctionBegin;
  if(icntl <= 64){
    mat_pardiso->iparm[icntl - 1] = ival;
  } else {
    if(icntl == 65)
      mkl_set_num_threads((int)ival);
    else if(icntl == 66)
      mat_pardiso->maxfct = ival;
    else if(icntl == 67)
      mat_pardiso->mnum = ival;
    else if(icntl == 68)
      mat_pardiso->msglvl = ival;
    else if(icntl == 69){
      int pt[IPARM_SIZE];
      mat_pardiso->mtype = ival;
      PARDISO_INIT(&pt, &mat_pardiso->mtype, mat_pardiso->iparm);
#if defined(PETSC_USE_REAL_SINGLE)
      mat_pardiso->iparm[27] = 1;
#else
      mat_pardiso->iparm[27] = 0;
#endif
      mat_pardiso->iparm[34] = 1;
    } 
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatPardisoSetCntl"
/*@
  MatPardisoSetCntl - Set Pardiso parameters

   Logically Collective on Mat

   Input Parameters:
+  F - the factored matrix obtained by calling MatGetFactor()
.  icntl - index of Pardiso parameter
-  ival - value of Pardiso parameter

  Options Database:
.   -mat_pardiso_<icntl> <ival>

   Level: beginner

   References: Pardiso Users' Guide

.seealso: MatGetFactor()
@*/
PetscErrorCode MatPardisoSetCntl(Mat F,PetscInt icntl,PetscInt ival)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscTryMethod(F,"MatPardisoSetCntl_C",(Mat,PetscInt,PetscInt),(F,icntl,ival));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


/*MC
  MATSOLVERPARDISO -  A matrix type providing direct solvers (LU) for
  sequential matrices via the external package PARDISO.

  Works with MATSEQAIJ matrices

  Options Database Keys:
+ -mat_pardiso_65 - Number of thrads to use
. -mat_pardiso_66 - Maximum number of factors with identical sparsity structure that must be kept in memory at the same time
. -mat_pardiso_67 - Indicates the actual matrix for the solution phase
. -mat_pardiso_68 - Message level information
. -mat_pardiso_69 - Defines the matrix type. IMPORTANT: When you set this flag, iparm parameters are going to be set to the default ones for the matrix type
. -mat_pardiso_1 - Use default values
. -mat_pardiso_2 - Fill-in reducing ordering for the input matrix
. -mat_pardiso_4 - Preconditioned CGS/CG
. -mat_pardiso_5 - User permutation
. -mat_pardiso_6 - Write solution on x
. -mat_pardiso_8 - Iterative refinement step
. -mat_pardiso_10 - Pivoting perturbation
. -mat_pardiso_11 - Scaling vectors
. -mat_pardiso_12 - Solve with transposed or conjugate transposed matrix A
. -mat_pardiso_13 - Improved accuracy using (non-) symmetric weighted matching
. -mat_pardiso_18 - Numbers of non-zero elements
. -mat_pardiso_19 - Report number of floating point operations
. -mat_pardiso_21 - Pivoting for symmetric indefinite matrices
. -mat_pardiso_24 - Parallel factorization control
. -mat_pardiso_25 - Parallel forward/backward solve control
. -mat_pardiso_27 - Matrix checker
. -mat_pardiso_31 - Partial solve and computing selected components of the solution vectors
. -mat_pardiso_34 - Optimal number of threads for conditional numerical reproducibility (CNR) mode
- -mat_pardiso_60 - Intel MKL PARDISO mode

  Level: beginner 

  For more information please check  MKL-pardiso manual

.seealso: PCFactorSetMatSolverPackage(), MatSolverPackage

M*/


#undef __FUNCT__
#define __FUNCT__ "MatFactorGetSolverPackage_pardiso"
static PetscErrorCode MatFactorGetSolverPackage_pardiso(Mat A, const MatSolverPackage *type){
  PetscFunctionBegin;
  *type = MATSOLVERPARDISO;
  PetscFunctionReturn(0);
}


/* MatGetFactor for Seq AIJ matrices */
#undef __FUNCT__
#define __FUNCT__ "MatGetFactor_aij_pardiso"
PETSC_EXTERN PetscErrorCode MatGetFactor_aij_pardiso(Mat A,MatFactorType ftype,Mat *F){
  Mat            B;
  PetscErrorCode ierr;
  Mat_PARDISO   *mat_pardiso;
  PetscBool      isSeqAIJ;

  PetscFunctionBegin;
  /* Create the factorization matrix */


  ierr = PetscObjectTypeCompare((PetscObject)A,MATSEQAIJ,&isSeqAIJ);CHKERRQ(ierr);
  ierr = MatCreate(PetscObjectComm((PetscObject)A),&B);CHKERRQ(ierr);
  ierr = MatSetSizes(B,A->rmap->n,A->cmap->n,A->rmap->N,A->cmap->N);CHKERRQ(ierr);
  ierr = MatSetType(B,((PetscObject)A)->type_name);CHKERRQ(ierr);
  if (isSeqAIJ) {
    ierr = MatSeqAIJSetPreallocation(B,0,NULL);CHKERRQ(ierr);
  } else {
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Is not allowed other types of matrices apart from MATSEQAIJ.");
  }

  B->ops->lufactorsymbolic = MatLUFactorSymbolic_AIJPARDISO;
  B->ops->destroy = MatDestroy_PARDISO;
  B->ops->view    = MatView_PARDISO;
  B->factortype   = ftype;
  B->ops->getinfo = MatGetInfo_PARDISO;
  B->assembled    = PETSC_TRUE;           /* required by -ksp_view */

  ierr = PetscNewLog(B,&mat_pardiso);CHKERRQ(ierr);
  B->spptr = mat_pardiso;

  ierr = PetscObjectComposeFunction((PetscObject)B,"MatFactorGetSolverPackage_C",MatFactorGetSolverPackage_pardiso);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)B,"MatPardisoSetCntl_C",MatPardisoSetCntl_PARDISO);CHKERRQ(ierr);
  ierr = PetscInitializePARDISO(A, mat_pardiso);CHKERRQ(ierr);

  *F = B;
  PetscFunctionReturn(0);
}

