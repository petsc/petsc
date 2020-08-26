
#include <petscsys.h>
#include <../src/mat/impls/aij/mpi/mpiaij.h> /*I  "petscmat.h"  I*/
#include <../src/mat/impls/sbaij/mpi/mpisbaij.h>

#if defined(PETSC_HAVE_MKL_INTEL_ILP64)
#define MKL_ILP64
#endif
#include <mkl.h>
#include <mkl_cluster_sparse_solver.h>

/*
 *  Possible mkl_cpardiso phases that controls the execution of the solver.
 *  For more information check mkl_cpardiso manual.
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
#define INT_TYPE MKL_INT

static const char *Err_MSG_CPardiso(int errNo) {
  switch (errNo) {
    case -1:
      return "input inconsistent"; break;
    case -2:
      return "not enough memory"; break;
    case -3:
      return "reordering problem"; break;
    case -4:
      return "zero pivot, numerical factorization or iterative refinement problem"; break;
    case -5:
      return "unclassified (internal) error"; break;
    case -6:
      return "preordering failed (matrix types 11, 13 only)"; break;
    case -7:
      return "diagonal matrix problem"; break;
    case -8:
      return "32-bit integer overflow problem"; break;
    case -9:
      return "not enough memory for OOC"; break;
    case -10:
      return "problems with opening OOC temporary files"; break;
    case -11:
      return "read/write problems with the OOC data file"; break;
    default :
      return "unknown error";
  }
}

/*
 *  Internal data structure.
 *  For more information check mkl_cpardiso manual.
 */

typedef struct {

  /* Configuration vector */
  INT_TYPE     iparm[IPARM_SIZE];

  /*
   * Internal mkl_cpardiso memory location.
   * After the first call to mkl_cpardiso do not modify pt, as that could cause a serious memory leak.
   */
  void         *pt[IPARM_SIZE];

  MPI_Comm     comm_mkl_cpardiso;

  /* Basic mkl_cpardiso info*/
  INT_TYPE     phase, maxfct, mnum, mtype, n, nrhs, msglvl, err;

  /* Matrix structure */
  PetscScalar  *a;

  INT_TYPE     *ia, *ja;

  /* Number of non-zero elements */
  INT_TYPE     nz;

  /* Row permutaton vector*/
  INT_TYPE     *perm;

  /* Define is matrix preserve sparce structure. */
  MatStructure matstruc;

  PetscErrorCode (*ConvertToTriples)(Mat, MatReuse, PetscInt*, PetscInt**, PetscInt**, PetscScalar**);

  /* True if mkl_cpardiso function have been used. */
  PetscBool CleanUp;
} Mat_MKL_CPARDISO;

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
PetscErrorCode MatCopy_seqaij_seqaij_MKL_CPARDISO(Mat A, MatReuse reuse, PetscInt *nnz, PetscInt **r, PetscInt **c, PetscScalar **v)
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

PetscErrorCode MatConvertToTriples_mpiaij_mpiaij_MKL_CPARDISO(Mat A, MatReuse reuse, PetscInt *nnz, PetscInt **r, PetscInt **c, PetscScalar **v)
{
  const PetscInt    *ai, *aj, *bi, *bj,*garray,m=A->rmap->n,*ajj,*bjj;
  PetscErrorCode    ierr;
  PetscInt          rstart,nz,i,j,countA,countB;
  PetscInt          *row,*col;
  const PetscScalar *av, *bv;
  PetscScalar       *val;
  Mat_MPIAIJ        *mat = (Mat_MPIAIJ*)A->data;
  Mat_SeqAIJ        *aa  = (Mat_SeqAIJ*)(mat->A)->data;
  Mat_SeqAIJ        *bb  = (Mat_SeqAIJ*)(mat->B)->data;
  PetscInt          colA_start,jB,jcol;

  PetscFunctionBegin;
  ai=aa->i; aj=aa->j; bi=bb->i; bj=bb->j; rstart=A->rmap->rstart;
  av=aa->a; bv=bb->a;

  garray = mat->garray;

  if (reuse == MAT_INITIAL_MATRIX) {
    nz   = aa->nz + bb->nz;
    *nnz = nz;
    ierr = PetscMalloc3(m+1,&row,nz,&col,nz,&val);CHKERRQ(ierr);
    *r = row; *c = col; *v = val;
  } else {
    row = *r; col = *c; val = *v;
  }

  nz = 0;
  for (i=0; i<m; i++) {
    row[i] = nz;
    countA     = ai[i+1] - ai[i];
    countB     = bi[i+1] - bi[i];
    ajj        = aj + ai[i]; /* ptr to the beginning of this row */
    bjj        = bj + bi[i];

    /* B part, smaller col index */
    colA_start = rstart + ajj[0]; /* the smallest global col index of A */
    jB         = 0;
    for (j=0; j<countB; j++) {
      jcol = garray[bjj[j]];
      if (jcol > colA_start) break;
      col[nz]   = jcol;
      val[nz++] = *bv++;
    }
    jB = j;

    /* A part */
    for (j=0; j<countA; j++) {
      col[nz]   = rstart + ajj[j];
      val[nz++] = *av++;
    }

    /* B part, larger col index */
    for (j=jB; j<countB; j++) {
      col[nz]   = garray[bjj[j]];
      val[nz++] = *bv++;
    }
  }
  row[m] = nz;

  PetscFunctionReturn(0);
}

PetscErrorCode MatConvertToTriples_mpibaij_mpibaij_MKL_CPARDISO(Mat A, MatReuse reuse, PetscInt *nnz, PetscInt **r, PetscInt **c, PetscScalar **v)
{
  const PetscInt    *ai, *aj, *bi, *bj,*garray,bs=A->rmap->bs,bs2=bs*bs,m=A->rmap->n/bs,*ajj,*bjj;
  PetscErrorCode    ierr;
  PetscInt          rstart,nz,i,j,countA,countB;
  PetscInt          *row,*col;
  const PetscScalar *av, *bv;
  PetscScalar       *val;
  Mat_MPIBAIJ       *mat = (Mat_MPIBAIJ*)A->data;
  Mat_SeqBAIJ       *aa  = (Mat_SeqBAIJ*)(mat->A)->data;
  Mat_SeqBAIJ       *bb  = (Mat_SeqBAIJ*)(mat->B)->data;
  PetscInt          colA_start,jB,jcol;

  PetscFunctionBegin;
  ai=aa->i; aj=aa->j; bi=bb->i; bj=bb->j; rstart=A->rmap->rstart/bs;
  av=aa->a; bv=bb->a;

  garray = mat->garray;

  if (reuse == MAT_INITIAL_MATRIX) {
    nz   = aa->nz + bb->nz;
    *nnz = nz;
    ierr = PetscMalloc3(m+1,&row,nz,&col,nz*bs2,&val);CHKERRQ(ierr);
    *r = row; *c = col; *v = val;
  } else {
    row = *r; col = *c; val = *v;
  }

  nz = 0;
  for (i=0; i<m; i++) {
    row[i]     = nz+1;
    countA     = ai[i+1] - ai[i];
    countB     = bi[i+1] - bi[i];
    ajj        = aj + ai[i]; /* ptr to the beginning of this row */
    bjj        = bj + bi[i];

    /* B part, smaller col index */
    colA_start = rstart + (countA > 0 ? ajj[0] : 0); /* the smallest global col index of A */
    jB         = 0;
    for (j=0; j<countB; j++) {
      jcol = garray[bjj[j]];
      if (jcol > colA_start) break;
      col[nz++] = jcol + 1;
    }
    jB = j;
    ierr = PetscArraycpy(val,bv,jB*bs2);CHKERRQ(ierr);
    val += jB*bs2;
    bv  += jB*bs2;

    /* A part */
    for (j=0; j<countA; j++) col[nz++] = rstart + ajj[j] + 1;
    ierr = PetscArraycpy(val,av,countA*bs2);CHKERRQ(ierr);
    val += countA*bs2;
    av  += countA*bs2;

    /* B part, larger col index */
    for (j=jB; j<countB; j++) col[nz++] = garray[bjj[j]] + 1;
    ierr = PetscArraycpy(val,bv,(countB-jB)*bs2);CHKERRQ(ierr);
    val += (countB-jB)*bs2;
    bv  += (countB-jB)*bs2;
  }
  row[m] = nz+1;

  PetscFunctionReturn(0);
}

PetscErrorCode MatConvertToTriples_mpisbaij_mpisbaij_MKL_CPARDISO(Mat A, MatReuse reuse, PetscInt *nnz, PetscInt **r, PetscInt **c, PetscScalar **v)
{
  const PetscInt    *ai, *aj, *bi, *bj,*garray,bs=A->rmap->bs,bs2=bs*bs,m=A->rmap->n/bs,*ajj,*bjj;
  PetscErrorCode    ierr;
  PetscInt          rstart,nz,i,j,countA,countB;
  PetscInt          *row,*col;
  const PetscScalar *av, *bv;
  PetscScalar       *val;
  Mat_MPISBAIJ      *mat = (Mat_MPISBAIJ*)A->data;
  Mat_SeqSBAIJ      *aa  = (Mat_SeqSBAIJ*)(mat->A)->data;
  Mat_SeqBAIJ       *bb  = (Mat_SeqBAIJ*)(mat->B)->data;

  PetscFunctionBegin;
  ai=aa->i; aj=aa->j; bi=bb->i; bj=bb->j; rstart=A->rmap->rstart/bs;
  av=aa->a; bv=bb->a;

  garray = mat->garray;

  if (reuse == MAT_INITIAL_MATRIX) {
    nz   = aa->nz + bb->nz;
    *nnz = nz;
    ierr = PetscMalloc3(m+1,&row,nz,&col,nz*bs2,&val);CHKERRQ(ierr);
    *r = row; *c = col; *v = val;
  } else {
    row = *r; col = *c; val = *v;
  }

  nz = 0;
  for (i=0; i<m; i++) {
    row[i]     = nz+1;
    countA     = ai[i+1] - ai[i];
    countB     = bi[i+1] - bi[i];
    ajj        = aj + ai[i]; /* ptr to the beginning of this row */
    bjj        = bj + bi[i];

    /* A part */
    for (j=0; j<countA; j++) col[nz++] = rstart + ajj[j] + 1;
    ierr = PetscArraycpy(val,av,countA*bs2);CHKERRQ(ierr);
    val += countA*bs2;
    av  += countA*bs2;

    /* B part, larger col index */
    for (j=0; j<countB; j++) col[nz++] = garray[bjj[j]] + 1;
    ierr = PetscArraycpy(val,bv,countB*bs2);CHKERRQ(ierr);
    val += countB*bs2;
    bv  += countB*bs2;
  }
  row[m] = nz+1;

  PetscFunctionReturn(0);
}

/*
 * Free memory for Mat_MKL_CPARDISO structure and pointers to objects.
 */
PetscErrorCode MatDestroy_MKL_CPARDISO(Mat A)
{
  Mat_MKL_CPARDISO *mat_mkl_cpardiso=(Mat_MKL_CPARDISO*)A->data;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  /* Terminate instance, deallocate memories */
  if (mat_mkl_cpardiso->CleanUp) {
    mat_mkl_cpardiso->phase = JOB_RELEASE_OF_ALL_MEMORY;

    cluster_sparse_solver (
      mat_mkl_cpardiso->pt,
      &mat_mkl_cpardiso->maxfct,
      &mat_mkl_cpardiso->mnum,
      &mat_mkl_cpardiso->mtype,
      &mat_mkl_cpardiso->phase,
      &mat_mkl_cpardiso->n,
      NULL,
      NULL,
      NULL,
      mat_mkl_cpardiso->perm,
      &mat_mkl_cpardiso->nrhs,
      mat_mkl_cpardiso->iparm,
      &mat_mkl_cpardiso->msglvl,
      NULL,
      NULL,
      &mat_mkl_cpardiso->comm_mkl_cpardiso,
      (PetscInt*)&mat_mkl_cpardiso->err);
  }

  if (mat_mkl_cpardiso->ConvertToTriples != MatCopy_seqaij_seqaij_MKL_CPARDISO) {
    ierr = PetscFree3(mat_mkl_cpardiso->ia,mat_mkl_cpardiso->ja,mat_mkl_cpardiso->a);CHKERRQ(ierr);
  }
  ierr = MPI_Comm_free(&(mat_mkl_cpardiso->comm_mkl_cpardiso));CHKERRQ(ierr);
  ierr = PetscFree(A->data);CHKERRQ(ierr);

  /* clear composed functions */
  ierr = PetscObjectComposeFunction((PetscObject)A,"MatFactorGetSolverType_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)A,"MatMkl_CPardisoSetCntl_C",NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
 * Computes Ax = b
 */
PetscErrorCode MatSolve_MKL_CPARDISO(Mat A,Vec b,Vec x)
{
  Mat_MKL_CPARDISO   *mat_mkl_cpardiso=(Mat_MKL_CPARDISO*)(A)->data;
  PetscErrorCode    ierr;
  PetscScalar       *xarray;
  const PetscScalar *barray;

  PetscFunctionBegin;
  mat_mkl_cpardiso->nrhs = 1;
  ierr = VecGetArray(x,&xarray);CHKERRQ(ierr);
  ierr = VecGetArrayRead(b,&barray);CHKERRQ(ierr);

  /* solve phase */
  /*-------------*/
  mat_mkl_cpardiso->phase = JOB_SOLVE_ITERATIVE_REFINEMENT;
  cluster_sparse_solver (
    mat_mkl_cpardiso->pt,
    &mat_mkl_cpardiso->maxfct,
    &mat_mkl_cpardiso->mnum,
    &mat_mkl_cpardiso->mtype,
    &mat_mkl_cpardiso->phase,
    &mat_mkl_cpardiso->n,
    mat_mkl_cpardiso->a,
    mat_mkl_cpardiso->ia,
    mat_mkl_cpardiso->ja,
    mat_mkl_cpardiso->perm,
    &mat_mkl_cpardiso->nrhs,
    mat_mkl_cpardiso->iparm,
    &mat_mkl_cpardiso->msglvl,
    (void*)barray,
    (void*)xarray,
    &mat_mkl_cpardiso->comm_mkl_cpardiso,
    (PetscInt*)&mat_mkl_cpardiso->err);

  if (mat_mkl_cpardiso->err < 0) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_LIB,"Error reported by MKL_CPARDISO: err=%d, msg = \"%s\". Please check manual\n",mat_mkl_cpardiso->err,Err_MSG_CPardiso(mat_mkl_cpardiso->err));

  ierr = VecRestoreArray(x,&xarray);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(b,&barray);CHKERRQ(ierr);
  mat_mkl_cpardiso->CleanUp = PETSC_TRUE;
  PetscFunctionReturn(0);
}

PetscErrorCode MatSolveTranspose_MKL_CPARDISO(Mat A,Vec b,Vec x)
{
  Mat_MKL_CPARDISO *mat_mkl_cpardiso=(Mat_MKL_CPARDISO*)A->data;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
#if defined(PETSC_USE_COMPLEX)
  mat_mkl_cpardiso->iparm[12 - 1] = 1;
#else
  mat_mkl_cpardiso->iparm[12 - 1] = 2;
#endif
  ierr = MatSolve_MKL_CPARDISO(A,b,x);CHKERRQ(ierr);
  mat_mkl_cpardiso->iparm[12 - 1] = 0;
  PetscFunctionReturn(0);
}

PetscErrorCode MatMatSolve_MKL_CPARDISO(Mat A,Mat B,Mat X)
{
  Mat_MKL_CPARDISO  *mat_mkl_cpardiso=(Mat_MKL_CPARDISO*)(A)->data;
  PetscErrorCode    ierr;
  PetscScalar       *xarray;
  const PetscScalar *barray;

  PetscFunctionBegin;
  ierr = MatGetSize(B,NULL,(PetscInt*)&mat_mkl_cpardiso->nrhs);CHKERRQ(ierr);

  if (mat_mkl_cpardiso->nrhs > 0) {
    ierr = MatDenseGetArrayRead(B,&barray);
    ierr = MatDenseGetArray(X,&xarray);

    if (barray == xarray) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"B and X cannot share the same memory location");

    /* solve phase */
    /*-------------*/
    mat_mkl_cpardiso->phase = JOB_SOLVE_ITERATIVE_REFINEMENT;
    cluster_sparse_solver (
      mat_mkl_cpardiso->pt,
      &mat_mkl_cpardiso->maxfct,
      &mat_mkl_cpardiso->mnum,
      &mat_mkl_cpardiso->mtype,
      &mat_mkl_cpardiso->phase,
      &mat_mkl_cpardiso->n,
      mat_mkl_cpardiso->a,
      mat_mkl_cpardiso->ia,
      mat_mkl_cpardiso->ja,
      mat_mkl_cpardiso->perm,
      &mat_mkl_cpardiso->nrhs,
      mat_mkl_cpardiso->iparm,
      &mat_mkl_cpardiso->msglvl,
      (void*)barray,
      (void*)xarray,
      &mat_mkl_cpardiso->comm_mkl_cpardiso,
      (PetscInt*)&mat_mkl_cpardiso->err);
    if (mat_mkl_cpardiso->err < 0) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_LIB,"Error reported by MKL_CPARDISO: err=%d, msg = \"%s\". Please check manual\n",mat_mkl_cpardiso->err,Err_MSG_CPardiso(mat_mkl_cpardiso->err));
    ierr = MatDenseRestoreArrayRead(B,&barray);
    ierr = MatDenseRestoreArray(X,&xarray);

  }
  mat_mkl_cpardiso->CleanUp = PETSC_TRUE;
  PetscFunctionReturn(0);
}

/*
 * LU Decomposition
 */
PetscErrorCode MatFactorNumeric_MKL_CPARDISO(Mat F,Mat A,const MatFactorInfo *info)
{
  Mat_MKL_CPARDISO *mat_mkl_cpardiso=(Mat_MKL_CPARDISO*)(F)->data;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  mat_mkl_cpardiso->matstruc = SAME_NONZERO_PATTERN;
  ierr = (*mat_mkl_cpardiso->ConvertToTriples)(A, MAT_REUSE_MATRIX,&mat_mkl_cpardiso->nz,&mat_mkl_cpardiso->ia,&mat_mkl_cpardiso->ja,&mat_mkl_cpardiso->a);CHKERRQ(ierr);

  mat_mkl_cpardiso->phase = JOB_NUMERICAL_FACTORIZATION;
  cluster_sparse_solver (
    mat_mkl_cpardiso->pt,
    &mat_mkl_cpardiso->maxfct,
    &mat_mkl_cpardiso->mnum,
    &mat_mkl_cpardiso->mtype,
    &mat_mkl_cpardiso->phase,
    &mat_mkl_cpardiso->n,
    mat_mkl_cpardiso->a,
    mat_mkl_cpardiso->ia,
    mat_mkl_cpardiso->ja,
    mat_mkl_cpardiso->perm,
    &mat_mkl_cpardiso->nrhs,
    mat_mkl_cpardiso->iparm,
    &mat_mkl_cpardiso->msglvl,
    NULL,
    NULL,
    &mat_mkl_cpardiso->comm_mkl_cpardiso,
    &mat_mkl_cpardiso->err);
  if (mat_mkl_cpardiso->err < 0) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_LIB,"Error reported by MKL_CPARDISO: err=%d, msg = \"%s\". Please check manual\n",mat_mkl_cpardiso->err,Err_MSG_CPardiso(mat_mkl_cpardiso->err));

  mat_mkl_cpardiso->matstruc = SAME_NONZERO_PATTERN;
  mat_mkl_cpardiso->CleanUp  = PETSC_TRUE;
  PetscFunctionReturn(0);
}

/* Sets mkl_cpardiso options from the options database */
PetscErrorCode PetscSetMKL_CPARDISOFromOptions(Mat F, Mat A)
{
  Mat_MKL_CPARDISO    *mat_mkl_cpardiso = (Mat_MKL_CPARDISO*)F->data;
  PetscErrorCode      ierr;
  PetscInt            icntl,threads;
  PetscBool           flg;

  PetscFunctionBegin;
  ierr = PetscOptionsBegin(PetscObjectComm((PetscObject)A),((PetscObject)A)->prefix,"MKL_CPARDISO Options","Mat");CHKERRQ(ierr);
  ierr = PetscOptionsInt("-mat_mkl_cpardiso_65","Number of threads to use","None",threads,&threads,&flg);CHKERRQ(ierr);
  if (flg) mkl_set_num_threads((int)threads);

  ierr = PetscOptionsInt("-mat_mkl_cpardiso_66","Maximum number of factors with identical sparsity structure that must be kept in memory at the same time","None",mat_mkl_cpardiso->maxfct,&icntl,&flg);CHKERRQ(ierr);
  if (flg) mat_mkl_cpardiso->maxfct = icntl;

  ierr = PetscOptionsInt("-mat_mkl_cpardiso_67","Indicates the actual matrix for the solution phase","None",mat_mkl_cpardiso->mnum,&icntl,&flg);CHKERRQ(ierr);
  if (flg) mat_mkl_cpardiso->mnum = icntl;

  ierr = PetscOptionsInt("-mat_mkl_cpardiso_68","Message level information","None",mat_mkl_cpardiso->msglvl,&icntl,&flg);CHKERRQ(ierr);
  if (flg) mat_mkl_cpardiso->msglvl = icntl;

  ierr = PetscOptionsInt("-mat_mkl_cpardiso_69","Defines the matrix type","None",mat_mkl_cpardiso->mtype,&icntl,&flg);CHKERRQ(ierr);
  if (flg) mat_mkl_cpardiso->mtype = icntl;
  ierr = PetscOptionsInt("-mat_mkl_cpardiso_1","Use default values","None",mat_mkl_cpardiso->iparm[0],&icntl,&flg);CHKERRQ(ierr);

  if (flg && icntl != 0) {
    ierr = PetscOptionsInt("-mat_mkl_cpardiso_2","Fill-in reducing ordering for the input matrix","None",mat_mkl_cpardiso->iparm[1],&icntl,&flg);CHKERRQ(ierr);
    if (flg) mat_mkl_cpardiso->iparm[1] = icntl;

    ierr = PetscOptionsInt("-mat_mkl_cpardiso_4","Preconditioned CGS/CG","None",mat_mkl_cpardiso->iparm[3],&icntl,&flg);CHKERRQ(ierr);
    if (flg) mat_mkl_cpardiso->iparm[3] = icntl;

    ierr = PetscOptionsInt("-mat_mkl_cpardiso_5","User permutation","None",mat_mkl_cpardiso->iparm[4],&icntl,&flg);CHKERRQ(ierr);
    if (flg) mat_mkl_cpardiso->iparm[4] = icntl;

    ierr = PetscOptionsInt("-mat_mkl_cpardiso_6","Write solution on x","None",mat_mkl_cpardiso->iparm[5],&icntl,&flg);CHKERRQ(ierr);
    if (flg) mat_mkl_cpardiso->iparm[5] = icntl;

    ierr = PetscOptionsInt("-mat_mkl_cpardiso_8","Iterative refinement step","None",mat_mkl_cpardiso->iparm[7],&icntl,&flg);CHKERRQ(ierr);
    if (flg) mat_mkl_cpardiso->iparm[7] = icntl;

    ierr = PetscOptionsInt("-mat_mkl_cpardiso_10","Pivoting perturbation","None",mat_mkl_cpardiso->iparm[9],&icntl,&flg);CHKERRQ(ierr);
    if (flg) mat_mkl_cpardiso->iparm[9] = icntl;

    ierr = PetscOptionsInt("-mat_mkl_cpardiso_11","Scaling vectors","None",mat_mkl_cpardiso->iparm[10],&icntl,&flg);CHKERRQ(ierr);
    if (flg) mat_mkl_cpardiso->iparm[10] = icntl;

    ierr = PetscOptionsInt("-mat_mkl_cpardiso_12","Solve with transposed or conjugate transposed matrix A","None",mat_mkl_cpardiso->iparm[11],&icntl,&flg);CHKERRQ(ierr);
    if (flg) mat_mkl_cpardiso->iparm[11] = icntl;

    ierr = PetscOptionsInt("-mat_mkl_cpardiso_13","Improved accuracy using (non-) symmetric weighted matching","None",mat_mkl_cpardiso->iparm[12],&icntl,
      &flg);CHKERRQ(ierr);
    if (flg) mat_mkl_cpardiso->iparm[12] = icntl;

    ierr = PetscOptionsInt("-mat_mkl_cpardiso_18","Numbers of non-zero elements","None",mat_mkl_cpardiso->iparm[17],&icntl,
      &flg);CHKERRQ(ierr);
    if (flg) mat_mkl_cpardiso->iparm[17] = icntl;

    ierr = PetscOptionsInt("-mat_mkl_cpardiso_19","Report number of floating point operations","None",mat_mkl_cpardiso->iparm[18],&icntl,&flg);CHKERRQ(ierr);
    if (flg) mat_mkl_cpardiso->iparm[18] = icntl;

    ierr = PetscOptionsInt("-mat_mkl_cpardiso_21","Pivoting for symmetric indefinite matrices","None",mat_mkl_cpardiso->iparm[20],&icntl,&flg);CHKERRQ(ierr);
    if (flg) mat_mkl_cpardiso->iparm[20] = icntl;

    ierr = PetscOptionsInt("-mat_mkl_cpardiso_24","Parallel factorization control","None",mat_mkl_cpardiso->iparm[23],&icntl,&flg);CHKERRQ(ierr);
    if (flg) mat_mkl_cpardiso->iparm[23] = icntl;

    ierr = PetscOptionsInt("-mat_mkl_cpardiso_25","Parallel forward/backward solve control","None",mat_mkl_cpardiso->iparm[24],&icntl,&flg);CHKERRQ(ierr);
    if (flg) mat_mkl_cpardiso->iparm[24] = icntl;

    ierr = PetscOptionsInt("-mat_mkl_cpardiso_27","Matrix checker","None",mat_mkl_cpardiso->iparm[26],&icntl,&flg);CHKERRQ(ierr);
    if (flg) mat_mkl_cpardiso->iparm[26] = icntl;

    ierr = PetscOptionsInt("-mat_mkl_cpardiso_31","Partial solve and computing selected components of the solution vectors","None",mat_mkl_cpardiso->iparm[30],&icntl,&flg);CHKERRQ(ierr);
    if (flg) mat_mkl_cpardiso->iparm[30] = icntl;

    ierr = PetscOptionsInt("-mat_mkl_cpardiso_34","Optimal number of threads for conditional numerical reproducibility (CNR) mode","None",mat_mkl_cpardiso->iparm[33],&icntl,&flg);CHKERRQ(ierr);
    if (flg) mat_mkl_cpardiso->iparm[33] = icntl;

    ierr = PetscOptionsInt("-mat_mkl_cpardiso_60","Intel MKL_CPARDISO mode","None",mat_mkl_cpardiso->iparm[59],&icntl,&flg);CHKERRQ(ierr);
    if (flg) mat_mkl_cpardiso->iparm[59] = icntl;
  }

  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode PetscInitialize_MKL_CPARDISO(Mat A, Mat_MKL_CPARDISO *mat_mkl_cpardiso)
{
  PetscErrorCode  ierr;
  PetscInt        bs;
  PetscBool       match;
  PetscMPIInt     size;

  PetscFunctionBegin;

  ierr = MPI_Comm_dup(PetscObjectComm((PetscObject)A),&(mat_mkl_cpardiso->comm_mkl_cpardiso));CHKERRQ(ierr);
  ierr = MPI_Comm_size(mat_mkl_cpardiso->comm_mkl_cpardiso, &size);CHKERRQ(ierr);

  mat_mkl_cpardiso->CleanUp = PETSC_FALSE;
  mat_mkl_cpardiso->maxfct = 1;
  mat_mkl_cpardiso->mnum = 1;
  mat_mkl_cpardiso->n = A->rmap->N;
  if (mat_mkl_cpardiso->iparm[36]) mat_mkl_cpardiso->n /= mat_mkl_cpardiso->iparm[36];
  mat_mkl_cpardiso->msglvl = 0;
  mat_mkl_cpardiso->nrhs = 1;
  mat_mkl_cpardiso->err = 0;
  mat_mkl_cpardiso->phase = -1;
#if defined(PETSC_USE_COMPLEX)
  mat_mkl_cpardiso->mtype = 13;
#else
  mat_mkl_cpardiso->mtype = 11;
#endif

#if defined(PETSC_USE_REAL_SINGLE)
  mat_mkl_cpardiso->iparm[27] = 1;
#else
  mat_mkl_cpardiso->iparm[27] = 0;
#endif

  mat_mkl_cpardiso->iparm[ 0] =  1; /* Solver default parameters overriden with provided by iparm */
  mat_mkl_cpardiso->iparm[ 1] =  2; /* Use METIS for fill-in reordering */
  mat_mkl_cpardiso->iparm[ 5] =  0; /* Write solution into x */
  mat_mkl_cpardiso->iparm[ 7] =  2; /* Max number of iterative refinement steps */
  mat_mkl_cpardiso->iparm[ 9] = 13; /* Perturb the pivot elements with 1E-13 */
  mat_mkl_cpardiso->iparm[10] =  1; /* Use nonsymmetric permutation and scaling MPS */
  mat_mkl_cpardiso->iparm[12] =  1; /* Switch on Maximum Weighted Matching algorithm (default for non-symmetric) */
  mat_mkl_cpardiso->iparm[17] = -1; /* Output: Number of nonzeros in the factor LU */
  mat_mkl_cpardiso->iparm[18] = -1; /* Output: Mflops for LU factorization */
  mat_mkl_cpardiso->iparm[26] =  1; /* Check input data for correctness */

  mat_mkl_cpardiso->iparm[39] = 0;
  if (size > 1) {
    mat_mkl_cpardiso->iparm[39] = 2;
    mat_mkl_cpardiso->iparm[40] = A->rmap->rstart;
    mat_mkl_cpardiso->iparm[41] = A->rmap->rend-1;
  }
  ierr = PetscObjectTypeCompareAny((PetscObject)A,&match,MATMPIBAIJ,MATMPISBAIJ,"");CHKERRQ(ierr);
  if (match) {
    ierr = MatGetBlockSize(A,&bs);CHKERRQ(ierr);
    mat_mkl_cpardiso->iparm[36] = bs;
    mat_mkl_cpardiso->iparm[40] /= bs;
    mat_mkl_cpardiso->iparm[41] /= bs;
    mat_mkl_cpardiso->iparm[40]++;
    mat_mkl_cpardiso->iparm[41]++;
    mat_mkl_cpardiso->iparm[34] = 0;  /* Fortran style */
  } else {
    mat_mkl_cpardiso->iparm[34] = 1;  /* C style */
  }

  mat_mkl_cpardiso->perm = 0;
  PetscFunctionReturn(0);
}

/*
 * Symbolic decomposition. Mkl_Pardiso analysis phase.
 */
PetscErrorCode MatLUFactorSymbolic_AIJMKL_CPARDISO(Mat F,Mat A,IS r,IS c,const MatFactorInfo *info)
{
  Mat_MKL_CPARDISO *mat_mkl_cpardiso = (Mat_MKL_CPARDISO*)F->data;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  mat_mkl_cpardiso->matstruc = DIFFERENT_NONZERO_PATTERN;

  /* Set MKL_CPARDISO options from the options database */
  ierr = PetscSetMKL_CPARDISOFromOptions(F,A);CHKERRQ(ierr);
  ierr = (*mat_mkl_cpardiso->ConvertToTriples)(A,MAT_INITIAL_MATRIX,&mat_mkl_cpardiso->nz,&mat_mkl_cpardiso->ia,&mat_mkl_cpardiso->ja,&mat_mkl_cpardiso->a);CHKERRQ(ierr);

  mat_mkl_cpardiso->n = A->rmap->N;
  if (mat_mkl_cpardiso->iparm[36]) mat_mkl_cpardiso->n /= mat_mkl_cpardiso->iparm[36];

  /* analysis phase */
  /*----------------*/
  mat_mkl_cpardiso->phase = JOB_ANALYSIS;

  cluster_sparse_solver (
    mat_mkl_cpardiso->pt,
    &mat_mkl_cpardiso->maxfct,
    &mat_mkl_cpardiso->mnum,
    &mat_mkl_cpardiso->mtype,
    &mat_mkl_cpardiso->phase,
    &mat_mkl_cpardiso->n,
    mat_mkl_cpardiso->a,
    mat_mkl_cpardiso->ia,
    mat_mkl_cpardiso->ja,
    mat_mkl_cpardiso->perm,
    &mat_mkl_cpardiso->nrhs,
    mat_mkl_cpardiso->iparm,
    &mat_mkl_cpardiso->msglvl,
    NULL,
    NULL,
    &mat_mkl_cpardiso->comm_mkl_cpardiso,
    (PetscInt*)&mat_mkl_cpardiso->err);

  if (mat_mkl_cpardiso->err < 0) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_LIB,"Error reported by MKL_CPARDISO: err=%d, msg = \"%s\".Check manual\n",mat_mkl_cpardiso->err,Err_MSG_CPardiso(mat_mkl_cpardiso->err));

  mat_mkl_cpardiso->CleanUp = PETSC_TRUE;
  F->ops->lufactornumeric = MatFactorNumeric_MKL_CPARDISO;
  F->ops->solve           = MatSolve_MKL_CPARDISO;
  F->ops->solvetranspose  = MatSolveTranspose_MKL_CPARDISO;
  F->ops->matsolve        = MatMatSolve_MKL_CPARDISO;
  PetscFunctionReturn(0);
}

PetscErrorCode MatCholeskyFactorSymbolic_AIJMKL_CPARDISO(Mat F,Mat A,IS perm,const MatFactorInfo *info)
{
  Mat_MKL_CPARDISO *mat_mkl_cpardiso = (Mat_MKL_CPARDISO*)F->data;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  mat_mkl_cpardiso->matstruc = DIFFERENT_NONZERO_PATTERN;


  /* Set MKL_CPARDISO options from the options database */
  ierr = PetscSetMKL_CPARDISOFromOptions(F,A);CHKERRQ(ierr);
  ierr = (*mat_mkl_cpardiso->ConvertToTriples)(A,MAT_INITIAL_MATRIX,&mat_mkl_cpardiso->nz,&mat_mkl_cpardiso->ia,&mat_mkl_cpardiso->ja,&mat_mkl_cpardiso->a);CHKERRQ(ierr);

  mat_mkl_cpardiso->n = A->rmap->N;
  if (mat_mkl_cpardiso->iparm[36]) mat_mkl_cpardiso->n /= mat_mkl_cpardiso->iparm[36];
#if defined(PETSC_USE_COMPLEX)
  SETERRQ1(PetscObjectComm((PetscObject)A),PETSC_ERR_SUP,"No support for PARDISO CHOLESKY with complex scalars! Use MAT_FACTOR_LU instead",((PetscObject)A)->type_name);
#endif
  if (A->spd_set && A->spd) mat_mkl_cpardiso->mtype = 2;
  else                      mat_mkl_cpardiso->mtype = -2;

  /* analysis phase */
  /*----------------*/
  mat_mkl_cpardiso->phase = JOB_ANALYSIS;

  cluster_sparse_solver (
    mat_mkl_cpardiso->pt,
    &mat_mkl_cpardiso->maxfct,
    &mat_mkl_cpardiso->mnum,
    &mat_mkl_cpardiso->mtype,
    &mat_mkl_cpardiso->phase,
    &mat_mkl_cpardiso->n,
    mat_mkl_cpardiso->a,
    mat_mkl_cpardiso->ia,
    mat_mkl_cpardiso->ja,
    mat_mkl_cpardiso->perm,
    &mat_mkl_cpardiso->nrhs,
    mat_mkl_cpardiso->iparm,
    &mat_mkl_cpardiso->msglvl,
    NULL,
    NULL,
    &mat_mkl_cpardiso->comm_mkl_cpardiso,
    (PetscInt*)&mat_mkl_cpardiso->err);

  if (mat_mkl_cpardiso->err < 0) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_LIB,"Error reported by MKL_CPARDISO: err=%d, msg = \"%s\".Check manual\n",mat_mkl_cpardiso->err,Err_MSG_CPardiso(mat_mkl_cpardiso->err));

  mat_mkl_cpardiso->CleanUp = PETSC_TRUE;
  F->ops->choleskyfactornumeric = MatFactorNumeric_MKL_CPARDISO;
  F->ops->solve                 = MatSolve_MKL_CPARDISO;
  F->ops->solvetranspose        = MatSolveTranspose_MKL_CPARDISO;
  F->ops->matsolve              = MatMatSolve_MKL_CPARDISO;
  PetscFunctionReturn(0);
}

PetscErrorCode MatView_MKL_CPARDISO(Mat A, PetscViewer viewer)
{
  PetscErrorCode    ierr;
  PetscBool         iascii;
  PetscViewerFormat format;
  Mat_MKL_CPARDISO  *mat_mkl_cpardiso=(Mat_MKL_CPARDISO*)A->data;
  PetscInt          i;

  PetscFunctionBegin;
  /* check if matrix is mkl_cpardiso type */
  if (A->ops->solve != MatSolve_MKL_CPARDISO) PetscFunctionReturn(0);

  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii);CHKERRQ(ierr);
  if (iascii) {
    ierr = PetscViewerGetFormat(viewer,&format);CHKERRQ(ierr);
    if (format == PETSC_VIEWER_ASCII_INFO) {
      ierr = PetscViewerASCIIPrintf(viewer,"MKL_CPARDISO run parameters:\n");CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer,"MKL_CPARDISO phase:             %d \n",mat_mkl_cpardiso->phase);CHKERRQ(ierr);
      for (i = 1; i <= 64; i++) {
        ierr = PetscViewerASCIIPrintf(viewer,"MKL_CPARDISO iparm[%d]:     %d \n",i, mat_mkl_cpardiso->iparm[i - 1]);CHKERRQ(ierr);
      }
      ierr = PetscViewerASCIIPrintf(viewer,"MKL_CPARDISO maxfct:     %d \n", mat_mkl_cpardiso->maxfct);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer,"MKL_CPARDISO mnum:     %d \n", mat_mkl_cpardiso->mnum);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer,"MKL_CPARDISO mtype:     %d \n", mat_mkl_cpardiso->mtype);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer,"MKL_CPARDISO n:     %d \n", mat_mkl_cpardiso->n);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer,"MKL_CPARDISO nrhs:     %d \n", mat_mkl_cpardiso->nrhs);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer,"MKL_CPARDISO msglvl:     %d \n", mat_mkl_cpardiso->msglvl);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

PetscErrorCode MatGetInfo_MKL_CPARDISO(Mat A, MatInfoType flag, MatInfo *info)
{
  Mat_MKL_CPARDISO *mat_mkl_cpardiso=(Mat_MKL_CPARDISO*)A->data;

  PetscFunctionBegin;
  info->block_size        = 1.0;
  info->nz_allocated      = mat_mkl_cpardiso->nz + 0.0;
  info->nz_unneeded       = 0.0;
  info->assemblies        = 0.0;
  info->mallocs           = 0.0;
  info->memory            = 0.0;
  info->fill_ratio_given  = 0;
  info->fill_ratio_needed = 0;
  info->factor_mallocs    = 0;
  PetscFunctionReturn(0);
}

PetscErrorCode MatMkl_CPardisoSetCntl_MKL_CPARDISO(Mat F,PetscInt icntl,PetscInt ival)
{
  Mat_MKL_CPARDISO *mat_mkl_cpardiso=(Mat_MKL_CPARDISO*)F->data;

  PetscFunctionBegin;
  if (icntl <= 64) {
    mat_mkl_cpardiso->iparm[icntl - 1] = ival;
  } else {
    if (icntl == 65) mkl_set_num_threads((int)ival);
    else if (icntl == 66) mat_mkl_cpardiso->maxfct = ival;
    else if (icntl == 67) mat_mkl_cpardiso->mnum = ival;
    else if (icntl == 68) mat_mkl_cpardiso->msglvl = ival;
    else if (icntl == 69) mat_mkl_cpardiso->mtype = ival;
  }
  PetscFunctionReturn(0);
}

/*@
  MatMkl_CPardisoSetCntl - Set Mkl_Pardiso parameters

   Logically Collective on Mat

   Input Parameters:
+  F - the factored matrix obtained by calling MatGetFactor()
.  icntl - index of Mkl_Pardiso parameter
-  ival - value of Mkl_Pardiso parameter

  Options Database:
.   -mat_mkl_cpardiso_<icntl> <ival>

   Level: Intermediate

   Notes:
    This routine cannot be used if you are solving the linear system with TS, SNES, or KSP, only if you directly call MatGetFactor() so use the options
          database approach when working with TS, SNES, or KSP.

   References:
.      Mkl_Pardiso Users' Guide

.seealso: MatGetFactor()
@*/
PetscErrorCode MatMkl_CPardisoSetCntl(Mat F,PetscInt icntl,PetscInt ival)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscTryMethod(F,"MatMkl_CPardisoSetCntl_C",(Mat,PetscInt,PetscInt),(F,icntl,ival));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatFactorGetSolverType_mkl_cpardiso(Mat A, MatSolverType *type)
{
  PetscFunctionBegin;
  *type = MATSOLVERMKL_CPARDISO;
  PetscFunctionReturn(0);
}

/* MatGetFactor for MPI AIJ matrices */
static PetscErrorCode MatGetFactor_mpiaij_mkl_cpardiso(Mat A,MatFactorType ftype,Mat *F)
{
  Mat              B;
  PetscErrorCode   ierr;
  Mat_MKL_CPARDISO *mat_mkl_cpardiso;
  PetscBool        isSeqAIJ,isMPIBAIJ,isMPISBAIJ;

  PetscFunctionBegin;
  /* Create the factorization matrix */

  ierr = PetscObjectTypeCompare((PetscObject)A,MATSEQAIJ,&isSeqAIJ);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject)A,MATMPIBAIJ,&isMPIBAIJ);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject)A,MATMPISBAIJ,&isMPISBAIJ);CHKERRQ(ierr);
  ierr = MatCreate(PetscObjectComm((PetscObject)A),&B);CHKERRQ(ierr);
  ierr = MatSetSizes(B,A->rmap->n,A->cmap->n,A->rmap->N,A->cmap->N);CHKERRQ(ierr);
  ierr = PetscStrallocpy("mkl_cpardiso",&((PetscObject)B)->type_name);CHKERRQ(ierr);
  ierr = MatSetUp(B);CHKERRQ(ierr);

  ierr = PetscNewLog(B,&mat_mkl_cpardiso);CHKERRQ(ierr);

  if (isSeqAIJ) mat_mkl_cpardiso->ConvertToTriples = MatCopy_seqaij_seqaij_MKL_CPARDISO;
  else if (isMPIBAIJ) mat_mkl_cpardiso->ConvertToTriples = MatConvertToTriples_mpibaij_mpibaij_MKL_CPARDISO;
  else if (isMPISBAIJ) mat_mkl_cpardiso->ConvertToTriples = MatConvertToTriples_mpisbaij_mpisbaij_MKL_CPARDISO;
  else          mat_mkl_cpardiso->ConvertToTriples = MatConvertToTriples_mpiaij_mpiaij_MKL_CPARDISO;

  if (ftype == MAT_FACTOR_LU) B->ops->lufactorsymbolic = MatLUFactorSymbolic_AIJMKL_CPARDISO;
  else B->ops->choleskyfactorsymbolic = MatCholeskyFactorSymbolic_AIJMKL_CPARDISO;
  B->ops->destroy = MatDestroy_MKL_CPARDISO;

  B->ops->view    = MatView_MKL_CPARDISO;
  B->ops->getinfo = MatGetInfo_MKL_CPARDISO;

  B->factortype   = ftype;
  B->assembled    = PETSC_TRUE;           /* required by -ksp_view */

  B->data = mat_mkl_cpardiso;

  /* set solvertype */
  ierr = PetscFree(B->solvertype);CHKERRQ(ierr);
  ierr = PetscStrallocpy(MATSOLVERMKL_CPARDISO,&B->solvertype);CHKERRQ(ierr);

  ierr = PetscObjectComposeFunction((PetscObject)B,"MatFactorGetSolverType_C",MatFactorGetSolverType_mkl_cpardiso);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)B,"MatMkl_CPardisoSetCntl_C",MatMkl_CPardisoSetCntl_MKL_CPARDISO);CHKERRQ(ierr);
  ierr = PetscInitialize_MKL_CPARDISO(A, mat_mkl_cpardiso);CHKERRQ(ierr);

  *F = B;
  PetscFunctionReturn(0);
}

PETSC_EXTERN PetscErrorCode MatSolverTypeRegister_MKL_CPardiso(void)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatSolverTypeRegister(MATSOLVERMKL_CPARDISO,MATMPIAIJ,MAT_FACTOR_LU,MatGetFactor_mpiaij_mkl_cpardiso);CHKERRQ(ierr);
  ierr = MatSolverTypeRegister(MATSOLVERMKL_CPARDISO,MATSEQAIJ,MAT_FACTOR_LU,MatGetFactor_mpiaij_mkl_cpardiso);CHKERRQ(ierr);
  ierr = MatSolverTypeRegister(MATSOLVERMKL_CPARDISO,MATMPIBAIJ,MAT_FACTOR_LU,MatGetFactor_mpiaij_mkl_cpardiso);CHKERRQ(ierr);
  ierr = MatSolverTypeRegister(MATSOLVERMKL_CPARDISO,MATMPISBAIJ,MAT_FACTOR_CHOLESKY,MatGetFactor_mpiaij_mkl_cpardiso);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
