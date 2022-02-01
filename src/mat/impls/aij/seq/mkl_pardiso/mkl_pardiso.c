#include <../src/mat/impls/aij/seq/aij.h>        /*I "petscmat.h" I*/
#include <../src/mat/impls/sbaij/seq/sbaij.h>
#include <../src/mat/impls/dense/seq/dense.h>

#if defined(PETSC_HAVE_MKL_INTEL_ILP64)
#define MKL_ILP64
#endif
#include <mkl_pardiso.h>

PETSC_EXTERN void PetscSetMKL_PARDISOThreads(int);

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
 #if defined(PETSC_HAVE_MKL_INTEL_ILP64)
  #define INT_TYPE long long int
  #define MKL_PARDISO pardiso
  #define MKL_PARDISO_INIT pardisoinit
 #else
  /* this is the case where the MKL BLAS/LAPACK are 32 bit integers but the 64 bit integer version of
     of Pardiso code is used; hence the need for the 64 below*/
  #define INT_TYPE long long int
  #define MKL_PARDISO pardiso_64
  #define MKL_PARDISO_INIT pardiso_64init
void pardiso_64init(void *pt, INT_TYPE *mtype, INT_TYPE iparm [])
{
  int iparm_copy[IPARM_SIZE], mtype_copy, i;

  mtype_copy = *mtype;
  pardisoinit(pt, &mtype_copy, iparm_copy);
  for (i=0; i<IPARM_SIZE; i++) iparm[i] = iparm_copy[i];
}
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

  PetscBool    needsym;
  PetscBool    freeaij;

  /* Schur complement */
  PetscScalar  *schur;
  PetscInt     schur_size;
  PetscInt     *schur_idxs;
  PetscScalar  *schur_work;
  PetscBLASInt schur_work_size;
  PetscBool    solve_interior;

  /* True if mkl_pardiso function have been used.*/
  PetscBool CleanUp;

  /* Conversion to a format suitable for MKL */
  PetscErrorCode (*Convert)(Mat, PetscBool, MatReuse, PetscBool*, INT_TYPE*, INT_TYPE**, INT_TYPE**, PetscScalar**);
} Mat_MKL_PARDISO;

PetscErrorCode MatMKLPardiso_Convert_seqsbaij(Mat A,PetscBool sym,MatReuse reuse,PetscBool *free,INT_TYPE *nnz,INT_TYPE **r,INT_TYPE **c,PetscScalar **v)
{
  Mat_SeqSBAIJ   *aa = (Mat_SeqSBAIJ*)A->data;
  PetscInt       bs  = A->rmap->bs,i;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!sym) SETERRQ(PetscObjectComm((PetscObject)A),PETSC_ERR_PLIB,"This should not happen");
  *v      = aa->a;
  if (bs == 1) { /* already in the correct format */
    /* though PetscInt and INT_TYPE are of the same size since they are defined differently the Intel compiler requires a cast */
    *r    = (INT_TYPE*)aa->i;
    *c    = (INT_TYPE*)aa->j;
    *nnz  = (INT_TYPE)aa->nz;
    *free = PETSC_FALSE;
  } else if (reuse == MAT_INITIAL_MATRIX) {
    PetscInt m = A->rmap->n,nz = aa->nz;
    PetscInt *row,*col;
    ierr = PetscMalloc2(m+1,&row,nz,&col);CHKERRQ(ierr);
    for (i=0; i<m+1; i++) {
      row[i] = aa->i[i]+1;
    }
    for (i=0; i<nz; i++) {
      col[i] = aa->j[i]+1;
    }
    *r    = (INT_TYPE*)row;
    *c    = (INT_TYPE*)col;
    *nnz  = (INT_TYPE)nz;
    *free = PETSC_TRUE;
  }
  PetscFunctionReturn(0);
}

PetscErrorCode MatMKLPardiso_Convert_seqbaij(Mat A,PetscBool sym,MatReuse reuse,PetscBool *free,INT_TYPE *nnz,INT_TYPE **r,INT_TYPE **c,PetscScalar **v)
{
  Mat_SeqBAIJ    *aa = (Mat_SeqBAIJ*)A->data;
  PetscInt       bs  = A->rmap->bs,i;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!sym) {
    *v      = aa->a;
    if (bs == 1) { /* already in the correct format */
      /* though PetscInt and INT_TYPE are of the same size since they are defined differently the Intel compiler requires a cast */
      *r    = (INT_TYPE*)aa->i;
      *c    = (INT_TYPE*)aa->j;
      *nnz  = (INT_TYPE)aa->nz;
      *free = PETSC_FALSE;
      PetscFunctionReturn(0);
    } else if (reuse == MAT_INITIAL_MATRIX) {
      PetscInt m = A->rmap->n,nz = aa->nz;
      PetscInt *row,*col;
      ierr = PetscMalloc2(m+1,&row,nz,&col);CHKERRQ(ierr);
      for (i=0; i<m+1; i++) {
        row[i] = aa->i[i]+1;
      }
      for (i=0; i<nz; i++) {
        col[i] = aa->j[i]+1;
      }
      *r    = (INT_TYPE*)row;
      *c    = (INT_TYPE*)col;
      *nnz  = (INT_TYPE)nz;
    }
    *free = PETSC_TRUE;
  } else {
    SETERRQ(PetscObjectComm((PetscObject)A),PETSC_ERR_PLIB,"This should not happen");
  }
  PetscFunctionReturn(0);
}

PetscErrorCode MatMKLPardiso_Convert_seqaij(Mat A,PetscBool sym,MatReuse reuse,PetscBool *free,INT_TYPE *nnz,INT_TYPE **r,INT_TYPE **c,PetscScalar **v)
{
  Mat_SeqAIJ     *aa = (Mat_SeqAIJ*)A->data;
  PetscScalar    *aav;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatSeqAIJGetArrayRead(A,(const PetscScalar**)&aav);CHKERRQ(ierr);
  if (!sym) { /* already in the correct format */
    *v    = aav;
    *r    = (INT_TYPE*)aa->i;
    *c    = (INT_TYPE*)aa->j;
    *nnz  = (INT_TYPE)aa->nz;
    *free = PETSC_FALSE;
  } else if (reuse == MAT_INITIAL_MATRIX) { /* need to get the triangular part */
    PetscScalar *vals,*vv;
    PetscInt    *row,*col,*jj;
    PetscInt    m = A->rmap->n,nz,i;

    nz = 0;
    for (i=0; i<m; i++) nz += aa->i[i+1] - aa->diag[i];
    ierr = PetscMalloc2(m+1,&row,nz,&col);CHKERRQ(ierr);
    ierr = PetscMalloc1(nz,&vals);CHKERRQ(ierr);
    jj = col;
    vv = vals;

    row[0] = 0;
    for (i=0; i<m; i++) {
      PetscInt    *aj = aa->j + aa->diag[i];
      PetscScalar *av = aav + aa->diag[i];
      PetscInt    rl  = aa->i[i+1] - aa->diag[i],j;

      for (j=0; j<rl; j++) {
        *jj = *aj; jj++; aj++;
        *vv = *av; vv++; av++;
      }
      row[i+1] = row[i] + rl;
    }
    *v    = vals;
    *r    = (INT_TYPE*)row;
    *c    = (INT_TYPE*)col;
    *nnz  = (INT_TYPE)nz;
    *free = PETSC_TRUE;
  } else {
    PetscScalar *vv;
    PetscInt    m = A->rmap->n,i;

    vv = *v;
    for (i=0; i<m; i++) {
      PetscScalar *av = aav + aa->diag[i];
      PetscInt    rl  = aa->i[i+1] - aa->diag[i],j;
      for (j=0; j<rl; j++) {
        *vv = *av; vv++; av++;
      }
    }
    *free = PETSC_TRUE;
  }
  ierr = MatSeqAIJRestoreArrayRead(A,(const PetscScalar**)&aav);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatMKLPardisoSolveSchur_Private(Mat F, PetscScalar *B, PetscScalar *X)
{
  Mat_MKL_PARDISO      *mpardiso = (Mat_MKL_PARDISO*)F->data;
  Mat                  S,Xmat,Bmat;
  MatFactorSchurStatus schurstatus;
  PetscErrorCode       ierr;

  PetscFunctionBegin;
  ierr = MatFactorGetSchurComplement(F,&S,&schurstatus);CHKERRQ(ierr);
  if (X == B && schurstatus == MAT_FACTOR_SCHUR_INVERTED) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"X and B cannot point to the same address");
  ierr = MatCreateSeqDense(PETSC_COMM_SELF,mpardiso->schur_size,mpardiso->nrhs,B,&Bmat);CHKERRQ(ierr);
  ierr = MatCreateSeqDense(PETSC_COMM_SELF,mpardiso->schur_size,mpardiso->nrhs,X,&Xmat);CHKERRQ(ierr);
  ierr = MatSetType(Bmat,((PetscObject)S)->type_name);CHKERRQ(ierr);
  ierr = MatSetType(Xmat,((PetscObject)S)->type_name);CHKERRQ(ierr);
#if defined(PETSC_HAVE_VIENNACL) || defined(PETSC_HAVE_CUDA)
  ierr = MatBindToCPU(Xmat,S->boundtocpu);CHKERRQ(ierr);
  ierr = MatBindToCPU(Bmat,S->boundtocpu);CHKERRQ(ierr);
#endif

#if defined(PETSC_USE_COMPLEX)
  if (mpardiso->iparm[12-1] == 1) SETERRQ(PetscObjectComm((PetscObject)F),PETSC_ERR_SUP,"Hermitian solve not implemented yet");
#endif

  switch (schurstatus) {
  case MAT_FACTOR_SCHUR_FACTORED:
    if (!mpardiso->iparm[12-1]) {
      ierr = MatMatSolve(S,Bmat,Xmat);CHKERRQ(ierr);
    } else { /* transpose solve */
      ierr = MatMatSolveTranspose(S,Bmat,Xmat);CHKERRQ(ierr);
    }
    break;
  case MAT_FACTOR_SCHUR_INVERTED:
    ierr = MatProductCreateWithMat(S,Bmat,NULL,Xmat);CHKERRQ(ierr);
    if (!mpardiso->iparm[12-1]) {
      ierr = MatProductSetType(Xmat,MATPRODUCT_AB);CHKERRQ(ierr);
    } else { /* transpose solve */
      ierr = MatProductSetType(Xmat,MATPRODUCT_AtB);CHKERRQ(ierr);
    }
    ierr = MatProductSetFromOptions(Xmat);CHKERRQ(ierr);
    ierr = MatProductSymbolic(Xmat);CHKERRQ(ierr);
    ierr = MatProductNumeric(Xmat);CHKERRQ(ierr);
    ierr = MatProductClear(Xmat);CHKERRQ(ierr);
    break;
  default:
    SETERRQ1(PetscObjectComm((PetscObject)F),PETSC_ERR_SUP,"Unhandled MatFactorSchurStatus %" PetscInt_FMT,F->schur_status);
    break;
  }
  ierr = MatFactorRestoreSchurComplement(F,&S,schurstatus);CHKERRQ(ierr);
  ierr = MatDestroy(&Bmat);CHKERRQ(ierr);
  ierr = MatDestroy(&Xmat);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatFactorSetSchurIS_MKL_PARDISO(Mat F, IS is)
{
  Mat_MKL_PARDISO   *mpardiso = (Mat_MKL_PARDISO*)F->data;
  const PetscScalar *arr;
  const PetscInt    *idxs;
  PetscInt          size,i;
  PetscMPIInt       csize;
  PetscBool         sorted;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ierr = MPI_Comm_size(PetscObjectComm((PetscObject)F),&csize);CHKERRMPI(ierr);
  if (csize > 1) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"MKL_PARDISO parallel Schur complements not yet supported from PETSc");
  ierr = ISSorted(is,&sorted);CHKERRQ(ierr);
  if (!sorted) {
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"IS for MKL_PARDISO Schur complements needs to be sorted");
  }
  ierr = ISGetLocalSize(is,&size);CHKERRQ(ierr);
  ierr = PetscFree(mpardiso->schur_work);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(PetscMax(mpardiso->n,2*size),&mpardiso->schur_work_size);CHKERRQ(ierr);
  ierr = PetscMalloc1(mpardiso->schur_work_size,&mpardiso->schur_work);CHKERRQ(ierr);
  ierr = MatDestroy(&F->schur);CHKERRQ(ierr);
  ierr = MatCreateSeqDense(PETSC_COMM_SELF,size,size,NULL,&F->schur);CHKERRQ(ierr);
  ierr = MatDenseGetArrayRead(F->schur,&arr);CHKERRQ(ierr);
  mpardiso->schur      = (PetscScalar*)arr;
  mpardiso->schur_size = size;
  ierr = MatDenseRestoreArrayRead(F->schur,&arr);CHKERRQ(ierr);
  if (mpardiso->mtype == 2) {
    ierr = MatSetOption(F->schur,MAT_SPD,PETSC_TRUE);CHKERRQ(ierr);
  }

  ierr = PetscFree(mpardiso->schur_idxs);CHKERRQ(ierr);
  ierr = PetscMalloc1(size,&mpardiso->schur_idxs);CHKERRQ(ierr);
  ierr = PetscArrayzero(mpardiso->perm,mpardiso->n);CHKERRQ(ierr);
  ierr = ISGetIndices(is,&idxs);CHKERRQ(ierr);
  ierr = PetscArraycpy(mpardiso->schur_idxs,idxs,size);CHKERRQ(ierr);
  for (i=0;i<size;i++) mpardiso->perm[idxs[i]] = 1;
  ierr = ISRestoreIndices(is,&idxs);CHKERRQ(ierr);
  if (size) { /* turn on Schur switch if the set of indices is not empty */
    mpardiso->iparm[36-1] = 2;
  }
  PetscFunctionReturn(0);
}

PetscErrorCode MatDestroy_MKL_PARDISO(Mat A)
{
  Mat_MKL_PARDISO *mat_mkl_pardiso=(Mat_MKL_PARDISO*)A->data;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
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
      NULL,
      &mat_mkl_pardiso->nrhs,
      mat_mkl_pardiso->iparm,
      &mat_mkl_pardiso->msglvl,
      NULL,
      NULL,
      &mat_mkl_pardiso->err);
  }
  ierr = PetscFree(mat_mkl_pardiso->perm);CHKERRQ(ierr);
  ierr = PetscFree(mat_mkl_pardiso->schur_work);CHKERRQ(ierr);
  ierr = PetscFree(mat_mkl_pardiso->schur_idxs);CHKERRQ(ierr);
  if (mat_mkl_pardiso->freeaij) {
    ierr = PetscFree2(mat_mkl_pardiso->ia,mat_mkl_pardiso->ja);CHKERRQ(ierr);
    if (mat_mkl_pardiso->iparm[34] == 1) {
      ierr = PetscFree(mat_mkl_pardiso->a);CHKERRQ(ierr);
    }
  }
  ierr = PetscFree(A->data);CHKERRQ(ierr);

  /* clear composed functions */
  ierr = PetscObjectComposeFunction((PetscObject)A,"MatFactorGetSolverType_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)A,"MatFactorSetSchurIS_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)A,"MatMkl_PardisoSetCntl_C",NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatMKLPardisoScatterSchur_Private(Mat_MKL_PARDISO *mpardiso, PetscScalar *whole, PetscScalar *schur, PetscBool reduce)
{
  PetscFunctionBegin;
  if (reduce) { /* data given for the whole matrix */
    PetscInt i,m=0,p=0;
    for (i=0;i<mpardiso->nrhs;i++) {
      PetscInt j;
      for (j=0;j<mpardiso->schur_size;j++) {
        schur[p+j] = whole[m+mpardiso->schur_idxs[j]];
      }
      m += mpardiso->n;
      p += mpardiso->schur_size;
    }
  } else { /* from Schur to whole */
    PetscInt i,m=0,p=0;
    for (i=0;i<mpardiso->nrhs;i++) {
      PetscInt j;
      for (j=0;j<mpardiso->schur_size;j++) {
        whole[m+mpardiso->schur_idxs[j]] = schur[p+j];
      }
      m += mpardiso->n;
      p += mpardiso->schur_size;
    }
  }
  PetscFunctionReturn(0);
}

PetscErrorCode MatSolve_MKL_PARDISO(Mat A,Vec b,Vec x)
{
  Mat_MKL_PARDISO   *mat_mkl_pardiso=(Mat_MKL_PARDISO*)A->data;
  PetscErrorCode    ierr;
  PetscScalar       *xarray;
  const PetscScalar *barray;

  PetscFunctionBegin;
  mat_mkl_pardiso->nrhs = 1;
  ierr = VecGetArrayWrite(x,&xarray);CHKERRQ(ierr);
  ierr = VecGetArrayRead(b,&barray);CHKERRQ(ierr);

  if (!mat_mkl_pardiso->schur) mat_mkl_pardiso->phase = JOB_SOLVE_ITERATIVE_REFINEMENT;
  else mat_mkl_pardiso->phase = JOB_SOLVE_FORWARD_SUBSTITUTION;

  if (barray == xarray) { /* if the two vectors share the same memory */
    PetscScalar *work;
    if (!mat_mkl_pardiso->schur_work) {
      ierr = PetscMalloc1(mat_mkl_pardiso->n,&work);CHKERRQ(ierr);
    } else {
      work = mat_mkl_pardiso->schur_work;
    }
    mat_mkl_pardiso->iparm[6-1] = 1;
    MKL_PARDISO (mat_mkl_pardiso->pt,
      &mat_mkl_pardiso->maxfct,
      &mat_mkl_pardiso->mnum,
      &mat_mkl_pardiso->mtype,
      &mat_mkl_pardiso->phase,
      &mat_mkl_pardiso->n,
      mat_mkl_pardiso->a,
      mat_mkl_pardiso->ia,
      mat_mkl_pardiso->ja,
      NULL,
      &mat_mkl_pardiso->nrhs,
      mat_mkl_pardiso->iparm,
      &mat_mkl_pardiso->msglvl,
      (void*)xarray,
      (void*)work,
      &mat_mkl_pardiso->err);
    if (!mat_mkl_pardiso->schur_work) {
      ierr = PetscFree(work);CHKERRQ(ierr);
    }
  } else {
    mat_mkl_pardiso->iparm[6-1] = 0;
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
  }
  ierr = VecRestoreArrayRead(b,&barray);CHKERRQ(ierr);

  if (mat_mkl_pardiso->err < 0) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Error reported by MKL_PARDISO: err=%d. Please check manual",mat_mkl_pardiso->err);

  if (mat_mkl_pardiso->schur) { /* solve Schur complement and expand solution */
    if (!mat_mkl_pardiso->solve_interior) {
      PetscInt shift = mat_mkl_pardiso->schur_size;

      ierr = MatFactorFactorizeSchurComplement(A);CHKERRQ(ierr);
      /* if inverted, uses BLAS *MM subroutines, otherwise LAPACK *TRS */
      if (A->schur_status != MAT_FACTOR_SCHUR_INVERTED) shift = 0;

      /* solve Schur complement */
      ierr = MatMKLPardisoScatterSchur_Private(mat_mkl_pardiso,xarray,mat_mkl_pardiso->schur_work,PETSC_TRUE);CHKERRQ(ierr);
      ierr = MatMKLPardisoSolveSchur_Private(A,mat_mkl_pardiso->schur_work,mat_mkl_pardiso->schur_work+shift);CHKERRQ(ierr);
      ierr = MatMKLPardisoScatterSchur_Private(mat_mkl_pardiso,xarray,mat_mkl_pardiso->schur_work+shift,PETSC_FALSE);CHKERRQ(ierr);
    } else { /* if we are solving for the interior problem, any value in barray[schur] forward-substituted to xarray[schur] will be neglected */
      PetscInt i;
      for (i=0;i<mat_mkl_pardiso->schur_size;i++) {
        xarray[mat_mkl_pardiso->schur_idxs[i]] = 0.;
      }
    }

    /* expansion phase */
    mat_mkl_pardiso->iparm[6-1] = 1;
    mat_mkl_pardiso->phase = JOB_SOLVE_BACKWARD_SUBSTITUTION;
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
      (void*)xarray,
      (void*)mat_mkl_pardiso->schur_work, /* according to the specs, the solution vector is always used */
      &mat_mkl_pardiso->err);

    if (mat_mkl_pardiso->err < 0) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Error reported by MKL_PARDISO: err=%d. Please check manual",mat_mkl_pardiso->err);
    mat_mkl_pardiso->iparm[6-1] = 0;
  }
  ierr = VecRestoreArrayWrite(x,&xarray);CHKERRQ(ierr);
  mat_mkl_pardiso->CleanUp = PETSC_TRUE;
  PetscFunctionReturn(0);
}

PetscErrorCode MatSolveTranspose_MKL_PARDISO(Mat A,Vec b,Vec x)
{
  Mat_MKL_PARDISO *mat_mkl_pardiso=(Mat_MKL_PARDISO*)A->data;
  PetscInt        oiparm12;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  oiparm12 = mat_mkl_pardiso->iparm[12 - 1];
  mat_mkl_pardiso->iparm[12 - 1] = 2;
  ierr = MatSolve_MKL_PARDISO(A,b,x);CHKERRQ(ierr);
  mat_mkl_pardiso->iparm[12 - 1] = oiparm12;
  PetscFunctionReturn(0);
}

PetscErrorCode MatMatSolve_MKL_PARDISO(Mat A,Mat B,Mat X)
{
  Mat_MKL_PARDISO   *mat_mkl_pardiso=(Mat_MKL_PARDISO*)(A)->data;
  PetscErrorCode    ierr;
  const PetscScalar *barray;
  PetscScalar       *xarray;
  PetscBool         flg;

  PetscFunctionBegin;
  ierr = PetscObjectBaseTypeCompare((PetscObject)B,MATSEQDENSE,&flg);CHKERRQ(ierr);
  if (!flg) SETERRQ(PetscObjectComm((PetscObject)A),PETSC_ERR_ARG_WRONG,"Matrix B must be MATSEQDENSE matrix");
  if (X != B) {
    ierr = PetscObjectBaseTypeCompare((PetscObject)X,MATSEQDENSE,&flg);CHKERRQ(ierr);
    if (!flg) SETERRQ(PetscObjectComm((PetscObject)A),PETSC_ERR_ARG_WRONG,"Matrix X must be MATSEQDENSE matrix");
  }

  ierr = MatGetSize(B,NULL,(PetscInt*)&mat_mkl_pardiso->nrhs);CHKERRQ(ierr);

  if (mat_mkl_pardiso->nrhs > 0) {
    ierr = MatDenseGetArrayRead(B,&barray);CHKERRQ(ierr);
    ierr = MatDenseGetArrayWrite(X,&xarray);CHKERRQ(ierr);

    if (barray == xarray) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"B and X cannot share the same memory location");
    if (!mat_mkl_pardiso->schur) mat_mkl_pardiso->phase = JOB_SOLVE_ITERATIVE_REFINEMENT;
    else mat_mkl_pardiso->phase = JOB_SOLVE_FORWARD_SUBSTITUTION;

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
    if (mat_mkl_pardiso->err < 0) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Error reported by MKL_PARDISO: err=%d. Please check manual",mat_mkl_pardiso->err);

    ierr = MatDenseRestoreArrayRead(B,&barray);CHKERRQ(ierr);
    if (mat_mkl_pardiso->schur) { /* solve Schur complement and expand solution */
      PetscScalar *o_schur_work = NULL;

      /* solve Schur complement */
      if (!mat_mkl_pardiso->solve_interior) {
        PetscInt shift = mat_mkl_pardiso->schur_size*mat_mkl_pardiso->nrhs,scale;
        PetscInt mem = mat_mkl_pardiso->n*mat_mkl_pardiso->nrhs;

        ierr = MatFactorFactorizeSchurComplement(A);CHKERRQ(ierr);
        /* allocate extra memory if it is needed */
        scale = 1;
        if (A->schur_status == MAT_FACTOR_SCHUR_INVERTED) scale = 2;
        mem *= scale;
        if (mem > mat_mkl_pardiso->schur_work_size) {
          o_schur_work = mat_mkl_pardiso->schur_work;
          ierr = PetscMalloc1(mem,&mat_mkl_pardiso->schur_work);CHKERRQ(ierr);
        }
        /* if inverted, uses BLAS *MM subroutines, otherwise LAPACK *TRS */
        if (A->schur_status != MAT_FACTOR_SCHUR_INVERTED) shift = 0;
        ierr = MatMKLPardisoScatterSchur_Private(mat_mkl_pardiso,xarray,mat_mkl_pardiso->schur_work,PETSC_TRUE);CHKERRQ(ierr);
        ierr = MatMKLPardisoSolveSchur_Private(A,mat_mkl_pardiso->schur_work,mat_mkl_pardiso->schur_work+shift);CHKERRQ(ierr);
        ierr = MatMKLPardisoScatterSchur_Private(mat_mkl_pardiso,xarray,mat_mkl_pardiso->schur_work+shift,PETSC_FALSE);CHKERRQ(ierr);
      } else { /* if we are solving for the interior problem, any value in barray[schur,n] forward-substituted to xarray[schur,n] will be neglected */
        PetscInt i,n,m=0;
        for (n=0;n<mat_mkl_pardiso->nrhs;n++) {
          for (i=0;i<mat_mkl_pardiso->schur_size;i++) {
            xarray[mat_mkl_pardiso->schur_idxs[i]+m] = 0.;
          }
          m += mat_mkl_pardiso->n;
        }
      }

      /* expansion phase */
      mat_mkl_pardiso->iparm[6-1] = 1;
      mat_mkl_pardiso->phase = JOB_SOLVE_BACKWARD_SUBSTITUTION;
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
        (void*)xarray,
        (void*)mat_mkl_pardiso->schur_work, /* according to the specs, the solution vector is always used */
        &mat_mkl_pardiso->err);
      if (o_schur_work) { /* restore original schur_work (minimal size) */
        ierr = PetscFree(mat_mkl_pardiso->schur_work);CHKERRQ(ierr);
        mat_mkl_pardiso->schur_work = o_schur_work;
      }
      if (mat_mkl_pardiso->err < 0) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Error reported by MKL_PARDISO: err=%d. Please check manual",mat_mkl_pardiso->err);
      mat_mkl_pardiso->iparm[6-1] = 0;
    }
    ierr = MatDenseRestoreArrayWrite(X,&xarray);CHKERRQ(ierr);
  }
  mat_mkl_pardiso->CleanUp = PETSC_TRUE;
  PetscFunctionReturn(0);
}

PetscErrorCode MatFactorNumeric_MKL_PARDISO(Mat F,Mat A,const MatFactorInfo *info)
{
  Mat_MKL_PARDISO *mat_mkl_pardiso=(Mat_MKL_PARDISO*)(F)->data;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  mat_mkl_pardiso->matstruc = SAME_NONZERO_PATTERN;
  ierr = (*mat_mkl_pardiso->Convert)(A,mat_mkl_pardiso->needsym,MAT_REUSE_MATRIX,&mat_mkl_pardiso->freeaij,&mat_mkl_pardiso->nz,&mat_mkl_pardiso->ia,&mat_mkl_pardiso->ja,(PetscScalar**)&mat_mkl_pardiso->a);CHKERRQ(ierr);

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
    (void*)mat_mkl_pardiso->schur,
    &mat_mkl_pardiso->err);
  if (mat_mkl_pardiso->err < 0) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Error reported by MKL_PARDISO: err=%d. Please check manual",mat_mkl_pardiso->err);

  /* report flops */
  if (mat_mkl_pardiso->iparm[18] > 0) {
    ierr = PetscLogFlops(PetscPowRealInt(10.,6)*mat_mkl_pardiso->iparm[18]);CHKERRQ(ierr);
  }

  if (F->schur) { /* schur output from pardiso is in row major format */
#if defined(PETSC_HAVE_CUDA)
    F->schur->offloadmask = PETSC_OFFLOAD_CPU;
#endif
    ierr = MatFactorRestoreSchurComplement(F,NULL,MAT_FACTOR_SCHUR_UNFACTORED);CHKERRQ(ierr);
    ierr = MatTranspose(F->schur,MAT_INPLACE_MATRIX,&F->schur);CHKERRQ(ierr);
  }
  mat_mkl_pardiso->matstruc = SAME_NONZERO_PATTERN;
  mat_mkl_pardiso->CleanUp  = PETSC_TRUE;
  PetscFunctionReturn(0);
}

PetscErrorCode PetscSetMKL_PARDISOFromOptions(Mat F, Mat A)
{
  Mat_MKL_PARDISO     *mat_mkl_pardiso = (Mat_MKL_PARDISO*)F->data;
  PetscErrorCode      ierr;
  PetscInt            icntl,bs,threads=1;
  PetscBool           flg;

  PetscFunctionBegin;
  ierr = PetscOptionsBegin(PetscObjectComm((PetscObject)A),((PetscObject)A)->prefix,"MKL_PARDISO Options","Mat");CHKERRQ(ierr);

  ierr = PetscOptionsInt("-mat_mkl_pardiso_65","Suggested number of threads to use within PARDISO","None",threads,&threads,&flg);CHKERRQ(ierr);
  if (flg) PetscSetMKL_PARDISOThreads((int)threads);

  ierr = PetscOptionsInt("-mat_mkl_pardiso_66","Maximum number of factors with identical sparsity structure that must be kept in memory at the same time","None",mat_mkl_pardiso->maxfct,&icntl,&flg);CHKERRQ(ierr);
  if (flg) mat_mkl_pardiso->maxfct = icntl;

  ierr = PetscOptionsInt("-mat_mkl_pardiso_67","Indicates the actual matrix for the solution phase","None",mat_mkl_pardiso->mnum,&icntl,&flg);CHKERRQ(ierr);
  if (flg) mat_mkl_pardiso->mnum = icntl;

  ierr = PetscOptionsInt("-mat_mkl_pardiso_68","Message level information","None",mat_mkl_pardiso->msglvl,&icntl,&flg);CHKERRQ(ierr);
  if (flg) mat_mkl_pardiso->msglvl = icntl;

  ierr = PetscOptionsInt("-mat_mkl_pardiso_69","Defines the matrix type","None",mat_mkl_pardiso->mtype,&icntl,&flg);CHKERRQ(ierr);
  if (flg) {
    void *pt[IPARM_SIZE];
    mat_mkl_pardiso->mtype = icntl;
    icntl = mat_mkl_pardiso->iparm[34];
    bs = mat_mkl_pardiso->iparm[36];
    MKL_PARDISO_INIT(pt, &mat_mkl_pardiso->mtype, mat_mkl_pardiso->iparm);
#if defined(PETSC_USE_REAL_SINGLE)
    mat_mkl_pardiso->iparm[27] = 1;
#else
    mat_mkl_pardiso->iparm[27] = 0;
#endif
    mat_mkl_pardiso->iparm[34] = icntl;
    mat_mkl_pardiso->iparm[36] = bs;
  }

  ierr = PetscOptionsInt("-mat_mkl_pardiso_1","Use default values (if 0)","None",mat_mkl_pardiso->iparm[0],&icntl,&flg);CHKERRQ(ierr);
  if (flg) mat_mkl_pardiso->iparm[0] = icntl;

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

  ierr = PetscOptionsInt("-mat_mkl_pardiso_19","Report number of floating point operations (0 to disable)","None",mat_mkl_pardiso->iparm[18],&icntl,&flg);CHKERRQ(ierr);
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
  PetscOptionsEnd();
  PetscFunctionReturn(0);
}

PetscErrorCode MatFactorMKL_PARDISOInitialize_Private(Mat A, MatFactorType ftype, Mat_MKL_PARDISO *mat_mkl_pardiso)
{
  PetscErrorCode ierr;
  PetscInt       i,bs;
  PetscBool      match;

  PetscFunctionBegin;
  for (i=0; i<IPARM_SIZE; i++) mat_mkl_pardiso->iparm[i] = 0;
  for (i=0; i<IPARM_SIZE; i++) mat_mkl_pardiso->pt[i] = 0;
#if defined(PETSC_USE_REAL_SINGLE)
  mat_mkl_pardiso->iparm[27] = 1;
#else
  mat_mkl_pardiso->iparm[27] = 0;
#endif
  /* Default options for both sym and unsym */
  mat_mkl_pardiso->iparm[ 0] =  1; /* Solver default parameters overriden with provided by iparm */
  mat_mkl_pardiso->iparm[ 1] =  2; /* Metis reordering */
  mat_mkl_pardiso->iparm[ 5] =  0; /* Write solution into x */
  mat_mkl_pardiso->iparm[ 7] =  0; /* Max number of iterative refinement steps */
  mat_mkl_pardiso->iparm[17] = -1; /* Output: Number of nonzeros in the factor LU */
  mat_mkl_pardiso->iparm[18] = -1; /* Output: Mflops for LU factorization */
#if 0
  mat_mkl_pardiso->iparm[23] =  1; /* Parallel factorization control*/
#endif
  ierr = PetscObjectTypeCompareAny((PetscObject)A,&match,MATSEQBAIJ,MATSEQSBAIJ,"");CHKERRQ(ierr);
  ierr = MatGetBlockSize(A,&bs);CHKERRQ(ierr);
  if (!match || bs == 1) {
    mat_mkl_pardiso->iparm[34] = 1; /* Cluster Sparse Solver use C-style indexing for ia and ja arrays */
    mat_mkl_pardiso->n         = A->rmap->N;
  } else {
    mat_mkl_pardiso->iparm[34] = 0; /* Cluster Sparse Solver use Fortran-style indexing for ia and ja arrays */
    mat_mkl_pardiso->iparm[36] = bs;
    mat_mkl_pardiso->n         = A->rmap->N/bs;
  }
  mat_mkl_pardiso->iparm[39] =  0; /* Input: matrix/rhs/solution stored on rank-0 */

  mat_mkl_pardiso->CleanUp   = PETSC_FALSE;
  mat_mkl_pardiso->maxfct    = 1; /* Maximum number of numerical factorizations. */
  mat_mkl_pardiso->mnum      = 1; /* Which factorization to use. */
  mat_mkl_pardiso->msglvl    = 0; /* 0: do not print 1: Print statistical information in file */
  mat_mkl_pardiso->phase     = -1;
  mat_mkl_pardiso->err       = 0;

  mat_mkl_pardiso->nrhs      = 1;
  mat_mkl_pardiso->err       = 0;
  mat_mkl_pardiso->phase     = -1;

  if (ftype == MAT_FACTOR_LU) {
    mat_mkl_pardiso->iparm[ 9] = 13; /* Perturb the pivot elements with 1E-13 */
    mat_mkl_pardiso->iparm[10] =  1; /* Use nonsymmetric permutation and scaling MPS */
    mat_mkl_pardiso->iparm[12] =  1; /* Switch on Maximum Weighted Matching algorithm (default for non-symmetric) */
  } else {
    mat_mkl_pardiso->iparm[ 9] = 8; /* Perturb the pivot elements with 1E-8 */
    mat_mkl_pardiso->iparm[10] = 0; /* Use nonsymmetric permutation and scaling MPS */
    mat_mkl_pardiso->iparm[12] = 1; /* Switch on Maximum Weighted Matching algorithm (default for non-symmetric) */
#if defined(PETSC_USE_DEBUG)
    mat_mkl_pardiso->iparm[26] = 1; /* Matrix checker */
#endif
  }
  ierr = PetscCalloc1(A->rmap->N*sizeof(INT_TYPE), &mat_mkl_pardiso->perm);CHKERRQ(ierr);
  mat_mkl_pardiso->schur_size = 0;
  PetscFunctionReturn(0);
}

PetscErrorCode MatFactorSymbolic_AIJMKL_PARDISO_Private(Mat F,Mat A,const MatFactorInfo *info)
{
  Mat_MKL_PARDISO *mat_mkl_pardiso = (Mat_MKL_PARDISO*)F->data;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  mat_mkl_pardiso->matstruc = DIFFERENT_NONZERO_PATTERN;
  ierr = PetscSetMKL_PARDISOFromOptions(F,A);CHKERRQ(ierr);
  /* throw away any previously computed structure */
  if (mat_mkl_pardiso->freeaij) {
    ierr = PetscFree2(mat_mkl_pardiso->ia,mat_mkl_pardiso->ja);CHKERRQ(ierr);
    if (mat_mkl_pardiso->iparm[34] == 1) {
      ierr = PetscFree(mat_mkl_pardiso->a);CHKERRQ(ierr);
    }
  }
  ierr = (*mat_mkl_pardiso->Convert)(A,mat_mkl_pardiso->needsym,MAT_INITIAL_MATRIX,&mat_mkl_pardiso->freeaij,&mat_mkl_pardiso->nz,&mat_mkl_pardiso->ia,&mat_mkl_pardiso->ja,(PetscScalar**)&mat_mkl_pardiso->a);CHKERRQ(ierr);
  if (mat_mkl_pardiso->iparm[34] == 1) mat_mkl_pardiso->n = A->rmap->N;
  else mat_mkl_pardiso->n = A->rmap->N/A->rmap->bs;

  mat_mkl_pardiso->phase = JOB_ANALYSIS;

  /* reset flops counting if requested */
  if (mat_mkl_pardiso->iparm[18]) mat_mkl_pardiso->iparm[18] = -1;

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
  if (mat_mkl_pardiso->err < 0) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Error reported by MKL_PARDISO: err=%d. Please check manual",mat_mkl_pardiso->err);

  mat_mkl_pardiso->CleanUp = PETSC_TRUE;

  if (F->factortype == MAT_FACTOR_LU) F->ops->lufactornumeric = MatFactorNumeric_MKL_PARDISO;
  else F->ops->choleskyfactornumeric = MatFactorNumeric_MKL_PARDISO;

  F->ops->solve           = MatSolve_MKL_PARDISO;
  F->ops->solvetranspose  = MatSolveTranspose_MKL_PARDISO;
  F->ops->matsolve        = MatMatSolve_MKL_PARDISO;
  PetscFunctionReturn(0);
}

PetscErrorCode MatLUFactorSymbolic_AIJMKL_PARDISO(Mat F,Mat A,IS r,IS c,const MatFactorInfo *info)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatFactorSymbolic_AIJMKL_PARDISO_Private(F, A, info);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#if !defined(PETSC_USE_COMPLEX)
PetscErrorCode MatGetInertia_MKL_PARDISO(Mat F,PetscInt *nneg,PetscInt *nzero,PetscInt *npos)
{
  Mat_MKL_PARDISO   *mat_mkl_pardiso=(Mat_MKL_PARDISO*)F->data;

  PetscFunctionBegin;
  if (nneg) *nneg = mat_mkl_pardiso->iparm[22];
  if (npos) *npos = mat_mkl_pardiso->iparm[21];
  if (nzero) *nzero = F->rmap->N - (mat_mkl_pardiso->iparm[22] + mat_mkl_pardiso->iparm[21]);
  PetscFunctionReturn(0);
}
#endif

PetscErrorCode MatCholeskyFactorSymbolic_AIJMKL_PARDISO(Mat F,Mat A,IS r,const MatFactorInfo *info)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatFactorSymbolic_AIJMKL_PARDISO_Private(F, A, info);CHKERRQ(ierr);
#if defined(PETSC_USE_COMPLEX)
  F->ops->getinertia = NULL;
#else
  F->ops->getinertia = MatGetInertia_MKL_PARDISO;
#endif
  PetscFunctionReturn(0);
}

PetscErrorCode MatView_MKL_PARDISO(Mat A, PetscViewer viewer)
{
  PetscErrorCode    ierr;
  PetscBool         iascii;
  PetscViewerFormat format;
  Mat_MKL_PARDISO   *mat_mkl_pardiso=(Mat_MKL_PARDISO*)A->data;
  PetscInt          i;

  PetscFunctionBegin;
  if (A->ops->solve != MatSolve_MKL_PARDISO) PetscFunctionReturn(0);

  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii);CHKERRQ(ierr);
  if (iascii) {
    ierr = PetscViewerGetFormat(viewer,&format);CHKERRQ(ierr);
    if (format == PETSC_VIEWER_ASCII_INFO) {
      ierr = PetscViewerASCIIPrintf(viewer,"MKL_PARDISO run parameters:\n");CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer,"MKL_PARDISO phase:             %d \n",mat_mkl_pardiso->phase);CHKERRQ(ierr);
      for (i=1; i<=64; i++) {
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

PetscErrorCode MatGetInfo_MKL_PARDISO(Mat A, MatInfoType flag, MatInfo *info)
{
  Mat_MKL_PARDISO *mat_mkl_pardiso = (Mat_MKL_PARDISO*)A->data;

  PetscFunctionBegin;
  info->block_size        = 1.0;
  info->nz_used           = mat_mkl_pardiso->iparm[17];
  info->nz_allocated      = mat_mkl_pardiso->iparm[17];
  info->nz_unneeded       = 0.0;
  info->assemblies        = 0.0;
  info->mallocs           = 0.0;
  info->memory            = 0.0;
  info->fill_ratio_given  = 0;
  info->fill_ratio_needed = 0;
  info->factor_mallocs    = 0;
  PetscFunctionReturn(0);
}

PetscErrorCode MatMkl_PardisoSetCntl_MKL_PARDISO(Mat F,PetscInt icntl,PetscInt ival)
{
  PetscInt        backup,bs;
  Mat_MKL_PARDISO *mat_mkl_pardiso = (Mat_MKL_PARDISO*)F->data;

  PetscFunctionBegin;
  if (icntl <= 64) {
    mat_mkl_pardiso->iparm[icntl - 1] = ival;
  } else {
    if (icntl == 65) PetscSetMKL_PARDISOThreads(ival);
    else if (icntl == 66) mat_mkl_pardiso->maxfct = ival;
    else if (icntl == 67) mat_mkl_pardiso->mnum = ival;
    else if (icntl == 68) mat_mkl_pardiso->msglvl = ival;
    else if (icntl == 69) {
      void *pt[IPARM_SIZE];
      backup = mat_mkl_pardiso->iparm[34];
      bs = mat_mkl_pardiso->iparm[36];
      mat_mkl_pardiso->mtype = ival;
      MKL_PARDISO_INIT(pt, &mat_mkl_pardiso->mtype, mat_mkl_pardiso->iparm);
#if defined(PETSC_USE_REAL_SINGLE)
      mat_mkl_pardiso->iparm[27] = 1;
#else
      mat_mkl_pardiso->iparm[27] = 0;
#endif
      mat_mkl_pardiso->iparm[34] = backup;
      mat_mkl_pardiso->iparm[36] = bs;
    } else if (icntl==70) mat_mkl_pardiso->solve_interior = (PetscBool)!!ival;
  }
  PetscFunctionReturn(0);
}

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

   References:
.      Mkl_Pardiso Users' Guide

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

  Use -pc_type lu -pc_factor_mat_solver_type mkl_pardiso to use this direct solver

  Options Database Keys:
+ -mat_mkl_pardiso_65 - Suggested number of threads to use within MKL_PARDISO
. -mat_mkl_pardiso_66 - Maximum number of factors with identical sparsity structure that must be kept in memory at the same time
. -mat_mkl_pardiso_67 - Indicates the actual matrix for the solution phase
. -mat_mkl_pardiso_68 - Message level information, use 1 to get detailed information on the solver options
. -mat_mkl_pardiso_69 - Defines the matrix type. IMPORTANT: When you set this flag, iparm parameters are going to be set to the default ones for the matrix type
. -mat_mkl_pardiso_1  - Use default values
. -mat_mkl_pardiso_2  - Fill-in reducing ordering for the input matrix
. -mat_mkl_pardiso_4  - Preconditioned CGS/CG
. -mat_mkl_pardiso_5  - User permutation
. -mat_mkl_pardiso_6  - Write solution on x
. -mat_mkl_pardiso_8  - Iterative refinement step
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

  Notes:
    Use -mat_mkl_pardiso_68 1 to display the number of threads the solver is using. MKL does not provide a way to directly access this
    information.

    For more information on the options check the MKL_Pardiso manual

.seealso: PCFactorSetMatSolverType(), MatSolverType

M*/
static PetscErrorCode MatFactorGetSolverType_mkl_pardiso(Mat A, MatSolverType *type)
{
  PetscFunctionBegin;
  *type = MATSOLVERMKL_PARDISO;
  PetscFunctionReturn(0);
}

PETSC_EXTERN PetscErrorCode MatGetFactor_aij_mkl_pardiso(Mat A,MatFactorType ftype,Mat *F)
{
  Mat             B;
  PetscErrorCode  ierr;
  Mat_MKL_PARDISO *mat_mkl_pardiso;
  PetscBool       isSeqAIJ,isSeqBAIJ,isSeqSBAIJ;

  PetscFunctionBegin;
  ierr = PetscObjectBaseTypeCompare((PetscObject)A,MATSEQAIJ,&isSeqAIJ);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject)A,MATSEQBAIJ,&isSeqBAIJ);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject)A,MATSEQSBAIJ,&isSeqSBAIJ);CHKERRQ(ierr);
  ierr = MatCreate(PetscObjectComm((PetscObject)A),&B);CHKERRQ(ierr);
  ierr = MatSetSizes(B,A->rmap->n,A->cmap->n,A->rmap->N,A->cmap->N);CHKERRQ(ierr);
  ierr = PetscStrallocpy("mkl_pardiso",&((PetscObject)B)->type_name);CHKERRQ(ierr);
  ierr = MatSetUp(B);CHKERRQ(ierr);

  ierr = PetscNewLog(B,&mat_mkl_pardiso);CHKERRQ(ierr);
  B->data = mat_mkl_pardiso;

  ierr = MatFactorMKL_PARDISOInitialize_Private(A, ftype, mat_mkl_pardiso);CHKERRQ(ierr);
  if (ftype == MAT_FACTOR_LU) {
    B->ops->lufactorsymbolic = MatLUFactorSymbolic_AIJMKL_PARDISO;
    B->factortype            = MAT_FACTOR_LU;
    mat_mkl_pardiso->needsym = PETSC_FALSE;
    if (isSeqAIJ) mat_mkl_pardiso->Convert = MatMKLPardiso_Convert_seqaij;
    else if (isSeqBAIJ) mat_mkl_pardiso->Convert = MatMKLPardiso_Convert_seqbaij;
    else if (isSeqSBAIJ) SETERRQ(PetscObjectComm((PetscObject)A),PETSC_ERR_SUP,"No support for PARDISO LU factor with SEQSBAIJ format! Use MAT_FACTOR_CHOLESKY instead");
    else SETERRQ1(PetscObjectComm((PetscObject)A),PETSC_ERR_SUP,"No support for PARDISO LU with %s format",((PetscObject)A)->type_name);
#if defined(PETSC_USE_COMPLEX)
    mat_mkl_pardiso->mtype = 13;
#else
    mat_mkl_pardiso->mtype = 11;
#endif
  } else {
    B->ops->choleskyfactorsymbolic = MatCholeskyFactorSymbolic_AIJMKL_PARDISO;
    B->factortype                  = MAT_FACTOR_CHOLESKY;
    if (isSeqAIJ) mat_mkl_pardiso->Convert = MatMKLPardiso_Convert_seqaij;
    else if (isSeqBAIJ) mat_mkl_pardiso->Convert = MatMKLPardiso_Convert_seqbaij;
    else if (isSeqSBAIJ) mat_mkl_pardiso->Convert = MatMKLPardiso_Convert_seqsbaij;
    else SETERRQ1(PetscObjectComm((PetscObject)A),PETSC_ERR_SUP,"No support for PARDISO CHOLESKY with %s format",((PetscObject)A)->type_name);

    mat_mkl_pardiso->needsym = PETSC_TRUE;
#if !defined(PETSC_USE_COMPLEX)
    if (A->spd_set && A->spd) mat_mkl_pardiso->mtype = 2;
    else                      mat_mkl_pardiso->mtype = -2;
#else
    mat_mkl_pardiso->mtype = 6;
    if (A->hermitian) SETERRQ(PetscObjectComm((PetscObject)A),PETSC_ERR_SUP,"No support for PARDISO CHOLESKY with Hermitian matrices! Use MAT_FACTOR_LU instead");
#endif
  }
  B->ops->destroy = MatDestroy_MKL_PARDISO;
  B->ops->view    = MatView_MKL_PARDISO;
  B->ops->getinfo = MatGetInfo_MKL_PARDISO;
  B->factortype   = ftype;
  B->assembled    = PETSC_TRUE;

  ierr = PetscFree(B->solvertype);CHKERRQ(ierr);
  ierr = PetscStrallocpy(MATSOLVERMKL_PARDISO,&B->solvertype);CHKERRQ(ierr);

  ierr = PetscObjectComposeFunction((PetscObject)B,"MatFactorGetSolverType_C",MatFactorGetSolverType_mkl_pardiso);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)B,"MatFactorSetSchurIS_C",MatFactorSetSchurIS_MKL_PARDISO);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)B,"MatMkl_PardisoSetCntl_C",MatMkl_PardisoSetCntl_MKL_PARDISO);CHKERRQ(ierr);

  *F = B;
  PetscFunctionReturn(0);
}

PETSC_EXTERN PetscErrorCode MatSolverTypeRegister_MKL_Pardiso(void)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatSolverTypeRegister(MATSOLVERMKL_PARDISO,MATSEQAIJ,MAT_FACTOR_LU,MatGetFactor_aij_mkl_pardiso);CHKERRQ(ierr);
  ierr = MatSolverTypeRegister(MATSOLVERMKL_PARDISO,MATSEQAIJ,MAT_FACTOR_CHOLESKY,MatGetFactor_aij_mkl_pardiso);CHKERRQ(ierr);
  ierr = MatSolverTypeRegister(MATSOLVERMKL_PARDISO,MATSEQBAIJ,MAT_FACTOR_LU,MatGetFactor_aij_mkl_pardiso);CHKERRQ(ierr);
  ierr = MatSolverTypeRegister(MATSOLVERMKL_PARDISO,MATSEQSBAIJ,MAT_FACTOR_CHOLESKY,MatGetFactor_aij_mkl_pardiso);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

