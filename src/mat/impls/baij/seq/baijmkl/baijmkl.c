/*
  Defines basic operations for the MATSEQBAIJMKL matrix class.
  Uses sparse BLAS operations from the Intel Math Kernel Library (MKL)
  wherever possible. If used MKL verion is older than 11.3 PETSc default
  code for sparse matrix operations is used.
*/

#include <../src/mat/impls/baij/seq/baij.h>
#include <../src/mat/impls/baij/seq/baijmkl/baijmkl.h>
#include <mkl_spblas.h>

static PetscBool PetscSeqBAIJSupportsZeroBased(void)
{
  static PetscBool set = PETSC_FALSE,value;
  int              n=1,ia[1],ja[1];
  float            a[1];
  sparse_status_t  status;
  sparse_matrix_t  A;

  if (!set) {
    status = mkl_sparse_s_create_bsr(&A,SPARSE_INDEX_BASE_ZERO,SPARSE_LAYOUT_COLUMN_MAJOR,n,n,n,ia,ia,ja,a);
    value  = (status != SPARSE_STATUS_NOT_SUPPORTED) ? PETSC_TRUE : PETSC_FALSE;
    (void)   mkl_sparse_destroy(A);
    set    = PETSC_TRUE;
  }
  return value;
}

typedef struct {
  PetscBool           sparse_optimized; /* If PETSC_TRUE, then mkl_sparse_optimize() has been called. */
  sparse_matrix_t     bsrA; /* "Handle" used by SpMV2 inspector-executor routines. */
  struct matrix_descr descr;
  PetscInt            *ai1;
  PetscInt            *aj1;
} Mat_SeqBAIJMKL;

static PetscErrorCode MatAssemblyEnd_SeqBAIJMKL(Mat A, MatAssemblyType mode);
extern PetscErrorCode MatAssemblyEnd_SeqBAIJ(Mat,MatAssemblyType);

PETSC_INTERN PetscErrorCode MatConvert_SeqBAIJMKL_SeqBAIJ(Mat A,MatType type,MatReuse reuse,Mat *newmat)
{
  /* This routine is only called to convert a MATBAIJMKL to its base PETSc type, */
  /* so we will ignore 'MatType type'. */
  PetscErrorCode ierr;
  Mat            B        = *newmat;
  Mat_SeqBAIJMKL *baijmkl = (Mat_SeqBAIJMKL*)A->spptr;

  PetscFunctionBegin;
  if (reuse == MAT_INITIAL_MATRIX) {
    ierr = MatDuplicate(A,MAT_COPY_VALUES,&B);CHKERRQ(ierr);
  }

  /* Reset the original function pointers. */
  B->ops->duplicate        = MatDuplicate_SeqBAIJ;
  B->ops->assemblyend      = MatAssemblyEnd_SeqBAIJ;
  B->ops->destroy          = MatDestroy_SeqBAIJ;
  B->ops->multtranspose    = MatMultTranspose_SeqBAIJ;
  B->ops->multtransposeadd = MatMultTransposeAdd_SeqBAIJ;
  B->ops->scale            = MatScale_SeqBAIJ;
  B->ops->diagonalscale    = MatDiagonalScale_SeqBAIJ;
  B->ops->axpy             = MatAXPY_SeqBAIJ;

  switch (A->rmap->bs) {
    case 1:
      B->ops->mult    = MatMult_SeqBAIJ_1;
      B->ops->multadd = MatMultAdd_SeqBAIJ_1;
      break;
    case 2:
      B->ops->mult    = MatMult_SeqBAIJ_2;
      B->ops->multadd = MatMultAdd_SeqBAIJ_2;
      break;
    case 3:
      B->ops->mult    = MatMult_SeqBAIJ_3;
      B->ops->multadd = MatMultAdd_SeqBAIJ_3;
      break;
    case 4:
      B->ops->mult    = MatMult_SeqBAIJ_4;
      B->ops->multadd = MatMultAdd_SeqBAIJ_4;
      break;
    case 5:
      B->ops->mult    = MatMult_SeqBAIJ_5;
      B->ops->multadd = MatMultAdd_SeqBAIJ_5;
      break;
    case 6:
      B->ops->mult    = MatMult_SeqBAIJ_6;
      B->ops->multadd = MatMultAdd_SeqBAIJ_6;
      break;
    case 7:
      B->ops->mult    = MatMult_SeqBAIJ_7;
      B->ops->multadd = MatMultAdd_SeqBAIJ_7;
      break;
    case 11:
      B->ops->mult    = MatMult_SeqBAIJ_11;
      B->ops->multadd = MatMultAdd_SeqBAIJ_11;
      break;
    case 15:
      B->ops->mult    = MatMult_SeqBAIJ_15_ver1;
      B->ops->multadd = MatMultAdd_SeqBAIJ_N;
      break;
    default:
      B->ops->mult    = MatMult_SeqBAIJ_N;
      B->ops->multadd = MatMultAdd_SeqBAIJ_N;
      break;
  }
  ierr = PetscObjectComposeFunction((PetscObject)B,"MatConvert_seqbaijmkl_seqbaij_C",NULL);CHKERRQ(ierr);

  /* Free everything in the Mat_SeqBAIJMKL data structure. Currently, this
   * simply involves destroying the MKL sparse matrix handle and then freeing
   * the spptr pointer. */
  if (reuse == MAT_INITIAL_MATRIX) baijmkl = (Mat_SeqBAIJMKL*)B->spptr;

  if (baijmkl->sparse_optimized) {
    sparse_status_t stat;
    stat = mkl_sparse_destroy(baijmkl->bsrA);
    PetscAssertFalse(stat != SPARSE_STATUS_SUCCESS,PETSC_COMM_SELF,PETSC_ERR_LIB,"Intel MKL error: error in mkl_sparse_destroy");
  }
  ierr = PetscFree2(baijmkl->ai1,baijmkl->aj1);CHKERRQ(ierr);
  ierr = PetscFree(B->spptr);CHKERRQ(ierr);

  /* Change the type of B to MATSEQBAIJ. */
  ierr = PetscObjectChangeTypeName((PetscObject)B, MATSEQBAIJ);CHKERRQ(ierr);

  *newmat = B;
  PetscFunctionReturn(0);
}

static PetscErrorCode MatDestroy_SeqBAIJMKL(Mat A)
{
  PetscErrorCode ierr;
  Mat_SeqBAIJMKL *baijmkl = (Mat_SeqBAIJMKL*) A->spptr;

  PetscFunctionBegin;
  if (baijmkl) {
    /* Clean up everything in the Mat_SeqBAIJMKL data structure, then free A->spptr. */
    if (baijmkl->sparse_optimized) {
      sparse_status_t stat = SPARSE_STATUS_SUCCESS;
      stat = mkl_sparse_destroy(baijmkl->bsrA);
      PetscAssertFalse(stat != SPARSE_STATUS_SUCCESS,PETSC_COMM_SELF,PETSC_ERR_LIB,"Intel MKL error: error in mkl_sparse_destroy");
    }
    ierr = PetscFree2(baijmkl->ai1,baijmkl->aj1);CHKERRQ(ierr);
    ierr = PetscFree(A->spptr);CHKERRQ(ierr);
  }

  /* Change the type of A back to SEQBAIJ and use MatDestroy_SeqBAIJ()
   * to destroy everything that remains. */
  ierr = PetscObjectChangeTypeName((PetscObject)A, MATSEQBAIJ);CHKERRQ(ierr);
  ierr = MatDestroy_SeqBAIJ(A);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatSeqBAIJMKL_create_mkl_handle(Mat A)
{
  Mat_SeqBAIJ     *a = (Mat_SeqBAIJ*)A->data;
  Mat_SeqBAIJMKL  *baijmkl = (Mat_SeqBAIJMKL*)A->spptr;
  PetscInt        mbs, nbs, nz, bs;
  MatScalar       *aa;
  PetscInt        *aj,*ai;
  sparse_status_t stat;
  PetscErrorCode  ierr;
  PetscInt        i;

  PetscFunctionBegin;
  if (baijmkl->sparse_optimized) {
    /* Matrix has been previously assembled and optimized. Must destroy old
     * matrix handle before running the optimization step again. */
    ierr = PetscFree2(baijmkl->ai1,baijmkl->aj1);CHKERRQ(ierr);
    stat = mkl_sparse_destroy(baijmkl->bsrA);CHKERRMKL(stat);
  }
  baijmkl->sparse_optimized = PETSC_FALSE;

  /* Now perform the SpMV2 setup and matrix optimization. */
  baijmkl->descr.type        = SPARSE_MATRIX_TYPE_GENERAL;
  baijmkl->descr.mode        = SPARSE_FILL_MODE_LOWER;
  baijmkl->descr.diag        = SPARSE_DIAG_NON_UNIT;
  mbs = a->mbs;
  nbs = a->nbs;
  nz  = a->nz;
  bs  = A->rmap->bs;
  aa  = a->a;

  if ((nz!=0) & !(A->structure_only)) {
    /* Create a new, optimized sparse matrix handle only if the matrix has nonzero entries.
     * The MKL sparse-inspector executor routines don't like being passed an empty matrix. */
    if (PetscSeqBAIJSupportsZeroBased()) {
      aj   = a->j;
      ai   = a->i;
      stat = mkl_sparse_x_create_bsr(&(baijmkl->bsrA),SPARSE_INDEX_BASE_ZERO,SPARSE_LAYOUT_COLUMN_MAJOR,mbs,nbs,bs,ai,ai+1,aj,aa);CHKERRMKL(stat);
    } else {
      ierr = PetscMalloc2(mbs+1,&ai,nz,&aj);CHKERRQ(ierr);
      for (i=0;i<mbs+1;i++) ai[i] = a->i[i]+1;
      for (i=0;i<nz;i++) aj[i] = a->j[i]+1;
      aa   = a->a;
      stat = mkl_sparse_x_create_bsr(&baijmkl->bsrA,SPARSE_INDEX_BASE_ONE,SPARSE_LAYOUT_COLUMN_MAJOR,mbs,nbs,bs,ai,ai+1,aj,aa);CHKERRMKL(stat);
      baijmkl->ai1 = ai;
      baijmkl->aj1 = aj;
    }
    stat = mkl_sparse_set_mv_hint(baijmkl->bsrA,SPARSE_OPERATION_NON_TRANSPOSE,baijmkl->descr,1000);CHKERRMKL(stat);
    stat = mkl_sparse_set_memory_hint(baijmkl->bsrA,SPARSE_MEMORY_AGGRESSIVE);CHKERRMKL(stat);
    stat = mkl_sparse_optimize(baijmkl->bsrA);CHKERRMKL(stat);
    baijmkl->sparse_optimized = PETSC_TRUE;
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode MatDuplicate_SeqBAIJMKL(Mat A, MatDuplicateOption op, Mat *M)
{
  PetscErrorCode ierr;
  Mat_SeqBAIJMKL *baijmkl;
  Mat_SeqBAIJMKL *baijmkl_dest;

  PetscFunctionBegin;
  ierr = MatDuplicate_SeqBAIJ(A,op,M);CHKERRQ(ierr);
  baijmkl = (Mat_SeqBAIJMKL*) A->spptr;
  ierr = PetscNewLog((*M),&baijmkl_dest);CHKERRQ(ierr);
  (*M)->spptr = (void*)baijmkl_dest;
  ierr = PetscMemcpy(baijmkl_dest,baijmkl,sizeof(Mat_SeqBAIJMKL));CHKERRQ(ierr);
  baijmkl_dest->sparse_optimized = PETSC_FALSE;
  ierr = MatSeqBAIJMKL_create_mkl_handle(A);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatMult_SeqBAIJMKL_SpMV2(Mat A,Vec xx,Vec yy)
{
  Mat_SeqBAIJ        *a = (Mat_SeqBAIJ*)A->data;
  Mat_SeqBAIJMKL     *baijmkl=(Mat_SeqBAIJMKL*)A->spptr;
  const PetscScalar  *x;
  PetscScalar        *y;
  PetscErrorCode     ierr;
  sparse_status_t    stat = SPARSE_STATUS_SUCCESS;

  PetscFunctionBegin;
  /* If there are no nonzero entries, zero yy and return immediately. */
  if (!a->nz) {
    ierr = VecSet(yy,0.0);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }

  ierr = VecGetArrayRead(xx,&x);CHKERRQ(ierr);
  ierr = VecGetArray(yy,&y);CHKERRQ(ierr);

  /* In some cases, we get to this point without mkl_sparse_optimize() having been called, so we check and then call
   * it if needed. Eventually, when everything in PETSc is properly updating the matrix state, we should probably
   * take a "lazy" approach to creation/updating of the MKL matrix handle and plan to always do it here (when needed). */
  if (!baijmkl->sparse_optimized) {
    ierr = MatSeqBAIJMKL_create_mkl_handle(A);CHKERRQ(ierr);
  }

  /* Call MKL SpMV2 executor routine to do the MatMult. */
  stat = mkl_sparse_x_mv(SPARSE_OPERATION_NON_TRANSPOSE,1.0,baijmkl->bsrA,baijmkl->descr,x,0.0,y);CHKERRMKL(stat);

  ierr = PetscLogFlops(2.0*a->bs2*a->nz - a->nonzerorowcnt*A->rmap->bs);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(xx,&x);CHKERRQ(ierr);
  ierr = VecRestoreArray(yy,&y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatMultTranspose_SeqBAIJMKL_SpMV2(Mat A,Vec xx,Vec yy)
{
  Mat_SeqBAIJ       *a       = (Mat_SeqBAIJ*)A->data;
  Mat_SeqBAIJMKL    *baijmkl = (Mat_SeqBAIJMKL*)A->spptr;
  const PetscScalar *x;
  PetscScalar       *y;
  PetscErrorCode    ierr;
  sparse_status_t   stat;

  PetscFunctionBegin;
  /* If there are no nonzero entries, zero yy and return immediately. */
  if (!a->nz) {
    ierr = VecSet(yy,0.0);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }

  ierr = VecGetArrayRead(xx,&x);CHKERRQ(ierr);
  ierr = VecGetArray(yy,&y);CHKERRQ(ierr);

  /* In some cases, we get to this point without mkl_sparse_optimize() having been called, so we check and then call
   * it if needed. Eventually, when everything in PETSc is properly updating the matrix state, we should probably
   * take a "lazy" approach to creation/updating of the MKL matrix handle and plan to always do it here (when needed). */
  if (!baijmkl->sparse_optimized) {
    ierr = MatSeqBAIJMKL_create_mkl_handle(A);CHKERRQ(ierr);
  }

  /* Call MKL SpMV2 executor routine to do the MatMultTranspose. */
  stat = mkl_sparse_x_mv(SPARSE_OPERATION_TRANSPOSE,1.0,baijmkl->bsrA,baijmkl->descr,x,0.0,y);CHKERRMKL(stat);

  ierr = PetscLogFlops(2.0*a->bs2*a->nz - a->nonzerorowcnt*A->rmap->bs);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(xx,&x);CHKERRQ(ierr);
  ierr = VecRestoreArray(yy,&y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatMultAdd_SeqBAIJMKL_SpMV2(Mat A,Vec xx,Vec yy,Vec zz)
{
  Mat_SeqBAIJ        *a       = (Mat_SeqBAIJ*)A->data;
  Mat_SeqBAIJMKL     *baijmkl = (Mat_SeqBAIJMKL*)A->spptr;
  const PetscScalar  *x;
  PetscScalar        *y,*z;
  PetscErrorCode     ierr;
  PetscInt           m=a->mbs*A->rmap->bs;
  PetscInt           i;

  sparse_status_t stat = SPARSE_STATUS_SUCCESS;

  PetscFunctionBegin;
  /* If there are no nonzero entries, set zz = yy and return immediately. */
  if (!a->nz) {
    ierr = VecCopy(yy,zz);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }

  ierr = VecGetArrayRead(xx,&x);CHKERRQ(ierr);
  ierr = VecGetArrayPair(yy,zz,&y,&z);CHKERRQ(ierr);

  /* In some cases, we get to this point without mkl_sparse_optimize() having been called, so we check and then call
   * it if needed. Eventually, when everything in PETSc is properly updating the matrix state, we should probably
   * take a "lazy" approach to creation/updating of the MKL matrix handle and plan to always do it here (when needed). */
  if (!baijmkl->sparse_optimized) {
    ierr = MatSeqBAIJMKL_create_mkl_handle(A);CHKERRQ(ierr);
  }

  /* Call MKL sparse BLAS routine to do the MatMult. */
  if (zz == yy) {
    /* If zz and yy are the same vector, we can use mkl_sparse_x_mv, which calculates y = alpha*A*x + beta*y,
     * with alpha and beta both set to 1.0. */
    stat = mkl_sparse_x_mv(SPARSE_OPERATION_NON_TRANSPOSE,1.0,baijmkl->bsrA,baijmkl->descr,x,1.0,z);CHKERRMKL(stat);
  } else {
    /* zz and yy are different vectors, so we call mkl_sparse_x_mv with alpha=1.0 and beta=0.0, and then
     * we add the contents of vector yy to the result; MKL sparse BLAS does not have a MatMultAdd equivalent. */
    stat = mkl_sparse_x_mv(SPARSE_OPERATION_NON_TRANSPOSE,1.0,baijmkl->bsrA,baijmkl->descr,x,0.0,z);CHKERRMKL(stat);
    for (i=0; i<m; i++) {
      z[i] += y[i];
    }
  }

  ierr = PetscLogFlops(2.0*a->bs2*a->nz);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(xx,&x);CHKERRQ(ierr);
  ierr = VecRestoreArrayPair(yy,zz,&y,&z);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatMultTransposeAdd_SeqBAIJMKL_SpMV2(Mat A,Vec xx,Vec yy,Vec zz)
{
  Mat_SeqBAIJ       *a       = (Mat_SeqBAIJ*)A->data;
  Mat_SeqBAIJMKL    *baijmkl = (Mat_SeqBAIJMKL*)A->spptr;
  const PetscScalar *x;
  PetscScalar       *y,*z;
  PetscErrorCode    ierr;
  PetscInt          n=a->nbs*A->rmap->bs;
  PetscInt          i;
  /* Variables not in MatMultTransposeAdd_SeqBAIJ. */
  sparse_status_t   stat = SPARSE_STATUS_SUCCESS;

  PetscFunctionBegin;
  /* If there are no nonzero entries, set zz = yy and return immediately. */
  if (!a->nz) {
    ierr = VecCopy(yy,zz);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }

  ierr = VecGetArrayRead(xx,&x);CHKERRQ(ierr);
  ierr = VecGetArrayPair(yy,zz,&y,&z);CHKERRQ(ierr);

  /* In some cases, we get to this point without mkl_sparse_optimize() having been called, so we check and then call
   * it if needed. Eventually, when everything in PETSc is properly updating the matrix state, we should probably
   * take a "lazy" approach to creation/updating of the MKL matrix handle and plan to always do it here (when needed). */
  if (!baijmkl->sparse_optimized) {
    ierr = MatSeqBAIJMKL_create_mkl_handle(A);CHKERRQ(ierr);
  }

  /* Call MKL sparse BLAS routine to do the MatMult. */
  if (zz == yy) {
    /* If zz and yy are the same vector, we can use mkl_sparse_x_mv, which calculates y = alpha*A*x + beta*y,
     * with alpha and beta both set to 1.0. */
    stat = mkl_sparse_x_mv(SPARSE_OPERATION_TRANSPOSE,1.0,baijmkl->bsrA,baijmkl->descr,x,1.0,z);CHKERRMKL(stat);
  } else {
    /* zz and yy are different vectors, so we call mkl_sparse_x_mv with alpha=1.0 and beta=0.0, and then
     * we add the contents of vector yy to the result; MKL sparse BLAS does not have a MatMultAdd equivalent. */
    stat = mkl_sparse_x_mv(SPARSE_OPERATION_TRANSPOSE,1.0,baijmkl->bsrA,baijmkl->descr,x,0.0,z);CHKERRMKL(stat);
    for (i=0; i<n; i++) {
      z[i] += y[i];
    }
  }

  ierr = PetscLogFlops(2.0*a->bs2*a->nz);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(xx,&x);CHKERRQ(ierr);
  ierr = VecRestoreArrayPair(yy,zz,&y,&z);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatScale_SeqBAIJMKL(Mat inA,PetscScalar alpha)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatScale_SeqBAIJ(inA,alpha);CHKERRQ(ierr);
  ierr = MatSeqBAIJMKL_create_mkl_handle(inA);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatDiagonalScale_SeqBAIJMKL(Mat A,Vec ll,Vec rr)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatDiagonalScale_SeqBAIJ(A,ll,rr);CHKERRQ(ierr);
  ierr = MatSeqBAIJMKL_create_mkl_handle(A);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatAXPY_SeqBAIJMKL(Mat Y,PetscScalar a,Mat X,MatStructure str)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatAXPY_SeqBAIJ(Y,a,X,str);CHKERRQ(ierr);
  if (str == SAME_NONZERO_PATTERN) {
    /* MatAssemblyEnd() is not called if SAME_NONZERO_PATTERN, so we need to force update of the MKL matrix handle. */
    ierr = MatSeqBAIJMKL_create_mkl_handle(Y);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}
/* MatConvert_SeqBAIJ_SeqBAIJMKL converts a SeqBAIJ matrix into a
 * SeqBAIJMKL matrix.  This routine is called by the MatCreate_SeqMKLBAIJ()
 * routine, but can also be used to convert an assembled SeqBAIJ matrix
 * into a SeqBAIJMKL one. */
PETSC_INTERN PetscErrorCode MatConvert_SeqBAIJ_SeqBAIJMKL(Mat A,MatType type,MatReuse reuse,Mat *newmat)
{
  PetscErrorCode ierr;
  Mat            B = *newmat;
  Mat_SeqBAIJMKL *baijmkl;
  PetscBool      sametype;

  PetscFunctionBegin;
  if (reuse == MAT_INITIAL_MATRIX) {
    ierr = MatDuplicate(A,MAT_COPY_VALUES,&B);CHKERRQ(ierr);
  }

  ierr = PetscObjectTypeCompare((PetscObject)A,type,&sametype);CHKERRQ(ierr);
  if (sametype) PetscFunctionReturn(0);

  ierr     = PetscNewLog(B,&baijmkl);CHKERRQ(ierr);
  B->spptr = (void*)baijmkl;

  /* Set function pointers for methods that we inherit from BAIJ but override.
   * We also parse some command line options below, since those determine some of the methods we point to. */
  B->ops->assemblyend      = MatAssemblyEnd_SeqBAIJMKL;

  baijmkl->sparse_optimized = PETSC_FALSE;

  ierr = PetscObjectComposeFunction((PetscObject)B,"MatScale_SeqBAIJMKL_C",MatScale_SeqBAIJMKL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)B,"MatConvert_seqbaijmkl_seqbaij_C",MatConvert_SeqBAIJMKL_SeqBAIJ);CHKERRQ(ierr);

  ierr    = PetscObjectChangeTypeName((PetscObject)B,MATSEQBAIJMKL);CHKERRQ(ierr);
  *newmat = B;
  PetscFunctionReturn(0);
}

static PetscErrorCode MatAssemblyEnd_SeqBAIJMKL(Mat A, MatAssemblyType mode)
{
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  if (mode == MAT_FLUSH_ASSEMBLY) PetscFunctionReturn(0);
  ierr = MatAssemblyEnd_SeqBAIJ(A, mode);CHKERRQ(ierr);
  ierr = MatSeqBAIJMKL_create_mkl_handle(A);CHKERRQ(ierr);
  A->ops->destroy          = MatDestroy_SeqBAIJMKL;
  A->ops->mult             = MatMult_SeqBAIJMKL_SpMV2;
  A->ops->multtranspose    = MatMultTranspose_SeqBAIJMKL_SpMV2;
  A->ops->multadd          = MatMultAdd_SeqBAIJMKL_SpMV2;
  A->ops->multtransposeadd = MatMultTransposeAdd_SeqBAIJMKL_SpMV2;
  A->ops->scale            = MatScale_SeqBAIJMKL;
  A->ops->diagonalscale    = MatDiagonalScale_SeqBAIJMKL;
  A->ops->axpy             = MatAXPY_SeqBAIJMKL;
  A->ops->duplicate        = MatDuplicate_SeqBAIJMKL;
  PetscFunctionReturn(0);
}

/*@C
   MatCreateSeqBAIJMKL - Creates a sparse matrix of type SEQBAIJMKL.
   This type inherits from BAIJ and is largely identical, but uses sparse BLAS
   routines from Intel MKL whenever possible.
   MatMult, MatMultAdd, MatMultTranspose, and MatMultTransposeAdd
   operations are currently supported.
   If the installed version of MKL supports the "SpMV2" sparse
   inspector-executor routines, then those are used by default.
   Default PETSc kernels are used otherwise.

   Input Parameters:
+  comm - MPI communicator, set to PETSC_COMM_SELF
.  bs - size of block, the blocks are ALWAYS square. One can use MatSetBlockSizes() to set a different row and column blocksize but the row
          blocksize always defines the size of the blocks. The column blocksize sets the blocksize of the vectors obtained with MatCreateVecs()
.  m - number of rows
.  n - number of columns
.  nz - number of nonzero blocks  per block row (same for all rows)
-  nnz - array containing the number of nonzero blocks in the various block rows
         (possibly different for each block row) or NULL

   Output Parameter:
.  A - the matrix

   It is recommended that one use the MatCreate(), MatSetType() and/or MatSetFromOptions(),
   MatXXXXSetPreallocation() paradigm instead of this routine directly.
   [MatXXXXSetPreallocation() is, for example, MatSeqAIJSetPreallocation]

   Options Database Keys:
+   -mat_no_unroll - uses code that does not unroll the loops in the
                     block calculations (much slower)
-   -mat_block_size - size of the blocks to use

   Level: intermediate

   Notes:
   The number of rows and columns must be divisible by blocksize.

   If the nnz parameter is given then the nz parameter is ignored

   A nonzero block is any block that as 1 or more nonzeros in it

   The block AIJ format is fully compatible with standard Fortran 77
   storage.  That is, the stored row and column indices can begin at
   either one (as in Fortran) or zero.  See the users' manual for details.

   Specify the preallocated storage with either nz or nnz (not both).
   Set nz=PETSC_DEFAULT and nnz=NULL for PETSc to control dynamic memory
   allocation.  See Users-Manual: ch_mat for details.
   matrices.

.seealso: MatCreate(), MatCreateSeqAIJ(), MatSetValues(), MatCreateBAIJ()

@*/
PetscErrorCode  MatCreateSeqBAIJMKL(MPI_Comm comm,PetscInt bs,PetscInt m,PetscInt n,PetscInt nz,const PetscInt nnz[],Mat *A)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatCreate(comm,A);CHKERRQ(ierr);
  ierr = MatSetSizes(*A,m,n,m,n);CHKERRQ(ierr);
  ierr = MatSetType(*A,MATSEQBAIJMKL);CHKERRQ(ierr);
  ierr = MatSeqBAIJSetPreallocation_SeqBAIJ(*A,bs,nz,(PetscInt*)nnz);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PETSC_EXTERN PetscErrorCode MatCreate_SeqBAIJMKL(Mat A)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatSetType(A,MATSEQBAIJ);CHKERRQ(ierr);
  ierr = MatConvert_SeqBAIJ_SeqBAIJMKL(A,MATSEQBAIJMKL,MAT_INPLACE_MATRIX,&A);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
