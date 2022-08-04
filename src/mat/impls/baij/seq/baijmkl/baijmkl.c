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
  Mat            B        = *newmat;
  Mat_SeqBAIJMKL *baijmkl = (Mat_SeqBAIJMKL*)A->spptr;

  PetscFunctionBegin;
  if (reuse == MAT_INITIAL_MATRIX) PetscCall(MatDuplicate(A,MAT_COPY_VALUES,&B));

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
  PetscCall(PetscObjectComposeFunction((PetscObject)B,"MatConvert_seqbaijmkl_seqbaij_C",NULL));

  /* Free everything in the Mat_SeqBAIJMKL data structure. Currently, this
   * simply involves destroying the MKL sparse matrix handle and then freeing
   * the spptr pointer. */
  if (reuse == MAT_INITIAL_MATRIX) baijmkl = (Mat_SeqBAIJMKL*)B->spptr;

  if (baijmkl->sparse_optimized) PetscCallExternal(mkl_sparse_destroy,baijmkl->bsrA);
  PetscCall(PetscFree2(baijmkl->ai1,baijmkl->aj1));
  PetscCall(PetscFree(B->spptr));

  /* Change the type of B to MATSEQBAIJ. */
  PetscCall(PetscObjectChangeTypeName((PetscObject)B, MATSEQBAIJ));

  *newmat = B;
  PetscFunctionReturn(0);
}

static PetscErrorCode MatDestroy_SeqBAIJMKL(Mat A)
{
  Mat_SeqBAIJMKL *baijmkl = (Mat_SeqBAIJMKL*) A->spptr;

  PetscFunctionBegin;
  if (baijmkl) {
    /* Clean up everything in the Mat_SeqBAIJMKL data structure, then free A->spptr. */
    if (baijmkl->sparse_optimized) PetscCallExternal(mkl_sparse_destroy,baijmkl->bsrA);
    PetscCall(PetscFree2(baijmkl->ai1,baijmkl->aj1));
    PetscCall(PetscFree(A->spptr));
  }

  /* Change the type of A back to SEQBAIJ and use MatDestroy_SeqBAIJ()
   * to destroy everything that remains. */
  PetscCall(PetscObjectChangeTypeName((PetscObject)A, MATSEQBAIJ));
  PetscCall(MatDestroy_SeqBAIJ(A));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatSeqBAIJMKL_create_mkl_handle(Mat A)
{
  Mat_SeqBAIJ     *a = (Mat_SeqBAIJ*)A->data;
  Mat_SeqBAIJMKL  *baijmkl = (Mat_SeqBAIJMKL*)A->spptr;
  PetscInt        mbs, nbs, nz, bs;
  MatScalar       *aa;
  PetscInt        *aj,*ai;
  PetscInt        i;

  PetscFunctionBegin;
  if (baijmkl->sparse_optimized) {
    /* Matrix has been previously assembled and optimized. Must destroy old
     * matrix handle before running the optimization step again. */
    PetscCall(PetscFree2(baijmkl->ai1,baijmkl->aj1));
    PetscCallMKL(mkl_sparse_destroy(baijmkl->bsrA));
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
      PetscCallMKL(mkl_sparse_x_create_bsr(&(baijmkl->bsrA),SPARSE_INDEX_BASE_ZERO,SPARSE_LAYOUT_COLUMN_MAJOR,mbs,nbs,bs,ai,ai+1,aj,aa));
    } else {
      PetscCall(PetscMalloc2(mbs+1,&ai,nz,&aj));
      for (i=0;i<mbs+1;i++) ai[i] = a->i[i]+1;
      for (i=0;i<nz;i++) aj[i] = a->j[i]+1;
      aa   = a->a;
      PetscCallMKL(mkl_sparse_x_create_bsr(&baijmkl->bsrA,SPARSE_INDEX_BASE_ONE,SPARSE_LAYOUT_COLUMN_MAJOR,mbs,nbs,bs,ai,ai+1,aj,aa));
      baijmkl->ai1 = ai;
      baijmkl->aj1 = aj;
    }
    PetscCallMKL(mkl_sparse_set_mv_hint(baijmkl->bsrA,SPARSE_OPERATION_NON_TRANSPOSE,baijmkl->descr,1000));
    PetscCallMKL(mkl_sparse_set_memory_hint(baijmkl->bsrA,SPARSE_MEMORY_AGGRESSIVE));
    PetscCallMKL(mkl_sparse_optimize(baijmkl->bsrA));
    baijmkl->sparse_optimized = PETSC_TRUE;
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode MatDuplicate_SeqBAIJMKL(Mat A, MatDuplicateOption op, Mat *M)
{
  Mat_SeqBAIJMKL *baijmkl;
  Mat_SeqBAIJMKL *baijmkl_dest;

  PetscFunctionBegin;
  PetscCall(MatDuplicate_SeqBAIJ(A,op,M));
  baijmkl = (Mat_SeqBAIJMKL*) A->spptr;
  PetscCall(PetscNewLog((*M),&baijmkl_dest));
  (*M)->spptr = (void*)baijmkl_dest;
  PetscCall(PetscMemcpy(baijmkl_dest,baijmkl,sizeof(Mat_SeqBAIJMKL)));
  baijmkl_dest->sparse_optimized = PETSC_FALSE;
  PetscCall(MatSeqBAIJMKL_create_mkl_handle(A));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatMult_SeqBAIJMKL_SpMV2(Mat A,Vec xx,Vec yy)
{
  Mat_SeqBAIJ        *a = (Mat_SeqBAIJ*)A->data;
  Mat_SeqBAIJMKL     *baijmkl=(Mat_SeqBAIJMKL*)A->spptr;
  const PetscScalar  *x;
  PetscScalar        *y;

  PetscFunctionBegin;
  /* If there are no nonzero entries, zero yy and return immediately. */
  if (!a->nz) {
    PetscCall(VecSet(yy,0.0));
    PetscFunctionReturn(0);
  }

  PetscCall(VecGetArrayRead(xx,&x));
  PetscCall(VecGetArray(yy,&y));

  /* In some cases, we get to this point without mkl_sparse_optimize() having been called, so we check and then call
   * it if needed. Eventually, when everything in PETSc is properly updating the matrix state, we should probably
   * take a "lazy" approach to creation/updating of the MKL matrix handle and plan to always do it here (when needed). */
  if (!baijmkl->sparse_optimized) {
    PetscCall(MatSeqBAIJMKL_create_mkl_handle(A));
  }

  /* Call MKL SpMV2 executor routine to do the MatMult. */
  PetscCallMKL(mkl_sparse_x_mv(SPARSE_OPERATION_NON_TRANSPOSE,1.0,baijmkl->bsrA,baijmkl->descr,x,0.0,y));

  PetscCall(PetscLogFlops(2.0*a->bs2*a->nz - a->nonzerorowcnt*A->rmap->bs));
  PetscCall(VecRestoreArrayRead(xx,&x));
  PetscCall(VecRestoreArray(yy,&y));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatMultTranspose_SeqBAIJMKL_SpMV2(Mat A,Vec xx,Vec yy)
{
  Mat_SeqBAIJ       *a       = (Mat_SeqBAIJ*)A->data;
  Mat_SeqBAIJMKL    *baijmkl = (Mat_SeqBAIJMKL*)A->spptr;
  const PetscScalar *x;
  PetscScalar       *y;

  PetscFunctionBegin;
  /* If there are no nonzero entries, zero yy and return immediately. */
  if (!a->nz) {
    PetscCall(VecSet(yy,0.0));
    PetscFunctionReturn(0);
  }

  PetscCall(VecGetArrayRead(xx,&x));
  PetscCall(VecGetArray(yy,&y));

  /* In some cases, we get to this point without mkl_sparse_optimize() having been called, so we check and then call
   * it if needed. Eventually, when everything in PETSc is properly updating the matrix state, we should probably
   * take a "lazy" approach to creation/updating of the MKL matrix handle and plan to always do it here (when needed). */
  if (!baijmkl->sparse_optimized) {
    PetscCall(MatSeqBAIJMKL_create_mkl_handle(A));
  }

  /* Call MKL SpMV2 executor routine to do the MatMultTranspose. */
  PetscCallMKL(mkl_sparse_x_mv(SPARSE_OPERATION_TRANSPOSE,1.0,baijmkl->bsrA,baijmkl->descr,x,0.0,y));

  PetscCall(PetscLogFlops(2.0*a->bs2*a->nz - a->nonzerorowcnt*A->rmap->bs));
  PetscCall(VecRestoreArrayRead(xx,&x));
  PetscCall(VecRestoreArray(yy,&y));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatMultAdd_SeqBAIJMKL_SpMV2(Mat A,Vec xx,Vec yy,Vec zz)
{
  Mat_SeqBAIJ        *a       = (Mat_SeqBAIJ*)A->data;
  Mat_SeqBAIJMKL     *baijmkl = (Mat_SeqBAIJMKL*)A->spptr;
  const PetscScalar  *x;
  PetscScalar        *y,*z;
  PetscInt           m=a->mbs*A->rmap->bs;
  PetscInt           i;

  PetscFunctionBegin;
  /* If there are no nonzero entries, set zz = yy and return immediately. */
  if (!a->nz) {
    PetscCall(VecCopy(yy,zz));
    PetscFunctionReturn(0);
  }

  PetscCall(VecGetArrayRead(xx,&x));
  PetscCall(VecGetArrayPair(yy,zz,&y,&z));

  /* In some cases, we get to this point without mkl_sparse_optimize() having been called, so we check and then call
   * it if needed. Eventually, when everything in PETSc is properly updating the matrix state, we should probably
   * take a "lazy" approach to creation/updating of the MKL matrix handle and plan to always do it here (when needed). */
  if (!baijmkl->sparse_optimized) {
    PetscCall(MatSeqBAIJMKL_create_mkl_handle(A));
  }

  /* Call MKL sparse BLAS routine to do the MatMult. */
  if (zz == yy) {
    /* If zz and yy are the same vector, we can use mkl_sparse_x_mv, which calculates y = alpha*A*x + beta*y,
     * with alpha and beta both set to 1.0. */
    PetscCallMKL(mkl_sparse_x_mv(SPARSE_OPERATION_NON_TRANSPOSE,1.0,baijmkl->bsrA,baijmkl->descr,x,1.0,z));
  } else {
    /* zz and yy are different vectors, so we call mkl_sparse_x_mv with alpha=1.0 and beta=0.0, and then
     * we add the contents of vector yy to the result; MKL sparse BLAS does not have a MatMultAdd equivalent. */
    PetscCallMKL(mkl_sparse_x_mv(SPARSE_OPERATION_NON_TRANSPOSE,1.0,baijmkl->bsrA,baijmkl->descr,x,0.0,z));
    for (i=0; i<m; i++) {
      z[i] += y[i];
    }
  }

  PetscCall(PetscLogFlops(2.0*a->bs2*a->nz));
  PetscCall(VecRestoreArrayRead(xx,&x));
  PetscCall(VecRestoreArrayPair(yy,zz,&y,&z));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatMultTransposeAdd_SeqBAIJMKL_SpMV2(Mat A,Vec xx,Vec yy,Vec zz)
{
  Mat_SeqBAIJ       *a       = (Mat_SeqBAIJ*)A->data;
  Mat_SeqBAIJMKL    *baijmkl = (Mat_SeqBAIJMKL*)A->spptr;
  const PetscScalar *x;
  PetscScalar       *y,*z;
  PetscInt          n=a->nbs*A->rmap->bs;
  PetscInt          i;
  /* Variables not in MatMultTransposeAdd_SeqBAIJ. */

  PetscFunctionBegin;
  /* If there are no nonzero entries, set zz = yy and return immediately. */
  if (!a->nz) {
    PetscCall(VecCopy(yy,zz));
    PetscFunctionReturn(0);
  }

  PetscCall(VecGetArrayRead(xx,&x));
  PetscCall(VecGetArrayPair(yy,zz,&y,&z));

  /* In some cases, we get to this point without mkl_sparse_optimize() having been called, so we check and then call
   * it if needed. Eventually, when everything in PETSc is properly updating the matrix state, we should probably
   * take a "lazy" approach to creation/updating of the MKL matrix handle and plan to always do it here (when needed). */
  if (!baijmkl->sparse_optimized) {
    PetscCall(MatSeqBAIJMKL_create_mkl_handle(A));
  }

  /* Call MKL sparse BLAS routine to do the MatMult. */
  if (zz == yy) {
    /* If zz and yy are the same vector, we can use mkl_sparse_x_mv, which calculates y = alpha*A*x + beta*y,
     * with alpha and beta both set to 1.0. */
    PetscCallMKL(mkl_sparse_x_mv(SPARSE_OPERATION_TRANSPOSE,1.0,baijmkl->bsrA,baijmkl->descr,x,1.0,z));
  } else {
    /* zz and yy are different vectors, so we call mkl_sparse_x_mv with alpha=1.0 and beta=0.0, and then
     * we add the contents of vector yy to the result; MKL sparse BLAS does not have a MatMultAdd equivalent. */
    PetscCallMKL(mkl_sparse_x_mv(SPARSE_OPERATION_TRANSPOSE,1.0,baijmkl->bsrA,baijmkl->descr,x,0.0,z));
    for (i=0; i<n; i++) {
      z[i] += y[i];
    }
  }

  PetscCall(PetscLogFlops(2.0*a->bs2*a->nz));
  PetscCall(VecRestoreArrayRead(xx,&x));
  PetscCall(VecRestoreArrayPair(yy,zz,&y,&z));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatScale_SeqBAIJMKL(Mat inA,PetscScalar alpha)
{
  PetscFunctionBegin;
  PetscCall(MatScale_SeqBAIJ(inA,alpha));
  PetscCall(MatSeqBAIJMKL_create_mkl_handle(inA));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatDiagonalScale_SeqBAIJMKL(Mat A,Vec ll,Vec rr)
{
  PetscFunctionBegin;
  PetscCall(MatDiagonalScale_SeqBAIJ(A,ll,rr));
  PetscCall(MatSeqBAIJMKL_create_mkl_handle(A));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatAXPY_SeqBAIJMKL(Mat Y,PetscScalar a,Mat X,MatStructure str)
{
  PetscFunctionBegin;
  PetscCall(MatAXPY_SeqBAIJ(Y,a,X,str));
  if (str == SAME_NONZERO_PATTERN) {
    /* MatAssemblyEnd() is not called if SAME_NONZERO_PATTERN, so we need to force update of the MKL matrix handle. */
    PetscCall(MatSeqBAIJMKL_create_mkl_handle(Y));
  }
  PetscFunctionReturn(0);
}
/* MatConvert_SeqBAIJ_SeqBAIJMKL converts a SeqBAIJ matrix into a
 * SeqBAIJMKL matrix.  This routine is called by the MatCreate_SeqMKLBAIJ()
 * routine, but can also be used to convert an assembled SeqBAIJ matrix
 * into a SeqBAIJMKL one. */
PETSC_INTERN PetscErrorCode MatConvert_SeqBAIJ_SeqBAIJMKL(Mat A,MatType type,MatReuse reuse,Mat *newmat)
{
  Mat            B = *newmat;
  Mat_SeqBAIJMKL *baijmkl;
  PetscBool      sametype;

  PetscFunctionBegin;
  if (reuse == MAT_INITIAL_MATRIX) PetscCall(MatDuplicate(A,MAT_COPY_VALUES,&B));

  PetscCall(PetscObjectTypeCompare((PetscObject)A,type,&sametype));
  if (sametype) PetscFunctionReturn(0);

  PetscCall(PetscNewLog(B,&baijmkl));
  B->spptr = (void*)baijmkl;

  /* Set function pointers for methods that we inherit from BAIJ but override.
   * We also parse some command line options below, since those determine some of the methods we point to. */
  B->ops->assemblyend      = MatAssemblyEnd_SeqBAIJMKL;

  baijmkl->sparse_optimized = PETSC_FALSE;

  PetscCall(PetscObjectComposeFunction((PetscObject)B,"MatScale_SeqBAIJMKL_C",MatScale_SeqBAIJMKL));
  PetscCall(PetscObjectComposeFunction((PetscObject)B,"MatConvert_seqbaijmkl_seqbaij_C",MatConvert_SeqBAIJMKL_SeqBAIJ));

  PetscCall(PetscObjectChangeTypeName((PetscObject)B,MATSEQBAIJMKL));
  *newmat = B;
  PetscFunctionReturn(0);
}

static PetscErrorCode MatAssemblyEnd_SeqBAIJMKL(Mat A, MatAssemblyType mode)
{
  PetscFunctionBegin;
  if (mode == MAT_FLUSH_ASSEMBLY) PetscFunctionReturn(0);
  PetscCall(MatAssemblyEnd_SeqBAIJ(A, mode));
  PetscCall(MatSeqBAIJMKL_create_mkl_handle(A));
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

.seealso: `MatCreate()`, `MatCreateSeqAIJ()`, `MatSetValues()`, `MatCreateBAIJ()`

@*/
PetscErrorCode  MatCreateSeqBAIJMKL(MPI_Comm comm,PetscInt bs,PetscInt m,PetscInt n,PetscInt nz,const PetscInt nnz[],Mat *A)
{
  PetscFunctionBegin;
  PetscCall(MatCreate(comm,A));
  PetscCall(MatSetSizes(*A,m,n,m,n));
  PetscCall(MatSetType(*A,MATSEQBAIJMKL));
  PetscCall(MatSeqBAIJSetPreallocation_SeqBAIJ(*A,bs,nz,(PetscInt*)nnz));
  PetscFunctionReturn(0);
}

PETSC_EXTERN PetscErrorCode MatCreate_SeqBAIJMKL(Mat A)
{
  PetscFunctionBegin;
  PetscCall(MatSetType(A,MATSEQBAIJ));
  PetscCall(MatConvert_SeqBAIJ_SeqBAIJMKL(A,MATSEQBAIJMKL,MAT_INPLACE_MATRIX,&A));
  PetscFunctionReturn(0);
}
