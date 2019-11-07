
/*
  Defines basic operations for the MATSEQAIJMKL matrix class.
  This class is derived from the MATSEQAIJ class and retains the
  compressed row storage (aka Yale sparse matrix format) but uses 
  sparse BLAS operations from the Intel Math Kernel Library (MKL) 
  wherever possible.
*/

#include <../src/mat/impls/aij/seq/aij.h>
#include <../src/mat/impls/aij/seq/aijmkl/aijmkl.h>
#include <mkl_spblas.h>

typedef struct {
  PetscBool           no_SpMV2;  /* If PETSC_TRUE, then don't use the MKL SpMV2 inspector-executor routines. */
  PetscBool           eager_inspection; /* If PETSC_TRUE, then call mkl_sparse_optimize() in MatDuplicate()/MatAssemblyEnd(). */
  PetscBool           sparse_optimized; /* If PETSC_TRUE, then mkl_sparse_optimize() has been called. */
  PetscObjectState    state;
#if defined(PETSC_HAVE_MKL_SPARSE_OPTIMIZE)
  sparse_matrix_t     csrA; /* "Handle" used by SpMV2 inspector-executor routines. */
  struct matrix_descr descr;
#endif
} Mat_SeqAIJMKL;

extern PetscErrorCode MatAssemblyEnd_SeqAIJ(Mat,MatAssemblyType);

PETSC_INTERN PetscErrorCode MatConvert_SeqAIJMKL_SeqAIJ(Mat A,MatType type,MatReuse reuse,Mat *newmat)
{
  /* This routine is only called to convert a MATAIJMKL to its base PETSc type, */
  /* so we will ignore 'MatType type'. */
  PetscErrorCode ierr;
  Mat            B       = *newmat;
#if defined(PETSC_HAVE_MKL_SPARSE_OPTIMIZE)
  Mat_SeqAIJMKL  *aijmkl=(Mat_SeqAIJMKL*)A->spptr;
#endif

  PetscFunctionBegin;
  if (reuse == MAT_INITIAL_MATRIX) {
    ierr = MatDuplicate(A,MAT_COPY_VALUES,&B);CHKERRQ(ierr);
  }

  /* Reset the original function pointers. */
  B->ops->duplicate        = MatDuplicate_SeqAIJ;
  B->ops->assemblyend      = MatAssemblyEnd_SeqAIJ;
  B->ops->destroy          = MatDestroy_SeqAIJ;
  B->ops->mult             = MatMult_SeqAIJ;
  B->ops->multtranspose    = MatMultTranspose_SeqAIJ;
  B->ops->multadd          = MatMultAdd_SeqAIJ;
  B->ops->multtransposeadd = MatMultTransposeAdd_SeqAIJ;
  B->ops->matmult          = MatMatMult_SeqAIJ_SeqAIJ;
  B->ops->matmultnumeric   = MatMatMultNumeric_SeqAIJ_SeqAIJ;
  B->ops->ptap             = MatPtAP_SeqAIJ_SeqAIJ;
  B->ops->ptapnumeric      = MatPtAPNumeric_SeqAIJ_SeqAIJ;
  B->ops->transposematmult = MatTransposeMatMult_SeqAIJ_SeqAIJ;

  ierr = PetscObjectComposeFunction((PetscObject)B,"MatConvert_seqaijmkl_seqaij_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)B,"MatMatMult_seqdense_seqaijmkl_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)B,"MatMatMultSymbolic_seqdense_seqaijmkl_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)B,"MatMatMultNumeric_seqdense_seqaijmkl_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)B,"MatPtAP_is_seqaijmkl_C",NULL);CHKERRQ(ierr);
#if defined(PETSC_HAVE_HYPRE)
  ierr = PetscObjectComposeFunction((PetscObject)B,"MatMatMatMult_transpose_seqaijmkl_seqaijmkl_C",NULL);CHKERRQ(ierr);
#endif
#if defined(PETSC_HAVE_MKL_SPARSE_OPTIMIZE)
  if(!aijmkl->no_SpMV2) {
    ierr = PetscObjectComposeFunction((PetscObject)B,"MatMatMult_seqaijmkl_seqaijmkl_C",NULL);CHKERRQ(ierr);
#if defined(PETSC_HAVE_MKL_SPARSE_SP2M_FEATURE)
    ierr = PetscObjectComposeFunction((PetscObject)B,"MatMatMultNumeric_seqaijmkl_seqaijmkl_C",NULL);CHKERRQ(ierr);
#endif
    ierr = PetscObjectComposeFunction((PetscObject)B,"MatTransposeMatMult_seqaijmkl_seqaijmkl_C",NULL);CHKERRQ(ierr);
  }

  /* Free everything in the Mat_SeqAIJMKL data structure. Currently, this 
   * simply involves destroying the MKL sparse matrix handle and then freeing 
   * the spptr pointer. */
  if (reuse == MAT_INITIAL_MATRIX) aijmkl = (Mat_SeqAIJMKL*)B->spptr;

  if (aijmkl->sparse_optimized) {
    sparse_status_t stat;
    stat = mkl_sparse_destroy(aijmkl->csrA);
    if (stat != SPARSE_STATUS_SUCCESS) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_LIB,"Intel MKL error: unable to set hints/complete mkl_sparse_optimize"); 
  }
#endif /* PETSC_HAVE_MKL_SPARSE_OPTIMIZE */
  ierr = PetscFree(B->spptr);CHKERRQ(ierr);

  /* Change the type of B to MATSEQAIJ. */
  ierr = PetscObjectChangeTypeName((PetscObject)B, MATSEQAIJ);CHKERRQ(ierr);

  *newmat = B;
  PetscFunctionReturn(0);
}

PetscErrorCode MatDestroy_SeqAIJMKL(Mat A)
{
  PetscErrorCode ierr;
  Mat_SeqAIJMKL  *aijmkl = (Mat_SeqAIJMKL*) A->spptr;

  PetscFunctionBegin;

  /* If MatHeaderMerge() was used, then this SeqAIJMKL matrix will not have an 
   * spptr pointer. */
  if (aijmkl) {
    /* Clean up everything in the Mat_SeqAIJMKL data structure, then free A->spptr. */
#if defined(PETSC_HAVE_MKL_SPARSE_OPTIMIZE)
    if (aijmkl->sparse_optimized) {
      sparse_status_t stat = SPARSE_STATUS_SUCCESS;
      stat = mkl_sparse_destroy(aijmkl->csrA);
      if (stat != SPARSE_STATUS_SUCCESS) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_LIB,"Intel MKL error: error in mkl_sparse_destroy"); 
    }
#endif /* PETSC_HAVE_MKL_SPARSE_OPTIMIZE */
    ierr = PetscFree(A->spptr);CHKERRQ(ierr);
  }

  /* Change the type of A back to SEQAIJ and use MatDestroy_SeqAIJ()
   * to destroy everything that remains. */
  ierr = PetscObjectChangeTypeName((PetscObject)A, MATSEQAIJ);CHKERRQ(ierr);
  /* Note that I don't call MatSetType().  I believe this is because that
   * is only to be called when *building* a matrix.  I could be wrong, but
   * that is how things work for the SuperLU matrix class. */
  ierr = MatDestroy_SeqAIJ(A);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* MatSeqAIJKL_create_mkl_handle(), if called with an AIJMKL matrix that has not had mkl_sparse_optimize() called for it, 
 * creates an MKL sparse matrix handle from the AIJ arrays and calls mkl_sparse_optimize().
 * If called with an AIJMKL matrix for which aijmkl->sparse_optimized == PETSC_TRUE, then it destroys the old matrix 
 * handle, creates a new one, and then calls mkl_sparse_optimize().
 * Although in normal MKL usage it is possible to have a valid matrix handle on which mkl_sparse_optimize() has not been 
 * called, for AIJMKL the handle creation and optimization step always occur together, so we don't handle the case of 
 * an unoptimized matrix handle here. */
PETSC_INTERN PetscErrorCode MatSeqAIJMKL_create_mkl_handle(Mat A)
{
#if !defined(PETSC_HAVE_MKL_SPARSE_OPTIMIZE)
  /* If the MKL library does not have mkl_sparse_optimize(), then this routine 
   * does nothing. We make it callable anyway in this case because it cuts 
   * down on littering the code with #ifdefs. */
  PetscFunctionBegin;
  PetscFunctionReturn(0);
#else
  Mat_SeqAIJ       *a = (Mat_SeqAIJ*)A->data;
  Mat_SeqAIJMKL    *aijmkl = (Mat_SeqAIJMKL*)A->spptr;
  PetscInt         m,n;
  MatScalar        *aa;
  PetscInt         *aj,*ai;
  sparse_status_t  stat;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
#if !defined(PETSC_MKL_SPBLAS_DEPRECATED)
  /* For MKL versions that still support the old, non-inspector-executor interfaces versions, we simply exit here if the no_SpMV2
   * option has been specified. For versions that have deprecated the old interfaces (version 18, update 2 and later), we must
   * use the new inspector-executor interfaces, but we can still use the old, non-inspector-executor code by not calling
   * mkl_sparse_optimize() later. */
  if (aijmkl->no_SpMV2) PetscFunctionReturn(0);
#endif

  if (aijmkl->sparse_optimized) {
    /* Matrix has been previously assembled and optimized. Must destroy old
     * matrix handle before running the optimization step again. */
    stat = mkl_sparse_destroy(aijmkl->csrA);
    if (stat != SPARSE_STATUS_SUCCESS) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_LIB,"Intel MKL error: error in mkl_sparse_destroy"); 
  }
  aijmkl->sparse_optimized = PETSC_FALSE;

  /* Now perform the SpMV2 setup and matrix optimization. */
  aijmkl->descr.type        = SPARSE_MATRIX_TYPE_GENERAL;
  aijmkl->descr.mode        = SPARSE_FILL_MODE_LOWER;
  aijmkl->descr.diag        = SPARSE_DIAG_NON_UNIT;
  m = A->rmap->n;
  n = A->cmap->n;
  aj   = a->j;  /* aj[k] gives column index for element aa[k]. */
  aa   = a->a;  /* Nonzero elements stored row-by-row. */
  ai   = a->i;  /* ai[k] is the position in aa and aj where row k starts. */
  if ((a->nz!=0) & !(A->structure_only)) {
    /* Create a new, optimized sparse matrix handle only if the matrix has nonzero entries.
     * The MKL sparse-inspector executor routines don't like being passed an empty matrix. */
    stat = mkl_sparse_x_create_csr(&aijmkl->csrA,SPARSE_INDEX_BASE_ZERO,m,n,ai,ai+1,aj,aa);
    if (stat != SPARSE_STATUS_SUCCESS) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_LIB,"Intel MKL error: unable to create matrix handle");
    stat = mkl_sparse_set_mv_hint(aijmkl->csrA,SPARSE_OPERATION_NON_TRANSPOSE,aijmkl->descr,1000);
    if (stat != SPARSE_STATUS_SUCCESS) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_LIB,"Intel MKL error: unable to set mv_hint");
    stat = mkl_sparse_set_memory_hint(aijmkl->csrA,SPARSE_MEMORY_AGGRESSIVE);
    if (stat != SPARSE_STATUS_SUCCESS) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_LIB,"Intel MKL error: unable to set memory_hint");
    if (!aijmkl->no_SpMV2) {
      stat = mkl_sparse_optimize(aijmkl->csrA);
      if (stat != SPARSE_STATUS_SUCCESS) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_LIB,"Intel MKL error: unable to complete mkl_sparse_optimize");
    }
    aijmkl->sparse_optimized = PETSC_TRUE;
    ierr = PetscObjectStateGet((PetscObject)A,&(aijmkl->state));CHKERRQ(ierr);
  }

  PetscFunctionReturn(0);
#endif
}

/* MatSeqAIJMKL_create_from_mkl_handle() creates a sequential AIJMKL matrix from an MKL sparse matrix handle. 
 * We need this to implement MatMatMult() using the MKL inspector-executor routines, which return an (unoptimized) 
 * matrix handle.
 * Note: This routine simply destroys and replaces the original matrix if MAT_REUSE_MATRIX has been specified, as
 * there is no good alternative. */
#if defined(PETSC_HAVE_MKL_SPARSE_OPTIMIZE)
PETSC_INTERN PetscErrorCode MatSeqAIJMKL_create_from_mkl_handle(MPI_Comm comm,sparse_matrix_t csrA,MatReuse reuse,Mat *mat)
{
  PetscErrorCode      ierr;
  sparse_status_t     stat;
  sparse_index_base_t indexing;
  PetscInt            nrows, ncols;
  PetscInt            *aj,*ai,*dummy;
  MatScalar           *aa;
  Mat                 A;
  Mat_SeqAIJMKL       *aijmkl;

  /* Note: Must pass in &dummy below since MKL can't accept NULL for this output array we don't actually want. */
  stat = mkl_sparse_x_export_csr(csrA,&indexing,&nrows,&ncols,&ai,&dummy,&aj,&aa);
  if (stat != SPARSE_STATUS_SUCCESS) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_LIB,"Intel MKL error: unable to complete mkl_sparse_x_export_csr()");

  if (reuse == MAT_REUSE_MATRIX) {
    ierr = MatDestroy(mat);CHKERRQ(ierr);
  }
  ierr = MatCreate(comm,&A);CHKERRQ(ierr);
  ierr = MatSetType(A,MATSEQAIJ);CHKERRQ(ierr);
  ierr = MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,nrows,ncols);CHKERRQ(ierr);
  /* We use MatSeqAIJSetPreallocationCSR() instead of MatCreateSeqAIJWithArrays() because we must copy the arrays exported 
   * from MKL; MKL developers tell us that modifying the arrays may cause unexpected results when using the MKL handle, and
   * they will be destroyed when the MKL handle is destroyed.
   * (In the interest of reducing memory consumption in future, can we figure out good ways to deal with this?) */
  ierr = MatSeqAIJSetPreallocationCSR(A,ai,aj,aa);CHKERRQ(ierr);

  /* We now have an assembled sequential AIJ matrix created from copies of the exported arrays from the MKL matrix handle.
   * Now turn it into a MATSEQAIJMKL. */
  ierr = MatConvert_SeqAIJ_SeqAIJMKL(A,MATSEQAIJMKL,MAT_INPLACE_MATRIX,&A);CHKERRQ(ierr);

  aijmkl = (Mat_SeqAIJMKL*) A->spptr;
  aijmkl->csrA = csrA;

  /* The below code duplicates much of what is in MatSeqAIJKL_create_mkl_handle(). I dislike this code duplication, but
   * MatSeqAIJMKL_create_mkl_handle() cannot be used because we don't need to create a handle -- we've already got one, 
   * and just need to be able to run the MKL optimization step. */
  aijmkl->descr.type        = SPARSE_MATRIX_TYPE_GENERAL;
  aijmkl->descr.mode        = SPARSE_FILL_MODE_LOWER;
  aijmkl->descr.diag        = SPARSE_DIAG_NON_UNIT;
  stat = mkl_sparse_set_mv_hint(aijmkl->csrA,SPARSE_OPERATION_NON_TRANSPOSE,aijmkl->descr,1000);
  if (stat != SPARSE_STATUS_SUCCESS) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_LIB,"Intel MKL error: unable to set mv_hint");
  stat = mkl_sparse_set_memory_hint(aijmkl->csrA,SPARSE_MEMORY_AGGRESSIVE);
  if (stat != SPARSE_STATUS_SUCCESS) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_LIB,"Intel MKL error: unable to set memory_hint");
  if (!aijmkl->no_SpMV2) {
    stat = mkl_sparse_optimize(aijmkl->csrA);
    if (stat != SPARSE_STATUS_SUCCESS) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_LIB,"Intel MKL error: unable to complete mkl_sparse_optimize");
  }
  aijmkl->sparse_optimized = PETSC_TRUE;
  ierr = PetscObjectStateGet((PetscObject)A,&(aijmkl->state));CHKERRQ(ierr);

  *mat = A;
  PetscFunctionReturn(0);
}
#endif /* PETSC_HAVE_MKL_SPARSE_OPTIMIZE */

/* MatSeqAIJMKL_update_from_mkl_handle() updates the matrix values array from the contents of the associated MKL sparse matrix handle.
 * This is needed after mkl_sparse_sp2m() with SPARSE_STAGE_FINALIZE_MULT has been used to compute new values of the matrix in 
 * MatMatMultNumeric(). */
#if defined(PETSC_HAVE_MKL_SPARSE_OPTIMIZE)
PETSC_INTERN PetscErrorCode MatSeqAIJMKL_update_from_mkl_handle(Mat A)
{
  PetscInt            i;
  PetscInt            nrows,ncols;
  PetscInt            nz;
  PetscInt            *ai,*aj,*dummy;
  PetscScalar         *aa;
  PetscErrorCode      ierr;
  Mat_SeqAIJMKL       *aijmkl;
  sparse_status_t     stat;
  sparse_index_base_t indexing;

  aijmkl = (Mat_SeqAIJMKL*) A->spptr;

  /* Note: Must pass in &dummy below since MKL can't accept NULL for this output array we don't actually want. */
  stat = mkl_sparse_x_export_csr(aijmkl->csrA,&indexing,&nrows,&ncols,&ai,&dummy,&aj,&aa);
  if (stat != SPARSE_STATUS_SUCCESS) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_LIB,"Intel MKL error: unable to complete mkl_sparse_x_export_csr()");

  /* We can't just do a copy from the arrays exported by MKL to those used for the PETSc AIJ storage, because the MKL and PETSc 
   * representations differ in small ways (e.g., more explicit nonzeros per row due to preallocation). */
  for (i=0; i<nrows; i++) {
    nz = ai[i+1] - ai[i];
    ierr = MatSetValues_SeqAIJ(A, 1, &i, nz, aj+ai[i], aa+ai[i], INSERT_VALUES);CHKERRQ(ierr);
  }

  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  ierr = PetscObjectStateGet((PetscObject)A,&(aijmkl->state));CHKERRQ(ierr);
  /* We mark our matrix as having a valid, optimized MKL handle.
   * TODO: It is valid, but I am not sure if it is optimized. Need to ask MKL developers. */
  aijmkl->sparse_optimized = PETSC_TRUE;

  PetscFunctionReturn(0);
}
#endif /* PETSC_HAVE_MKL_SPARSE_OPTIMIZE */

PetscErrorCode MatDuplicate_SeqAIJMKL(Mat A, MatDuplicateOption op, Mat *M)
{
  PetscErrorCode ierr;
  Mat_SeqAIJMKL  *aijmkl;
  Mat_SeqAIJMKL  *aijmkl_dest;

  PetscFunctionBegin;
  ierr = MatDuplicate_SeqAIJ(A,op,M);CHKERRQ(ierr);
  aijmkl      = (Mat_SeqAIJMKL*) A->spptr;
  aijmkl_dest = (Mat_SeqAIJMKL*) (*M)->spptr;
  ierr = PetscArraycpy(aijmkl_dest,aijmkl,1);CHKERRQ(ierr);
  aijmkl_dest->sparse_optimized = PETSC_FALSE;
  if (aijmkl->eager_inspection) {
    ierr = MatSeqAIJMKL_create_mkl_handle(A);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode MatAssemblyEnd_SeqAIJMKL(Mat A, MatAssemblyType mode)
{
  PetscErrorCode  ierr;
  Mat_SeqAIJ      *a = (Mat_SeqAIJ*)A->data;
  Mat_SeqAIJMKL   *aijmkl;

  PetscFunctionBegin;
  if (mode == MAT_FLUSH_ASSEMBLY) PetscFunctionReturn(0);

  /* Since a MATSEQAIJMKL matrix is really just a MATSEQAIJ with some
   * extra information and some different methods, call the AssemblyEnd 
   * routine for a MATSEQAIJ.
   * I'm not sure if this is the best way to do this, but it avoids
   * a lot of code duplication. */
  a->inode.use = PETSC_FALSE;  /* Must disable: otherwise the MKL routines won't get used. */
  ierr = MatAssemblyEnd_SeqAIJ(A, mode);CHKERRQ(ierr);

  /* If the user has requested "eager" inspection, create the optimized MKL sparse handle (if needed; the function checks).
   * (The default is to do "lazy" inspection, deferring this until something like MatMult() is called.) */
  aijmkl = (Mat_SeqAIJMKL*) A->spptr;
  if (aijmkl->eager_inspection) {
    ierr = MatSeqAIJMKL_create_mkl_handle(A);CHKERRQ(ierr);
  }

  PetscFunctionReturn(0);
}

#if !defined(PETSC_MKL_SPBLAS_DEPRECATED)
PetscErrorCode MatMult_SeqAIJMKL(Mat A,Vec xx,Vec yy)
{
  Mat_SeqAIJ        *a = (Mat_SeqAIJ*)A->data;
  const PetscScalar *x;
  PetscScalar       *y;
  const MatScalar   *aa;
  PetscErrorCode    ierr;
  PetscInt          m=A->rmap->n;
  PetscInt          n=A->cmap->n;
  PetscScalar       alpha = 1.0;
  PetscScalar       beta = 0.0;
  const PetscInt    *aj,*ai;
  char              matdescra[6];


  /* Variables not in MatMult_SeqAIJ. */
  char transa = 'n';  /* Used to indicate to MKL that we are not computing the transpose product. */

  PetscFunctionBegin;
  matdescra[0] = 'g';  /* Indicates to MKL that we using a general CSR matrix. */
  matdescra[3] = 'c';  /* Indicates to MKL that we use C-style (0-based) indexing. */
  ierr = VecGetArrayRead(xx,&x);CHKERRQ(ierr);
  ierr = VecGetArray(yy,&y);CHKERRQ(ierr);
  aj   = a->j;  /* aj[k] gives column index for element aa[k]. */
  aa   = a->a;  /* Nonzero elements stored row-by-row. */
  ai   = a->i;  /* ai[k] is the position in aa and aj where row k starts. */

  /* Call MKL sparse BLAS routine to do the MatMult. */
  mkl_xcsrmv(&transa,&m,&n,&alpha,matdescra,aa,aj,ai,ai+1,x,&beta,y);

  ierr = PetscLogFlops(2.0*a->nz - a->nonzerorowcnt);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(xx,&x);CHKERRQ(ierr);
  ierr = VecRestoreArray(yy,&y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
#endif

#if defined(PETSC_HAVE_MKL_SPARSE_OPTIMIZE)
PetscErrorCode MatMult_SeqAIJMKL_SpMV2(Mat A,Vec xx,Vec yy)
{
  Mat_SeqAIJ        *a = (Mat_SeqAIJ*)A->data;
  Mat_SeqAIJMKL     *aijmkl=(Mat_SeqAIJMKL*)A->spptr;
  const PetscScalar *x;
  PetscScalar       *y;
  PetscErrorCode    ierr;
  sparse_status_t   stat = SPARSE_STATUS_SUCCESS;
  PetscObjectState  state;

  PetscFunctionBegin;

  /* If there are no nonzero entries, zero yy and return immediately. */
  if(!a->nz) {
    PetscInt i;
    PetscInt m=A->rmap->n;
    ierr = VecGetArray(yy,&y);CHKERRQ(ierr);
    for (i=0; i<m; i++) {
      y[i] = 0.0;
    }
    ierr = VecRestoreArray(yy,&y);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }

  ierr = VecGetArrayRead(xx,&x);CHKERRQ(ierr);
  ierr = VecGetArray(yy,&y);CHKERRQ(ierr);

  /* In some cases, we get to this point without mkl_sparse_optimize() having been called, so we check and then call 
   * it if needed. Eventually, when everything in PETSc is properly updating the matrix state, we should probably 
   * take a "lazy" approach to creation/updating of the MKL matrix handle and plan to always do it here (when needed). */
  ierr = PetscObjectStateGet((PetscObject)A,&state);CHKERRQ(ierr);
  if (!aijmkl->sparse_optimized || aijmkl->state != state) {
    MatSeqAIJMKL_create_mkl_handle(A);
  }

  /* Call MKL SpMV2 executor routine to do the MatMult. */
  stat = mkl_sparse_x_mv(SPARSE_OPERATION_NON_TRANSPOSE,1.0,aijmkl->csrA,aijmkl->descr,x,0.0,y);
  if (stat != SPARSE_STATUS_SUCCESS) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_LIB,"Intel MKL error: error in mkl_sparse_x_mv"); 
  
  ierr = PetscLogFlops(2.0*a->nz - a->nonzerorowcnt);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(xx,&x);CHKERRQ(ierr);
  ierr = VecRestoreArray(yy,&y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
#endif /* PETSC_HAVE_MKL_SPARSE_OPTIMIZE */

#if !defined(PETSC_MKL_SPBLAS_DEPRECATED)
PetscErrorCode MatMultTranspose_SeqAIJMKL(Mat A,Vec xx,Vec yy)
{
  Mat_SeqAIJ        *a = (Mat_SeqAIJ*)A->data;
  const PetscScalar *x;
  PetscScalar       *y;
  const MatScalar   *aa;
  PetscErrorCode    ierr;
  PetscInt          m=A->rmap->n;
  PetscInt          n=A->cmap->n;
  PetscScalar       alpha = 1.0;
  PetscScalar       beta = 0.0;
  const PetscInt    *aj,*ai;
  char              matdescra[6];

  /* Variables not in MatMultTranspose_SeqAIJ. */
  char transa = 't';  /* Used to indicate to MKL that we are computing the transpose product. */

  PetscFunctionBegin;
  matdescra[0] = 'g';  /* Indicates to MKL that we using a general CSR matrix. */
  matdescra[3] = 'c';  /* Indicates to MKL that we use C-style (0-based) indexing. */
  ierr = VecGetArrayRead(xx,&x);CHKERRQ(ierr);
  ierr = VecGetArray(yy,&y);CHKERRQ(ierr);
  aj   = a->j;  /* aj[k] gives column index for element aa[k]. */
  aa   = a->a;  /* Nonzero elements stored row-by-row. */
  ai   = a->i;  /* ai[k] is the position in aa and aj where row k starts. */

  /* Call MKL sparse BLAS routine to do the MatMult. */
  mkl_xcsrmv(&transa,&m,&n,&alpha,matdescra,aa,aj,ai,ai+1,x,&beta,y);

  ierr = PetscLogFlops(2.0*a->nz - a->nonzerorowcnt);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(xx,&x);CHKERRQ(ierr);
  ierr = VecRestoreArray(yy,&y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
#endif

#if defined(PETSC_HAVE_MKL_SPARSE_OPTIMIZE)
PetscErrorCode MatMultTranspose_SeqAIJMKL_SpMV2(Mat A,Vec xx,Vec yy)
{
  Mat_SeqAIJ        *a = (Mat_SeqAIJ*)A->data;
  Mat_SeqAIJMKL     *aijmkl=(Mat_SeqAIJMKL*)A->spptr;
  const PetscScalar *x;
  PetscScalar       *y;
  PetscErrorCode    ierr;
  sparse_status_t   stat;
  PetscObjectState  state;

  PetscFunctionBegin;

  /* If there are no nonzero entries, zero yy and return immediately. */
  if(!a->nz) {
    PetscInt i;
    PetscInt n=A->cmap->n;
    ierr = VecGetArray(yy,&y);CHKERRQ(ierr);
    for (i=0; i<n; i++) {
      y[i] = 0.0;
    }
    ierr = VecRestoreArray(yy,&y);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }

  ierr = VecGetArrayRead(xx,&x);CHKERRQ(ierr);
  ierr = VecGetArray(yy,&y);CHKERRQ(ierr);

  /* In some cases, we get to this point without mkl_sparse_optimize() having been called, so we check and then call 
   * it if needed. Eventually, when everything in PETSc is properly updating the matrix state, we should probably 
   * take a "lazy" approach to creation/updating of the MKL matrix handle and plan to always do it here (when needed). */
  ierr = PetscObjectStateGet((PetscObject)A,&state);CHKERRQ(ierr);
  if (!aijmkl->sparse_optimized || aijmkl->state != state) {
    MatSeqAIJMKL_create_mkl_handle(A);
  }

  /* Call MKL SpMV2 executor routine to do the MatMultTranspose. */
  stat = mkl_sparse_x_mv(SPARSE_OPERATION_TRANSPOSE,1.0,aijmkl->csrA,aijmkl->descr,x,0.0,y);
  if (stat != SPARSE_STATUS_SUCCESS) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_LIB,"Intel MKL error: error in mkl_sparse_x_mv"); 
  
  ierr = PetscLogFlops(2.0*a->nz - a->nonzerorowcnt);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(xx,&x);CHKERRQ(ierr);
  ierr = VecRestoreArray(yy,&y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
#endif /* PETSC_HAVE_MKL_SPARSE_OPTIMIZE */

#if !defined(PETSC_MKL_SPBLAS_DEPRECATED)
PetscErrorCode MatMultAdd_SeqAIJMKL(Mat A,Vec xx,Vec yy,Vec zz)
{
  Mat_SeqAIJ        *a = (Mat_SeqAIJ*)A->data;
  const PetscScalar *x;
  PetscScalar       *y,*z;
  const MatScalar   *aa;
  PetscErrorCode    ierr;
  PetscInt          m=A->rmap->n;
  PetscInt          n=A->cmap->n;
  const PetscInt    *aj,*ai;
  PetscInt          i;

  /* Variables not in MatMultAdd_SeqAIJ. */
  char              transa = 'n';  /* Used to indicate to MKL that we are not computing the transpose product. */
  PetscScalar       alpha = 1.0;
  PetscScalar       beta;
  char              matdescra[6];

  PetscFunctionBegin;
  matdescra[0] = 'g';  /* Indicates to MKL that we using a general CSR matrix. */
  matdescra[3] = 'c';  /* Indicates to MKL that we use C-style (0-based) indexing. */

  ierr = VecGetArrayRead(xx,&x);CHKERRQ(ierr);
  ierr = VecGetArrayPair(yy,zz,&y,&z);CHKERRQ(ierr);
  aj   = a->j;  /* aj[k] gives column index for element aa[k]. */
  aa   = a->a;  /* Nonzero elements stored row-by-row. */
  ai   = a->i;  /* ai[k] is the position in aa and aj where row k starts. */

  /* Call MKL sparse BLAS routine to do the MatMult. */
  if (zz == yy) {
    /* If zz and yy are the same vector, we can use MKL's mkl_xcsrmv(), which calculates y = alpha*A*x + beta*y. */
    beta = 1.0;
    mkl_xcsrmv(&transa,&m,&n,&alpha,matdescra,aa,aj,ai,ai+1,x,&beta,z);
  } else {
    /* zz and yy are different vectors, so call MKL's mkl_xcsrmv() with beta=0, then add the result to z. 
     * MKL sparse BLAS does not have a MatMultAdd equivalent. */
    beta = 0.0;
    mkl_xcsrmv(&transa,&m,&n,&alpha,matdescra,aa,aj,ai,ai+1,x,&beta,z);
    for (i=0; i<m; i++) {
      z[i] += y[i];
    }
  }

  ierr = PetscLogFlops(2.0*a->nz);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(xx,&x);CHKERRQ(ierr);
  ierr = VecRestoreArrayPair(yy,zz,&y,&z);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
#endif

#if defined(PETSC_HAVE_MKL_SPARSE_OPTIMIZE)
PetscErrorCode MatMultAdd_SeqAIJMKL_SpMV2(Mat A,Vec xx,Vec yy,Vec zz)
{
  Mat_SeqAIJ        *a = (Mat_SeqAIJ*)A->data;
  Mat_SeqAIJMKL     *aijmkl=(Mat_SeqAIJMKL*)A->spptr;
  const PetscScalar *x;
  PetscScalar       *y,*z;
  PetscErrorCode    ierr;
  PetscInt          m=A->rmap->n;
  PetscInt          i;

  /* Variables not in MatMultAdd_SeqAIJ. */
  sparse_status_t   stat = SPARSE_STATUS_SUCCESS;
  PetscObjectState  state;

  PetscFunctionBegin;

  /* If there are no nonzero entries, set zz = yy and return immediately. */
  if(!a->nz) {
    PetscInt i;
    ierr = VecGetArrayPair(yy,zz,&y,&z);CHKERRQ(ierr);
    for (i=0; i<m; i++) {
      z[i] = y[i];
    }
    ierr = VecRestoreArrayPair(yy,zz,&y,&z);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }

  ierr = VecGetArrayRead(xx,&x);CHKERRQ(ierr);
  ierr = VecGetArrayPair(yy,zz,&y,&z);CHKERRQ(ierr);

  /* In some cases, we get to this point without mkl_sparse_optimize() having been called, so we check and then call 
   * it if needed. Eventually, when everything in PETSc is properly updating the matrix state, we should probably 
   * take a "lazy" approach to creation/updating of the MKL matrix handle and plan to always do it here (when needed). */
  ierr = PetscObjectStateGet((PetscObject)A,&state);CHKERRQ(ierr);
  if (!aijmkl->sparse_optimized || aijmkl->state != state) {
    MatSeqAIJMKL_create_mkl_handle(A);
  }

  /* Call MKL sparse BLAS routine to do the MatMult. */
  if (zz == yy) {
    /* If zz and yy are the same vector, we can use mkl_sparse_x_mv, which calculates y = alpha*A*x + beta*y, 
     * with alpha and beta both set to 1.0. */
    stat = mkl_sparse_x_mv(SPARSE_OPERATION_NON_TRANSPOSE,1.0,aijmkl->csrA,aijmkl->descr,x,1.0,z);
    if (stat != SPARSE_STATUS_SUCCESS) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_LIB,"Intel MKL error: error in mkl_sparse_x_mv"); 
  } else {
    /* zz and yy are different vectors, so we call mkl_sparse_x_mv with alpha=1.0 and beta=0.0, and then 
     * we add the contents of vector yy to the result; MKL sparse BLAS does not have a MatMultAdd equivalent. */
    stat = mkl_sparse_x_mv(SPARSE_OPERATION_NON_TRANSPOSE,1.0,aijmkl->csrA,aijmkl->descr,x,0.0,z);
    if (stat != SPARSE_STATUS_SUCCESS) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_LIB,"Intel MKL error: error in mkl_sparse_x_mv"); 
    for (i=0; i<m; i++) {
      z[i] += y[i];
    }
  }

  ierr = PetscLogFlops(2.0*a->nz);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(xx,&x);CHKERRQ(ierr);
  ierr = VecRestoreArrayPair(yy,zz,&y,&z);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
#endif /* PETSC_HAVE_MKL_SPARSE_OPTIMIZE */

#if !defined(PETSC_MKL_SPBLAS_DEPRECATED)
PetscErrorCode MatMultTransposeAdd_SeqAIJMKL(Mat A,Vec xx,Vec yy,Vec zz)
{
  Mat_SeqAIJ        *a = (Mat_SeqAIJ*)A->data;
  const PetscScalar *x;
  PetscScalar       *y,*z;
  const MatScalar   *aa;
  PetscErrorCode    ierr;
  PetscInt          m=A->rmap->n;
  PetscInt          n=A->cmap->n;
  const PetscInt    *aj,*ai;
  PetscInt          i;

  /* Variables not in MatMultTransposeAdd_SeqAIJ. */
  char transa = 't';  /* Used to indicate to MKL that we are computing the transpose product. */
  PetscScalar       alpha = 1.0;
  PetscScalar       beta;
  char              matdescra[6];

  PetscFunctionBegin;
  matdescra[0] = 'g';  /* Indicates to MKL that we using a general CSR matrix. */
  matdescra[3] = 'c';  /* Indicates to MKL that we use C-style (0-based) indexing. */

  ierr = VecGetArrayRead(xx,&x);CHKERRQ(ierr);
  ierr = VecGetArrayPair(yy,zz,&y,&z);CHKERRQ(ierr);
  aj   = a->j;  /* aj[k] gives column index for element aa[k]. */
  aa   = a->a;  /* Nonzero elements stored row-by-row. */
  ai   = a->i;  /* ai[k] is the position in aa and aj where row k starts. */

  /* Call MKL sparse BLAS routine to do the MatMult. */
  if (zz == yy) {
    /* If zz and yy are the same vector, we can use MKL's mkl_xcsrmv(), which calculates y = alpha*A*x + beta*y. */
    beta = 1.0;
    mkl_xcsrmv(&transa,&m,&n,&alpha,matdescra,aa,aj,ai,ai+1,x,&beta,z);
  } else {
    /* zz and yy are different vectors, so call MKL's mkl_xcsrmv() with beta=0, then add the result to z. 
     * MKL sparse BLAS does not have a MatMultAdd equivalent. */
    beta = 0.0;
    mkl_xcsrmv(&transa,&m,&n,&alpha,matdescra,aa,aj,ai,ai+1,x,&beta,z);
    for (i=0; i<n; i++) {
      z[i] += y[i];
    }
  }

  ierr = PetscLogFlops(2.0*a->nz);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(xx,&x);CHKERRQ(ierr);
  ierr = VecRestoreArrayPair(yy,zz,&y,&z);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
#endif

#if defined(PETSC_HAVE_MKL_SPARSE_OPTIMIZE)
PetscErrorCode MatMultTransposeAdd_SeqAIJMKL_SpMV2(Mat A,Vec xx,Vec yy,Vec zz)
{
  Mat_SeqAIJ        *a = (Mat_SeqAIJ*)A->data;
  Mat_SeqAIJMKL     *aijmkl=(Mat_SeqAIJMKL*)A->spptr;
  const PetscScalar *x;
  PetscScalar       *y,*z;
  PetscErrorCode    ierr;
  PetscInt          n=A->cmap->n;
  PetscInt          i;
  PetscObjectState  state;

  /* Variables not in MatMultTransposeAdd_SeqAIJ. */
  sparse_status_t stat = SPARSE_STATUS_SUCCESS;

  PetscFunctionBegin;

  /* If there are no nonzero entries, set zz = yy and return immediately. */
  if(!a->nz) {
    PetscInt i;
    ierr = VecGetArrayPair(yy,zz,&y,&z);CHKERRQ(ierr);
    for (i=0; i<n; i++) {
      z[i] = y[i];
    }
    ierr = VecRestoreArrayPair(yy,zz,&y,&z);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }

  ierr = VecGetArrayRead(xx,&x);CHKERRQ(ierr);
  ierr = VecGetArrayPair(yy,zz,&y,&z);CHKERRQ(ierr);

  /* In some cases, we get to this point without mkl_sparse_optimize() having been called, so we check and then call 
   * it if needed. Eventually, when everything in PETSc is properly updating the matrix state, we should probably 
   * take a "lazy" approach to creation/updating of the MKL matrix handle and plan to always do it here (when needed). */
  ierr = PetscObjectStateGet((PetscObject)A,&state);CHKERRQ(ierr);
  if (!aijmkl->sparse_optimized || aijmkl->state != state) {
    MatSeqAIJMKL_create_mkl_handle(A);
  }

  /* Call MKL sparse BLAS routine to do the MatMult. */
  if (zz == yy) {
    /* If zz and yy are the same vector, we can use mkl_sparse_x_mv, which calculates y = alpha*A*x + beta*y, 
     * with alpha and beta both set to 1.0. */
    stat = mkl_sparse_x_mv(SPARSE_OPERATION_TRANSPOSE,1.0,aijmkl->csrA,aijmkl->descr,x,1.0,z);
    if (stat != SPARSE_STATUS_SUCCESS) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_LIB,"Intel MKL error: error in mkl_sparse_x_mv"); 
  } else {
    /* zz and yy are different vectors, so we call mkl_sparse_x_mv with alpha=1.0 and beta=0.0, and then 
     * we add the contents of vector yy to the result; MKL sparse BLAS does not have a MatMultAdd equivalent. */
    stat = mkl_sparse_x_mv(SPARSE_OPERATION_TRANSPOSE,1.0,aijmkl->csrA,aijmkl->descr,x,0.0,z);
    if (stat != SPARSE_STATUS_SUCCESS) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_LIB,"Intel MKL error: error in mkl_sparse_x_mv"); 
    for (i=0; i<n; i++) {
      z[i] += y[i];
    }
  }

  ierr = PetscLogFlops(2.0*a->nz);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(xx,&x);CHKERRQ(ierr);
  ierr = VecRestoreArrayPair(yy,zz,&y,&z);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
#endif /* PETSC_HAVE_MKL_SPARSE_OPTIMIZE */

#if defined(PETSC_HAVE_MKL_SPARSE_OPTIMIZE)
/* Note that this code currently doesn't actually get used when MatMatMult() is called with MAT_REUSE_MATRIX, because 
 * the MatMatMult() interface code calls MatMatMultNumeric() in this case. 
 * For releases of MKL prior to version 18, update 2:
 * MKL has no notion of separately callable symbolic vs. numeric phases of sparse matrix-matrix multiply, so in the 
 * MAT_REUSE_MATRIX case, the SeqAIJ routines end up being used. Even though this means that the (hopefully more 
 * optimized) MKL routines do not get used, this probably is best because the MKL routines would waste time re-computing 
 * the symbolic portion, whereas the native PETSc SeqAIJ routines will avoid this. */
PetscErrorCode MatMatMult_SeqAIJMKL_SeqAIJMKL_SpMV2(Mat A,Mat B,MatReuse scall,PetscReal fill,Mat*C)
{
  Mat_SeqAIJMKL    *a, *b;
  sparse_matrix_t  csrA, csrB, csrC;
  PetscErrorCode   ierr;
  sparse_status_t  stat = SPARSE_STATUS_SUCCESS;
  PetscObjectState state;

  PetscFunctionBegin;
  a = (Mat_SeqAIJMKL*)A->spptr;
  b = (Mat_SeqAIJMKL*)B->spptr;
  ierr = PetscObjectStateGet((PetscObject)A,&state);CHKERRQ(ierr);
  if (!a->sparse_optimized || a->state != state) {
    MatSeqAIJMKL_create_mkl_handle(A);
  }
  ierr = PetscObjectStateGet((PetscObject)B,&state);CHKERRQ(ierr);
  if (!b->sparse_optimized || b->state != state) {
    MatSeqAIJMKL_create_mkl_handle(B);
  }
  csrA = a->csrA;
  csrB = b->csrA;

  stat = mkl_sparse_spmm(SPARSE_OPERATION_NON_TRANSPOSE,csrA,csrB,&csrC);
  if (stat != SPARSE_STATUS_SUCCESS) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_LIB,"Intel MKL error: unable to complete sparse matrix-matrix multiply");

  ierr = MatSeqAIJMKL_create_from_mkl_handle(PETSC_COMM_SELF,csrC,scall,C);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}
#endif /* PETSC_HAVE_MKL_SPARSE_OPTIMIZE */

#if defined(PETSC_HAVE_MKL_SPARSE_SP2M_FEATURE)
PetscErrorCode MatMatMultNumeric_SeqAIJMKL_SeqAIJMKL_SpMV2(Mat A,Mat B,Mat C)
{
  Mat_SeqAIJMKL       *a, *b, *c;
  sparse_matrix_t     csrA, csrB, csrC;
  PetscErrorCode      ierr;
  sparse_status_t     stat = SPARSE_STATUS_SUCCESS;
  struct matrix_descr descr_type_gen;
  PetscObjectState    state;

  PetscFunctionBegin;
  a = (Mat_SeqAIJMKL*)A->spptr;
  b = (Mat_SeqAIJMKL*)B->spptr;
  c = (Mat_SeqAIJMKL*)C->spptr;
  ierr = PetscObjectStateGet((PetscObject)A,&state);CHKERRQ(ierr);
  if (!a->sparse_optimized || a->state != state) {
    MatSeqAIJMKL_create_mkl_handle(A);
  }
  ierr = PetscObjectStateGet((PetscObject)B,&state);CHKERRQ(ierr);
  if (!b->sparse_optimized || b->state != state) {
    MatSeqAIJMKL_create_mkl_handle(B);
  }
  csrA = a->csrA;
  csrB = b->csrA;
  csrC = c->csrA;
  descr_type_gen.type = SPARSE_MATRIX_TYPE_GENERAL;

  stat = mkl_sparse_sp2m(SPARSE_OPERATION_NON_TRANSPOSE,descr_type_gen,csrA,
                         SPARSE_OPERATION_NON_TRANSPOSE,descr_type_gen,csrB,
                         SPARSE_STAGE_FINALIZE_MULT,&csrC);

  if (stat != SPARSE_STATUS_SUCCESS) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_LIB,"Intel MKL error: unable to complete numerical stage of sparse matrix-matrix multiply");

  /* Have to update the PETSc AIJ representation for matrix C from contents of MKL handle. */
  ierr = MatSeqAIJMKL_update_from_mkl_handle(C);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}
#endif /* PETSC_HAVE_MKL_SPARSE_SP2M_FEATURE */

#if defined(PETSC_HAVE_MKL_SPARSE_OPTIMIZE)
PetscErrorCode MatTransposeMatMult_SeqAIJMKL_SeqAIJMKL_SpMV2(Mat A,Mat B,MatReuse scall,PetscReal fill,Mat*C)
{
  Mat_SeqAIJMKL    *a, *b;
  sparse_matrix_t  csrA, csrB, csrC;
  PetscErrorCode   ierr;
  sparse_status_t  stat = SPARSE_STATUS_SUCCESS;
  PetscObjectState state;

  PetscFunctionBegin;
  a = (Mat_SeqAIJMKL*)A->spptr;
  b = (Mat_SeqAIJMKL*)B->spptr;
  ierr = PetscObjectStateGet((PetscObject)A,&state);CHKERRQ(ierr);
  if (!a->sparse_optimized || a->state != state) {
    MatSeqAIJMKL_create_mkl_handle(A);
  }
  ierr = PetscObjectStateGet((PetscObject)B,&state);CHKERRQ(ierr);
  if (!b->sparse_optimized || b->state != state) {
    MatSeqAIJMKL_create_mkl_handle(B);
  }
  csrA = a->csrA;
  csrB = b->csrA;

  stat = mkl_sparse_spmm(SPARSE_OPERATION_TRANSPOSE,csrA,csrB,&csrC);
  if (stat != SPARSE_STATUS_SUCCESS) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_LIB,"Intel MKL error: unable to complete sparse matrix-matrix multiply");

  ierr = MatSeqAIJMKL_create_from_mkl_handle(PETSC_COMM_SELF,csrC,scall,C);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}
#endif /* PETSC_HAVE_MKL_SPARSE_OPTIMIZE */

#if defined(PETSC_HAVE_MKL_SPARSE_SP2M_FEATURE)
PetscErrorCode MatPtAPNumeric_SeqAIJMKL_SeqAIJMKL_SpMV2(Mat A,Mat P,Mat C)
{
  Mat_SeqAIJMKL       *a, *p, *c;
  sparse_matrix_t     csrA, csrP, csrC;
  PetscBool           set, flag;
  sparse_status_t     stat = SPARSE_STATUS_SUCCESS;
  struct matrix_descr descr_type_sym;
  PetscObjectState    state;
  PetscErrorCode      ierr;

  PetscFunctionBegin;
  ierr = MatIsSymmetricKnown(A,&set,&flag);
  if (!set || (set && !flag)) {
    ierr = MatPtAPNumeric_SeqAIJ_SeqAIJ(A,P,C);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }

  a = (Mat_SeqAIJMKL*)A->spptr;
  p = (Mat_SeqAIJMKL*)P->spptr;
  c = (Mat_SeqAIJMKL*)C->spptr;
  ierr = PetscObjectStateGet((PetscObject)A,&state);CHKERRQ(ierr);
  if (!a->sparse_optimized || a->state != state) {
    MatSeqAIJMKL_create_mkl_handle(A);
  }
  ierr = PetscObjectStateGet((PetscObject)P,&state);CHKERRQ(ierr);
  if (!p->sparse_optimized || p->state != state) {
    MatSeqAIJMKL_create_mkl_handle(P);
  }
  csrA = a->csrA;
  csrP = p->csrA;
  csrC = c->csrA;
  descr_type_sym.type = SPARSE_MATRIX_TYPE_SYMMETRIC;
  descr_type_sym.mode = SPARSE_FILL_MODE_LOWER;
  descr_type_sym.diag = SPARSE_DIAG_NON_UNIT;

  /* Note that the call below won't work for complex matrices. (We protect this when pointers are assigned in MatConvert.) */
  stat = mkl_sparse_sypr(SPARSE_OPERATION_TRANSPOSE,csrP,csrA,descr_type_sym,&csrC,SPARSE_STAGE_FINALIZE_MULT);
  if (stat != SPARSE_STATUS_SUCCESS) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_LIB,"Intel MKL error: unable to finalize mkl_sparse_sypr");

  /* Have to update the PETSc AIJ representation for matrix C from contents of MKL handle. */
  ierr = MatSeqAIJMKL_update_from_mkl_handle(C);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}
#endif

#if defined(PETSC_HAVE_MKL_SPARSE_SP2M_FEATURE)
PetscErrorCode MatPtAP_SeqAIJMKL_SeqAIJMKL_SpMV2(Mat A,Mat P,MatReuse scall,PetscReal fill,Mat *C)
{
  Mat_SeqAIJMKL       *a, *p;
  sparse_matrix_t     csrA, csrP, csrC;
  PetscBool           set, flag;
  sparse_status_t     stat = SPARSE_STATUS_SUCCESS;
  struct matrix_descr descr_type_sym;
  PetscObjectState    state;
  PetscErrorCode      ierr;

  PetscFunctionBegin;
  ierr = MatIsSymmetricKnown(A,&set,&flag);
  if (!set || (set && !flag)) {
    ierr = MatPtAP_SeqAIJ_SeqAIJ(A,P,scall,fill,C);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }

  if (scall == MAT_REUSE_MATRIX) {
    ierr = MatPtAPNumeric_SeqAIJMKL_SeqAIJMKL_SpMV2(A,P,*C);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }

  a = (Mat_SeqAIJMKL*)A->spptr;
  p = (Mat_SeqAIJMKL*)P->spptr;
  ierr = PetscObjectStateGet((PetscObject)A,&state);CHKERRQ(ierr);
  if (!a->sparse_optimized || a->state != state) {
    MatSeqAIJMKL_create_mkl_handle(A);
  }
  ierr = PetscObjectStateGet((PetscObject)P,&state);CHKERRQ(ierr);
  if (!p->sparse_optimized || p->state != state) {
    MatSeqAIJMKL_create_mkl_handle(P);
  }
  csrA = a->csrA;
  csrP = p->csrA;
  descr_type_sym.type = SPARSE_MATRIX_TYPE_SYMMETRIC;
  descr_type_sym.mode = SPARSE_FILL_MODE_LOWER;
  descr_type_sym.diag = SPARSE_DIAG_NON_UNIT;

  /* Note that the call below won't work for complex matrices. (We protect this when pointers are assigned in MatConvert.) */
  stat = mkl_sparse_sypr(SPARSE_OPERATION_TRANSPOSE,csrP,csrA,descr_type_sym,&csrC,SPARSE_STAGE_FULL_MULT);
  if (stat != SPARSE_STATUS_SUCCESS) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_LIB,"Intel MKL error: unable to complete full mkl_sparse_sypr");

  ierr = MatSeqAIJMKL_create_from_mkl_handle(PETSC_COMM_SELF,csrC,scall,C);CHKERRQ(ierr);
  ierr = MatSetOption(*C,MAT_SYMMETRIC,PETSC_TRUE);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}
#endif

/* These function prototypes are needed in MatConvert_SeqAIJ_SeqAIJMKL(), below. */
PETSC_INTERN PetscErrorCode MatPtAP_IS_XAIJ(Mat,Mat,MatReuse,PetscReal,Mat*);
#if defined(PETSC_HAVE_HYPRE)
PETSC_INTERN PetscErrorCode MatMatMatMult_Transpose_AIJ_AIJ(Mat,Mat,Mat,MatReuse,PetscReal,Mat*);
#endif

/* MatConvert_SeqAIJ_SeqAIJMKL converts a SeqAIJ matrix into a
 * SeqAIJMKL matrix.  This routine is called by the MatCreate_SeqAIJMKL()
 * routine, but can also be used to convert an assembled SeqAIJ matrix
 * into a SeqAIJMKL one. */
PETSC_INTERN PetscErrorCode MatConvert_SeqAIJ_SeqAIJMKL(Mat A,MatType type,MatReuse reuse,Mat *newmat)
{
  PetscErrorCode ierr;
  Mat            B = *newmat;
  Mat_SeqAIJMKL  *aijmkl;
  PetscBool      set;
  PetscBool      sametype;

  PetscFunctionBegin;
  if (reuse == MAT_INITIAL_MATRIX) {
    ierr = MatDuplicate(A,MAT_COPY_VALUES,&B);CHKERRQ(ierr);
  }

  ierr = PetscObjectTypeCompare((PetscObject)A,type,&sametype);CHKERRQ(ierr);
  if (sametype) PetscFunctionReturn(0);

  ierr     = PetscNewLog(B,&aijmkl);CHKERRQ(ierr);
  B->spptr = (void*) aijmkl;

  /* Set function pointers for methods that we inherit from AIJ but override. 
   * We also parse some command line options below, since those determine some of the methods we point to. */
  B->ops->duplicate        = MatDuplicate_SeqAIJMKL;
  B->ops->assemblyend      = MatAssemblyEnd_SeqAIJMKL;
  B->ops->destroy          = MatDestroy_SeqAIJMKL;

  aijmkl->sparse_optimized = PETSC_FALSE;
#if defined(PETSC_HAVE_MKL_SPARSE_OPTIMIZE)
  aijmkl->no_SpMV2 = PETSC_FALSE;  /* Default to using the SpMV2 routines if our MKL supports them. */
#else
  aijmkl->no_SpMV2 = PETSC_TRUE;
#endif
  aijmkl->eager_inspection = PETSC_FALSE;

  /* Parse command line options. */
  ierr = PetscOptionsBegin(PetscObjectComm((PetscObject)A),((PetscObject)A)->prefix,"AIJMKL Options","Mat");CHKERRQ(ierr);
  ierr = PetscOptionsBool("-mat_aijmkl_no_spmv2","Disable use of inspector-executor (SpMV 2) routines","None",(PetscBool)aijmkl->no_SpMV2,(PetscBool*)&aijmkl->no_SpMV2,&set);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-mat_aijmkl_eager_inspection","Run inspection at matrix assembly time, instead of waiting until needed by an operation","None",(PetscBool)aijmkl->eager_inspection,(PetscBool*)&aijmkl->eager_inspection,&set);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
#if !defined(PETSC_HAVE_MKL_SPARSE_OPTIMIZE)
  if(!aijmkl->no_SpMV2) {
    ierr = PetscInfo(B,"User requested use of MKL SpMV2 routines, but MKL version does not support mkl_sparse_optimize();  defaulting to non-SpMV2 routines.\n");
    aijmkl->no_SpMV2 = PETSC_TRUE;
  }
#endif

#if defined(PETSC_HAVE_MKL_SPARSE_OPTIMIZE)
  B->ops->mult             = MatMult_SeqAIJMKL_SpMV2;
  B->ops->multtranspose    = MatMultTranspose_SeqAIJMKL_SpMV2;
  B->ops->multadd          = MatMultAdd_SeqAIJMKL_SpMV2;
  B->ops->multtransposeadd = MatMultTransposeAdd_SeqAIJMKL_SpMV2;
  B->ops->matmult          = MatMatMult_SeqAIJMKL_SeqAIJMKL_SpMV2;
# if defined(PETSC_HAVE_MKL_SPARSE_SP2M_FEATURE)
  B->ops->matmultnumeric   = MatMatMultNumeric_SeqAIJMKL_SeqAIJMKL_SpMV2;
#   if !defined(PETSC_USE_COMPLEX)
  B->ops->ptap             = MatPtAP_SeqAIJMKL_SeqAIJMKL_SpMV2;
  B->ops->ptapnumeric      = MatPtAPNumeric_SeqAIJMKL_SeqAIJMKL_SpMV2;
#   endif
# endif
  B->ops->transposematmult = MatTransposeMatMult_SeqAIJMKL_SeqAIJMKL_SpMV2;
#endif /* PETSC_HAVE_MKL_SPARSE_OPTIMIZE */

#if !defined(PETSC_MKL_SPBLAS_DEPRECATED)
  /* In MKL version 18, update 2, the old sparse BLAS interfaces were marked as deprecated. If "no_SpMV2" has been specified by the
   * user and the old SpBLAS interfaces are deprecated in our MKL version, we use the new _SpMV2 routines (set above), but do not
   * call mkl_sparse_optimize(), which results in the old numerical kernels (without the inspector-executor model) being used. For
   * versions in which the older interface has not been deprecated, we use the old interface. */
  if (aijmkl->no_SpMV2) {
    B->ops->mult             = MatMult_SeqAIJMKL;
    B->ops->multtranspose    = MatMultTranspose_SeqAIJMKL;
    B->ops->multadd          = MatMultAdd_SeqAIJMKL;
    B->ops->multtransposeadd = MatMultTransposeAdd_SeqAIJMKL;
  }
#endif

  ierr = PetscObjectComposeFunction((PetscObject)B,"MatConvert_seqaijmkl_seqaij_C",MatConvert_SeqAIJMKL_SeqAIJ);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)B,"MatMatMult_seqdense_seqaijmkl_C",MatMatMult_SeqDense_SeqAIJ);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)B,"MatMatMultSymbolic_seqdense_seqaijmkl_C",MatMatMultSymbolic_SeqDense_SeqAIJ);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)B,"MatMatMultNumeric_seqdense_seqaijmkl_C",MatMatMultNumeric_SeqDense_SeqAIJ);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)B,"MatPtAP_is_seqaijmkl_C",MatPtAP_IS_XAIJ);CHKERRQ(ierr);
#if defined(PETSC_HAVE_HYPRE)
  ierr = PetscObjectComposeFunction((PetscObject)B,"MatMatMatMult_transpose_seqaijmkl_seqaijmkl_C",MatMatMatMult_Transpose_AIJ_AIJ);CHKERRQ(ierr);
#endif
  if(!aijmkl->no_SpMV2) {
#if defined(PETSC_HAVE_MKL_SPARSE_OPTIMIZE)
    ierr = PetscObjectComposeFunction((PetscObject)B,"MatMatMult_seqaijmkl_seqaijmkl_C",MatMatMult_SeqAIJMKL_SeqAIJMKL_SpMV2);CHKERRQ(ierr);
#if defined(PETSC_HAVE_MKL_SPARSE_SP2M_FEATURE)
    ierr = PetscObjectComposeFunction((PetscObject)B,"MatMatMultNumeric_seqaijmkl_seqaijmkl_C",MatMatMultNumeric_SeqAIJMKL_SeqAIJMKL_SpMV2);CHKERRQ(ierr);
#endif
    ierr = PetscObjectComposeFunction((PetscObject)B,"MatTransposeMatMult_seqaijmkl_seqaijmkl_C",MatTransposeMatMult_SeqAIJMKL_SeqAIJMKL_SpMV2);CHKERRQ(ierr);
#endif
  }

  ierr    = PetscObjectChangeTypeName((PetscObject)B,MATSEQAIJMKL);CHKERRQ(ierr);
  *newmat = B;
  PetscFunctionReturn(0);
}

/*@C
   MatCreateSeqAIJMKL - Creates a sparse matrix of type SEQAIJMKL.
   This type inherits from AIJ and is largely identical, but uses sparse BLAS 
   routines from Intel MKL whenever possible.
   If the installed version of MKL supports the "SpMV2" sparse 
   inspector-executor routines, then those are used by default.
   MatMult, MatMultAdd, MatMultTranspose, MatMultTransposeAdd, MatMatMult, MatTransposeMatMult, and MatPtAP (for
   symmetric A) operations are currently supported.
   Note that MKL version 18, update 2 or later is required for MatPtAP/MatPtAPNumeric and MatMatMultNumeric.

   Collective

   Input Parameters:
+  comm - MPI communicator, set to PETSC_COMM_SELF
.  m - number of rows
.  n - number of columns
.  nz - number of nonzeros per row (same for all rows)
-  nnz - array containing the number of nonzeros in the various rows
         (possibly different for each row) or NULL

   Output Parameter:
.  A - the matrix

   Options Database Keys:
+  -mat_aijmkl_no_spmv2 - disable use of the SpMV2 inspector-executor routines
-  -mat_aijmkl_eager_inspection - perform MKL "inspection" phase upon matrix assembly; default is to do "lazy" inspection, performing this step the first time the matrix is applied

   Notes:
   If nnz is given then nz is ignored

   Level: intermediate

.seealso: MatCreate(), MatCreateMPIAIJMKL(), MatSetValues()
@*/
PetscErrorCode  MatCreateSeqAIJMKL(MPI_Comm comm,PetscInt m,PetscInt n,PetscInt nz,const PetscInt nnz[],Mat *A)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatCreate(comm,A);CHKERRQ(ierr);
  ierr = MatSetSizes(*A,m,n,m,n);CHKERRQ(ierr);
  ierr = MatSetType(*A,MATSEQAIJMKL);CHKERRQ(ierr);
  ierr = MatSeqAIJSetPreallocation_SeqAIJ(*A,nz,nnz);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PETSC_EXTERN PetscErrorCode MatCreate_SeqAIJMKL(Mat A)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatSetType(A,MATSEQAIJ);CHKERRQ(ierr);
  ierr = MatConvert_SeqAIJ_SeqAIJMKL(A,MATSEQAIJMKL,MAT_INPLACE_MATRIX,&A);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
