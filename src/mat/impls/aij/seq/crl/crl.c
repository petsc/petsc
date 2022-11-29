
/*
  Defines a matrix-vector product for the MATSEQAIJCRL matrix class.
  This class is derived from the MATSEQAIJ class and retains the
  compressed row storage (aka Yale sparse matrix format) but augments
  it with a column oriented storage that is more efficient for
  matrix vector products on Vector machines.

  CRL stands for constant row length (that is the same number of columns
  is kept (padded with zeros) for each row of the sparse matrix.
*/
#include <../src/mat/impls/aij/seq/crl/crl.h>

PetscErrorCode MatDestroy_SeqAIJCRL(Mat A)
{
  Mat_AIJCRL *aijcrl = (Mat_AIJCRL *)A->spptr;

  PetscFunctionBegin;
  /* Free everything in the Mat_AIJCRL data structure. */
  if (aijcrl) PetscCall(PetscFree2(aijcrl->acols, aijcrl->icols));
  PetscCall(PetscFree(A->spptr));
  PetscCall(PetscObjectChangeTypeName((PetscObject)A, MATSEQAIJ));
  PetscCall(MatDestroy_SeqAIJ(A));
  PetscFunctionReturn(0);
}

PetscErrorCode MatDuplicate_AIJCRL(Mat A, MatDuplicateOption op, Mat *M)
{
  SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Cannot duplicate AIJCRL matrices yet");
}

PetscErrorCode MatSeqAIJCRL_create_aijcrl(Mat A)
{
  Mat_SeqAIJ  *a      = (Mat_SeqAIJ *)(A)->data;
  Mat_AIJCRL  *aijcrl = (Mat_AIJCRL *)A->spptr;
  PetscInt     m      = A->rmap->n; /* Number of rows in the matrix. */
  PetscInt    *aj     = a->j;       /* From the CSR representation; points to the beginning  of each row. */
  PetscInt     i, j, rmax = a->rmax, *icols, *ilen = a->ilen;
  MatScalar   *aa = a->a;
  PetscScalar *acols;

  PetscFunctionBegin;
  aijcrl->nz   = a->nz;
  aijcrl->m    = A->rmap->n;
  aijcrl->rmax = rmax;

  PetscCall(PetscFree2(aijcrl->acols, aijcrl->icols));
  PetscCall(PetscMalloc2(rmax * m, &aijcrl->acols, rmax * m, &aijcrl->icols));
  acols = aijcrl->acols;
  icols = aijcrl->icols;
  for (i = 0; i < m; i++) {
    for (j = 0; j < ilen[i]; j++) {
      acols[j * m + i] = *aa++;
      icols[j * m + i] = *aj++;
    }
    for (; j < rmax; j++) { /* empty column entries */
      acols[j * m + i] = 0.0;
      icols[j * m + i] = (j) ? icols[(j - 1) * m + i] : 0; /* handle case where row is EMPTY */
    }
  }
  PetscCall(PetscInfo(A, "Percentage of 0's introduced for vectorized multiply %g. Rmax= %" PetscInt_FMT "\n", 1.0 - ((double)a->nz) / ((double)(rmax * m)), rmax));
  PetscFunctionReturn(0);
}

PetscErrorCode MatAssemblyEnd_SeqAIJCRL(Mat A, MatAssemblyType mode)
{
  Mat_SeqAIJ *a = (Mat_SeqAIJ *)A->data;

  PetscFunctionBegin;
  a->inode.use = PETSC_FALSE;

  PetscCall(MatAssemblyEnd_SeqAIJ(A, mode));
  if (mode == MAT_FLUSH_ASSEMBLY) PetscFunctionReturn(0);

  /* Now calculate the permutation and grouping information. */
  PetscCall(MatSeqAIJCRL_create_aijcrl(A));
  PetscFunctionReturn(0);
}

#include <../src/mat/impls/aij/seq/crl/ftn-kernels/fmultcrl.h>

/*
    Shared by both sequential and parallel versions of CRL matrix: MATMPIAIJCRL and MATSEQAIJCRL
    - the scatter is used only in the parallel version

*/
PetscErrorCode MatMult_AIJCRL(Mat A, Vec xx, Vec yy)
{
  Mat_AIJCRL        *aijcrl = (Mat_AIJCRL *)A->spptr;
  PetscInt           m      = aijcrl->m; /* Number of rows in the matrix. */
  PetscInt           rmax = aijcrl->rmax, *icols = aijcrl->icols;
  PetscScalar       *acols = aijcrl->acols;
  PetscScalar       *y;
  const PetscScalar *x;
#if !defined(PETSC_USE_FORTRAN_KERNEL_MULTCRL)
  PetscInt i, j, ii;
#endif

#if defined(PETSC_HAVE_PRAGMA_DISJOINT)
  #pragma disjoint(*x, *y, *aa)
#endif

  PetscFunctionBegin;
  if (aijcrl->xscat) {
    PetscCall(VecCopy(xx, aijcrl->xwork));
    /* get remote values needed for local part of multiply */
    PetscCall(VecScatterBegin(aijcrl->xscat, xx, aijcrl->fwork, INSERT_VALUES, SCATTER_FORWARD));
    PetscCall(VecScatterEnd(aijcrl->xscat, xx, aijcrl->fwork, INSERT_VALUES, SCATTER_FORWARD));
    xx = aijcrl->xwork;
  }

  PetscCall(VecGetArrayRead(xx, &x));
  PetscCall(VecGetArray(yy, &y));

#if defined(PETSC_USE_FORTRAN_KERNEL_MULTCRL)
  fortranmultcrl_(&m, &rmax, x, y, icols, acols);
#else

  /* first column */
  for (j = 0; j < m; j++) y[j] = acols[j] * x[icols[j]];

    /* other columns */
  #if defined(PETSC_HAVE_CRAY_VECTOR)
    #pragma _CRI preferstream
  #endif
  for (i = 1; i < rmax; i++) {
    ii = i * m;
  #if defined(PETSC_HAVE_CRAY_VECTOR)
    #pragma _CRI prefervector
  #endif
    for (j = 0; j < m; j++) y[j] = y[j] + acols[ii + j] * x[icols[ii + j]];
  }
  #if defined(PETSC_HAVE_CRAY_VECTOR)
    #pragma _CRI ivdep
  #endif

#endif
  PetscCall(PetscLogFlops(2.0 * aijcrl->nz - m));
  PetscCall(VecRestoreArrayRead(xx, &x));
  PetscCall(VecRestoreArray(yy, &y));
  PetscFunctionReturn(0);
}

/* MatConvert_SeqAIJ_SeqAIJCRL converts a SeqAIJ matrix into a
 * SeqAIJCRL matrix.  This routine is called by the MatCreate_SeqAIJCRL()
 * routine, but can also be used to convert an assembled SeqAIJ matrix
 * into a SeqAIJCRL one. */
PETSC_INTERN PetscErrorCode MatConvert_SeqAIJ_SeqAIJCRL(Mat A, MatType type, MatReuse reuse, Mat *newmat)
{
  Mat         B = *newmat;
  Mat_AIJCRL *aijcrl;
  PetscBool   sametype;

  PetscFunctionBegin;
  if (reuse == MAT_INITIAL_MATRIX) PetscCall(MatDuplicate(A, MAT_COPY_VALUES, &B));
  PetscCall(PetscObjectTypeCompare((PetscObject)A, type, &sametype));
  if (sametype) PetscFunctionReturn(0);

  PetscCall(PetscNew(&aijcrl));
  B->spptr = (void *)aijcrl;

  /* Set function pointers for methods that we inherit from AIJ but override. */
  B->ops->duplicate   = MatDuplicate_AIJCRL;
  B->ops->assemblyend = MatAssemblyEnd_SeqAIJCRL;
  B->ops->destroy     = MatDestroy_SeqAIJCRL;
  B->ops->mult        = MatMult_AIJCRL;

  /* If A has already been assembled, compute the permutation. */
  if (A->assembled) PetscCall(MatSeqAIJCRL_create_aijcrl(B));
  PetscCall(PetscObjectChangeTypeName((PetscObject)B, MATSEQAIJCRL));
  *newmat = B;
  PetscFunctionReturn(0);
}

/*@C
   MatCreateSeqAIJCRL - Creates a sparse matrix of type `MATSEQAIJCRL`.
   This type inherits from `MATSEQAIJ`, but stores some additional
   information that is used to allow better vectorization of
   the matrix-vector product. At the cost of increased storage, the `MATSEQAIJ` formatted
   matrix can be copied to a format in which pieces of the matrix are
   stored in ELLPACK format, allowing the vectorized matrix multiply
   routine to use stride-1 memory accesses.  As with the `MATSEQAIJ` type, it is
   important to preallocate matrix storage in order to get good assembly
   performance.

   Collective

   Input Parameters:
+  comm - MPI communicator, set to `PETSC_COMM_SELF`
.  m - number of rows
.  n - number of columns
.  nz - number of nonzeros per row (same for all rows)
-  nnz - array containing the number of nonzeros in the various rows
         (possibly different for each row) or NULL

   Output Parameter:
.  A - the matrix

   Note:
   If nnz is given then nz is ignored

   Level: intermediate

.seealso: `MatCreate()`, `MatCreateMPIAIJPERM()`, `MatSetValues()`
@*/
PetscErrorCode MatCreateSeqAIJCRL(MPI_Comm comm, PetscInt m, PetscInt n, PetscInt nz, const PetscInt nnz[], Mat *A)
{
  PetscFunctionBegin;
  PetscCall(MatCreate(comm, A));
  PetscCall(MatSetSizes(*A, m, n, m, n));
  PetscCall(MatSetType(*A, MATSEQAIJCRL));
  PetscCall(MatSeqAIJSetPreallocation_SeqAIJ(*A, nz, nnz));
  PetscFunctionReturn(0);
}

PETSC_EXTERN PetscErrorCode MatCreate_SeqAIJCRL(Mat A)
{
  PetscFunctionBegin;
  PetscCall(MatSetType(A, MATSEQAIJ));
  PetscCall(MatConvert_SeqAIJ_SeqAIJCRL(A, MATSEQAIJCRL, MAT_INPLACE_MATRIX, &A));
  PetscFunctionReturn(0);
}
