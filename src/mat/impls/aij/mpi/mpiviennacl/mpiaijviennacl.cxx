#define PETSC_SKIP_IMMINTRIN_H_CUDAWORKAROUND 1

#include <petscconf.h>
#include <../src/mat/impls/aij/mpi/mpiaij.h> /*I "petscmat.h" I*/
#include <../src/mat/impls/aij/seq/seqviennacl/viennaclmatimpl.h>

PetscErrorCode MatMPIAIJSetPreallocation_MPIAIJViennaCL(Mat B, PetscInt d_nz, const PetscInt d_nnz[], PetscInt o_nz, const PetscInt o_nnz[])
{
  Mat_MPIAIJ *b = (Mat_MPIAIJ *)B->data;

  PetscFunctionBegin;
  PetscCall(PetscLayoutSetUp(B->rmap));
  PetscCall(PetscLayoutSetUp(B->cmap));
  if (!B->preallocated) {
    /* Explicitly create the two MATSEQAIJVIENNACL matrices. */
    PetscCall(MatCreate(PETSC_COMM_SELF, &b->A));
    PetscCall(MatSetSizes(b->A, B->rmap->n, B->cmap->n, B->rmap->n, B->cmap->n));
    PetscCall(MatSetType(b->A, MATSEQAIJVIENNACL));
    PetscCall(MatCreate(PETSC_COMM_SELF, &b->B));
    PetscCall(MatSetSizes(b->B, B->rmap->n, B->cmap->N, B->rmap->n, B->cmap->N));
    PetscCall(MatSetType(b->B, MATSEQAIJVIENNACL));
  }
  PetscCall(MatSeqAIJSetPreallocation(b->A, d_nz, d_nnz));
  PetscCall(MatSeqAIJSetPreallocation(b->B, o_nz, o_nnz));
  B->preallocated = PETSC_TRUE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MatAssemblyEnd_MPIAIJViennaCL(Mat A, MatAssemblyType mode)
{
  Mat_MPIAIJ *b = (Mat_MPIAIJ *)A->data;
  PetscBool   v;

  PetscFunctionBegin;
  PetscCall(MatAssemblyEnd_MPIAIJ(A, mode));
  PetscCall(PetscObjectTypeCompare((PetscObject)b->lvec, VECSEQVIENNACL, &v));
  if (!v) {
    PetscInt m;
    PetscCall(VecGetSize(b->lvec, &m));
    PetscCall(VecDestroy(&b->lvec));
    PetscCall(VecCreateSeqViennaCL(PETSC_COMM_SELF, m, &b->lvec));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MatDestroy_MPIAIJViennaCL(Mat A)
{
  PetscFunctionBegin;
  PetscCall(MatDestroy_MPIAIJ(A));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_EXTERN PetscErrorCode MatCreate_MPIAIJViennaCL(Mat A)
{
  PetscFunctionBegin;
  PetscCall(MatCreate_MPIAIJ(A));
  A->boundtocpu = PETSC_FALSE;
  PetscCall(PetscFree(A->defaultvectype));
  PetscCall(PetscStrallocpy(VECVIENNACL, &A->defaultvectype));
  PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatMPIAIJSetPreallocation_C", MatMPIAIJSetPreallocation_MPIAIJViennaCL));
  A->ops->assemblyend = MatAssemblyEnd_MPIAIJViennaCL;
  PetscCall(PetscObjectChangeTypeName((PetscObject)A, MATMPIAIJVIENNACL));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
   MatCreateAIJViennaCL - Creates a sparse matrix in `MATAIJ` (compressed row) format
   (the default parallel PETSc format).  This matrix will ultimately be pushed down
   to GPUs and use the ViennaCL library for calculations.

   Collective

   Input Parameters:
+  comm - MPI communicator, set to `PETSC_COMM_SELF`
.  m - number of rows, or `PETSC_DECIDE` if `M` is provided
.  n - number of columns, or `PETSC_DECIDE` if `N` is provided
.  M - number of global rows in the matrix, or `PETSC_DETERMINE` if `m` is provided
.  N - number of global columns in the matrix, or `PETSC_DETERMINE` if `n` is provided
.  d_nz  - number of nonzeros per row in DIAGONAL portion of local submatrix
           (same value is used for all local rows)
.  d_nnz - array containing the number of nonzeros in the various rows of the
           DIAGONAL portion of the local submatrix (possibly different for each row)
           or `NULL`, if `d_nz` is used to specify the nonzero structure.
           The size of this array is equal to the number of local rows, i.e `m`.
           For matrices you plan to factor you must leave room for the diagonal entry and
           put in the entry even if it is zero.
.  o_nz  - number of nonzeros per row in the OFF-DIAGONAL portion of local
           submatrix (same value is used for all local rows).
-  o_nnz - array containing the number of nonzeros in the various rows of the
           OFF-DIAGONAL portion of the local submatrix (possibly different for
           each row) or `NULL`, if `o_nz` is used to specify the nonzero
           structure. The size of this array is equal to the number
           of local rows, i.e `m`.

   Output Parameter:
.  A - the matrix

   Level: intermediate

   Notes:
   It is recommended that one use the `MatCreate()`, `MatSetType()` and/or `MatSetFromOptions()`,
   MatXXXXSetPreallocation() paradigm instead of this routine directly.
   [MatXXXXSetPreallocation() is, for example, `MatSeqAIJSetPreallocation()`]

   The AIJ format, also called
   compressed row storage), is fully compatible with standard Fortran
   storage.  That is, the stored row and column indices can begin at
   either one (as in Fortran) or zero.

.seealso: `Mat`, `MatCreate()`, `MatCreateAIJ()`, `MatCreateAIJCUSPARSE()`, `MatSetValues()`, `MatSeqAIJSetColumnIndices()`, `MatCreateSeqAIJWithArrays()`,
          `MatCreateAIJ()`, `MATMPIAIJVIENNACL`, `MATAIJVIENNACL`
@*/
PetscErrorCode MatCreateAIJViennaCL(MPI_Comm comm, PetscInt m, PetscInt n, PetscInt M, PetscInt N, PetscInt d_nz, const PetscInt d_nnz[], PetscInt o_nz, const PetscInt o_nnz[], Mat *A)
{
  PetscMPIInt size;

  PetscFunctionBegin;
  PetscCall(MatCreate(comm, A));
  PetscCall(MatSetSizes(*A, m, n, M, N));
  PetscCallMPI(MPI_Comm_size(comm, &size));
  if (size > 1) {
    PetscCall(MatSetType(*A, MATMPIAIJVIENNACL));
    PetscCall(MatMPIAIJSetPreallocation(*A, d_nz, d_nnz, o_nz, o_nnz));
  } else {
    PetscCall(MatSetType(*A, MATSEQAIJVIENNACL));
    PetscCall(MatSeqAIJSetPreallocation(*A, d_nz, d_nnz));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*MC
   MATAIJVIENNACL - MATMPIAIJVIENNACL= "aijviennacl" = "mpiaijviennacl" - A matrix type to be used for sparse matrices.

   A matrix type (CSR format) whose data resides on GPUs.
   All matrix calculations are performed using the ViennaCL library.

   This matrix type is identical to `MATSEQAIJVIENNACL` when constructed with a single process communicator,
   and `MATMPIAIJVIENNACL` otherwise.  As a result, for single process communicators,
   `MatSeqAIJSetPreallocation()` is supported, and similarly `MatMPIAIJSetPreallocation()` is supported
   for communicators controlling multiple processes.  It is recommended that you call both of
   the above preallocation routines for simplicity.

   Options Database Keys:
.  -mat_type mpiaijviennacl - sets the matrix type to `MATAIJVIENNACL` during a call to `MatSetFromOptions()`

  Level: beginner

.seealso: `Mat`, `MatType`, `MatCreateAIJViennaCL()`, `MATSEQAIJVIENNACL`, `MatCreateSeqAIJVIENNACL()`
M*/
