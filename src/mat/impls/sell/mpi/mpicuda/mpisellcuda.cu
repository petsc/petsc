#include <petscconf.h>
#include <petscdevice.h>
#include <../src/mat/impls/sell/mpi/mpisell.h> /*I "petscmat.h" I*/

static PetscErrorCode MatMPISELLSetPreallocation_MPISELLCUDA(Mat B, PetscInt d_rlenmax, const PetscInt d_rlen[], PetscInt o_rlenmax, const PetscInt o_rlen[])
{
  Mat_MPISELL *b = (Mat_MPISELL *)B->data;

  PetscFunctionBegin;
  PetscCall(PetscLayoutSetUp(B->rmap));
  PetscCall(PetscLayoutSetUp(B->cmap));

  if (!B->preallocated) {
    /* Explicitly create 2 MATSEQSELLCUDA matrices. */
    PetscCall(MatCreate(PETSC_COMM_SELF, &b->A));
    PetscCall(MatBindToCPU(b->A, B->boundtocpu));
    PetscCall(MatSetSizes(b->A, B->rmap->n, B->cmap->n, B->rmap->n, B->cmap->n));
    PetscCall(MatSetType(b->A, MATSEQSELLCUDA));
    PetscCall(MatCreate(PETSC_COMM_SELF, &b->B));
    PetscCall(MatBindToCPU(b->B, B->boundtocpu));
    PetscCall(MatSetSizes(b->B, B->rmap->n, B->cmap->N, B->rmap->n, B->cmap->N));
    PetscCall(MatSetType(b->B, MATSEQSELLCUDA));
  }
  PetscCall(MatSeqSELLSetPreallocation(b->A, d_rlenmax, d_rlen));
  PetscCall(MatSeqSELLSetPreallocation(b->B, o_rlenmax, o_rlen));
  B->preallocated  = PETSC_TRUE;
  B->was_assembled = PETSC_FALSE;
  B->assembled     = PETSC_FALSE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatSetFromOptions_MPISELLCUDA(Mat, PetscOptionItems)
{
  return PETSC_SUCCESS;
}

static PetscErrorCode MatAssemblyEnd_MPISELLCUDA(Mat A, MatAssemblyType mode)
{
  PetscFunctionBegin;
  PetscCall(MatAssemblyEnd_MPISELL(A, mode));
  if (!A->was_assembled && mode == MAT_FINAL_ASSEMBLY) PetscCall(VecSetType(((Mat_MPISELL *)A->data)->lvec, VECSEQCUDA));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatDestroy_MPISELLCUDA(Mat A)
{
  PetscFunctionBegin;
  PetscCall(MatDestroy_MPISELL(A));
  PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatMPISELLSetPreallocation_C", NULL));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode MatConvert_MPISELL_MPISELLCUDA(Mat B, MatType, MatReuse reuse, Mat *newmat)
{
  Mat_MPISELL *a;
  Mat          A;

  PetscFunctionBegin;
  PetscCall(PetscDeviceInitialize(PETSC_DEVICE_CUDA));
  if (reuse == MAT_INITIAL_MATRIX) PetscCall(MatDuplicate(B, MAT_COPY_VALUES, newmat));
  else if (reuse == MAT_REUSE_MATRIX) PetscCall(MatCopy(B, *newmat, SAME_NONZERO_PATTERN));
  A             = *newmat;
  A->boundtocpu = PETSC_FALSE;
  PetscCall(PetscFree(A->defaultvectype));
  PetscCall(PetscStrallocpy(VECCUDA, &A->defaultvectype));

  a = (Mat_MPISELL *)A->data;
  if (a->A) PetscCall(MatSetType(a->A, MATSEQSELLCUDA));
  if (a->B) PetscCall(MatSetType(a->B, MATSEQSELLCUDA));
  if (a->lvec) PetscCall(VecSetType(a->lvec, VECSEQCUDA));

  A->ops->assemblyend    = MatAssemblyEnd_MPISELLCUDA;
  A->ops->setfromoptions = MatSetFromOptions_MPISELLCUDA;
  A->ops->destroy        = MatDestroy_MPISELLCUDA;

  PetscCall(PetscObjectChangeTypeName((PetscObject)A, MATMPISELLCUDA));
  PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatMPISELLSetPreallocation_C", MatMPISELLSetPreallocation_MPISELLCUDA));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_EXTERN PetscErrorCode MatCreate_MPISELLCUDA(Mat A)
{
  PetscFunctionBegin;
  PetscCall(PetscDeviceInitialize(PETSC_DEVICE_CUDA));
  PetscCall(MatCreate_MPISELL(A));
  PetscCall(MatConvert_MPISELL_MPISELLCUDA(A, MATMPISELLCUDA, MAT_INPLACE_MATRIX, &A));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  MatCreateSELLCUDA - Creates a sparse matrix in SELL format.
  This matrix will ultimately pushed down to NVIDIA GPUs.

  Collective

  Input Parameters:
+ comm  - MPI communicator, set to `PETSC_COMM_SELF`
. m     - number of local rows (or `PETSC_DECIDE` to have calculated if `M` is given)
           This value should be the same as the local size used in creating the
           y vector for the matrix-vector product y = Ax.
. n     - This value should be the same as the local size used in creating the
       x vector for the matrix-vector product y = Ax. (or PETSC_DECIDE to have
       calculated if `N` is given) For square matrices `n` is almost always `m`.
. M     - number of global rows (or `PETSC_DETERMINE` to have calculated if `m` is given)
. N     - number of global columns (or `PETSC_DETERMINE` to have calculated if `n` is given)
. d_nz  - number of nonzeros per row in DIAGONAL portion of local submatrix
           (same value is used for all local rows)
. d_nnz - array containing the number of nonzeros in the various rows of the
           DIAGONAL portion of the local submatrix (possibly different for each row)
           or `NULL`, if `d_nz` is used to specify the nonzero structure.
           The size of this array is equal to the number of local rows, i.e `m`.
           For matrices you plan to factor you must leave room for the diagonal entry and
           put in the entry even if it is zero.
. o_nz  - number of nonzeros per row in the OFF-DIAGONAL portion of local
           submatrix (same value is used for all local rows).
- o_nnz - array containing the number of nonzeros in the various rows of the
           OFF-DIAGONAL portion of the local submatrix (possibly different for
           each row) or `NULL`, if `o_nz` is used to specify the nonzero
           structure. The size of this array is equal to the number
           of local rows, i.e `m`.

  Output Parameter:
. A - the matrix

  Level: intermediate

  Notes:
  If `nnz` is given then `nz` is ignored

  Specify the preallocated storage with either `nz` or `nnz` (not both).
  Set `nz` = `PETSC_DEFAULT` and `nnz` = `NULL` for PETSc to control dynamic memory
  allocation.

.seealso: [](ch_matrices), `Mat`, `MatCreate()`, `MatCreateSELL()`, `MatSetValues()`, `MATMPISELLCUDA`, `MATSELLCUDA`
@*/
PetscErrorCode MatCreateSELLCUDA(MPI_Comm comm, PetscInt m, PetscInt n, PetscInt M, PetscInt N, PetscInt d_nz, const PetscInt d_nnz[], PetscInt o_nz, const PetscInt o_nnz[], Mat *A)
{
  PetscMPIInt size;

  PetscFunctionBegin;
  PetscCall(MatCreate(comm, A));
  PetscCall(MatSetSizes(*A, m, n, M, N));
  PetscCallMPI(MPI_Comm_size(comm, &size));
  if (size > 1) {
    PetscCall(MatSetType(*A, MATMPISELLCUDA));
    PetscCall(MatMPISELLSetPreallocation(*A, d_nz, d_nnz, o_nz, o_nnz));
  } else {
    PetscCall(MatSetType(*A, MATSEQSELLCUDA));
    PetscCall(MatSeqSELLSetPreallocation(*A, d_nz, d_nnz));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*MC
   MATSELLCUDA - "sellcuda" = "mpisellcuda" - A matrix type to be used for sparse matrices.

   Sliced ELLPACK matrix type whose data resides on NVIDIA GPUs.

   This matrix type is identical to `MATSEQSELLCUDA` when constructed with a single process communicator,
   and `MATMPISELLCUDA` otherwise.  As a result, for single process communicators,
   `MatSeqSELLSetPreallocation()` is supported, and similarly `MatMPISELLSetPreallocation()` is supported
   for communicators controlling multiple processes.  It is recommended that you call both of
   the above preallocation routines for simplicity.

   Options Database Key:
.  -mat_type mpisellcuda - sets the matrix type to `MATMPISELLCUDA` during a call to MatSetFromOptions()

  Level: beginner

.seealso: `MatCreateSELLCUDA()`, `MATSEQSELLCUDA`, `MatCreateSeqSELLCUDA()`, `MatCUDAFormatOperation()`
M*/
