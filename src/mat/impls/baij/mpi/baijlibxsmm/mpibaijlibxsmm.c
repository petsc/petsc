#include <../src/mat/impls/baij/mpi/mpibaij.h> /*I   "petscmat.h"   I*/

static PetscErrorCode MatMPIBAIJSetPreallocation_MPIBAIJLIBXSMM(Mat B, PetscInt bs, PetscInt d_nz, const PetscInt *d_nnz, PetscInt o_nz, const PetscInt *o_nnz)
{
  Mat_MPIBAIJ *baij = (Mat_MPIBAIJ *)B->data;

  PetscFunctionBegin;
  PetscCall(MatMPIBAIJSetPreallocation_MPIBAIJ(B, bs, d_nz, d_nnz, o_nz, o_nnz));
  PetscCall(MatConvert(baij->A, MATSEQBAIJLIBXSMM, MAT_INPLACE_MATRIX, &baij->A));
  PetscCall(MatConvert(baij->B, MATSEQBAIJLIBXSMM, MAT_INPLACE_MATRIX, &baij->B));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode MatConvert_MPIBAIJLIBXSMM_MPIBAIJ(Mat A, MatType type, MatReuse reuse, Mat *newmat)
{
  Mat          B = *newmat;
  Mat_MPIBAIJ *baij;

  PetscFunctionBegin;
  if (reuse == MAT_INITIAL_MATRIX) PetscCall(MatDuplicate(A, MAT_COPY_VALUES, &B));
  baij = (Mat_MPIBAIJ *)B->data;
  if (baij->A) PetscCall(MatConvert(baij->A, MATSEQBAIJ, MAT_INPLACE_MATRIX, &baij->A));
  if (baij->B) PetscCall(MatConvert(baij->B, MATSEQBAIJ, MAT_INPLACE_MATRIX, &baij->B));
  PetscCall(PetscObjectComposeFunction((PetscObject)B, "MatMPIBAIJSetPreallocation_C", MatMPIBAIJSetPreallocation_MPIBAIJ));
  PetscCall(PetscObjectComposeFunction((PetscObject)B, "MatProductSetFromOptions_mpibaijlibxsmm_mpidense_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)B, "MatConvert_mpibaijlibxsmm_mpibaij_C", NULL));
  PetscCall(PetscObjectChangeTypeName((PetscObject)B, MATMPIBAIJ));
  *newmat = B;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode MatConvert_MPIBAIJ_MPIBAIJLIBXSMM(Mat A, MatType type, MatReuse reuse, Mat *newmat)
{
  Mat          B = *newmat;
  Mat_MPIBAIJ *baij;
  PetscBool    sametype;

  PetscFunctionBegin;
  if (reuse == MAT_INITIAL_MATRIX) PetscCall(MatDuplicate(A, MAT_COPY_VALUES, &B));
  PetscCall(PetscObjectTypeCompare((PetscObject)B, MATMPIBAIJLIBXSMM, &sametype));
  if (!sametype) {
    baij = (Mat_MPIBAIJ *)B->data;
    if (baij->A) PetscCall(MatConvert(baij->A, MATSEQBAIJLIBXSMM, MAT_INPLACE_MATRIX, &baij->A));
    if (baij->B) PetscCall(MatConvert(baij->B, MATSEQBAIJLIBXSMM, MAT_INPLACE_MATRIX, &baij->B));
    PetscCall(PetscObjectChangeTypeName((PetscObject)B, MATMPIBAIJLIBXSMM));
    PetscCall(PetscObjectComposeFunction((PetscObject)B, "MatMPIBAIJSetPreallocation_C", MatMPIBAIJSetPreallocation_MPIBAIJLIBXSMM));
    PetscCall(PetscObjectComposeFunction((PetscObject)B, "MatProductSetFromOptions_mpibaijlibxsmm_mpidense_C", MatProductSetFromOptions_MPIBAIJ_MPIDense));
    PetscCall(PetscObjectComposeFunction((PetscObject)B, "MatConvert_mpibaijlibxsmm_mpibaij_C", MatConvert_MPIBAIJLIBXSMM_MPIBAIJ));
  }
  *newmat = B;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*MC
   MATMPIBAIJLIBXSMM - "mpibaijlibxsmm" - A distributed block sparse matrix that uses LIBXSMM kernels for products with `MATMPIDENSE` matrices

   Options Database Key:
. -mat_type mpibaijlibxsmm - sets the matrix type to `MATMPIBAIJLIBXSMM` during a call to `MatSetFromOptions()`

   Level: beginner

   Notes:
   This matrix type is available when PETSc is configured with `--download-libxsmm` or `--with-libxsmm-dir=directory`.

   It has the same storage format and supports the same operations as `MATMPIBAIJ`.

.seealso: [](ch_matrices), `Mat`, `MATBAIJLIBXSMM`, `MATSEQBAIJLIBXSMM`, `MATMPIBAIJ`, `MatMatMult()`
M*/

/*MC
   MATBAIJLIBXSMM - "baijlibxsmm" - A block sparse matrix that uses LIBXSMM kernels for products with dense matrices

   Options Database Key:
. -mat_type baijlibxsmm - sets the matrix type to `MATBAIJLIBXSMM` during a call to `MatSetFromOptions()`

   Level: beginner

   Notes:
   This matrix type is `MATSEQBAIJLIBXSMM` on a single-process communicator and `MATMPIBAIJLIBXSMM` otherwise.

   This matrix type is available when PETSc is configured with `--download-libxsmm` or `--with-libxsmm-dir=directory`.

.seealso: [](ch_matrices), `Mat`, `MATSEQBAIJLIBXSMM`, `MATMPIBAIJLIBXSMM`, `MATBAIJ`, `MatMatMult()`
M*/
PETSC_EXTERN PetscErrorCode MatCreate_MPIBAIJLIBXSMM(Mat A)
{
  PetscFunctionBegin;
  PetscCall(MatSetType(A, MATMPIBAIJ));
  PetscCall(MatConvert_MPIBAIJ_MPIBAIJLIBXSMM(A, MATMPIBAIJLIBXSMM, MAT_INPLACE_MATRIX, &A));
  PetscFunctionReturn(PETSC_SUCCESS);
}
