#include "../matseqdensecupm.hpp"

using namespace Petsc::mat::cupm;
using Petsc::device::cupm::DeviceType;

static constexpr impl::MatDense_Seq_CUPM<DeviceType::CUDA> cupm_mat{};

/*MC
  MATSEQDENSECUDA - "seqdensecuda" - A matrix type to be used for sequential dense matrices on
  GPUs.

  Options Database Keys:
. -mat_type seqdensecuda - sets the matrix type to `MATSEQDENSECUDA` during a call to
                           `MatSetFromOptions()`

  Level: beginner

.seealso: `MATSEQDENSE`
M*/
PETSC_INTERN PetscErrorCode MatCreate_SeqDenseCUDA(Mat A)
{
  PetscFunctionBegin;
  PetscCall(cupm_mat.Create(A));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode MatSolverTypeRegister_DENSECUDA(void)
{
  PetscFunctionBegin;
  PetscCall(impl::MatSolverTypeRegister_DENSECUPM<DeviceType::CUDA>());
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MatConvert_SeqDense_SeqDenseCUDA(Mat A, MatType newtype, MatReuse reuse, Mat *newmat)
{
  PetscFunctionBegin;
  PetscCall(cupm_mat.Convert_SeqDense_SeqDenseCUPM(A, newtype, reuse, newmat));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MatMatMultNumeric_SeqDenseCUDA_SeqDenseCUDA_Internal(Mat A, Mat B, Mat C, PetscBool TA, PetscBool TB)
{
  PetscFunctionBegin;
  PetscCall(impl::MatMatMultNumeric_SeqDenseCUPM_SeqDenseCUPM<DeviceType::CUDA>(A, B, C, TA, TB));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MatSeqDenseCUDAInvertFactors_Internal(Mat A)
{
  PetscFunctionBegin;
  PetscCall(cupm_mat.InvertFactors(A));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  MatCreateSeqDenseCUDA - Creates a sequential matrix in dense format using CUDA.

  Collective

  Input Parameters:
+ comm - MPI communicator
. m    - number of rows
. n    - number of columns
- data - optional location of GPU matrix data. Pass `NULL` to let PETSc to control matrix
         memory allocation

  Output Parameter:
. A - the matrix

  Level: intermediate

.seealso: `MATSEQDENSE`, `MatCreate()`, `MatCreateSeqDense()`
@*/
PetscErrorCode MatCreateSeqDenseCUDA(MPI_Comm comm, PetscInt m, PetscInt n, PetscScalar *data, Mat *A)
{
  PetscFunctionBegin;
  PetscCall(MatCreateSeqDenseCUPM<DeviceType::CUDA>(comm, m, n, data, A));
  PetscFunctionReturn(PETSC_SUCCESS);
}
