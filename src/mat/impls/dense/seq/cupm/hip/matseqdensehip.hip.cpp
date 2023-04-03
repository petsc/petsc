#include "../matseqdensecupm.hpp"

using namespace Petsc::mat::cupm;
using Petsc::device::cupm::DeviceType;

static constexpr impl::MatDense_Seq_CUPM<DeviceType::HIP> cupm_mat{};

/*MC
  MATSEQDENSEHIP - "seqdensehip" - A matrix type to be used for sequential dense matrices on
  GPUs.

  Options Database Keys:
. -mat_type seqdensehip - sets the matrix type to `MATSEQDENSEHIP` during a call to
                           `MatSetFromOptions()`

  Level: beginner

.seealso: `MATSEQDENSE`
M*/
PETSC_INTERN PetscErrorCode MatCreate_SeqDenseHIP(Mat A)
{
  PetscFunctionBegin;
  PetscCall(cupm_mat.Create(A));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode MatSolverTypeRegister_DENSEHIP(void)
{
  PetscFunctionBegin;
  PetscCall(impl::MatSolverTypeRegister_DENSECUPM<DeviceType::HIP>());
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MatConvert_SeqDense_SeqDenseHIP(Mat A, MatType newtype, MatReuse reuse, Mat *newmat)
{
  PetscFunctionBegin;
  PetscCall(cupm_mat.Convert_SeqDense_SeqDenseCUPM(A, newtype, reuse, newmat));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MatMatMultNumeric_SeqDenseHIP_SeqDenseHIP_Internal(Mat A, Mat B, Mat C, PetscBool TA, PetscBool TB)
{
  PetscFunctionBegin;
  PetscCall(impl::MatMatMultNumeric_SeqDenseCUPM_SeqDenseCUPM<DeviceType::HIP>(A, B, C, TA, TB));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MatSeqDenseHIPInvertFactors_Internal(Mat A)
{
  PetscFunctionBegin;
  PetscCall(cupm_mat.InvertFactors(A));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  MatCreateSeqDenseHIP - Creates a sequential matrix in dense format using HIP.

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
PetscErrorCode MatCreateSeqDenseHIP(MPI_Comm comm, PetscInt m, PetscInt n, PetscScalar *data, Mat *A)
{
  PetscFunctionBegin;
  PetscCall(MatCreateSeqDenseCUPM<DeviceType::HIP>(comm, m, n, data, A));
  PetscFunctionReturn(PETSC_SUCCESS);
}
