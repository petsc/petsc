#include "../matmpidensecupm.hpp"

using namespace Petsc::mat::cupm;
using Petsc::device::cupm::DeviceType;

static constexpr impl::MatDense_MPI_CUPM<DeviceType::CUDA> mat_cupm{};

/*MC
  MATDENSECUDA - "densecuda" - A matrix type to be used for dense matrices on GPUs.

  This matrix type is identical to `MATSEQDENSECUDA` when constructed with a single process
  communicator, and `MATMPIDENSECUDA` otherwise.

  Options Database Key:
. -mat_type densecuda - sets the matrix type to `MATDENSECUDA` during a call to
                        `MatSetFromOptions()`

  Level: beginner

.seealso: [](chapter_matrices), `Mat`, `MATSEQDENSECUDA`, `MATMPIDENSECUDA`, `MATSEQDENSEHIP`,
`MATMPIDENSEHIP`, `MATDENSE`
M*/

/*MC
  MATMPIDENSECUDA - "mpidensecuda" - A matrix type to be used for distributed dense matrices on
  GPUs.

  Options Database Key:
. -mat_type mpidensecuda - sets the matrix type to `MATMPIDENSECUDA` during a call to
                           `MatSetFromOptions()`

  Level: beginner

.seealso: [](chapter_matrices), `Mat`, `MATDENSECUDA`, `MATMPIDENSE`, `MATSEQDENSE`,
`MATSEQDENSECUDA`, `MATSEQDENSEHIP`
M*/
PETSC_INTERN PetscErrorCode MatCreate_MPIDenseCUDA(Mat A)
{
  PetscFunctionBegin;
  PetscCall(mat_cupm.Create(A));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MatConvert_MPIDense_MPIDenseCUDA(Mat A, MatType type, MatReuse reuse, Mat *ret)
{
  PetscFunctionBegin;
  PetscCall(mat_cupm.Convert_MPIDense_MPIDenseCUPM(A, type, reuse, ret));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  MatCreateDenseCUDA - Creates a matrix in `MATDENSECUDA` format using CUDA.

  Collective

  Input Parameters:
+ comm - MPI communicator
. m    - number of local rows (or `PETSC_DECIDE` to have calculated if `M` is given)
. n    - number of local columns (or `PETSC_DECIDE` to have calculated if `N` is given)
. M    - number of global rows (or `PETSC_DECIDE` to have calculated if `m` is given)
. N    - number of global columns (or `PETSC_DECIDE` to have calculated if `n` is given)
- data - optional location of GPU matrix data. Pass `NULL` to have PETSc to control matrix memory allocation.

  Output Parameter:
. A - the matrix

  Level: intermediate

.seealso: `MATDENSECUDA`, `MatCreate()`, `MatCreateDense()`
@*/
PetscErrorCode MatCreateDenseCUDA(MPI_Comm comm, PetscInt m, PetscInt n, PetscInt M, PetscInt N, PetscScalar *data, Mat *A)
{
  PetscFunctionBegin;
  PetscCall(MatCreateDenseCUPM<DeviceType::CUDA>(comm, m, n, M, N, data, A));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  MatDenseCUDAPlaceArray - Allows one to replace the GPU array in a `MATDENSECUDA` matrix with an
  array provided by the user. This is useful to avoid copying an array into a matrix.

  Not Collective

  Input Parameters:
+ mat   - the matrix
- array - the array in column major order

  Level: developer

  Note:
  You can return to the original array with a call to `MatDenseCUDAResetArray()`. The user is
  responsible for freeing this array; it will not be freed when the matrix is destroyed. The
  array must have been allocated with `cudaMalloc()`.

.seealso: `MATDENSECUDA`, `MatDenseCUDAGetArray()`, `MatDenseCUDAResetArray()`,
          `MatDenseCUDAReplaceArray()`
@*/
PetscErrorCode MatDenseCUDAPlaceArray(Mat mat, const PetscScalar *array)
{
  PetscFunctionBegin;
  PetscCall(MatDenseCUPMPlaceArray<DeviceType::CUDA>(mat, array));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  MatDenseCUDAResetArray - Resets the matrix array to that it previously had before the call to
  `MatDenseCUDAPlaceArray()`

  Not Collective

  Input Parameter:
. mat - the matrix

  Level: developer

  Note:
  You can only call this after a call to `MatDenseCUDAPlaceArray()`

.seealso: `MATDENSECUDA`, `MatDenseCUDAGetArray()`, `MatDenseCUDAPlaceArray()`
@*/
PetscErrorCode MatDenseCUDAResetArray(Mat mat)
{
  PetscFunctionBegin;
  PetscCall(MatDenseCUPMResetArray<DeviceType::CUDA>(mat));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  MatDenseCUDAReplaceArray - Allows one to replace the GPU array in a `MATDENSECUDA` matrix
  with an array provided by the user. This is useful to avoid copying an array into a matrix.

  Not Collective

  Input Parameters:
+ mat   - the matrix
- array - the array in column major order

  Level: developer

  Note:
  This permanently replaces the GPU array and frees the memory associated with the old GPU
  array. The memory passed in CANNOT be freed by the user. It will be freed when the matrix is
  destroyed. The array should respect the matrix leading dimension.

.seealso: `MatDenseCUDAGetArray()`, `MatDenseCUDAPlaceArray()`, `MatDenseCUDAResetArray()`
@*/
PetscErrorCode MatDenseCUDAReplaceArray(Mat mat, const PetscScalar *array)
{
  PetscFunctionBegin;
  PetscCall(MatDenseCUPMReplaceArray<DeviceType::CUDA>(mat, array));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  MatDenseCUDAGetArrayWrite - Provides write access to the CUDA buffer inside a `MATDENSECUDA`
  matrix.

  Not Collective

  Input Parameter:
. A - the matrix

  Output Parameter:
. a - the GPU array in column major order

  Level: developer

  Notes:
  The data on the GPU may not be updated due to operations done on the CPU. If you need updated
  data, use `MatDenseCUDAGetArray()`.

  The array must be restored with `MatDenseCUDARestoreArrayWrite()` when no longer needed.

.seealso: `MATDENSECUDA`, `MatDenseCUDAGetArray()`, `MatDenseCUDARestoreArray()`,
          `MatDenseCUDARestoreArrayWrite()`, `MatDenseCUDAGetArrayRead()`,
          `MatDenseCUDARestoreArrayRead()`
@*/
PetscErrorCode MatDenseCUDAGetArrayWrite(Mat A, PetscScalar **a)
{
  PetscFunctionBegin;
  PetscCall(MatDenseCUPMGetArrayWrite<DeviceType::CUDA>(A, a));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  MatDenseCUDARestoreArrayWrite - Restore write access to the CUDA buffer inside a
  `MATDENSECUDA` matrix previously obtained with `MatDenseCUDAGetArrayWrite()`.

  Not Collective

  Input Parameters:
+ A     - the matrix
- a - the GPU array in column major order

  Level: developer

.seealso: `MATDENSECUDA`, `MatDenseCUDAGetArray()`, `MatDenseCUDARestoreArray()`,
`MatDenseCUDAGetArrayWrite()`, `MatDenseCUDARestoreArrayRead()`, `MatDenseCUDAGetArrayRead()`
@*/
PetscErrorCode MatDenseCUDARestoreArrayWrite(Mat A, PetscScalar **a)
{
  PetscFunctionBegin;
  PetscCall(MatDenseCUPMRestoreArrayWrite<DeviceType::CUDA>(A, a));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  MatDenseCUDAGetArrayRead - Provides read-only access to the CUDA buffer inside a
  `MATDENSECUDA` matrix. The array must be restored with `MatDenseCUDARestoreArrayRead()` when
  no longer needed.

  Not Collective

  Input Parameter:
. A - the matrix

  Output Parameter:
. a - the GPU array in column major order

  Level: developer

  Note:
  Data may be copied to the GPU due to operations done on the CPU. If you need write only
  access, use `MatDenseCUDAGetArrayWrite()`.

.seealso: `MATDENSECUDA`, `MatDenseCUDAGetArray()`, `MatDenseCUDARestoreArray()`,
          `MatDenseCUDARestoreArrayWrite()`, `MatDenseCUDAGetArrayWrite()`,
          `MatDenseCUDARestoreArrayRead()`
@*/
PetscErrorCode MatDenseCUDAGetArrayRead(Mat A, const PetscScalar **a)
{
  PetscFunctionBegin;
  PetscCall(MatDenseCUPMGetArrayRead<DeviceType::CUDA>(A, a));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  MatDenseCUDARestoreArrayRead - Restore read-only access to the CUDA buffer inside a
  `MATDENSECUDA` matrix previously obtained with a call to `MatDenseCUDAGetArrayRead()`.

  Not Collective

  Input Parameters:
+ A     - the matrix
- a - the GPU array in column major order

  Level: developer

  Note:
  Data can be copied to the GPU due to operations done on the CPU. If you need write only
  access, use `MatDenseCUDAGetArrayWrite()`.

.seealso: `MATDENSECUDA`, `MatDenseCUDAGetArray()`, `MatDenseCUDARestoreArray()`,
          `MatDenseCUDARestoreArrayWrite()`, `MatDenseCUDAGetArrayWrite()`, `MatDenseCUDAGetArrayRead()`
@*/
PetscErrorCode MatDenseCUDARestoreArrayRead(Mat A, const PetscScalar **a)
{
  PetscFunctionBegin;
  PetscCall(MatDenseCUPMRestoreArrayRead<DeviceType::CUDA>(A, a));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  MatDenseCUDAGetArray - Provides access to the CUDA buffer inside a `MATDENSECUDA` matrix. The
  array must be restored with `MatDenseCUDARestoreArray()` when no longer needed.

  Not Collective

  Input Parameter:
. A - the matrix

  Output Parameter:
. a - the GPU array in column major order

  Level: developer

  Note:
  Data can be copied to the GPU due to operations done on the CPU. If you need write only
  access, use `MatDenseCUDAGetArrayWrite()`. For read-only access, use
  `MatDenseCUDAGetArrayRead()`.

.seealso: `MATDENSECUDA`, `MatDenseCUDAGetArrayRead()`, `MatDenseCUDARestoreArray()`,
          `MatDenseCUDARestoreArrayWrite()`, `MatDenseCUDAGetArrayWrite()`,
          `MatDenseCUDARestoreArrayRead()`
@*/
PetscErrorCode MatDenseCUDAGetArray(Mat A, PetscScalar **a)
{
  PetscFunctionBegin;
  PetscCall(MatDenseCUPMGetArray<DeviceType::CUDA>(A, a));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  MatDenseCUDARestoreArray - Restore access to the CUDA buffer inside a `MATDENSECUDA` matrix
  previously obtained with `MatDenseCUDAGetArray()`.

  Not Collective

  Level: developer

  Input Parameters:
+ A - the matrix
- a - the GPU array in column major order

.seealso: `MATDENSECUDA`, `MatDenseCUDAGetArray()`, `MatDenseCUDARestoreArrayWrite()`,
          `MatDenseCUDAGetArrayWrite()`, `MatDenseCUDARestoreArrayRead()`, `MatDenseCUDAGetArrayRead()`
@*/
PetscErrorCode MatDenseCUDARestoreArray(Mat A, PetscScalar **a)
{
  PetscFunctionBegin;
  PetscCall(MatDenseCUPMRestoreArray<DeviceType::CUDA>(A, a));
  PetscFunctionReturn(PETSC_SUCCESS);
}
