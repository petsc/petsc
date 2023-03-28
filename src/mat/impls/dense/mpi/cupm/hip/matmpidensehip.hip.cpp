#include "../matmpidensecupm.hpp"

using namespace Petsc::mat::cupm;
using Petsc::device::cupm::DeviceType;

static constexpr impl::MatDense_MPI_CUPM<DeviceType::HIP> mat_cupm{};

/*MC
  MATDENSEHIP - "densehip" - A matrix type to be used for dense matrices on GPUs.

  This matrix type is identical to `MATSEQDENSEHIP` when constructed with a single process
  communicator, and `MATMPIDENSEHIP` otherwise.

  Options Database Key:
. -mat_type densehip - sets the matrix type to `MATDENSEHIP` during a call to
                        `MatSetFromOptions()`

  Level: beginner

.seealso: [](chapter_matrices), `Mat`, `MATSEQDENSEHIP`, `MATMPIDENSEHIP`, `MATSEQDENSECUDA`,
`MATMPIDENSECUDA`, `MATDENSE`
M*/

/*MC
  MATMPIDENSEHIP - "mpidensehip" - A matrix type to be used for distributed dense matrices on
  GPUs.

  Options Database Key:
. -mat_type mpidensehip - sets the matrix type to `MATMPIDENSEHIP` during a call to
                           `MatSetFromOptions()`

  Level: beginner

.seealso: [](chapter_matrices), `Mat`, `MATDENSEHIP`, `MATMPIDENSE`, `MATSEQDENSE`,
`MATSEQDENSEHIP`, `MATSEQDENSECUDA`
M*/
PETSC_INTERN PetscErrorCode MatCreate_MPIDenseHIP(Mat A)
{
  PetscFunctionBegin;
  PetscCall(mat_cupm.Create(A));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MatConvert_MPIDense_MPIDenseHIP(Mat A, MatType type, MatReuse reuse, Mat *ret)
{
  PetscFunctionBegin;
  PetscCall(mat_cupm.Convert_MPIDense_MPIDenseCUPM(A, type, reuse, ret));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  MatCreateDenseHIP - Creates a matrix in `MATDENSEHIP` format using HIP.

  Collective

  Input Parameters:
+ comm - MPI communicator
. m    - number of local rows (or `PETSC_DECIDE` to have calculated if `M` is given)
. n    - number of local columns (or `PETSC_DECIDE` to have calculated if `N` is given)
. M    - number of global rows (or `PETSC_DECIDE` to have calculated if `m` is given)
. N    - number of global columns (or `PETSC_DECIDE` to have calculated if `n` is given)
- data - optional location of GPU matrix data. Pass`NULL` to have PETSc to control matrix
         memory allocation.

  Output Parameter:
. A - the matrix

  Level: intermediate

.seealso: `MATDENSEHIP`, `MatCreate()`, `MatCreateDense()`
@*/
PetscErrorCode MatCreateDenseHIP(MPI_Comm comm, PetscInt m, PetscInt n, PetscInt M, PetscInt N, PetscScalar *data, Mat *A)
{
  PetscFunctionBegin;
  PetscCall(MatCreateDenseCUPM<DeviceType::HIP>(comm, m, n, M, N, data, A));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  MatDenseHIPPlaceArray - Allows one to replace the GPU array in a `MATDENSEHIP` matrix with an
  array provided by the user. This is useful to avoid copying an array into a matrix.

  Not Collective

  Input Parameters:
+ mat   - the matrix
- array - the array in column major order

  Level: developer

  Note:
  You can return to the original array with a call to `MatDenseHIPResetArray()`. The user is
  responsible for freeing this array; it will not be freed when the matrix is destroyed. The
  array must have been allocated with `hipMalloc()`.

.seealso: `MATDENSEHIP`, `MatDenseHIPGetArray()`, `MatDenseHIPResetArray()`,
`MatDenseHIPReplaceArray()`
@*/
PetscErrorCode MatDenseHIPPlaceArray(Mat mat, const PetscScalar *array)
{
  PetscFunctionBegin;
  PetscCall(MatDenseCUPMPlaceArray<DeviceType::HIP>(mat, array));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  MatDenseHIPResetArray - Resets the matrix array to that it previously had before the call to
  `MatDenseHIPPlaceArray()`

  Not Collective

  Input Parameter:
. mat - the matrix

  Level: developer

  Note:
  You can only call this after a call to `MatDenseHIPPlaceArray()`

.seealso: `MATDENSEHIP`, `MatDenseHIPGetArray()`, `MatDenseHIPPlaceArray()`
@*/
PetscErrorCode MatDenseHIPResetArray(Mat mat)
{
  PetscFunctionBegin;
  PetscCall(MatDenseCUPMResetArray<DeviceType::HIP>(mat));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  MatDenseHIPReplaceArray - Allows one to replace the GPU array in a `MATDENSEHIP` matrix
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

.seealso: `MatDenseHIPGetArray()`, `MatDenseHIPPlaceArray()`, `MatDenseHIPResetArray()`
@*/
PetscErrorCode MatDenseHIPReplaceArray(Mat mat, const PetscScalar *array)
{
  PetscFunctionBegin;
  PetscCall(MatDenseCUPMReplaceArray<DeviceType::HIP>(mat, array));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  MatDenseHIPGetArrayWrite - Provides write access to the HIP buffer inside a `MATDENSEHIP`
  matrix.

  Not Collective

  Input Parameter:
. A - the matrix

  Output Parameter:
. a - the GPU array in column major order

  Level: developer

  Notes:
  The data on the GPU may not be updated due to operations done on the CPU. If you need updated
  data, use `MatDenseHIPGetArray()`.

  The array must be restored with `MatDenseHIPRestoreArrayWrite()` when no longer needed.

.seealso: `MATDENSEHIP`, `MatDenseHIPGetArray()`, `MatDenseHIPRestoreArray()`,
`MatDenseHIPRestoreArrayWrite()`, `MatDenseHIPGetArrayRead()`,
`MatDenseHIPRestoreArrayRead()`
@*/
PetscErrorCode MatDenseHIPGetArrayWrite(Mat A, PetscScalar **a)
{
  PetscFunctionBegin;
  PetscCall(MatDenseCUPMGetArrayWrite<DeviceType::HIP>(A, a));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  MatDenseHIPRestoreArrayWrite - Restore write access to the HIP buffer inside a
  `MATDENSEHIP` matrix previously obtained with `MatDenseHIPGetArrayWrite()`.

  Not Collective

  Input Parameters:
+ A     - the matrix
- a - the GPU array in column major order

  Level: developer

.seealso: `MATDENSEHIP`, `MatDenseHIPGetArray()`, `MatDenseHIPRestoreArray()`,
`MatDenseHIPGetArrayWrite()`, `MatDenseHIPRestoreArrayRead()`, `MatDenseHIPGetArrayRead()`
@*/
PetscErrorCode MatDenseHIPRestoreArrayWrite(Mat A, PetscScalar **a)
{
  PetscFunctionBegin;
  PetscCall(MatDenseCUPMRestoreArrayWrite<DeviceType::HIP>(A, a));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  MatDenseHIPGetArrayRead - Provides read-only access to the HIP buffer inside a
  `MATDENSEHIP` matrix. The array must be restored with `MatDenseHIPRestoreArrayRead()` when
  no longer needed.

  Not Collective

  Input Parameter:
. A - the matrix

  Output Parameter:
. a - the GPU array in column major order

  Level: developer

  Note:
  Data may be copied to the GPU due to operations done on the CPU. If you need write only
  access, use `MatDenseHIPGetArrayWrite()`.

.seealso: `MATDENSEHIP`, `MatDenseHIPGetArray()`, `MatDenseHIPRestoreArray()`,
`MatDenseHIPRestoreArrayWrite()`, `MatDenseHIPGetArrayWrite()`,
`MatDenseHIPRestoreArrayRead()`
@*/
PetscErrorCode MatDenseHIPGetArrayRead(Mat A, const PetscScalar **a)
{
  PetscFunctionBegin;
  PetscCall(MatDenseCUPMGetArrayRead<DeviceType::HIP>(A, a));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  MatDenseHIPRestoreArrayRead - Restore read-only access to the HIP buffer inside a
  `MATDENSEHIP` matrix previously obtained with a call to `MatDenseHIPGetArrayRead()`.

  Not Collective

  Input Parameters:
+ A     - the matrix
- a - the GPU array in column major order

  Level: developer

  Note:
  Data can be copied to the GPU due to operations done on the CPU. If you need write only
  access, use `MatDenseHIPGetArrayWrite()`.

.seealso: `MATDENSEHIP`, `MatDenseHIPGetArray()`, `MatDenseHIPRestoreArray()`,
`MatDenseHIPRestoreArrayWrite()`, `MatDenseHIPGetArrayWrite()`, `MatDenseHIPGetArrayRead()`
@*/
PetscErrorCode MatDenseHIPRestoreArrayRead(Mat A, const PetscScalar **a)
{
  PetscFunctionBegin;
  PetscCall(MatDenseCUPMRestoreArrayRead<DeviceType::HIP>(A, a));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  MatDenseHIPGetArray - Provides access to the HIP buffer inside a `MATDENSEHIP` matrix. The
  array must be restored with `MatDenseHIPRestoreArray()` when no longer needed.

  Not Collective

  Input Parameter:
. A - the matrix

  Output Parameter:
. a - the GPU array in column major order

  Level: developer

  Note:
  Data can be copied to the GPU due to operations done on the CPU. If you need write only
  access, use `MatDenseHIPGetArrayWrite()`. For read-only access, use
  `MatDenseHIPGetArrayRead()`.

.seealso: `MATDENSEHIP`, `MatDenseHIPGetArrayRead()`, `MatDenseHIPRestoreArray()`,
`MatDenseHIPRestoreArrayWrite()`, `MatDenseHIPGetArrayWrite()`,
`MatDenseHIPRestoreArrayRead()`
@*/
PetscErrorCode MatDenseHIPGetArray(Mat A, PetscScalar **a)
{
  PetscFunctionBegin;
  PetscCall(MatDenseCUPMGetArray<DeviceType::HIP>(A, a));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  MatDenseHIPRestoreArray - Restore access to the HIP buffer inside a `MATDENSEHIP` matrix
  previously obtained with `MatDenseHIPGetArray()`.

  Not Collective

  Level: developer

  Input Parameters:
+ A     - the matrix
- a - the GPU array in column major order

.seealso: `MATDENSEHIP`, `MatDenseHIPGetArray()`, `MatDenseHIPRestoreArrayWrite()`,
`MatDenseHIPGetArrayWrite()`, `MatDenseHIPRestoreArrayRead()`, `MatDenseHIPGetArrayRead()`
@*/
PetscErrorCode MatDenseHIPRestoreArray(Mat A, PetscScalar **a)
{
  PetscFunctionBegin;
  PetscCall(MatDenseCUPMRestoreArray<DeviceType::HIP>(A, a));
  PetscFunctionReturn(PETSC_SUCCESS);
}
