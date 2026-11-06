static const char help[] = "Tests VecPointwiseMaxAbs(), VecPointwiseMax(), and VecPointwiseMin()\n\n";

#include <petscvec.h>

static PetscErrorCode TestPointwiseMaxAbs(Vec result, Vec x, Vec y, Vec ref)
{
  PetscInt           n;
  const PetscScalar *array, *ref_array;

  PetscFunctionBegin;
  PetscCall(VecPointwiseMaxAbs(result, x, y));
  PetscCall(VecGetLocalSize(result, &n));
  PetscCall(VecGetArrayRead(result, &array));
  PetscCall(VecGetArrayRead(ref, &ref_array));
  for (PetscInt i = 0; i < n; ++i) {
    const PetscReal expected    = PetscAbsScalar(ref_array[i]);
    const PetscReal actual_real = PetscRealPart(array[i]), actual_imag = PetscImaginaryPart(array[i]);

    PetscCheck(actual_imag == (PetscReal)0.0, PETSC_COMM_SELF, PETSC_ERR_PLIB, "VecPointwiseMaxAbs() did not properly take absolute value, imaginary part %g != 0.0", (double)actual_imag);
    PetscCheck(actual_real >= (PetscReal)0.0, PETSC_COMM_SELF, PETSC_ERR_PLIB, "VecPointwiseMaxAbs() did not properly take absolute value, real part %g < 0.0", (double)actual_real);
    PetscCheck(actual_real == expected, PETSC_COMM_SELF, PETSC_ERR_PLIB, "VecShift() returned array[%" PetscInt_FMT "] %g + %gi != expected_array[%" PetscInt_FMT "] %g + 0.0i", i, (double)actual_real, (double)actual_imag, i, (double)expected);
  }
  PetscCall(VecRestoreArrayRead(ref, &ref_array));
  PetscCall(VecRestoreArrayRead(result, &array));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode CheckPointwiseMaxMin(Vec result, Vec x, Vec y, PetscBool max)
{
  PetscInt           n;
  const PetscScalar *array, *x_array, *y_array;

  PetscFunctionBeginUser;
  PetscCall(VecGetLocalSize(result, &n));
  PetscCall(VecGetArrayRead(result, &array));
  PetscCall(VecGetArrayRead(x, &x_array));
  PetscCall(VecGetArrayRead(y, &y_array));
  for (PetscInt i = 0; i < n; ++i) {
    const PetscReal rx = PetscRealPart(x_array[i]), ry = PetscRealPart(y_array[i]);
    const PetscReal expected = max ? PetscMax(rx, ry) : PetscMin(rx, ry);

    PetscCheck(array[i] == (PetscScalar)expected, PETSC_COMM_SELF, PETSC_ERR_PLIB, "%s returned array[%" PetscInt_FMT "] %g + %gi != expected %g", max ? "VecPointwiseMax()" : "VecPointwiseMin()", i, (double)PetscRealPart(array[i]), (double)PetscImaginaryPart(array[i]), (double)expected);
  }
  PetscCall(VecRestoreArrayRead(y, &y_array));
  PetscCall(VecRestoreArrayRead(x, &x_array));
  PetscCall(VecRestoreArrayRead(result, &array));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TestPointwiseMaxMin(Vec z, Vec x, Vec y)
{
  PetscInt     rstart, rend;
  PetscScalar *x_array, *y_array;

  PetscFunctionBeginUser;
  // Entries of x and y cross over, so each of x and y supplies some of the max and min
  PetscCall(VecGetOwnershipRange(x, &rstart, &rend));
  PetscCall(VecGetArrayWrite(x, &x_array));
  PetscCall(VecGetArrayWrite(y, &y_array));
  for (PetscInt i = rstart; i < rend; ++i) {
    x_array[i - rstart] = (PetscReal)(i - 5);
    y_array[i - rstart] = (PetscReal)(4 - i);
  }
  PetscCall(VecRestoreArrayWrite(y, &y_array));
  PetscCall(VecRestoreArrayWrite(x, &x_array));

  // x and y are up to date only on the host
  PetscCall(VecPointwiseMax(z, x, y));
  PetscCall(CheckPointwiseMaxMin(z, x, y, PETSC_TRUE));
  PetscCall(VecPointwiseMin(z, x, y));
  PetscCall(CheckPointwiseMaxMin(z, x, y, PETSC_FALSE));

  // Scaling moves the data of device vectors to the device
  PetscCall(VecScale(x, 2.0));
  PetscCall(VecScale(y, 3.0));
  PetscCall(VecPointwiseMax(z, x, y));
  PetscCall(CheckPointwiseMaxMin(z, x, y, PETSC_TRUE));
  PetscCall(VecScale(x, 2.0));
  PetscCall(VecScale(y, 3.0));
  PetscCall(VecPointwiseMin(z, x, y));
  PetscCall(CheckPointwiseMaxMin(z, x, y, PETSC_FALSE));

  // Test that it works if the result is one of the inputs
  PetscCall(VecCopy(x, z));
  PetscCall(VecPointwiseMax(z, z, y));
  PetscCall(CheckPointwiseMaxMin(z, x, y, PETSC_TRUE));
  PetscCall(VecCopy(y, z));
  PetscCall(VecPointwiseMin(z, x, z));
  PetscCall(CheckPointwiseMaxMin(z, x, y, PETSC_FALSE));
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv)
{
  Vec       x, y, z;
  PetscBool is_kokkos;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));

  PetscCall(VecCreate(PETSC_COMM_WORLD, &x));
  PetscCall(VecSetSizes(x, PETSC_DECIDE, 10));
  PetscCall(VecSetFromOptions(x));
  PetscCall(VecDuplicate(x, &y));
  PetscCall(VecDuplicate(x, &z));

  PetscCall(VecSet(y, 10.0));

  // Basic correctness tests, z should always match abs(y) exactly
  PetscCall(TestPointwiseMaxAbs(z, x, y, y));
  PetscCall(VecSet(x, 1.0));
  PetscCall(TestPointwiseMaxAbs(z, x, y, y));
  PetscCall(VecSet(x, -1.0));
  PetscCall(TestPointwiseMaxAbs(z, x, y, y));
  PetscCall(VecSet(y, -10.0));
  PetscCall(TestPointwiseMaxAbs(z, x, y, y));

  // Test that it works if x and y are the same vector
  PetscCall(VecSet(x, 0.0));
  PetscCall(TestPointwiseMaxAbs(z, x, x, x));
  PetscCall(VecSet(x, 1.0));
  PetscCall(TestPointwiseMaxAbs(z, x, x, x));
  PetscCall(VecSet(x, -1.0));
  PetscCall(TestPointwiseMaxAbs(z, x, x, x));

  // Test that it works if z is one of x or y
  PetscCall(VecSet(z, 0.0));
  PetscCall(VecSet(x, 0.0));
  PetscCall(TestPointwiseMaxAbs(z, x, z, x));
  PetscCall(VecSet(x, 1.0));
  PetscCall(TestPointwiseMaxAbs(z, z, x, x));
  PetscCall(VecSet(x, -10.0));
  PetscCall(TestPointwiseMaxAbs(z, x, z, x));

  // Test that it works if all vectors are the same
  PetscCall(VecSet(z, 0.0));
  PetscCall(TestPointwiseMaxAbs(z, z, z, z));
  PetscCall(VecSet(z, 1.0));
  PetscCall(TestPointwiseMaxAbs(z, z, z, z));
  PetscCall(VecSet(z, -1.0));
  PetscCall(TestPointwiseMaxAbs(z, z, z, z));

  PetscCall(TestPointwiseMaxMin(z, x, y));

  // Test a Kokkos result with an input that is not a Kokkos vector
  PetscCall(PetscObjectTypeCompareAny((PetscObject)x, &is_kokkos, VECSEQKOKKOS, VECMPIKOKKOS, ""));
  if (is_kokkos) {
    Vec      y_standard;
    PetscInt n;

    PetscCall(VecGetLocalSize(x, &n));
    PetscCall(VecCreate(PETSC_COMM_WORLD, &y_standard));
    PetscCall(VecSetSizes(y_standard, n, PETSC_DECIDE));
    PetscCall(VecSetType(y_standard, VECSTANDARD));
    PetscCall(TestPointwiseMaxMin(z, x, y_standard));
    PetscCall(VecDestroy(&y_standard));
  }

  PetscCall(VecDestroy(&x));
  PetscCall(VecDestroy(&y));
  PetscCall(VecDestroy(&z));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

  testset:
    output_file: output/empty.out
    nsize: {{1 2}}
    test:
      suffix: standard
    test:
      requires: defined(PETSC_USE_SHARED_MEMORY)
      args: -vec_type shared
      suffix: shared
    test:
      requires: viennacl
      args: -vec_type viennacl
      suffix: viennacl
    test:
      requires: kokkos_kernels
      args: -vec_type kokkos
      suffix: kokkos
    test:
      requires: cuda
      args: -vec_type cuda
      suffix: cuda
    test:
      requires: hip
      args: -vec_type hip
      suffix: hip

TEST*/
