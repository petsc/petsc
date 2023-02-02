static const char help[] = "Tests VecPointwiseMaxAbs()\n\n";

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

int main(int argc, char **argv)
{
  Vec x, y, z;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));

  PetscCall(VecCreate(PETSC_COMM_WORLD, &x));
  PetscCall(VecSetSizes(x, PETSC_DECIDE, 10));
  PetscCall(VecSetFromOptions(x));
  PetscCall(VecDuplicate(x, &y));
  PetscCall(VecDuplicate(x, &z));

  PetscCall(VecSet(x, 0.0));
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

  PetscCall(VecDestroy(&x));
  PetscCall(VecDestroy(&y));
  PetscCall(VecDestroy(&z));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

  testset:
    output_file: ./output/empty.out
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
