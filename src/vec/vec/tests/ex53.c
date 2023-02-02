static const char help[] = "Tests VecShift()\n\n";

#include <petscvec.h>

static PetscBool PetscIsCloseAtTolScalar(PetscScalar l, PetscScalar r, PetscReal atol, PetscReal rtol)
{
  return (PetscBool)(PetscIsCloseAtTol(PetscRealPart(l), PetscRealPart(r), atol, rtol) && PetscIsCloseAtTol(PetscImaginaryPart(l), PetscImaginaryPart(r), atol, rtol));
}

static PetscErrorCode CheckVecShift(Vec v, PetscInt n, PetscScalar *array_copy, PetscScalar shift)
{
  const PetscScalar *array;

  PetscFunctionBegin;
  for (PetscInt i = 0; i < n; ++i) array_copy[i] += shift;
  PetscCall(VecShift(v, shift));
  PetscCall(VecGetArrayRead(v, &array));
  for (PetscInt i = 0; i < n; ++i) {
    const PetscScalar actual = array[i], expected = array_copy[i];

    PetscCheck(PetscIsCloseAtTolScalar(actual, expected, 1e-12, 0.0), PETSC_COMM_SELF, PETSC_ERR_PLIB, "VecShift() returned array[%" PetscInt_FMT "] %g + %gi != expected_array[%" PetscInt_FMT "] %g + %gi", i, (double)PetscRealPart(actual), (double)PetscImaginaryPart(actual), i, (double)PetscRealPart(expected), (double)PetscImaginaryPart(expected));
  }
  PetscCall(VecRestoreArrayRead(v, &array));
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv)
{
  Vec                x;
  PetscInt           n;
  const PetscScalar *array;
  PetscScalar       *array_copy;
  PetscReal          norm_before, norm_after;
  PetscBool          available;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));

  PetscCall(VecCreate(PETSC_COMM_WORLD, &x));
  PetscCall(VecSetSizes(x, PETSC_DECIDE, 10));
  PetscCall(VecSetFromOptions(x));

  PetscCall(VecZeroEntries(x));

  // get a copy of the vectors array, anything we do to the vector via VecShift we will also do
  // to the copy, and hence they should always match
  PetscCall(VecGetLocalSize(x, &n));
  PetscCall(PetscMalloc1(n, &array_copy));
  PetscCall(VecGetArrayRead(x, &array));
  PetscCall(PetscArraycpy(array_copy, array, n));
  PetscCall(VecRestoreArrayRead(x, &array));

  PetscCall(CheckVecShift(x, n, array_copy, 0.0));
  PetscCall(CheckVecShift(x, n, array_copy, 1.0));
  PetscCall(CheckVecShift(x, n, array_copy, -1.0));
  PetscCall(CheckVecShift(x, n, array_copy, 15.0));

  PetscCall(VecNorm(x, NORM_2, &norm_before));
  PetscCall(VecNormAvailable(x, NORM_2, &available, &norm_after));
  PetscCheck(available, PETSC_COMM_SELF, PETSC_ERR_PLIB, "VecNormAvailable() returned FALSE right after calling VecNorm()");
  // a shift of zero should not invalidate norms
  PetscCall(CheckVecShift(x, n, array_copy, 0.0));
  PetscCall(VecNormAvailable(x, NORM_2, &available, &norm_after));
  PetscCheck(available, PETSC_COMM_SELF, PETSC_ERR_PLIB, "VecNormAvailable() returned FALSE after calling VecShift() with a shift of 0.0!");
  // these can be compared with equality as the number should not change *at all*
  PetscCheck(norm_before == norm_after, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Norms differ before and after calling VecShift() with shift of 0.0: before %g after %g", (double)norm_before, (double)norm_after);

  PetscCall(PetscFree(array_copy));
  PetscCall(VecDestroy(&x));
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
