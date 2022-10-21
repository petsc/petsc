static const char help[] = "Tests VecMaxPointwiseDivide()\n\n";

#include <petscvec.h>

int main(int argc, char **argv)
{
  Vec          x, y;
  PetscScalar *x_array;
  PetscInt     n;
  PetscReal    max, expected;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));

  PetscCall(VecCreate(PETSC_COMM_WORLD, &x));
  PetscCall(VecSetSizes(x, PETSC_DECIDE, 10));
  PetscCall(VecSetFromOptions(x));

  PetscCall(VecGetLocalSize(x, &n));
  PetscCall(VecGetArrayWrite(x, &x_array));
  for (PetscInt i = 0; i < n; ++i) x_array[i] = (PetscScalar)(i + 1);
  PetscCall(VecRestoreArrayWrite(x, &x_array));
  expected = (PetscReal)n;

  PetscCall(VecDuplicate(x, &y));

  // check that it works at all
  PetscCall(VecSet(y, 1.0));
  PetscCall(VecMaxPointwiseDivide(x, y, &max));
  PetscCheck(PetscIsCloseAtTol(max, expected, 1e-12, 0.0), PETSC_COMM_SELF, PETSC_ERR_PLIB, "VecMaxPointwiseDivide() returned %g != expected %g for y = 1.0", (double)max, (double)expected);

  // check that it takes the absolute value
  PetscCall(VecSet(y, -1.0));
  PetscCall(VecMaxPointwiseDivide(x, y, &max));
  PetscCheck(PetscIsCloseAtTol(max, expected, 1e-12, 0.0), PETSC_COMM_SELF, PETSC_ERR_PLIB, "VecMaxPointwiseDivide() returned %g != expected %g for y = -1.0", (double)max, (double)expected);

  // check that it ignores zero entries in y (treats them as 1.0)
  PetscCall(VecZeroEntries(y));
  PetscCall(VecMaxPointwiseDivide(x, y, &max));
  PetscCheck(PetscIsCloseAtTol(max, expected, 1e-12, 0.0), PETSC_COMM_SELF, PETSC_ERR_PLIB, "VecMaxPointwiseDivide() returned %g != expected %g for y = 0.0", (double)max, (double)expected);

  PetscCall(VecDestroy(&y));
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
