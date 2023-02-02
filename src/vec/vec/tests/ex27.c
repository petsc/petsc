static char help[] = "Tests VecSetInf().\n\n";

#include <petscvec.h>

static PetscErrorCode TestSetInf(Vec v)
{
  PetscScalar *array;
  PetscInt     n;

  PetscFunctionBegin;
  // Zero the entries first this ensures a known initial state
  PetscCall(VecGetLocalSize(v, &n));
  PetscCall(VecGetArrayWrite(v, &array));
  PetscCall(PetscArrayzero(array, n));
  PetscCall(VecRestoreArrayWrite(v, &array));

  PetscCall(VecSetInf(v));
  // Check that it works to begin with
  PetscCall(VecGetLocalSize(v, &n));
  PetscCall(VecGetArrayRead(v, (const PetscScalar **)&array));
  for (PetscInt i = 0; i < n; ++i) {
    const PetscScalar x = array[i];

    PetscCheck(PetscIsInfOrNanScalar(x), PETSC_COMM_SELF, PETSC_ERR_PLIB, "array[%" PetscInt_FMT "] %g + %gi != infinity", i, (double)PetscRealPart(x), (double)PetscImaginaryPart(x));
  }
  PetscCall(VecRestoreArrayRead(v, (const PetscScalar **)&array));
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv)
{
  Vec v;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));

  PetscCall(VecCreate(PETSC_COMM_WORLD, &v));
  PetscCall(VecSetSizes(v, PETSC_DECIDE, 10));
  PetscCall(VecSetFromOptions(v));

  // Perform the test possibly calling v->ops->set
  PetscCall(TestSetInf(v));
  // Delete the function pointer to the implementation and do it again. This should now use the
  // "default" version
  PetscCall(VecSetOperation(v, VECOP_SET, NULL));
  PetscCall(TestSetInf(v));

  PetscCall(VecDestroy(&v));
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
