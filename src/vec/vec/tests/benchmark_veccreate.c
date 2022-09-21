static char help[] = "Benchmark VecCreate() for GPU vectors.\n\
  -n <length> : vector length\n\n";

#include <petscvec.h>
#include <petsctime.h>
#include <petscdevice_cuda.h>

int main(int argc, char **argv)
{
  PetscInt       i, n = 5, iter = 10;
  Vec            x;
  PetscLogDouble v0, v1;
  PetscMemType   memtype;
  PetscScalar   *array;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, (char *)0, help));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-n", &n, NULL));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-iter", &iter, NULL));

  for (i = 0; i < iter; i++) {
    PetscCall(PetscTime(&v0));
    PetscCall(VecCreate(PETSC_COMM_WORLD, &x));
    PetscCall(VecSetSizes(x, PETSC_DECIDE, n));
    PetscCall(VecSetFromOptions(x));
    /* make sure the vector's array exists */
    PetscCall(VecGetArrayAndMemType(x, &array, &memtype));
    PetscCall(VecRestoreArrayAndMemType(x, &array));
    PetscCall(WaitForCUDA());
    PetscCall(PetscTime(&v1));
    PetscCall(VecDestroy(&x));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Iteration %" PetscInt_FMT ": Time= %g\n", i, (double)(v1 - v0)));
  }
  PetscCall(PetscFinalize());
  return 0;
}
/*TEST
  build:
      requires: cuda
  test:
      args: -vec_type cuda
TEST*/
