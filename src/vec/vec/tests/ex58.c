
static char help[] = "Test VecCreate{Seq|MPI}CUDAWithArrays.\n\n";

#include "petsc.h"

int main(int argc, char **argv)
{
  Vec         x, y, z;
  PetscMPIInt size;
  PetscInt    n        = 5;
  PetscScalar xHost[5] = {0., 1., 2., 3., 4.};
  PetscScalar zHost[5];
  PetscBool   equal;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, (char *)0, help));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &size));

  PetscCall(PetscArraycpy(zHost, xHost, n));
  PetscCall(VecCreateMPIWithArray(PETSC_COMM_WORLD, 1, n, PETSC_DECIDE, zHost, &z)); /* build z for comparison */

  if (size == 1) PetscCall(VecCreateSeqCUDAWithArrays(PETSC_COMM_WORLD, 1, n, xHost, NULL, &x));
  else PetscCall(VecCreateMPICUDAWithArrays(PETSC_COMM_WORLD, 1, n, PETSC_DECIDE, xHost, NULL, &x));

  PetscCall(VecEqual(z, x, &equal));
  PetscCheck(equal, PETSC_COMM_WORLD, PETSC_ERR_PLIB, "x, z are different");

  PetscCall(VecSet(x, 42.0));
  PetscCall(VecSet(z, 42.0));
  PetscCall(VecEqual(z, x, &equal));
  PetscCheck(equal, PETSC_COMM_WORLD, PETSC_ERR_PLIB, "x, z are different");

  PetscCall(VecCreateMPIWithArray(PETSC_COMM_WORLD, 1, n, PETSC_DECIDE, xHost, &y));
  PetscCall(VecSetFromOptions(y)); /* changing y's type should not lose its value */
  PetscCall(VecEqual(z, y, &equal));
  PetscCheck(equal, PETSC_COMM_WORLD, PETSC_ERR_PLIB, "y, z are different");

  PetscCall(VecDestroy(&y));
  PetscCall(VecDestroy(&x));
  PetscCall(VecDestroy(&z));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

  build:
    requires: cuda

  testset:
    output_file: output/empty.out
    nsize: {{1 2}}

    test:
      suffix: y_host

    test:
      TODO: we need something like VecConvert()
      requires: kokkos_kernels
      suffix: y_dev
      args: -vec_type {{standard mpi cuda kokkos}}
TEST*/
