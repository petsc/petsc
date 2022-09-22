
static char help[] = "Scatters from a parallel vector into sequential vectors.\n\n";

#include <petscvec.h>

int main(int argc, char **argv)
{
  PetscMPIInt rank;
  PetscInt    n = 5, idx1[2] = {0, 3}, idx2[2] = {1, 4};
  PetscScalar one = 1.0, two = 2.0;
  Vec         x, y;
  IS          is1, is2;
  VecScatter  ctx = 0;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, (char *)0, help));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-n", &n, NULL));
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &rank));

  /* create two vectors */
  PetscCall(VecCreate(PETSC_COMM_WORLD, &x));
  PetscCall(VecSetSizes(x, n, PETSC_DECIDE));
  PetscCall(VecSetFromOptions(x));
  PetscCall(VecCreate(PETSC_COMM_SELF, &y));
  PetscCall(VecSetSizes(y, n, PETSC_DECIDE));
  PetscCall(VecSetFromOptions(y));

  /* create two index sets */
  PetscCall(ISCreateGeneral(PETSC_COMM_SELF, 2, idx1, PETSC_COPY_VALUES, &is1));
  PetscCall(ISCreateGeneral(PETSC_COMM_SELF, 2, idx2, PETSC_COPY_VALUES, &is2));

  PetscCall(VecSet(x, one));
  PetscCall(VecSet(y, two));
  PetscCall(VecScatterCreate(x, is1, y, is2, &ctx));
  PetscCall(VecScatterBegin(ctx, x, y, INSERT_VALUES, SCATTER_FORWARD));
  PetscCall(VecScatterEnd(ctx, x, y, INSERT_VALUES, SCATTER_FORWARD));
  PetscCall(VecScatterDestroy(&ctx));

  if (rank == 0) PetscCall(VecView(y, PETSC_VIEWER_STDOUT_SELF));

  PetscCall(ISDestroy(&is1));
  PetscCall(ISDestroy(&is2));

  PetscCall(VecDestroy(&x));
  PetscCall(VecDestroy(&y));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   test:
      nsize: 2
      filter: grep -v type
      diff_args: -j

   test:
      diff_args: -j
      suffix: cuda
      args: -vec_type cuda
      output_file: output/ex4_1.out
      filter: grep -v type
      requires: cuda

   test:
      diff_args: -j
      suffix: cuda2
      nsize: 2
      args: -vec_type cuda
      output_file: output/ex4_1.out
      filter: grep -v type
      requires: cuda

   test:
      diff_args: -j
      suffix: kokkos
      args: -vec_type kokkos
      output_file: output/ex4_1.out
      filter: grep -v type
      requires: kokkos_kernels

   test:
      diff_args: -j
      suffix: kokkos2
      nsize: 2
      args: -vec_type kokkos
      output_file: output/ex4_1.out
      filter: grep -v type
      requires: kokkos_kernels

   testset:
      diff_args: -j
      requires: hip
      filter: grep -v type
      args: -vec_type hip
      output_file: output/ex4_1.out
      test:
        suffix: hip
      test:
        suffix: hip2
        nsize: 2
TEST*/
