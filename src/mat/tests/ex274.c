static char help[] = "Test MatGetOrdering() on MPIAIJ and its derived matrices. The returned permutations should be on the same MPI communicator as the matrix.\n\n";

// Contributed by Steven Dargaville in issue #1897

#include <petscmat.h>

int main(int argc, char **argv)
{
  Mat      A, A_perm;
  IS       rperm, cperm;
  PetscInt i, n = 20, rstart, rend;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));

  PetscCall(MatCreate(PETSC_COMM_WORLD, &A));
  PetscCall(MatSetSizes(A, PETSC_DECIDE, PETSC_DECIDE, n, n));
  PetscCall(MatSetFromOptions(A));
  PetscCall(MatSetUp(A));
  PetscCall(MatGetOwnershipRange(A, &rstart, &rend));
  for (i = rstart; i < rend; i++) {
    PetscScalar two = 2.0, mone = -1.0;
    PetscInt    ip1 = i + 1, im1 = i - 1;
    PetscCall(MatSetValues(A, 1, &i, 1, &i, &two, INSERT_VALUES));

    if (i > 0) PetscCall(MatSetValues(A, 1, &i, 1, &im1, &mone, INSERT_VALUES));
    if (i < n - 1) PetscCall(MatSetValues(A, 1, &i, 1, &ip1, &mone, INSERT_VALUES));
  }
  PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));

  PetscCall(MatGetOrdering(A, MATORDERINGNATURAL, &rperm, &cperm));
  PetscCall(MatPermute(A, rperm, cperm, &A_perm));

  PetscCall(MatDestroy(&A_perm));
  PetscCall(ISDestroy(&rperm));
  PetscCall(ISDestroy(&cperm));
  PetscCall(MatDestroy(&A));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST
  testset:
    nsize: 2
    output_file: output/empty.out

    test:
      suffix: host
      args: -mat_type aij

    test:
      suffix: cuda
      requires: cuda
      args: -mat_type aijcusparse

    test:
      suffix: hip
      requires: hip
      args: -mat_type aijhipsparse

    test:
      suffix: kok
      requires: kokkos_kernels
      args: -mat_type aijkokkos

TEST*/
