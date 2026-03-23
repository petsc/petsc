static char help[] = "Tests MatADot() and MatANorm() for MATDIAGONAL matrices\n\n";

#include <petscmat.h>

int main(int argc, char **argv)
{
  Mat         A;
  Vec         d, x, y;
  PetscScalar dot;
  PetscReal   norm;
  PetscInt    use_case = 0;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-use_case", &use_case, NULL));
  PetscCall(VecCreate(PETSC_COMM_WORLD, &d));
  PetscCall(VecSetSizes(d, PETSC_DECIDE, 10));
  PetscCall(VecSetFromOptions(d));
  PetscCall(VecSet(d, 1.0));
  switch (use_case) {
  case 0:
    PetscCall(MatCreateDiagonal(d, &A));
    break;
  case 1:
    PetscCall(MatCreate(PETSC_COMM_WORLD, &A));
    PetscCall(MatSetType(A, MATDIAGONAL));
    PetscCall(MatDiagonalSetDiagonal(A, d));
    break;
  case 2:
    PetscCall(MatCreate(PETSC_COMM_WORLD, &A));
    PetscCall(MatSetType(A, MATDIAGONAL));
    PetscCall(MatSetFromOptions(A));
    PetscCall(MatSetSizes(A, PETSC_DECIDE, PETSC_DECIDE, 10, 10));
    PetscCall(MatSetUp(A));
    PetscCall(MatDiagonalSet(A, d, INSERT_VALUES));
    break;
  case 3:
  default:
    PetscCall(MatCreate(PETSC_COMM_WORLD, &A));
    PetscCall(MatSetType(A, MATAIJ));
    PetscCall(MatSetFromOptions(A));
    PetscCall(MatSetSizes(A, PETSC_DECIDE, PETSC_DECIDE, 10, 10));
    PetscCall(MatSetUp(A));
    PetscCall(MatDiagonalSet(A, d, INSERT_VALUES));
  }
  PetscCall(VecDuplicate(d, &x));
  PetscCall(VecSet(x, 2.0));
  PetscCall(VecDuplicate(d, &y));
  PetscCall(VecSet(y, 3.0));
  PetscCall(MatADot(A, x, y, &dot));
  PetscCall(MatANorm(A, x, &norm));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "The inner product is %g + %gi and the norm is %g\n", (double)PetscRealPart(dot), (double)PetscImaginaryPart(dot), (double)norm));
  PetscCall(MatDestroy(&A));
  PetscCall(VecDestroy(&d));
  PetscCall(VecDestroy(&x));
  PetscCall(VecDestroy(&y));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST
  testset:
    nsize: {{1 2}}
    output_file: output/ex271_seq.out

    test:
      suffix: cpu
      args: -use_case {{0 1 2 3}}

    test:
      requires: kokkos_kernels
      suffix: kokkos
      args: -vec_type kokkos -use_case {{0 1}}

    test:
      requires: kokkos_kernels
      suffix: kokkos_usecase2
      args: -vec_type kokkos -use_case 2 -mat_vec_type kokkos

    test:
      requires: kokkos_kernels
      suffix: kokkos_aij
      args: -vec_type kokkos -use_case 3 -mat_type aijkokkos

    test:
      requires: cuda
      suffix: cuda
      args: -vec_type cuda -use_case {{0 1}}

    test:
      requires: cuda
      suffix: cuda_aij
      args: -vec_type cuda -use_case 3 -mat_type aijcusparse

    test:
      requires: hip
      suffix: hip
      args: -vec_type hip -use_case {{0 1}}

    test:
      requires: hip
      suffix: hip_aij
      args: -vec_type hip -use_case 3 -mat_type aijhipsparse

TEST*/
