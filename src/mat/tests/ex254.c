static char help[] = "Test MatSetValuesCOO for MPIAIJ and its subclasses \n\n";

#include <petscmat.h>
int main(int argc, char **args)
{
  Mat            A, B, C;
  PetscInt       k;
  const PetscInt M = 18, N = 18;
  PetscBool      equal;
  PetscScalar   *vals;
  PetscBool      flg = PETSC_FALSE, freecoo = PETSC_FALSE;
  PetscInt       ncoos = 1;

  // clang-format off
  /* Construct 18 x 18 matrices, which are big enough to have complex communication patterns but still small enough for debugging */
  PetscInt i0[] = {7, 7, 8, 8,  9, 16, 17,  9, 10, 1, 1, -2, 2, 3, 3, 14, 4, 5, 10, 13,  9,  9, 10, 1, 0, 0, 5,  5,  6, 6, 13, 13, 14, -14, 4, 4, 5, 11, 11, 12, 15, 15, 16};
  PetscInt j0[] = {1, 6, 2, 4, 10, 15, 13, 16, 11, 2, 7,  3, 8, 4, 9, 13, 5, 2, 15, 14, 10, 16, 11, 2, 0, 1, 5, -11, 0, 7, 15, 17, 11,  13, 4, 8, 2, 12, 17, 13,  3, 16,  9};

  PetscInt i1[] = {8, 5, 15, 16, 6, 13, 4, 17, 8,  9, 9,  10, -6, 12, 7, 3, -4, 1, 1, 2, 5,  5, 6, 14, 17, 8,  9,  9, 10, 4,  5, 10, 11, 1, 2};
  PetscInt j1[] = {2, 3, 16,  9, 5, 17, 1, 13, 4, 10, 16, 11, -5, 12, 1, 7, -1, 2, 7, 3, 6, 11, 0, 11, 13, 4, 10, 16, 11, 8, -2, 15, 12, 7, 3};

  PetscInt i2[] = {3, 4, 1, 10, 0, 1, 1, 2, 1, 1, 2, 2, 3, 3, 4, 4, 1, 2, 5,  5, 6, 4, 17, 0, 1, 1, 8, 5,  5, 6, 4, 7, 8, 5};
  PetscInt j2[] = {7, 1, 2, 11, 5, 2, 7, 3, 2, 7, 3, 8, 4, 9, 3, 5, 7, 3, 6, 11, 0, 1, 13, 5, 2, 7, 4, 6, 11, 0, 1, 3, 4, 2};
  // clang-format on

  typedef struct {
    PetscInt *i, *j, n;
  } coo_data;

  coo_data coos[3] = {
    {i0, j0, PETSC_STATIC_ARRAY_LENGTH(i0)},
    {i1, j1, PETSC_STATIC_ARRAY_LENGTH(i1)},
    {i2, j2, PETSC_STATIC_ARRAY_LENGTH(i2)}
  };
  coo_data mycoo;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &args, (char *)0, help));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-ignore_remote", &flg, NULL));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-ncoos", &ncoos, NULL));

  mycoo.n = 0;
  if (ncoos > 1) {
    PetscLayout map;

    freecoo = PETSC_TRUE;
    PetscCall(PetscLayoutCreate(PETSC_COMM_WORLD, &map));
    PetscCall(PetscLayoutSetSize(map, ncoos));
    PetscCall(PetscLayoutSetUp(map));
    PetscCall(PetscLayoutGetLocalSize(map, &ncoos));
    for (PetscInt i = 0; i < ncoos; i++) mycoo.n += coos[i % 3].n;
    PetscCall(PetscMalloc2(mycoo.n, &mycoo.i, mycoo.n, &mycoo.j));
    mycoo.n = 0;
    for (PetscInt i = 0; i < ncoos; i++) {
      PetscCall(PetscArraycpy(mycoo.i + mycoo.n, coos[i % 3].i, coos[i % 3].n));
      PetscCall(PetscArraycpy(mycoo.j + mycoo.n, coos[i % 3].j, coos[i % 3].n));
      mycoo.n += coos[i % 3].n;
    }
    PetscCall(PetscLayoutDestroy(&map));
  } else if (ncoos == 1 && PetscGlobalRank < 3) mycoo = coos[PetscGlobalRank];

  PetscCall(MatCreate(PETSC_COMM_WORLD, &A));
  PetscCall(MatSetSizes(A, PETSC_DECIDE, PETSC_DECIDE, M, N));
  PetscCall(MatSetType(A, MATAIJ));
  // Do not preallocate A to also test MatHash with MAT_IGNORE_OFF_PROC_ENTRIES
  // PetscCall(MatSeqAIJSetPreallocation(A, 2, NULL));
  // PetscCall(MatMPIAIJSetPreallocation(A, 2, NULL, 2, NULL));
  PetscCall(MatSetOption(A, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE));
  PetscCall(MatSetOption(A, MAT_IGNORE_OFF_PROC_ENTRIES, flg));

  PetscCall(PetscMalloc1(mycoo.n, &vals));
  for (k = 0; k < mycoo.n; k++) {
    vals[k] = mycoo.j[k];
    PetscCall(MatSetValue(A, mycoo.i[k], mycoo.j[k], vals[k], ADD_VALUES));
  }
  PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatViewFromOptions(A, NULL, "-a_view"));

  PetscCall(MatCreate(PETSC_COMM_WORLD, &B));
  PetscCall(MatSetSizes(B, PETSC_DECIDE, PETSC_DECIDE, M, N));
  PetscCall(MatSetFromOptions(B));
  PetscCall(MatSetOption(B, MAT_IGNORE_OFF_PROC_ENTRIES, flg));
  PetscCall(MatSetPreallocationCOO(B, mycoo.n, mycoo.i, mycoo.j));

  /* Test with ADD_VALUES on a zeroed matrix */
  PetscCall(MatSetValuesCOO(B, vals, ADD_VALUES));
  PetscCall(MatMultEqual(A, B, 10, &equal));
  if (!equal) PetscCall(PetscPrintf(PETSC_COMM_WORLD, "MatSetValuesCOO() failed\n"));
  PetscCall(MatViewFromOptions(B, NULL, "-b_view"));

  /* Test with MatDuplicate on a zeroed matrix */
  PetscCall(MatDuplicate(B, MAT_DO_NOT_COPY_VALUES, &C));
  PetscCall(MatDestroy(&B));
  PetscCall(MatSetValuesCOO(C, vals, ADD_VALUES));
  PetscCall(MatMultEqual(A, C, 10, &equal));
  if (!equal) PetscCall(PetscPrintf(PETSC_COMM_WORLD, "MatSetValuesCOO() on duplicated matrix failed\n"));
  PetscCall(MatViewFromOptions(C, NULL, "-c_view"));

  PetscCall(PetscFree(vals));
  if (freecoo) PetscCall(PetscFree2(mycoo.i, mycoo.j));
  PetscCall(MatDestroy(&A));
  PetscCall(MatDestroy(&C));

  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

  testset:
    output_file: output/ex254_1.out
    nsize: {{1 2 3}}
    args: -ignore_remote {{0 1}}
    filter: grep -v type | grep -v "Mat Object"

    test:
      suffix: kokkos
      requires: kokkos_kernels
      args: -mat_type aijkokkos

    test:
      suffix: cuda
      requires: cuda
      args: -mat_type aijcusparse

    test:
      suffix: hip
      requires: hip
      args: -mat_type aijhipsparse

    test:
      suffix: aij
      args: -mat_type aij

    test:
      suffix: hypre
      requires: hypre
      args: -mat_type hypre

  testset:
    output_file: output/ex254_2.out
    nsize: 1
    args: -ncoos 3
    filter: grep -v type | grep -v "Mat Object"

    test:
      suffix: 2_kokkos
      requires: kokkos_kernels
      args: -mat_type aijkokkos

    test:
      suffix: 2_cuda
      requires: cuda
      args: -mat_type aijcusparse

    test:
      suffix: 2_hip
      requires: hip
      args: -mat_type aijhipsparse

    test:
      suffix: 2_aij
      args: -mat_type aij

    test:
      suffix: 2_hypre
      requires: hypre
      args: -mat_type hypre

TEST*/
