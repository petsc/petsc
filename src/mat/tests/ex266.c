static char help[] = "Test MatDuplicate() with new nonzeros on the duplicate\n\n";

#include <petscmat.h>
int main(int argc, char **args)
{
  Mat            A, B, C;
  PetscInt       k;
  const PetscInt M = 18, N = 18;
  PetscBool      equal;
  PetscScalar   *vals;
  PetscMPIInt    rank;

  // clang-format off
  // i0/j0[] has a dense diagonal
  PetscInt i0[] = {7, 7, 8, 8, 9, 16, 17,  9, 10, 1, 1, -2, 2, 3, 3, 14, 4, 5, 10, 13,  9,  9, 10, 1, 0, 0, 5,  5,  6, 6, 13, 13, 14, -14, 4, 4, 5, 11, 11, 12, 15, 15, 16};
  PetscInt j0[] = {7, 6, 8, 4, 9, 16, 17, 16, 10, 2, 1,  3, 2, 4, 3, 14, 4, 5, 15, 13, 10, 16, 11, 2, 0, 1, 5, -11, 0, 6, 15, 17, 11,  13, 4, 8, 2, 11, 17, 12,  3, 15,  9};

  // i0/j0[] miss some diagonals
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
  PetscCall(PetscInitialize(&argc, &args, NULL, help));
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &rank));

  mycoo = coos[rank / 3];

  PetscCall(MatCreate(PETSC_COMM_WORLD, &A));
  PetscCall(MatSetSizes(A, PETSC_DECIDE, PETSC_DECIDE, M, N));
  PetscCall(MatSetFromOptions(A));

  // Assemble matrix A with the full arrays
  PetscCall(PetscMalloc1(mycoo.n, &vals));
  for (k = 0; k < mycoo.n; k++) {
    vals[k] = mycoo.j[k];
    PetscCall(MatSetValue(A, mycoo.i[k], mycoo.j[k], vals[k], ADD_VALUES));
  }
  PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));

  // Assemble matrix B with the 1st half of the arrays
  PetscCall(MatCreate(PETSC_COMM_WORLD, &B));
  PetscCall(MatSetSizes(B, PETSC_DECIDE, PETSC_DECIDE, M, N));
  PetscCall(MatSetFromOptions(B));
  for (k = 0; k < mycoo.n / 2; k++) PetscCall(MatSetValue(B, mycoo.i[k], mycoo.j[k], vals[k], ADD_VALUES));
  PetscCall(MatAssemblyBegin(B, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(B, MAT_FINAL_ASSEMBLY));

  // Duplicate B to C and continue adding nozeros to C with the 2nd half
  PetscCall(MatDuplicate(B, MAT_COPY_VALUES, &C));
  for (k = mycoo.n / 2; k < mycoo.n; k++) PetscCall(MatSetValue(C, mycoo.i[k], mycoo.j[k], vals[k], ADD_VALUES));
  PetscCall(MatAssemblyBegin(C, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(C, MAT_FINAL_ASSEMBLY));

  // Test if A == C
  PetscCall(MatMultEqual(A, C, 10, &equal));
  if (!equal) PetscCall(PetscPrintf(PETSC_COMM_WORLD, "MatDuplicate() on regular matrices failed\n"));

  PetscCall(PetscFree(vals));
  PetscCall(MatDestroy(&A));
  PetscCall(MatDestroy(&B));
  PetscCall(MatDestroy(&C));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

  test:
    nsize: 3
    output_file: output/empty.out

TEST*/
