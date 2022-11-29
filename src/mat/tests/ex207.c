static char help[] = "Test MatCreateRedundantMatrix for a BAIJ matrix.\n\
                      Contributed by Lawrence Mitchell, Feb. 21, 2017\n\n";

#include <petscmat.h>
int main(int argc, char **args)
{
  Mat         A, B;
  Vec         diag;
  PetscMPIInt size, rank;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &args, (char *)0, help));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &size));
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &rank));

  PetscCall(MatCreate(PETSC_COMM_WORLD, &A));
  PetscCall(MatSetSizes(A, 2, 2, PETSC_DETERMINE, PETSC_DETERMINE));
  PetscCall(MatSetBlockSize(A, 2));
  PetscCall(MatSetType(A, MATBAIJ));
  PetscCall(MatSetUp(A));

  PetscCall(MatCreateVecs(A, &diag, NULL));
  PetscCall(VecSet(diag, 1.0));
  PetscCall(MatDiagonalSet(A, diag, INSERT_VALUES));
  PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatView(A, PETSC_VIEWER_STDOUT_WORLD));

  PetscCall(MatCreateRedundantMatrix(A, size, MPI_COMM_NULL, MAT_INITIAL_MATRIX, &B));
  if (rank == 0) PetscCall(MatView(B, PETSC_VIEWER_STDOUT_SELF));

  PetscCall(MatDestroy(&A));
  PetscCall(MatDestroy(&B));
  PetscCall(VecDestroy(&diag));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   test:

   test:
      suffix: 2
      nsize: 3

TEST*/
