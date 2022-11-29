static char help[] = "Tests MatShift for SeqAIJ matrices with some missing diagonal entries\n\n";

#include <petscmat.h>

int main(int argc, char **argv)
{
  Mat         A;
  PetscInt    coli[4], row;
  PetscScalar vali[4];
  PetscMPIInt size;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, (char *)0, help));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &size));
  PetscCheck(size == 1, PETSC_COMM_WORLD, PETSC_ERR_WRONG_MPI_SIZE, "This is a uniprocessor example only!");

  PetscCall(MatCreate(PETSC_COMM_SELF, &A));
  PetscCall(MatSetSizes(A, 4, 4, 4, 4));
  PetscCall(MatSetType(A, MATSEQAIJ));
  PetscCall(MatSeqAIJSetPreallocation(A, 4, NULL));

  row     = 0;
  coli[0] = 1;
  coli[1] = 3;
  vali[0] = 1.0;
  vali[1] = 2.0;
  PetscCall(MatSetValues(A, 1, &row, 2, coli, vali, ADD_VALUES));

  row     = 1;
  coli[0] = 0;
  coli[1] = 1;
  coli[2] = 2;
  coli[3] = 3;
  vali[0] = 3.0;
  vali[1] = 4.0;
  vali[2] = 5.0;
  vali[3] = 6.0;
  PetscCall(MatSetValues(A, 1, &row, 4, coli, vali, ADD_VALUES));

  row     = 2;
  coli[0] = 0;
  coli[1] = 3;
  vali[0] = 7.0;
  vali[1] = 8.0;
  PetscCall(MatSetValues(A, 1, &row, 2, coli, vali, ADD_VALUES));

  PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatView(A, PETSC_VIEWER_STDOUT_WORLD));

  PetscCall(MatShift(A, 0.0));
  PetscCall(MatView(A, PETSC_VIEWER_STDOUT_WORLD));

  PetscCall(MatDestroy(&A));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   test:

TEST*/
