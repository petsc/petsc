static char help[] = "Test MatCreateRedundantMatrix for a BAIJ matrix.\n\
                      Contributed by Lawrence Mitchell, Feb. 21, 2017\n\n";

#include <petscmat.h>
int main(int argc,char **args)
{
  Mat               A,B;
  Vec               diag;
  PetscErrorCode    ierr;
  PetscMPIInt       size,rank;

  ierr = PetscInitialize(&argc,&args,(char*)0,help);if (ierr) return ierr;
  CHKERRMPI(MPI_Comm_size(PETSC_COMM_WORLD, &size));
  CHKERRMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &rank));

  CHKERRQ(MatCreate(PETSC_COMM_WORLD, &A));
  CHKERRQ(MatSetSizes(A, 2, 2, PETSC_DETERMINE, PETSC_DETERMINE));
  CHKERRQ(MatSetBlockSize(A, 2));
  CHKERRQ(MatSetType(A, MATBAIJ));
  CHKERRQ(MatSetUp(A));

  CHKERRQ(MatCreateVecs(A, &diag, NULL));
  CHKERRQ(VecSet(diag, 1.0));
  CHKERRQ(MatDiagonalSet(A, diag, INSERT_VALUES));
  CHKERRQ(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatView(A,PETSC_VIEWER_STDOUT_WORLD));

  CHKERRQ(MatCreateRedundantMatrix(A, size, MPI_COMM_NULL, MAT_INITIAL_MATRIX, &B));
  if (rank == 0) {
    CHKERRQ(MatView(B,PETSC_VIEWER_STDOUT_SELF));
  }

  CHKERRQ(MatDestroy(&A));
  CHKERRQ(MatDestroy(&B));
  CHKERRQ(VecDestroy(&diag));
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

   test:

   test:
      suffix: 2
      nsize: 3

TEST*/
