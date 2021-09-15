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
  ierr = MPI_Comm_size(PETSC_COMM_WORLD, &size);CHKERRMPI(ierr);
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD, &rank);CHKERRMPI(ierr);

  ierr = MatCreate(PETSC_COMM_WORLD, &A);CHKERRQ(ierr);
  ierr = MatSetSizes(A, 2, 2, PETSC_DETERMINE, PETSC_DETERMINE);CHKERRQ(ierr);
  ierr = MatSetBlockSize(A, 2);CHKERRQ(ierr);
  ierr = MatSetType(A, MATBAIJ);CHKERRQ(ierr);
  ierr = MatSetUp(A);CHKERRQ(ierr);

  ierr = MatCreateVecs(A, &diag, NULL);CHKERRQ(ierr);
  ierr = VecSet(diag, 1.0);CHKERRQ(ierr);
  ierr = MatDiagonalSet(A, diag, INSERT_VALUES);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatView(A,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  ierr = MatCreateRedundantMatrix(A, size, MPI_COMM_NULL, MAT_INITIAL_MATRIX, &B);CHKERRQ(ierr);
  if (rank == 0) {
    ierr = MatView(B,PETSC_VIEWER_STDOUT_SELF);CHKERRQ(ierr);
  }

  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = MatDestroy(&B);CHKERRQ(ierr);
  ierr = VecDestroy(&diag);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

   test:

   test:
      suffix: 2
      nsize: 3

TEST*/
