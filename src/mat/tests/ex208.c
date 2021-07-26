static char help[] = "Test MatCreateRedundantMatrix for rectangular matrix.\n\
                      Contributed by Jose E. Roman, July 2017\n\n";

#include <petscmat.h>
int main(int argc,char **args)
{
  Mat               A,B;
  PetscErrorCode    ierr;
  PetscInt          m=3,n=4,i,nsubcomm;
  PetscMPIInt       size,rank;

  ierr = PetscInitialize(&argc,&args,(char*)0,help);if (ierr) return ierr;
  ierr = MPI_Comm_size(PETSC_COMM_WORLD, &size);CHKERRMPI(ierr);
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD, &rank);CHKERRMPI(ierr);

  nsubcomm = size;
  ierr = PetscOptionsGetInt(NULL,NULL,"-nsubcomm",&nsubcomm,NULL);CHKERRQ(ierr);

  ierr = MatCreate(PETSC_COMM_WORLD, &A);CHKERRQ(ierr);
  ierr = MatSetSizes(A, m, n, PETSC_DETERMINE, PETSC_DETERMINE);CHKERRQ(ierr);
  ierr = MatSetType(A, MATAIJ);CHKERRQ(ierr);
  ierr = MatSetFromOptions(A);CHKERRQ(ierr);
  ierr = MatSetUp(A);CHKERRQ(ierr);

  if (rank == 0) {
    for (i=0;i<size*PetscMin(m,n);i++) {
      ierr = MatSetValue(A, i, i, 1.0, INSERT_VALUES);CHKERRQ(ierr);
    }
  }
  ierr = MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatView(A,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  ierr = MatCreateRedundantMatrix(A, nsubcomm, MPI_COMM_NULL, MAT_INITIAL_MATRIX, &B);CHKERRQ(ierr);
  if (nsubcomm==size) { /* B is a sequential matrix */
    if (rank == 0) {
      ierr = MatView(B,PETSC_VIEWER_STDOUT_SELF);CHKERRQ(ierr);
    }
  } else {
    MPI_Comm comm;
    ierr = PetscObjectGetComm((PetscObject)B,&comm);CHKERRQ(ierr);
    ierr = MatView(B,PETSC_VIEWER_STDOUT_(comm));CHKERRQ(ierr);
  }

  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = MatDestroy(&B);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

   test:

   test:
      suffix: 2
      nsize: 3

   test:
      suffix: baij
      args: -mat_type baij

   test:
      suffix: baij_2
      nsize: 3
      args: -mat_type baij

   test:
      suffix: dense
      args: -mat_type dense

   test:
      suffix: dense_2
      nsize: 3
      args: -mat_type dense

TEST*/
