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
  CHKERRMPI(MPI_Comm_size(PETSC_COMM_WORLD, &size));
  CHKERRMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &rank));

  nsubcomm = size;
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-nsubcomm",&nsubcomm,NULL));

  CHKERRQ(MatCreate(PETSC_COMM_WORLD, &A));
  CHKERRQ(MatSetSizes(A, m, n, PETSC_DETERMINE, PETSC_DETERMINE));
  CHKERRQ(MatSetType(A, MATAIJ));
  CHKERRQ(MatSetFromOptions(A));
  CHKERRQ(MatSetUp(A));

  if (rank == 0) {
    for (i=0;i<size*PetscMin(m,n);i++) {
      CHKERRQ(MatSetValue(A, i, i, 1.0, INSERT_VALUES));
    }
  }
  CHKERRQ(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatView(A,PETSC_VIEWER_STDOUT_WORLD));

  CHKERRQ(MatCreateRedundantMatrix(A, nsubcomm, MPI_COMM_NULL, MAT_INITIAL_MATRIX, &B));
  if (nsubcomm==size) { /* B is a sequential matrix */
    if (rank == 0) {
      CHKERRQ(MatView(B,PETSC_VIEWER_STDOUT_SELF));
    }
  } else {
    MPI_Comm comm;
    CHKERRQ(PetscObjectGetComm((PetscObject)B,&comm));
    CHKERRQ(MatView(B,PETSC_VIEWER_STDOUT_(comm)));
  }

  CHKERRQ(MatDestroy(&A));
  CHKERRQ(MatDestroy(&B));
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
