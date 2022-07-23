static char help[] = "Test MatCreateRedundantMatrix for rectangular matrix.\n\
                      Contributed by Jose E. Roman, July 2017\n\n";

#include <petscmat.h>
int main(int argc,char **args)
{
  Mat               A,B;
  PetscInt          m=3,n=4,i,nsubcomm;
  PetscMPIInt       size,rank;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc,&args,(char*)0,help));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &size));
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &rank));

  nsubcomm = size;
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-nsubcomm",&nsubcomm,NULL));

  PetscCall(MatCreate(PETSC_COMM_WORLD, &A));
  PetscCall(MatSetSizes(A, m, n, PETSC_DETERMINE, PETSC_DETERMINE));
  PetscCall(MatSetType(A, MATAIJ));
  PetscCall(MatSetFromOptions(A));
  PetscCall(MatSetUp(A));

  if (rank == 0) {
    for (i=0;i<size*PetscMin(m,n);i++) {
      PetscCall(MatSetValue(A, i, i, 1.0, INSERT_VALUES));
    }
  }
  PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatView(A,PETSC_VIEWER_STDOUT_WORLD));

  PetscCall(MatCreateRedundantMatrix(A, nsubcomm, MPI_COMM_NULL, MAT_INITIAL_MATRIX, &B));
  if (nsubcomm==size) { /* B is a sequential matrix */
    if (rank == 0) {
      PetscCall(MatView(B,PETSC_VIEWER_STDOUT_SELF));
    }
  } else {
    MPI_Comm comm;
    PetscCall(PetscObjectGetComm((PetscObject)B,&comm));
    PetscCall(MatView(B,PETSC_VIEWER_STDOUT_(comm)));
  }

  PetscCall(MatDestroy(&A));
  PetscCall(MatDestroy(&B));
  PetscCall(PetscFinalize());
  return 0;
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
