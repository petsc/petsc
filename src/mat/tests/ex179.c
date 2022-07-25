
static char help[] = "Tests MatTranspose() with MAT_REUSE_MATRIX and different nonzero pattern\n\n";

#include <petscmat.h>

int main(int argc,char **argv)
{
  Mat         A,B;
  PetscMPIInt size;

  PetscCall(PetscInitialize(&argc,&argv,(char*) 0,help));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD,&size));
  PetscCheck(size == 1,PETSC_COMM_WORLD,PETSC_ERR_WRONG_MPI_SIZE,"This is a uniprocessor example only!");
  PetscCall(MatCreateSeqAIJ(PETSC_COMM_SELF,2,2,2,NULL,&A));
  PetscCall(MatSetValue(A,0,0,1.0,INSERT_VALUES));
  PetscCall(MatSetValue(A,0,1,2.0,INSERT_VALUES));
  PetscCall(MatSetValue(A,1,1,4.0,INSERT_VALUES));
  PetscCall(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));
  PetscCall(MatView(A,PETSC_VIEWER_STDOUT_SELF));
  PetscCall(MatTranspose(A,MAT_INITIAL_MATRIX,&B));
  PetscCall(MatView(B,PETSC_VIEWER_STDOUT_SELF));
  PetscCall(MatSetOption(A, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE));
  PetscCall(MatSetValue(A,1,0,3.0,INSERT_VALUES));
  PetscCall(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));
  PetscCall(MatView(A,PETSC_VIEWER_STDOUT_SELF));
  PetscCall(MatTranspose(A,MAT_REUSE_MATRIX,&B));
  PetscCall(MatView(B,PETSC_VIEWER_STDOUT_SELF));
  PetscCall(MatDestroy(&A));
  PetscCall(MatDestroy(&B));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   test:

TEST*/
