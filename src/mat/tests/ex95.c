static char help[] = "Testing MatCreateMPIAIJSumSeqAIJ().\n\n";

#include <petscmat.h>

int main(int argc,char **argv)
{
  Mat            A,B;
  MatScalar      a[1],alpha;
  PetscMPIInt    size,rank;
  PetscInt       m,n,i,col, prid;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc,&argv,(char*)0,help));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD,&size));
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD,&rank));
  prid = size;
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-prid",&prid,NULL));

  m    = n = 10*size;
  PetscCall(MatCreate(PETSC_COMM_SELF,&A));
  PetscCall(MatSetSizes(A,PETSC_DETERMINE,PETSC_DETERMINE,m,n));
  PetscCall(MatSetType(A,MATSEQAIJ));
  PetscCall(MatSetUp(A));

  a[0] = rank+1;
  for (i=0; i<m-rank; i++) {
    col  = i+rank;
    PetscCall(MatSetValues(A,1,&i,1,&col,a,INSERT_VALUES));
  }
  PetscCall(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));

  if (rank == prid) {
    PetscCall(PetscPrintf(PETSC_COMM_SELF,"[%d] A: \n",rank));
    PetscCall(MatView(A,PETSC_VIEWER_STDOUT_SELF));
  }

  /* Test MatCreateMPIAIJSumSeqAIJ */
  PetscCall(MatCreateMPIAIJSumSeqAIJ(PETSC_COMM_WORLD,A,PETSC_DECIDE,PETSC_DECIDE,MAT_INITIAL_MATRIX,&B));

  /* Test MAT_REUSE_MATRIX */
  alpha = 0.1;
  for (i=0; i<3; i++) {
    PetscCall(MatScale(A,alpha));
    PetscCall(MatCreateMPIAIJSumSeqAIJ(PETSC_COMM_WORLD,A,PETSC_DECIDE,PETSC_DECIDE,MAT_REUSE_MATRIX,&B));
  }
  PetscCall(MatView(B, PETSC_VIEWER_STDOUT_WORLD));
  PetscCall(MatDestroy(&B));
  PetscCall(MatDestroy(&A));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   test:
      nsize: 3
      filter: grep -v " MPI process"

   test:
      suffix: 2
      filter: grep -v " MPI process"

TEST*/
