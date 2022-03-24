static char help[] = "Testing MatCreateMPIAIJSumSeqAIJ().\n\n";

#include <petscmat.h>

int main(int argc,char **argv)
{
  Mat            A,B;
  MatScalar      a[1],alpha;
  PetscMPIInt    size,rank;
  PetscInt       m,n,i,col, prid;

  CHKERRQ(PetscInitialize(&argc,&argv,(char*)0,help));
  CHKERRMPI(MPI_Comm_size(PETSC_COMM_WORLD,&size));
  CHKERRMPI(MPI_Comm_rank(PETSC_COMM_WORLD,&rank));
  prid = size;
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-prid",&prid,NULL));

  m    = n = 10*size;
  CHKERRQ(MatCreate(PETSC_COMM_SELF,&A));
  CHKERRQ(MatSetSizes(A,PETSC_DETERMINE,PETSC_DETERMINE,m,n));
  CHKERRQ(MatSetType(A,MATSEQAIJ));
  CHKERRQ(MatSetUp(A));

  a[0] = rank+1;
  for (i=0; i<m-rank; i++) {
    col  = i+rank;
    CHKERRQ(MatSetValues(A,1,&i,1,&col,a,INSERT_VALUES));
  }
  CHKERRQ(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));

  if (rank == prid) {
    CHKERRQ(PetscPrintf(PETSC_COMM_SELF,"[%d] A: \n",rank));
    CHKERRQ(MatView(A,PETSC_VIEWER_STDOUT_SELF));
  }

  /* Test MatCreateMPIAIJSumSeqAIJ */
  CHKERRQ(MatCreateMPIAIJSumSeqAIJ(PETSC_COMM_WORLD,A,PETSC_DECIDE,PETSC_DECIDE,MAT_INITIAL_MATRIX,&B));

  /* Test MAT_REUSE_MATRIX */
  alpha = 0.1;
  for (i=0; i<3; i++) {
    CHKERRQ(MatScale(A,alpha));
    CHKERRQ(MatCreateMPIAIJSumSeqAIJ(PETSC_COMM_WORLD,A,PETSC_DECIDE,PETSC_DECIDE,MAT_REUSE_MATRIX,&B));
  }
  CHKERRQ(MatView(B, PETSC_VIEWER_STDOUT_WORLD));
  CHKERRQ(MatDestroy(&B));
  CHKERRQ(MatDestroy(&A));
  CHKERRQ(PetscFinalize());
  return 0;
}

/*TEST

   test:
      nsize: 3
      filter: grep -v "MPI processes"

   test:
      suffix: 2
      filter: grep -v "MPI processes"

TEST*/
