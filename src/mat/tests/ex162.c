static char help[] = "Tests MatShift for SeqAIJ matrices with some missing diagonal entries\n\n";

#include <petscmat.h>

int main(int argc,char **argv)
{
  Mat                  A;
  PetscInt             coli[4],row;
  PetscScalar          vali[4];
  PetscMPIInt          size;

  CHKERRQ(PetscInitialize(&argc,&argv,(char*)0,help));
  CHKERRMPI(MPI_Comm_size(PETSC_COMM_WORLD,&size));
  PetscCheckFalse(size != 1,PETSC_COMM_WORLD,PETSC_ERR_SUP,"This is a uniprocessor example only!");

  CHKERRQ(MatCreate(PETSC_COMM_SELF,&A));
  CHKERRQ(MatSetSizes(A,4,4,4,4));
  CHKERRQ(MatSetType(A,MATSEQAIJ));
  CHKERRQ(MatSeqAIJSetPreallocation(A,4,NULL));

  row = 0; coli[0] = 1; coli[1] = 3; vali[0] = 1.0; vali[1] = 2.0;
  CHKERRQ(MatSetValues(A,1,&row,2,coli,vali,ADD_VALUES));

  row = 1; coli[0] = 0; coli[1] = 1; coli[2] = 2; coli[3] = 3; vali[0] = 3.0; vali[1] = 4.0; vali[2] = 5.0; vali[3] = 6.0;
  CHKERRQ(MatSetValues(A,1,&row,4,coli,vali,ADD_VALUES));

  row = 2; coli[0] = 0; coli[1] = 3; vali[0] = 7.0; vali[1] = 8.0;
  CHKERRQ(MatSetValues(A,1,&row,2,coli,vali,ADD_VALUES));

  CHKERRQ(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatView(A,PETSC_VIEWER_STDOUT_WORLD));

  CHKERRQ(MatShift(A,0.0));
  CHKERRQ(MatView(A,PETSC_VIEWER_STDOUT_WORLD));

  CHKERRQ(MatDestroy(&A));
  CHKERRQ(PetscFinalize());
  return 0;
}

/*TEST

   test:

TEST*/
