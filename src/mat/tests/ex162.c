static char help[] = "Tests MatShift for SeqAIJ matrices with some missing diagonal entries\n\n";

#include <petscmat.h>

int main(int argc,char **argv)
{
  Mat                  A;
  PetscInt             coli[4],row;
  PetscScalar          vali[4];
  PetscErrorCode       ierr;
  PetscMPIInt          size;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRMPI(ierr);
  PetscAssertFalse(size != 1,PETSC_COMM_WORLD,PETSC_ERR_SUP,"This is a uniprocessor example only!");

  ierr = MatCreate(PETSC_COMM_SELF,&A);CHKERRQ(ierr);
  ierr = MatSetSizes(A,4,4,4,4);CHKERRQ(ierr);
  ierr = MatSetType(A,MATSEQAIJ);CHKERRQ(ierr);
  ierr = MatSeqAIJSetPreallocation(A,4,NULL);CHKERRQ(ierr);

  row = 0; coli[0] = 1; coli[1] = 3; vali[0] = 1.0; vali[1] = 2.0;
  ierr = MatSetValues(A,1,&row,2,coli,vali,ADD_VALUES);CHKERRQ(ierr);

  row = 1; coli[0] = 0; coli[1] = 1; coli[2] = 2; coli[3] = 3; vali[0] = 3.0; vali[1] = 4.0; vali[2] = 5.0; vali[3] = 6.0;
  ierr = MatSetValues(A,1,&row,4,coli,vali,ADD_VALUES);CHKERRQ(ierr);

  row = 2; coli[0] = 0; coli[1] = 3; vali[0] = 7.0; vali[1] = 8.0;
  ierr = MatSetValues(A,1,&row,2,coli,vali,ADD_VALUES);CHKERRQ(ierr);

  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatView(A,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  ierr = MatShift(A,0.0);CHKERRQ(ierr);
  ierr = MatView(A,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

   test:

TEST*/
