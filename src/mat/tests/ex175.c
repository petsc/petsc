
static char help[] = "Tests MatCreateHermitianTranspose().\n\n";

#include <petscmat.h>

int main(int argc,char **args)
{
  Mat            C,C_htransposed,Cht,C_empty;
  PetscInt       i,j,m = 10,n = 10;
  PetscErrorCode ierr;
  PetscScalar    v;

  ierr = PetscInitialize(&argc,&args,(char*)0,help);if (ierr) return ierr;
  /* Create a complex non-hermitian matrix */
  CHKERRQ(MatCreate(PETSC_COMM_SELF,&C));
  CHKERRQ(MatSetSizes(C,PETSC_DECIDE,PETSC_DECIDE,m,n));
  CHKERRQ(MatSetFromOptions(C));
  CHKERRQ(MatSetUp(C));
  for (i=0; i<m; i++) {
    for (j=0; j<n; j++) {
      v = 0.0 - 1.0*PETSC_i;
      if (i>j && i-j<2)   CHKERRQ(MatSetValues(C,1,&i,1,&j,&v,INSERT_VALUES));
    }
  }
  CHKERRQ(MatAssemblyBegin(C,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(C,MAT_FINAL_ASSEMBLY));

  CHKERRQ(MatCreateHermitianTranspose(C, &C_htransposed));

  CHKERRQ(MatView(C,PETSC_VIEWER_STDOUT_SELF));
  CHKERRQ(MatDuplicate(C_htransposed,MAT_COPY_VALUES,&Cht));
  CHKERRQ(MatView(Cht,PETSC_VIEWER_STDOUT_SELF));
  CHKERRQ(MatDuplicate(C_htransposed,MAT_DO_NOT_COPY_VALUES,&C_empty));
  CHKERRQ(MatView(C_empty,PETSC_VIEWER_STDOUT_SELF));

  CHKERRQ(MatDestroy(&C));
  CHKERRQ(MatDestroy(&C_htransposed));
  CHKERRQ(MatDestroy(&Cht));
  CHKERRQ(MatDestroy(&C_empty));
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

   build:
     requires: complex

   test:
     output_file: output/ex175.out

TEST*/
