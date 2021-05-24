
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
  ierr = MatCreate(PETSC_COMM_SELF,&C);CHKERRQ(ierr);
  ierr = MatSetSizes(C,PETSC_DECIDE,PETSC_DECIDE,m,n);CHKERRQ(ierr);
  ierr = MatSetFromOptions(C);CHKERRQ(ierr);
  ierr = MatSetUp(C);CHKERRQ(ierr);
  for (i=0; i<m; i++) {
    for (j=0; j<n; j++) {
      v = 0.0 - 1.0*PETSC_i;
      if (i>j && i-j<2)   {ierr = MatSetValues(C,1,&i,1,&j,&v,INSERT_VALUES);CHKERRQ(ierr);}
    }
  }
  ierr = MatAssemblyBegin(C,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(C,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  ierr = MatCreateHermitianTranspose(C, &C_htransposed);CHKERRQ(ierr);

  ierr = MatView(C,PETSC_VIEWER_STDOUT_SELF);CHKERRQ(ierr);
  ierr = MatDuplicate(C_htransposed,MAT_COPY_VALUES,&Cht);CHKERRQ(ierr);
  ierr = MatView(Cht,PETSC_VIEWER_STDOUT_SELF);CHKERRQ(ierr);
  ierr = MatDuplicate(C_htransposed,MAT_DO_NOT_COPY_VALUES,&C_empty);CHKERRQ(ierr);
  ierr = MatView(C_empty,PETSC_VIEWER_STDOUT_SELF);CHKERRQ(ierr);

  ierr = MatDestroy(&C);CHKERRQ(ierr);
  ierr = MatDestroy(&C_htransposed);CHKERRQ(ierr);
  ierr = MatDestroy(&Cht);CHKERRQ(ierr);
  ierr = MatDestroy(&C_empty);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

   build:
     requires: complex

   test:
     output_file: output/ex175.out

TEST*/
