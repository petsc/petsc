
static char help[] = "Tests PCView() before PCSetup() with -pc_type lu.\n\n";

#include <petscmat.h>
#include <petscpc.h>

int main(int argc,char **args)
{
  Mat            A;
  PC             pc;
  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc,&args,(char*)0,help);if (ierr) return ierr;
  ierr = MatCreate(PETSC_COMM_WORLD,&A);CHKERRQ(ierr);
  ierr = MatSetSizes(A,1,1,1,1);CHKERRQ(ierr);
  ierr = MatSetFromOptions(A);CHKERRQ(ierr);
  ierr = MatSetUp(A);CHKERRQ(ierr);
  ierr = MatSetValue(A,0,0,1,INSERT_VALUES);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatView(A,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = PCCreate(PETSC_COMM_WORLD,&pc);CHKERRQ(ierr);
  ierr = PCSetOperators(pc,A,A);CHKERRQ(ierr);
  ierr = PCSetType(pc,PCLU);CHKERRQ(ierr);
  ierr = PCView(pc,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = PCDestroy(&pc);CHKERRQ(ierr);
  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

   test:

TEST*/
