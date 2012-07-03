
static char help[] = "Loads matrix dumped by ex43.\n\n";

#include <petscmat.h>

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **args)
{
  Mat            C;
  PetscViewer    viewer;
  PetscErrorCode ierr;

  PetscInitialize(&argc,&args,0,help);
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,"matrix.dat",FILE_MODE_READ,&viewer);CHKERRQ(ierr);
  ierr = MatCreate(PETSC_COMM_WORLD,&C);CHKERRQ(ierr);
  ierr = MatSetType(C,MATMPIDENSE);CHKERRQ(ierr);
  ierr = MatLoad(C,viewer);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
  ierr = MatView(C,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = MatDestroy(&C);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return 0;
}


