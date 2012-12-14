
static char help[] = "Demonstrates PetscOptionsGetViewer().\n\n";

#include <petscviewer.h>

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **args)
{
  PetscViewer    viewer;
  PetscErrorCode ierr;

  PetscInitialize(&argc,&args,(char *)0,help);
  ierr = PetscOptionsGetViewer(PETSC_COMM_WORLD,PETSC_NULL,"-myviewer",&viewer,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscViewerView(viewer,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = PetscOptionsRestoreViewer(viewer);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return 0;
}
