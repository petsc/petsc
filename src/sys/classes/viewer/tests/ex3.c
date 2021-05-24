
static char help[] = "Tests dynamic loading of viewer.\n\n";

#include <petscsys.h>
#include <petscviewer.h>

int main(int argc,char **args)
{
  PetscErrorCode ierr;
  PetscViewer    viewer;

  ierr = PetscInitialize(&argc,&args,(char*)0,help);if (ierr) return ierr;
  ierr = PetscViewerCreate(PETSC_COMM_WORLD,&viewer);CHKERRQ(ierr);
  ierr = PetscViewerSetFromOptions(viewer);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);

  ierr = PetscViewerASCIIOpen(PETSC_COMM_WORLD,"stdout",&viewer);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

   test:

TEST*/
