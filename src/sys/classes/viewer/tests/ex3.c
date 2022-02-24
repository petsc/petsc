
static char help[] = "Tests dynamic loading of viewer.\n\n";

#include <petscsys.h>
#include <petscviewer.h>

int main(int argc,char **args)
{
  PetscErrorCode ierr;
  PetscViewer    viewer;

  ierr = PetscInitialize(&argc,&args,(char*)0,help);if (ierr) return ierr;
  CHKERRQ(PetscViewerCreate(PETSC_COMM_WORLD,&viewer));
  CHKERRQ(PetscViewerSetFromOptions(viewer));
  CHKERRQ(PetscViewerDestroy(&viewer));

  CHKERRQ(PetscViewerASCIIOpen(PETSC_COMM_WORLD,"stdout",&viewer));
  CHKERRQ(PetscViewerDestroy(&viewer));
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

   test:

TEST*/
