
static char help[] = "Tests dynamic loading of viewer.\n\n";

#include <petscsys.h>
#include <petscviewer.h>

int main(int argc,char **args)
{
  PetscViewer    viewer;

  CHKERRQ(PetscInitialize(&argc,&args,(char*)0,help));
  CHKERRQ(PetscViewerCreate(PETSC_COMM_WORLD,&viewer));
  CHKERRQ(PetscViewerSetFromOptions(viewer));
  CHKERRQ(PetscViewerDestroy(&viewer));

  CHKERRQ(PetscViewerASCIIOpen(PETSC_COMM_WORLD,"stdout",&viewer));
  CHKERRQ(PetscViewerDestroy(&viewer));
  CHKERRQ(PetscFinalize());
  return 0;
}

/*TEST

   test:

TEST*/
