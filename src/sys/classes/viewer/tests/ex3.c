
static char help[] = "Tests dynamic loading of viewer.\n\n";

#include <petscsys.h>
#include <petscviewer.h>

int main(int argc, char **args)
{
  PetscViewer viewer;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &args, (char *)0, help));
  PetscCall(PetscViewerCreate(PETSC_COMM_WORLD, &viewer));
  PetscCall(PetscViewerSetFromOptions(viewer));
  PetscCall(PetscViewerDestroy(&viewer));

  PetscCall(PetscViewerASCIIOpen(PETSC_COMM_WORLD, "stdout", &viewer));
  PetscCall(PetscViewerDestroy(&viewer));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   test:

TEST*/
