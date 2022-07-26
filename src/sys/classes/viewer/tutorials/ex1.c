
static char help[] = "Appends to an ASCII file.\n\n";

#include <petscviewer.h>

int main(int argc,char **args)
{
  PetscViewer    viewer;
  PetscInt       i;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc,&args,(char*)0,help));
  PetscCall(PetscViewerCreate(PETSC_COMM_WORLD, &viewer));
  PetscCall(PetscViewerSetType(viewer, PETSCVIEWERASCII));
  PetscCall(PetscViewerFileSetMode(viewer, FILE_MODE_APPEND));
  PetscCall(PetscViewerFileSetName(viewer, "test.txt"));
  for (i = 0; i < 10; ++i) {
    PetscCall(PetscViewerASCIIPrintf(viewer, "test line %" PetscInt_FMT "\n", i));
  }
  PetscCall(PetscViewerDestroy(&viewer));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   test:

TEST*/
