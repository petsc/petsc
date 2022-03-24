
static char help[] = "Appends to an ASCII file.\n\n";

/*T
   Concepts: viewers^append
T*/

#include <petscviewer.h>

int main(int argc,char **args)
{
  PetscViewer    viewer;
  PetscInt       i;

  CHKERRQ(PetscInitialize(&argc,&args,(char*)0,help));
  CHKERRQ(PetscViewerCreate(PETSC_COMM_WORLD, &viewer));
  CHKERRQ(PetscViewerSetType(viewer, PETSCVIEWERASCII));
  CHKERRQ(PetscViewerFileSetMode(viewer, FILE_MODE_APPEND));
  CHKERRQ(PetscViewerFileSetName(viewer, "test.txt"));
  for (i = 0; i < 10; ++i) {
    CHKERRQ(PetscViewerASCIIPrintf(viewer, "test line %" PetscInt_FMT "\n", i));
  }
  CHKERRQ(PetscViewerDestroy(&viewer));
  CHKERRQ(PetscFinalize());
  return 0;
}

/*TEST

   test:

TEST*/
