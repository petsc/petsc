
static char help[] = "Appends to an ASCII file.\n\n";

/*T
   Concepts: viewers^append
T*/

#include <petscviewer.h>

int main(int argc,char **args)
{
  PetscViewer    viewer;
  PetscInt       i;
  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc,&args,(char*)0,help);if (ierr) return ierr;
  CHKERRQ(PetscViewerCreate(PETSC_COMM_WORLD, &viewer));
  CHKERRQ(PetscViewerSetType(viewer, PETSCVIEWERASCII));
  CHKERRQ(PetscViewerFileSetMode(viewer, FILE_MODE_APPEND));
  CHKERRQ(PetscViewerFileSetName(viewer, "test.txt"));
  for (i = 0; i < 10; ++i) {
    CHKERRQ(PetscViewerASCIIPrintf(viewer, "test line %" PetscInt_FMT "\n", i));
  }
  CHKERRQ(PetscViewerDestroy(&viewer));
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

   test:

TEST*/
