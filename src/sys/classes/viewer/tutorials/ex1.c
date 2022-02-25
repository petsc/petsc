
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
  ierr = PetscViewerCreate(PETSC_COMM_WORLD, &viewer);CHKERRQ(ierr);
  ierr = PetscViewerSetType(viewer, PETSCVIEWERASCII);CHKERRQ(ierr);
  ierr = PetscViewerFileSetMode(viewer, FILE_MODE_APPEND);CHKERRQ(ierr);
  ierr = PetscViewerFileSetName(viewer, "test.txt");CHKERRQ(ierr);
  for (i = 0; i < 10; ++i) {
    ierr = PetscViewerASCIIPrintf(viewer, "test line %" PetscInt_FMT "\n", i);CHKERRQ(ierr);
  }
  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

   test:

TEST*/
