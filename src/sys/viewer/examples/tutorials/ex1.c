
/* Program usage:  mpiexec -n <procs> ex2 [-help] [all PETSc options] */ 

static char help[] = "Appends to an ASCII file.\n\n";

/*T
   Concepts: Viewer, append
T*/

#include <petscviewer.h>

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **args)
{
  PetscViewer    viewer;
  PetscInt       i;
  PetscErrorCode ierr;

  PetscInitialize(&argc,&args,(char *)0,help);
  ierr = PetscViewerCreate(PETSC_COMM_WORLD, &viewer);CHKERRQ(ierr);
  ierr = PetscViewerSetType(viewer, PETSCVIEWERASCII);CHKERRQ(ierr);
  ierr = PetscViewerFileSetMode(viewer, FILE_MODE_APPEND);CHKERRQ(ierr);
  ierr = PetscViewerFileSetName(viewer, "test.txt");CHKERRQ(ierr);
  for(i = 0; i < 10; ++i) {
    ierr = PetscViewerASCIIPrintf(viewer, "test line %d\n", i);CHKERRQ(ierr);
  }
  ierr = PetscFinalize();
  return 0;
}
