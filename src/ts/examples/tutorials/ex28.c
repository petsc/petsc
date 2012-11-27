
static char help[] ="Loads a previously saved TS.";

/*
   This example it not yet working. It loads a TS saved with TSView()

*/
/*
    Include "petscts.h" to use the PETSc timestepping routines. Note that
    this file automatically includes "petscsys.h" and other lower-level
    PETSc include files.
*/
#include <petscts.h>


#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **argv)
{
  TS             ts;                 /* timestepping context */
  PetscErrorCode ierr;
  PetscViewer    viewer;

  PetscInitialize(&argc,&argv,PETSC_NULL,help);
  ierr = TSCreate(PETSC_COMM_WORLD,&ts);CHKERRQ(ierr);
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,"binaryoutput",FILE_MODE_READ,&viewer);CHKERRQ(ierr);
  ierr = TSLoad(ts,viewer);CHKERRQ(ierr);
  ierr = TSSetUp(ts);CHKERRQ(ierr);
  ierr = TSView(ts,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = TSSolve(ts,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
  ierr = TSDestroy(&ts);CHKERRQ(ierr);
  ierr = PetscFinalize();

  return 0;
}
