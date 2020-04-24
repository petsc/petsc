
static char help[] ="Loads a previously saved TS.";

/*
   It loads a TS saved with TSView()

*/
/*
    Include "petscts.h" to use the PETSc timestepping routines. Note that
    this file automatically includes "petscsys.h" and other lower-level
    PETSc include files.
*/
#include <petscts.h>


int main(int argc,char **argv)
{
  TS             ts;                 /* timestepping context */
  PetscErrorCode ierr;
  PetscViewer    viewer;

  ierr = PetscInitialize(&argc,&argv,NULL,help);if (ierr) return ierr;
  ierr = PetscDLLibraryAppend(PETSC_COMM_WORLD,&PetscDLLibrariesLoaded,"advection-diffusion-reaction/ex1");CHKERRQ(ierr);
  ierr = TSCreate(PETSC_COMM_WORLD,&ts);CHKERRQ(ierr);
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,"advection-diffusion-reaction/binaryoutput",FILE_MODE_READ,&viewer);CHKERRQ(ierr);
  ierr = TSLoad(ts,viewer);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
  /* ierr = PetscFPTView(0);CHKERRQ(ierr); */
  ierr = TSSetFromOptions(ts);CHKERRQ(ierr);
  ierr = TSSetUp(ts);CHKERRQ(ierr);
  ierr = TSView(ts,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = TSSolve(ts,NULL);CHKERRQ(ierr);
  ierr = TSDestroy(&ts);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}
