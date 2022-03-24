
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
  PetscViewer    viewer;

  CHKERRQ(PetscInitialize(&argc,&argv,NULL,help));
  CHKERRQ(PetscDLLibraryAppend(PETSC_COMM_WORLD,&PetscDLLibrariesLoaded,"advection-diffusion-reaction/ex1"));
  CHKERRQ(TSCreate(PETSC_COMM_WORLD,&ts));
  CHKERRQ(PetscViewerBinaryOpen(PETSC_COMM_WORLD,"advection-diffusion-reaction/binaryoutput",FILE_MODE_READ,&viewer));
  CHKERRQ(TSLoad(ts,viewer));
  CHKERRQ(PetscViewerDestroy(&viewer));
  /* CHKERRQ(PetscFPTView(0)); */
  CHKERRQ(TSSetFromOptions(ts));
  CHKERRQ(TSSetUp(ts));
  CHKERRQ(TSView(ts,PETSC_VIEWER_STDOUT_WORLD));
  CHKERRQ(TSSolve(ts,NULL));
  CHKERRQ(TSDestroy(&ts));
  CHKERRQ(PetscFinalize());
  return 0;
}
