
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

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc,&argv,NULL,help));
  PetscCall(PetscDLLibraryAppend(PETSC_COMM_WORLD,&PetscDLLibrariesLoaded,"advection-diffusion-reaction/ex1"));
  PetscCall(TSCreate(PETSC_COMM_WORLD,&ts));
  PetscCall(PetscViewerBinaryOpen(PETSC_COMM_WORLD,"advection-diffusion-reaction/binaryoutput",FILE_MODE_READ,&viewer));
  PetscCall(TSLoad(ts,viewer));
  PetscCall(PetscViewerDestroy(&viewer));
  /* PetscCall(PetscFPTView(0)); */
  PetscCall(TSSetFromOptions(ts));
  PetscCall(TSSetUp(ts));
  PetscCall(TSView(ts,PETSC_VIEWER_STDOUT_WORLD));
  PetscCall(TSSolve(ts,NULL));
  PetscCall(TSDestroy(&ts));
  PetscCall(PetscFinalize());
  return 0;
}
