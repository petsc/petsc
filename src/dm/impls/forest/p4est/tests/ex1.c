#include "../src/dm/impls/forest/p4est/petsc_p4est_package.h"

static char help[] = "Test interaction with p4est/libsc error and logging routines\n";

int main(int argc, char **argv)
{
  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc,&argv,NULL,help);if (ierr) return ierr;
  ierr = PetscP4estInitialize();CHKERRQ(ierr);
  PetscStackCallP4est(sc_abort_verbose,(__FILE__,__LINE__,"Abort in main()\n"));
  ierr = PetscFinalize();
  return ierr;
}
