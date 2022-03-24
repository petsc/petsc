#include "../src/dm/impls/forest/p4est/petsc_p4est_package.h"

static char help[] = "Test interaction with p4est/libsc error and logging routines\n";

int main(int argc, char **argv)
{

  CHKERRQ(PetscInitialize(&argc,&argv,NULL,help));
  CHKERRQ(PetscP4estInitialize());
  PetscStackCallP4est(sc_abort_verbose,(__FILE__,__LINE__,"Abort in main()\n"));
  CHKERRQ(PetscFinalize());
  return 0;
}
