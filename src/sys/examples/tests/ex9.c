extern "C" {
#include "petsc.h"
}
main(int argc, char **argv) {
  int ierr;
  PetscInitialize(&argc,(char***)&argv,PETSC_NULL,PETSC_NULL);
  ierr = PetscSequentialPhaseBegin(MPI_COMM_WORLD , 1); CHKERRA(ierr);
  ierr = PetscSequentialPhaseEnd(MPI_COMM_WORLD , 1); CHKERRA(ierr);
  PetscFinalize();
}
