
#ifndef lint
static char vcid[] = "$Id: ex8.c,v 1.4 1997/04/10 00:01:45 bsmith Exp $";
#endif

/*
     Tests PetscSequentialPhaseBegin() and PetscSequentialPhaseEnd()
*/
#include "petsc.h"

int main(int argc, char **argv) {
  int ierr;

  PetscInitialize(&argc,&argv,PETSC_NULL,PETSC_NULL);
  ierr = PetscSequentialPhaseBegin(MPI_COMM_WORLD , 1); CHKERRA(ierr);
  ierr = PetscSequentialPhaseEnd(MPI_COMM_WORLD , 1); CHKERRA(ierr);
  PetscFinalize();
  return 0;
}
