#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: ex9.c,v 1.2 1997/05/28 23:19:54 bsmith Exp balay $";
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
