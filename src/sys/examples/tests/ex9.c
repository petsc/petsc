#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: ex9.c,v 1.3 1997/07/09 20:52:08 balay Exp balay $";
#endif

/*
     Tests PetscSequentialPhaseBegin() and PetscSequentialPhaseEnd()
*/
#include "petsc.h"

int main(int argc, char **argv) {
  int ierr;

  PetscInitialize(&argc,&argv,PETSC_NULL,PETSC_NULL);
  ierr = PetscSequentialPhaseBegin(PETSC_COMM_WORLD , 1); CHKERRA(ierr);
  ierr = PetscSequentialPhaseEnd(PETSC_COMM_WORLD , 1); CHKERRA(ierr);
  PetscFinalize();
  return 0;
}
