/*$Id: ex9.c,v 1.6 1999/05/04 20:29:49 balay Exp bsmith $*/

/*
     Tests PetscSequentialPhaseBegin() and PetscSequentialPhaseEnd()
*/
#include "petsc.h"

#undef __FUNC__
#define __FUNC__ "main"
int main(int argc, char **argv) {
  int ierr;

  PetscInitialize(&argc,&argv,PETSC_NULL,PETSC_NULL);
  ierr = PetscSequentialPhaseBegin(PETSC_COMM_WORLD , 1);CHKERRA(ierr);
  ierr = PetscSequentialPhaseEnd(PETSC_COMM_WORLD , 1);CHKERRA(ierr);
  PetscFinalize();
  return 0;
}
