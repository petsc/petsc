/*$Id: ex9.c,v 1.7 1999/10/24 14:01:38 bsmith Exp bsmith $*/

/*
     Tests PetscSequentialPhaseBegin() and PetscSequentialPhaseEnd()

*/
#include "petsc.h"

#undef __FUNC__
#define __FUNC__ "main"
int main(int argc,char **args){
  int ierr;

  ierr = PetscInitialize(&argc,&args,PETSC_NULL,PETSC_NULL);
  ierr = PetscSequentialPhaseBegin(PETSC_COMM_WORLD , 1);CHKERRA(ierr);
  ierr = PetscSequentialPhaseEnd(PETSC_COMM_WORLD , 1);CHKERRA(ierr);
  ierr = PetscFinalize();
  return 0;
}
