/*$Id: ex9.c,v 1.9 2000/01/11 20:59:47 bsmith Exp bsmith $*/

/*
     Tests PetscSequentialPhaseBegin() and PetscSequentialPhaseEnd()

*/
#include "petsc.h"

#undef __FUNC__
#define __FUNC__ "main"
int main(int argc,char **args){
  int ierr;

  ierr = PetscInitialize(&argc,&args,PETSC_NULL,PETSC_NULL);
  ierr = PetscSequentialPhaseBegin(PETSC_COMM_WORLD,1);CHKERRQ(ierr);
  ierr = PetscSequentialPhaseEnd(PETSC_COMM_WORLD,1);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return 0;
}
