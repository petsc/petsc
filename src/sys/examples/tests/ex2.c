/*$Id: ex2.c,v 1.15 1999/10/24 14:01:38 bsmith Exp bsmith $*/

/*
      Tests the signal handler.
*/
#include "petsc.h"

#undef __FUNC__
#define __FUNC__ "CreateError"
int CreateError(int n)
{
  int    ierr;
  double *x = 0;
  if (!n) x[0] = 100.; 
  ierr = CreateError(n-1);CHKERRQ(ierr);
  return 0;
}

#undef __FUNC__
#define __FUNC__ "main"
int main(int argc,char **argv)
{
  int ierr;
  PetscInitialize(&argc,&argv,(char *)0,0);
  ierr = PetscFPrintf(PETSC_COMM_WORLD,stdout,"Demonstrates how PETSc can trap error interrupts\n");CHKERRQ(ierr);
  ierr = PetscFPrintf(PETSC_COMM_WORLD,stdout,"The error below is contrived to test the code!\n");CHKERRQ(ierr);
  ierr = PetscSynchronizedFlush(PETSC_COMM_WORLD);CHKERRQ(ierr);
  ierr = CreateError(5);CHKERRQ(ierr);
  PetscFinalize();
  return 0;
}
 
