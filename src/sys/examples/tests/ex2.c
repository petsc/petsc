#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: ex2.c,v 1.13 1999/05/04 20:29:49 balay Exp bsmith $";
#endif

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
  ierr = PetscFPrintf(PETSC_COMM_WORLD,stdout,"Demonstrates how PETSc can trap error interrupts\n");CHKERRA(ierr);
  ierr = PetscFPrintf(PETSC_COMM_WORLD,stdout,"The error below is contrived to test the code!\n");CHKERRA(ierr);
  ierr = PetscSynchronizedFlush(PETSC_COMM_WORLD);CHKERRA(ierr);
  ierr = CreateError(5);CHKERRA(ierr);
  PetscFinalize();
  return 0;
}
 
