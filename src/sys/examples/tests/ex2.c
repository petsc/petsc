#ifndef lint
static char vcid[] = "$Id: ex2.c,v 1.8 1996/03/19 21:24:49 bsmith Exp curfman $";
#endif

/*
      Tests the signal handler.
*/
#include "petsc.h"
#include <stdio.h>

int CreateError(int n)
{
  int    ierr;
  double *x = 0;
  if (!n) x[0] = 100.; 
  ierr = CreateError(n-1); CHKERRQ(ierr);
  return 0;
}

int main(int argc,char **argv)
{
  int ierr;
  PetscInitialize(&argc,&argv,(char *)0,0);
  fprintf(stdout,"Demonstrates how PETSc can trap error interrupts\n");
  fprintf(stdout,"The error below is contrived to test the code!\n");
  fflush(stdout);
  ierr = CreateError(5); CHKERRA(ierr);
  PetscFinalize();
  return 0;
}
 
