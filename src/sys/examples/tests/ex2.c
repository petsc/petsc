#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: ex2.c,v 1.10 1997/07/09 20:52:08 balay Exp bsmith $";
#endif

/*
      Tests the signal handler.
*/
#include "petsc.h"

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
 
