#ifndef lint
static char vcid[] = "$Id: ex4.c,v 1.7 1995/03/10 04:45:17 bsmith Exp $";
#endif

/*
      Tests the signal handler.
*/
#include "petsc.h"
#include "options.h"
#include <stdio.h>

int CreateError(int n)
{
  int    ierr;
  double *x = 0;
  if (!n) x[0] = 100.; 
  if ((ierr = CreateError(n-1))) SETERR(ierr,"Error returned");
  return 0;
}

int main(int argc,char **argv)
{
  PetscInitialize(&argc,&argv,0,0);
  fprintf(stderr,"Demonstrates how PETSc may trap error interupts\n");
  fprintf(stderr,"The error below is contrived to test the code!\n");
  fflush(stderr);
  CreateError(5);
  PetscFinalize();
  return 0;
}
 
