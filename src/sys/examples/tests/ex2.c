/*
      This example tests the signal handler.
*/
#include "petsc.h"
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
 
