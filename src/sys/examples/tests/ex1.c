#ifndef lint
static char vcid[] = "$Id: ex3.c,v 1.5 1995/03/10 04:45:17 bsmith Exp $";
#endif

/*
      Example demonstrating some features of the options directory.
*/
#include "petsc.h"
#include "options.h"
#include <stdio.h>

int CreateError(int n)
{
  int ierr;
  if (!n) SETERR(1,"Error Created");
  if ((ierr = CreateError(n-1))) SETERR(ierr,"Error returned");
  return 0;
}

int main(int argc,char **argv)
{
  PetscInitialize(&argc,&argv,0,0);
  fprintf(stderr,"Demonstrates Petsc Error Handlers\n");
  fprintf(stderr,"The error below is a contrived error to test the code\n");
  CreateError(5);
  PetscFinalize();
  return 0;
}
 
