/* 
   This example demonstrates some features of the options directory.
 */

#include "petsc.h"
#include <stdio.h>

int CreateError(int n)
{
  int ierr;
  if (!n) SETERRQ(1,"Error Created");
  ierr = CreateError(n-1); CHKERRQ(ierr);
  return 0;
}

int main(int argc,char **argv)
{
  int ierr;
  PetscInitialize(&argc,&argv,0,0,0,0);
  fprintf(stdout,"Demonstrates Petsc Error Handlers\n");
  fprintf(stdout,"The error below is a contrived error to test the code\n");
  ierr = CreateError(5); CHKERRA(ierr);
  PetscFinalize();
  return 0;
}
 
