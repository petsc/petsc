#ifndef lint
static char vcid[] = "$Id: ex9.c,v 1.4 1995/09/30 19:31:28 bsmith Exp bsmith $";
#endif

/* 
   Demonstrates some features of the options directory.
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
  PetscInitialize(&argc,&argv,0,0,0);
  fprintf(stdout,"Demonstrates Petsc Error Handlers\n");
  fprintf(stdout,"The error below is a contrived error to test the code\n");
  ierr = CreateError(5); CHKERRA(ierr);
  PetscFinalize();
  return 0;
}
 
