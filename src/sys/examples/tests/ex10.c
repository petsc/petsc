#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: ex1.c,v 1.12 1997/07/09 20:52:08 balay Exp $";
#endif

/* 
   Demonstrates PETSc error handlers.
 */

#include "petsc.h"
#include <stdio.h>

int CreateError(int n)
{
  int ierr;
  if (!n) SETERRQ(1,0,"Error Created");
  ierr = CreateError(n-1); CHKERRQ(ierr);
  return 0;
}

int main(int argc,char **argv)
{
  int ierr;
  PetscInitialize(&argc,&argv,(char *)0,0);
  fprintf(stdout,"Demonstrates PETSc Error Handlers\n");
  fprintf(stdout,"The error below is a contrived error to test the code\n");
  ierr = CreateError(5); CHKERRA(ierr);
  PetscFinalize();
  return 0;
}
 
