#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: ex1.c,v 1.13 1997/10/19 03:24:14 bsmith Exp bsmith $";
#endif

/* 
   Demonstrates PETSc error handlers.
 */

#include "petsc.h"

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
  fprintf(stdout,"The error is a contrived error to test error handling\n");
  ierr = CreateError(5); CHKERRA(ierr);
  PetscFinalize();
  return 0;
}
 
