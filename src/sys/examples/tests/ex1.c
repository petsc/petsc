#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: ex1.c,v 1.17 1999/05/04 20:29:49 balay Exp bsmith $";
#endif

/* 
   Demonstrates PETSc error handlers.
 */

#include "petsc.h"

#undef __FUNC__
#define __FUNC__ "CreateError"
int CreateError(int n)
{
  int ierr;
  if (!n) SETERRQ(1,0,"Error Created");
  ierr = CreateError(n-1);CHKERRQ(ierr);
  return 0;
}

#undef __FUNC__
#define __FUNC__ "main"
int main(int argc,char **argv)
{
  int ierr;
  PetscInitialize(&argc,&argv,(char *)0,0);
  ierr = PetscFPrintf(PETSC_COMM_WORLD,stdout,"Demonstrates PETSc Error Handlers\n");CHKERRA(ierr);
  ierr = PetscFPrintf(PETSC_COMM_WORLD,stdout,"The error is a contrived error to test error handling\n");CHKERRA(ierr);
  ierr = PetscSynchronizedFlush(PETSC_COMM_WORLD);CHKERRA(ierr);
  ierr = CreateError(5);CHKERRA(ierr);
  PetscFinalize();
  return 0;
}
 
