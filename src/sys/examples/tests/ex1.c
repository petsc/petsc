/*$Id: ex1.c,v 1.19 1999/10/24 14:01:38 bsmith Exp bsmith $*/

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
  ierr = PetscFPrintf(PETSC_COMM_WORLD,stdout,"Demonstrates PETSc Error Handlers\n");CHKERRQ(ierr);
  ierr = PetscFPrintf(PETSC_COMM_WORLD,stdout,"The error is a contrived error to test error handling\n");CHKERRQ(ierr);
  ierr = PetscSynchronizedFlush(PETSC_COMM_WORLD);CHKERRQ(ierr);
  ierr = CreateError(5);CHKERRQ(ierr);
  PetscFinalize();
  return 0;
}
 
