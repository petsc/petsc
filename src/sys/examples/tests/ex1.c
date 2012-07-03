/* 
   Demonstrates PETSc error handlers.
 */

#include <petscsys.h>

#undef __FUNCT__
#define __FUNCT__ "CreateError"
int CreateError(int n)
{
  PetscErrorCode ierr;
  if (!n) SETERRQ(PETSC_COMM_SELF,1,"Error Created");
  ierr = CreateError(n-1);CHKERRQ(ierr);
  return 0;
}

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **argv)
{
  PetscErrorCode ierr;
  PetscInitialize(&argc,&argv,(char *)0,0);
  ierr = PetscFPrintf(PETSC_COMM_WORLD,stdout,"Demonstrates PETSc Error Handlers\n");CHKERRQ(ierr);
  ierr = PetscFPrintf(PETSC_COMM_WORLD,stdout,"The error is a contrived error to test error handling\n");CHKERRQ(ierr);
  ierr = PetscSynchronizedFlush(PETSC_COMM_WORLD);CHKERRQ(ierr);
  ierr = CreateError(5);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return 0;
}
 
