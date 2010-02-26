
static char help[] = "Tests catching of floating point exceptions.\n\n";

#include "petscsys.h"

#undef __FUNCT__
#define __FUNCT__ "CreateError"
int CreateError(PetscReal x)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  x = 1.0/x;
  ierr = PetscPrintf(PETSC_COMM_SELF,"x = %G\n",x);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **argv)
{
  PetscErrorCode ierr;
  PetscInitialize(&argc,&argv,(char *)0,help);
  ierr = PetscPrintf(PETSC_COMM_SELF,"This is a contrived example to test floating pointing\n");CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_SELF,"It is not a true error.\n");CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_SELF,"Run with -fp_trap to catch the floating point error\n");CHKERRQ(ierr);
  ierr = CreateError(0.0);CHKERRQ(ierr);
  return 0;
}
 
