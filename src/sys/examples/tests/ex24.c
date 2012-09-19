static char help[] = "Tests %D and %G formatting\n";
#include <petscsys.h>


#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **argv)
{
  PetscErrorCode ierr;

  PetscInitialize(&argc,&argv,(char *)0,help);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"A string followed by integer %D\n",22);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"A string followed by double %5G another %G\n",23.2,11.3);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"and then an int %D\n",30);CHKERRQ(ierr);

  ierr = PetscFinalize();
  return 0;
}

