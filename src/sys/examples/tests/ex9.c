
static char help[] = "Tests PetscSequentialPhaseBegin() and PetscSequentialPhaseEnd()\n";

#include <petscsys.h>

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **args)
{
  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc,&args,PETSC_NULL,help);
  ierr = PetscSequentialPhaseBegin(PETSC_COMM_WORLD,1);CHKERRQ(ierr);
  ierr = PetscSequentialPhaseEnd(PETSC_COMM_WORLD,1);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return 0;
}
