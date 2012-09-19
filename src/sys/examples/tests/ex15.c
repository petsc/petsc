
static char help[] = "Demonstrates PetscPopUpSelect()\n";

#include <petscsys.h>


#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **argv)
{
  int        ierr,choice;
  const char *choices[] = {"Say hello","Say goodbye"};

  PetscInitialize(&argc,&argv,(char *)0,help);
  ierr = PetscPopUpSelect(PETSC_COMM_WORLD,PETSC_NULL,"Select one of ",2,choices,&choice);CHKERRQ(ierr);
  ierr = PetscSynchronizedPrintf(PETSC_COMM_WORLD,"You selected %s\n",choices[choice]);CHKERRQ(ierr);
  ierr = PetscSynchronizedFlush(PETSC_COMM_WORLD);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return 0;
}

