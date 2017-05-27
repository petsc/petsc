
static char help[] = "Demonstrates PetscPopUpSelect()\n";

#include <petscsys.h>


int main(int argc,char **argv)
{
  int        ierr,choice;
  const char *choices[] = {"Say hello","Say goodbye"};

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  ierr = PetscPopUpSelect(PETSC_COMM_WORLD,NULL,"Select one of ",2,choices,&choice);CHKERRQ(ierr);
  ierr = PetscSynchronizedPrintf(PETSC_COMM_WORLD,"You selected %s\n",choices[choice]);CHKERRQ(ierr);
  ierr = PetscSynchronizedFlush(PETSC_COMM_WORLD,PETSC_STDOUT);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

