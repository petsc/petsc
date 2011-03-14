
static char help[] = "Tests PetscSynchronizedPrintf() and PetscSynchronizedFPrintf().\n\n";

#include <petscsys.h>

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **argv)
{
  PetscMPIInt    rank;
  PetscErrorCode ierr;

  PetscInitialize(&argc,&argv,(char *)0,help);
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRQ(ierr);

  ierr = PetscSynchronizedPrintf(PETSC_COMM_WORLD,"Greetings from %d\n",rank);CHKERRQ(ierr);
  ierr = PetscSynchronizedFlush(PETSC_COMM_WORLD);CHKERRQ(ierr);

  ierr = PetscSynchronizedFPrintf(PETSC_COMM_WORLD,stderr,"Greetings again from %d\n",rank);CHKERRQ(ierr);
  ierr = PetscSynchronizedFlush(PETSC_COMM_WORLD);CHKERRQ(ierr);
 
  ierr = PetscFinalize();
  return 0;
}
 
