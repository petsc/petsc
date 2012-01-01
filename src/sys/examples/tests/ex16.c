
/* 
   Demonstrates PetscMatlabEngineXXX()
*/

#include <petscsys.h>

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **argv)
{
  PetscErrorCode ierr;
  PetscMPIInt    rank;
  char           buffer[256],*output,user[256];
  PetscBool      userhappy = PETSC_FALSE;

  PetscInitialize(&argc,&argv,(char *)0,0);
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);

  ierr = PetscMatlabEngineGetOutput(PETSC_MATLAB_ENGINE_(PETSC_COMM_WORLD),&output);CHKERRQ(ierr);

  ierr = PetscMatlabEngineEvaluate(PETSC_MATLAB_ENGINE_(PETSC_COMM_WORLD),"MPI_Comm_rank");CHKERRQ(ierr);
  ierr = PetscSynchronizedPrintf(PETSC_COMM_WORLD,"[%d]Processor rank is %s",rank,output);CHKERRQ(ierr);
  ierr = PetscSynchronizedFlush(PETSC_COMM_WORLD);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,">>");CHKERRQ(ierr);
  ierr = PetscSynchronizedFGets(PETSC_COMM_WORLD,stdin,256,user);CHKERRQ(ierr); 
  ierr = PetscStrncmp(user,"exit",4,&userhappy);CHKERRQ(ierr);
  while (!userhappy) {
    ierr = PetscMatlabEngineEvaluate(PETSC_MATLAB_ENGINE_(PETSC_COMM_WORLD),user);CHKERRQ(ierr);
    ierr = PetscSynchronizedPrintf(PETSC_COMM_WORLD,"[%d]The result is %s",rank,output);CHKERRQ(ierr);
    ierr = PetscSynchronizedFlush(PETSC_COMM_WORLD);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD,">>");CHKERRQ(ierr);
    ierr = PetscSynchronizedFGets(PETSC_COMM_WORLD,stdin,256,user);CHKERRQ(ierr);
    ierr = PetscStrncmp(user,"exit",4,&userhappy);CHKERRQ(ierr);
  }
  ierr = PetscFinalize();
  return 0;
}
 
