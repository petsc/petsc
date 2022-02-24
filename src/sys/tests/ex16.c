
static char help[] = "Demonstrates PetscMatlabEngineXXX()\n";

#include <petscsys.h>
#include <petscmatlab.h>

int main(int argc,char **argv)
{
  PetscErrorCode ierr;
  PetscMPIInt    rank;
  char           buffer[256],*output,user[256];
  PetscBool      userhappy = PETSC_FALSE;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);

  CHKERRQ(PetscMatlabEngineGetOutput(PETSC_MATLAB_ENGINE_(PETSC_COMM_WORLD),&output));

  CHKERRQ(PetscMatlabEngineEvaluate(PETSC_MATLAB_ENGINE_(PETSC_COMM_WORLD),"MPI_Comm_rank"));
  CHKERRQ(PetscSynchronizedPrintf(PETSC_COMM_WORLD,"[%d]Processor rank is %s",rank,output));
  CHKERRQ(PetscSynchronizedFlush(PETSC_COMM_WORLD,PETSC_STDOUT));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,">>"));
  CHKERRQ(PetscSynchronizedFGets(PETSC_COMM_WORLD,stdin,256,user));
  CHKERRQ(PetscStrncmp(user,"exit",4,&userhappy));
  while (!userhappy) {
    CHKERRQ(PetscMatlabEngineEvaluate(PETSC_MATLAB_ENGINE_(PETSC_COMM_WORLD),user));
    CHKERRQ(PetscSynchronizedPrintf(PETSC_COMM_WORLD,"[%d]The result is %s",rank,output));
    CHKERRQ(PetscSynchronizedFlush(PETSC_COMM_WORLD,PETSC_STDOUT));
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,">>"));
    CHKERRQ(PetscSynchronizedFGets(PETSC_COMM_WORLD,stdin,256,user));
    CHKERRQ(PetscStrncmp(user,"exit",4,&userhappy));
  }
  ierr = PetscFinalize();
  return ierr;
}
