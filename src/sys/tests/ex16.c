
static char help[] = "Demonstrates PetscMatlabEngineXXX()\n";

#include <petscsys.h>
#include <petscmatlab.h>

int main(int argc,char **argv)
{
  PetscMPIInt    rank;
  char           buffer[256],*output,user[256];
  PetscBool      userhappy = PETSC_FALSE;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc,&argv,(char*)0,help));
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD,&rank));

  PetscCall(PetscMatlabEngineGetOutput(PETSC_MATLAB_ENGINE_(PETSC_COMM_WORLD),&output));

  PetscCall(PetscMatlabEngineEvaluate(PETSC_MATLAB_ENGINE_(PETSC_COMM_WORLD),"MPI_Comm_rank"));
  PetscCall(PetscSynchronizedPrintf(PETSC_COMM_WORLD,"[%d]Processor rank is %s",rank,output));
  PetscCall(PetscSynchronizedFlush(PETSC_COMM_WORLD,PETSC_STDOUT));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,">>"));
  PetscCall(PetscSynchronizedFGets(PETSC_COMM_WORLD,stdin,256,user));
  PetscCall(PetscStrncmp(user,"exit",4,&userhappy));
  while (!userhappy) {
    PetscCall(PetscMatlabEngineEvaluate(PETSC_MATLAB_ENGINE_(PETSC_COMM_WORLD),user));
    PetscCall(PetscSynchronizedPrintf(PETSC_COMM_WORLD,"[%d]The result is %s",rank,output));
    PetscCall(PetscSynchronizedFlush(PETSC_COMM_WORLD,PETSC_STDOUT));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,">>"));
    PetscCall(PetscSynchronizedFGets(PETSC_COMM_WORLD,stdin,256,user));
    PetscCall(PetscStrncmp(user,"exit",4,&userhappy));
  }
  PetscCall(PetscFinalize());
  return 0;
}
