static const char help[] = "Demonstrates PetscMatlabEngineXXX()\n";

#include <petscvec.h>
#include <petscmatlab.h>

int main(int argc,char **argv)
{
  PetscErrorCode ierr;
  PetscMPIInt    rank;
  PetscInt       n = 5;
  char           *output;
  Vec            x;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;

  CHKERRQ(VecCreate(PETSC_COMM_WORLD,&x));
  CHKERRQ(VecSetSizes(x,PETSC_DECIDE,n));
  CHKERRQ(VecSetFromOptions(x));

  CHKERRMPI(MPI_Comm_rank(PETSC_COMM_WORLD,&rank));
  CHKERRQ(PetscMatlabEngineGetOutput(PETSC_MATLAB_ENGINE_WORLD,&output));
  CHKERRQ(PetscMatlabEngineEvaluate(PETSC_MATLAB_ENGINE_WORLD,"MPI_Comm_rank"));
  CHKERRQ(PetscSynchronizedPrintf(PETSC_COMM_WORLD,"[%d]Processor rank is %s",rank,output));
  CHKERRQ(PetscSynchronizedFlush(PETSC_COMM_WORLD,PETSC_STDOUT));

  CHKERRQ(PetscObjectSetName((PetscObject)x,"x"));
  CHKERRQ(PetscMatlabEnginePut(PETSC_MATLAB_ENGINE_WORLD,(PetscObject)x));
  CHKERRQ(PetscMatlabEngineEvaluate(PETSC_MATLAB_ENGINE_WORLD,"x = x + MPI_Comm_rank;\n"));
  CHKERRQ(PetscMatlabEngineGet(PETSC_MATLAB_ENGINE_WORLD,(PetscObject)x));

  CHKERRQ(PetscMatlabEngineEvaluate(PETSC_MATLAB_ENGINE_WORLD,"whos\n"));
  CHKERRQ(PetscSynchronizedPrintf(PETSC_COMM_WORLD,"[%d]The result is %s",rank,output));
  CHKERRQ(PetscSynchronizedFlush(PETSC_COMM_WORLD,PETSC_STDOUT));

  CHKERRQ(VecView(x,PETSC_VIEWER_STDOUT_WORLD));
  CHKERRQ(VecDestroy(&x));
  ierr = PetscFinalize();
  return ierr;
}
