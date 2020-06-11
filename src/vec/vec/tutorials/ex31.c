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

  ierr = VecCreate(PETSC_COMM_WORLD,&x);CHKERRQ(ierr);
  ierr = VecSetSizes(x,PETSC_DECIDE,n);CHKERRQ(ierr);
  ierr = VecSetFromOptions(x);CHKERRQ(ierr);

  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);
  ierr = PetscMatlabEngineGetOutput(PETSC_MATLAB_ENGINE_WORLD,&output);
  ierr = PetscMatlabEngineEvaluate(PETSC_MATLAB_ENGINE_WORLD,"MPI_Comm_rank");
  ierr = PetscSynchronizedPrintf(PETSC_COMM_WORLD,"[%d]Processor rank is %s",rank,output);CHKERRQ(ierr);
  ierr = PetscSynchronizedFlush(PETSC_COMM_WORLD,PETSC_STDOUT);CHKERRQ(ierr);

  ierr = PetscObjectSetName((PetscObject)x,"x");CHKERRQ(ierr);
  ierr = PetscMatlabEnginePut(PETSC_MATLAB_ENGINE_WORLD,(PetscObject)x);CHKERRQ(ierr);
  ierr = PetscMatlabEngineEvaluate(PETSC_MATLAB_ENGINE_WORLD,"x = x + MPI_Comm_rank;\n");
  ierr = PetscMatlabEngineGet(PETSC_MATLAB_ENGINE_WORLD,(PetscObject)x);CHKERRQ(ierr);

  ierr = PetscMatlabEngineEvaluate(PETSC_MATLAB_ENGINE_WORLD,"whos\n");CHKERRQ(ierr);
  ierr = PetscSynchronizedPrintf(PETSC_COMM_WORLD,"[%d]The result is %s",rank,output);CHKERRQ(ierr);
  ierr = PetscSynchronizedFlush(PETSC_COMM_WORLD,PETSC_STDOUT);CHKERRQ(ierr);

  ierr = VecView(x,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = VecDestroy(&x);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

