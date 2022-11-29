static const char help[] = "Demonstrates PetscMatlabEngineXXX()\n";

#include <petscvec.h>
#include <petscmatlab.h>

int main(int argc, char **argv)
{
  PetscMPIInt rank;
  PetscInt    n = 5;
  char       *output;
  Vec         x;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, (char *)0, help));

  PetscCall(VecCreate(PETSC_COMM_WORLD, &x));
  PetscCall(VecSetSizes(x, PETSC_DECIDE, n));
  PetscCall(VecSetFromOptions(x));

  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &rank));
  PetscCall(PetscMatlabEngineGetOutput(PETSC_MATLAB_ENGINE_WORLD, &output));
  PetscCall(PetscMatlabEngineEvaluate(PETSC_MATLAB_ENGINE_WORLD, "MPI_Comm_rank"));
  PetscCall(PetscSynchronizedPrintf(PETSC_COMM_WORLD, "[%d]Processor rank is\n %s", rank, output));
  PetscCall(PetscSynchronizedFlush(PETSC_COMM_WORLD, PETSC_STDOUT));

  PetscCall(PetscObjectSetName((PetscObject)x, "x"));
  PetscCall(PetscMatlabEnginePut(PETSC_MATLAB_ENGINE_WORLD, (PetscObject)x));
  PetscCall(PetscMatlabEngineEvaluate(PETSC_MATLAB_ENGINE_WORLD, "x = x + MPI_Comm_rank;\n"));
  PetscCall(PetscMatlabEngineGet(PETSC_MATLAB_ENGINE_WORLD, (PetscObject)x));

  PetscCall(PetscMatlabEngineEvaluate(PETSC_MATLAB_ENGINE_WORLD, "whos\n"));
  PetscCall(PetscSynchronizedPrintf(PETSC_COMM_WORLD, "[%d]The result is\n %s", rank, output));
  PetscCall(PetscSynchronizedFlush(PETSC_COMM_WORLD, PETSC_STDOUT));

  PetscCall(VecView(x, PETSC_VIEWER_STDOUT_WORLD));
  PetscCall(VecDestroy(&x));
  PetscCall(PetscFinalize());
  return 0;
}
