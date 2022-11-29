
static char help[] = "Tests shared memory subcommunicators\n\n";
#include <petscsys.h>
#include <petscvec.h>

/*
   One can use petscmpiexec -n 3 -hosts localhost,Barrys-MacBook-Pro.local ./ex2 -info to mimic
  having two nodes that do not share common memory
*/

int main(int argc, char **args)
{
  PetscCommShared scomm;
  MPI_Comm        comm;
  PetscMPIInt     lrank, rank, size, i;
  Vec             x, y;
  VecScatter      vscat;
  IS              isstride, isblock;
  PetscViewer     singleton;
  PetscInt        indices[] = {0, 1, 2};

  PetscInitialize(&argc, &args, (char *)0, help);
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &rank));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &size));
  PetscCheck(size == 3, PETSC_COMM_WORLD, PETSC_ERR_WRONG_MPI_SIZE, "This example only works for 3 processes");

  PetscCall(PetscCommDuplicate(PETSC_COMM_WORLD, &comm, NULL));
  PetscCall(PetscCommSharedGet(comm, &scomm));

  for (i = 0; i < size; i++) {
    PetscCall(PetscCommSharedGlobalToLocal(scomm, i, &lrank));
    PetscCall(PetscSynchronizedPrintf(PETSC_COMM_WORLD, "[%d] Global rank %d shared memory comm rank %d\n", rank, i, lrank));
  }
  PetscCall(PetscSynchronizedFlush(PETSC_COMM_WORLD, stdout));
  PetscCall(PetscCommDestroy(&comm));

  PetscCall(VecCreateMPI(PETSC_COMM_WORLD, 2, PETSC_DETERMINE, &x));
  PetscCall(VecSetBlockSize(x, 2));
  PetscCall(VecSetValue(x, 2 * rank, (PetscScalar)(2 * rank + 10), INSERT_VALUES));
  PetscCall(VecSetValue(x, 2 * rank + 1, (PetscScalar)(2 * rank + 1 + 10), INSERT_VALUES));
  PetscCall(VecAssemblyBegin(x));
  PetscCall(VecAssemblyEnd(x));
  PetscCall(VecView(x, PETSC_VIEWER_STDOUT_WORLD));

  PetscCall(VecCreateSeq(PETSC_COMM_SELF, 6, &y));
  PetscCall(VecSetBlockSize(y, 2));
  PetscCall(ISCreateStride(PETSC_COMM_SELF, 6, 0, 1, &isstride));
  PetscCall(ISCreateBlock(PETSC_COMM_SELF, 2, 3, indices, PETSC_COPY_VALUES, &isblock));
  PetscCall(VecScatterCreate(x, isblock, y, isstride, &vscat));
  PetscCall(VecScatterBegin(vscat, x, y, INSERT_VALUES, SCATTER_FORWARD));
  PetscCall(VecScatterEnd(vscat, x, y, INSERT_VALUES, SCATTER_FORWARD));
  PetscCall(VecScatterDestroy(&vscat));
  PetscCall(PetscViewerGetSubViewer(PETSC_VIEWER_STDOUT_WORLD, PETSC_COMM_SELF, &singleton));
  PetscCall(VecView(y, singleton));
  PetscCall(PetscViewerRestoreSubViewer(PETSC_VIEWER_STDOUT_WORLD, PETSC_COMM_SELF, &singleton));

  PetscCall(ISDestroy(&isstride));
  PetscCall(ISDestroy(&isblock));
  PetscCall(VecDestroy(&x));
  PetscCall(VecDestroy(&y));
  PetscFinalize();
  return 0;
}
