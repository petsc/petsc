static char help[] = "Test MatZeroEntries() on unassembled matrices \n\n";

#include <petscmat.h>

int main(int argc, char **args)
{
  Mat             A;
  PetscInt        N = 32;
  MPI_Comm        comm;

  PetscCall(PetscInitialize(&argc, &args, (char*) 0, help));
  comm = PETSC_COMM_WORLD;
  PetscCall(PetscOptionsGetInt(NULL,NULL, "-N", &N, NULL));
  PetscCall(MatCreate(comm, &A));
  PetscCall(MatSetSizes(A, PETSC_DECIDE, PETSC_DECIDE, N, N));
  PetscCall(MatSetFromOptions(A));
  PetscCall(MatSeqAIJSetPreallocation(A, 3, NULL));
  PetscCall(MatMPIAIJSetPreallocation(A, 3, NULL, 2, NULL));
  PetscCall(MatZeroEntries(A));
  PetscCall(MatDestroy(&A));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST
   test:
      requires: kokkos_kernels
      nsize: {{1 2}}
      output_file: output/ex252_1.out
      args: -mat_type aijkokkos
TEST*/
