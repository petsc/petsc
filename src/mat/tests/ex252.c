static char help[] = "Test MatZeroEntries() on unassembled matrices \n\n";

#include <petscmat.h>

int main(int argc, char **args)
{
  Mat             A;
  PetscInt        N = 32;
  MPI_Comm        comm;

  CHKERRQ(PetscInitialize(&argc, &args, (char*) 0, help));
  comm = PETSC_COMM_WORLD;
  CHKERRQ(PetscOptionsGetInt(NULL,NULL, "-N", &N, NULL));
  CHKERRQ(MatCreate(comm, &A));
  CHKERRQ(MatSetSizes(A, PETSC_DECIDE, PETSC_DECIDE, N, N));
  CHKERRQ(MatSetFromOptions(A));
  CHKERRQ(MatSeqAIJSetPreallocation(A, 3, NULL));
  CHKERRQ(MatMPIAIJSetPreallocation(A, 3, NULL, 2, NULL));
  CHKERRQ(MatZeroEntries(A));
  CHKERRQ(MatDestroy(&A));
  CHKERRQ(PetscFinalize());
  return 0;
}

/*TEST
   test:
      requires: kokkos_kernels
      nsize: {{1 2}}
      output_file: output/ex252_1.out
      args: -mat_type aijkokkos
TEST*/
