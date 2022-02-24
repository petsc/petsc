static char help[] = "Test MatZeroEntries() on unassembled matrices \n\n";

#include <petscmat.h>

int main(int argc, char **args)
{
  Mat             A;
  PetscInt        N = 32;
  MPI_Comm        comm;
  PetscErrorCode  ierr;

  ierr = PetscInitialize(&argc, &args, (char*) 0, help);if (ierr) return ierr;
  comm = PETSC_COMM_WORLD;
  CHKERRQ(PetscOptionsGetInt(NULL,NULL, "-N", &N, NULL));
  CHKERRQ(MatCreate(comm, &A));
  CHKERRQ(MatSetSizes(A, PETSC_DECIDE, PETSC_DECIDE, N, N));
  CHKERRQ(MatSetFromOptions(A));
  CHKERRQ(MatSeqAIJSetPreallocation(A, 3, NULL));
  CHKERRQ(MatMPIAIJSetPreallocation(A, 3, NULL, 2, NULL));
  CHKERRQ(MatZeroEntries(A));
  CHKERRQ(MatDestroy(&A));
  ierr = PetscFinalize();
  return ierr;
}

/*TEST
   test:
      requires: kokkos_kernels
      nsize: {{1 2}}
      output_file: output/ex252_1.out
      args: -mat_type aijkokkos
TEST*/
