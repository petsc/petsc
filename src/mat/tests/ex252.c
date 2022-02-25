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
  ierr = PetscOptionsGetInt(NULL,NULL, "-N", &N, NULL);CHKERRQ(ierr);
  ierr = MatCreate(comm, &A);CHKERRQ(ierr);
  ierr = MatSetSizes(A, PETSC_DECIDE, PETSC_DECIDE, N, N);CHKERRQ(ierr);
  ierr = MatSetFromOptions(A);CHKERRQ(ierr);
  ierr = MatSeqAIJSetPreallocation(A, 3, NULL);CHKERRQ(ierr);
  ierr = MatMPIAIJSetPreallocation(A, 3, NULL, 2, NULL);CHKERRQ(ierr);
  ierr = MatZeroEntries(A);CHKERRQ(ierr);
  ierr = MatDestroy(&A);CHKERRQ(ierr);
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

