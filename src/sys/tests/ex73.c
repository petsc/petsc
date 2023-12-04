const char help[] = "Test dropping PetscLogEventEnd()";

#include <petsc.h>

int main(int argc, char **argv)
{
  FILE *file;

  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  file = fopen("test", "w");
  if (argc < 10) fclose(file);
  PetscCall(PetscFPrintf(PETSC_COMM_WORLD, file, "Testing error handling with bad \n"));
  if (argc >= 10) fclose(file);

  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

  test:
    suffix: 0
    requires: !defined(PETSCTEST_VALGRIND)
    args: -petsc_ci_portable_error_output -error_output_stdout
    filter: grep -E -v "(memory block|leaked context|not freed before MPI_Finalize|Could be the program crashed|PETSc Option Table entries|source: environment)"
    TODO: Too many odd ball failures that are not reproducible in alt files

TEST*/
