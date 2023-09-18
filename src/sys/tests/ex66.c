static const char help[] = "Tests error message when previous error was not fully handled\n";

#include <petscsys.h>

PetscErrorCode CreateError(int n)
{
  PetscCheck(n, PETSC_COMM_SELF, PETSC_ERR_USER, "Error Created");
  PetscCall(CreateError(n - 1));
  return PETSC_SUCCESS;
}

int main(int argc, char **argv)
{
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, (char *)0, help));
  PetscCall(PetscFPrintf(PETSC_COMM_WORLD, stdout, "Demonstrates PETSc Error Handlers\n"));
  PetscCall(PetscFPrintf(PETSC_COMM_WORLD, stdout, "The error is a contrived error to test error handling\n"));
  PetscCall(PetscSynchronizedFlush(PETSC_COMM_WORLD, PETSC_STDOUT));
  ierr = CreateError(5);
  (void)ierr; /* this prevents the compiler from warning about unused error return code */
  ierr = CreateError(5);
  (void)ierr;
  PetscCall(CreateError(5));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   test:
     requires: !defined(PETSCTEST_VALGRIND)
     args: -petsc_ci_portable_error_output -error_output_stdout
     filter: grep -E -v "(memory block|leaked context|not freed before MPI_Finalize|Could be the program crashed|PETSc Option Table entries|source: environment)"

TEST*/
