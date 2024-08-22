static char help[] = "Error handling for external library call in void function.\n";

#include <petscsys.h>

int ReturnAnError(PETSC_UNUSED int dummy)
{
  return 1;
}

void MakeAnError(void)
{
  PetscCallExternalAbort(ReturnAnError, 0);
}

int main(int argc, char **argv)
{
  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  MakeAnError();
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   test:
     requires: !defined(PETSCTEST_VALGRIND) defined(PETSC_USE_DEBUG) !defined(PETSC_HAVE_SANITIZER)
     args: -petsc_ci_portable_error_output -error_output_stdout
     filter: grep -E -v "(memory block|leaked context|not freed before MPI_Finalize|Could be the program crashed|PETSc Option Table entries|source: environment)"

TEST*/
