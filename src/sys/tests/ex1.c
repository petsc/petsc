static char help[] = "Demonstrates PETSc error handlers.\n";

#include <petscsys.h>

PetscErrorCode CreateError(int n)
{
  PetscCheck(n, PETSC_COMM_WORLD, PETSC_ERR_USER, "Error Created");
  PetscCall(CreateError(n - 1));
  return PETSC_SUCCESS;
}

int main(int argc, char **argv)
{
  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, (char *)0, help));
  PetscCall(PetscFPrintf(PETSC_COMM_WORLD, stdout, "Demonstrates PETSc Error Handlers\n"));
  PetscCall(PetscFPrintf(PETSC_COMM_WORLD, stdout, "The error is a contrived error to test error handling\n"));
  PetscCall(PetscSynchronizedFlush(PETSC_COMM_WORLD, PETSC_STDOUT));
  PetscCall(CreateError(5));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

 # Testing errors so only look for errors
   test:
     requires: !defined(PETSCTEST_VALGRIND)
     args: -petsc_ci_portable_error_output -error_output_stdout
     nsize: {{1 2 3}}
     filter: grep -E "(PETSC ERROR)" | egrep "(Error Created|CreateError\(\)|main\(\))"

TEST*/
