
static char help[] = "Demonstrates PETSc error handlers.\n";

#include <petscsys.h>

int CreateError(int n)
{
  PetscCheck(n, PETSC_COMM_SELF, PETSC_ERR_USER, "Error Created");
  PetscCall(CreateError(n - 1));
  return 0;
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
     args: -error_output_stdout
     filter: grep -E "(PETSC ERROR)" | egrep "(Error Created|CreateError\(\)|main\(\))" | cut -f1,2,3,4,5,6 -d " "
     TODO:  Does not always produce exactly expected output on all systems for all runs

TEST*/
