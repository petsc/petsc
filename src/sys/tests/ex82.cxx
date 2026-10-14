static char help[] = "Tests C++ access to the PETSc stack and logging thread-local variables.\n\n";

#include <petscsys.h>

int main(int argc, char **argv)
{
  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  PetscCheck(petscstack.currentsize == 1, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Incorrect C++ stack depth");
  PetscCall(PetscLogFlops(1.0));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

  build:
    requires: debug

  test:
    output_file: output/empty.out

TEST*/
