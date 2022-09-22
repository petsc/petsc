const char help[] = "Basic create / destroy for PetscDualSpace";

#include <petscfe.h>

int main(int argc, char **argv)
{
  PetscDualSpace dsp;
  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  PetscCall(PetscDualSpaceCreate(PETSC_COMM_WORLD, &dsp));
  PetscCall(PetscDualSpaceDestroy(&dsp));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

  test:

TEST*/
