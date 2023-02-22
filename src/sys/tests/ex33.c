static char help[] = "Tests PetscStack.\n\n";

#include <petscsys.h>

#if !defined(PETSCSTACKSIZE)
  #define PETSCSTACKSIZE 64
#endif

PetscErrorCode correct()
{
  PetscFunctionBegin;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode correctu()
{
  PetscFunctionBeginUser;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode foo()
{
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode bar()
{
  PetscFunctionBegin;
  return PETSC_SUCCESS;
}

PetscErrorCode baru()
{
  PetscFunctionBeginUser;
  return PETSC_SUCCESS;
}

int main(int argc, char **argv)
{
  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, (char *)0, help));
  for (PetscInt i = 0; i < PETSCSTACKSIZE + 1; i++) PetscCall(correct());
  for (PetscInt i = 0; i < PETSCSTACKSIZE + 1; i++) PetscCall(foo());
  for (PetscInt i = 0; i < PETSCSTACKSIZE + 1; i++) PetscCall(bar());
  for (PetscInt i = 0; i < PETSCSTACKSIZE + 1; i++) PetscCall(foo());
  for (PetscInt i = 0; i < PETSCSTACKSIZE + 1; i++) PetscCall(baru());
  for (PetscInt i = 0; i < PETSCSTACKSIZE + 1; i++) PetscCall(foo());
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

  test:
    TODO: Since this now errors out the test harness can chock on the output

TEST*/
