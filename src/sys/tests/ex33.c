static char help[] = "Tests PetscStack.\n\n";

#include <petscsys.h>

#if !defined(PETSCSTACKSIZE)
#define PETSCSTACKSIZE 64
#endif

PetscErrorCode correct()
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

PetscErrorCode correctu()
{
  PetscFunctionBeginUser;
  PetscFunctionReturn(0);
}

PetscErrorCode foo()
{
  PetscFunctionReturn(0);
}

PetscErrorCode bar()
{
  PetscFunctionBegin;
  return 0;
}

PetscErrorCode baru()
{
  PetscFunctionBeginUser;
  return 0;
}

int main(int argc,char **argv)
{
  PetscBool      flg = PETSC_FALSE;

  PetscCall(PetscInitialize(&argc,&argv,(char*)0,help));
#if defined(PETSC_USE_DEBUG)
  PetscCall(PetscOptionsGetBool(NULL,NULL,"-checkstack",&flg,NULL));
#endif
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,"%s for stack\n",flg ? "Checking" : "Not checking"));
  for (PetscInt i = 0; i < PETSCSTACKSIZE+1; i++) PetscCall(correct());
  for (PetscInt i = 0; i < PETSCSTACKSIZE+1; i++) PetscCall(foo());
  for (PetscInt i = 0; i < PETSCSTACKSIZE+1; i++) PetscCall(bar());
  for (PetscInt i = 0; i < PETSCSTACKSIZE+1; i++) PetscCall(foo());
  for (PetscInt i = 0; i < PETSCSTACKSIZE+1; i++) PetscCall(baru());
  for (PetscInt i = 0; i < PETSCSTACKSIZE+1; i++) PetscCall(foo());
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

  test:
    TODO: Since this now errors out the test harness can chock on the output

TEST*/
