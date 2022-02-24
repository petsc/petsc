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
  PetscErrorCode ierr;
  PetscBool      flg = PETSC_FALSE;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
#if defined(PETSC_USE_DEBUG)
  CHKERRQ(PetscOptionsGetBool(NULL,NULL,"-checkstack",&flg,NULL));
#endif
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"%s for stack\n",flg ? "Checking" : "Not checking"));
  for (PetscInt i = 0; i < PETSCSTACKSIZE+1; i++) CHKERRQ(correct());
  for (PetscInt i = 0; i < PETSCSTACKSIZE+1; i++) CHKERRQ(foo());
  for (PetscInt i = 0; i < PETSCSTACKSIZE+1; i++) CHKERRQ(bar());
  for (PetscInt i = 0; i < PETSCSTACKSIZE+1; i++) CHKERRQ(foo());
  for (PetscInt i = 0; i < PETSCSTACKSIZE+1; i++) CHKERRQ(baru());
  for (PetscInt i = 0; i < PETSCSTACKSIZE+1; i++) CHKERRQ(foo());
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

  test:

TEST*/
