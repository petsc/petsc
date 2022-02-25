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
  ierr = PetscOptionsGetBool(NULL,NULL,"-checkstack",&flg,NULL);CHKERRQ(ierr);
#endif
  ierr = PetscPrintf(PETSC_COMM_WORLD,"%s for stack\n",flg ? "Checking" : "Not checking");CHKERRQ(ierr);
  for (PetscInt i = 0; i < PETSCSTACKSIZE+1; i++) { ierr = correct();CHKERRQ(ierr); }
  for (PetscInt i = 0; i < PETSCSTACKSIZE+1; i++) { ierr = foo();CHKERRQ(ierr); }
  for (PetscInt i = 0; i < PETSCSTACKSIZE+1; i++) { ierr = bar();CHKERRQ(ierr); }
  for (PetscInt i = 0; i < PETSCSTACKSIZE+1; i++) { ierr = foo();CHKERRQ(ierr); }
  for (PetscInt i = 0; i < PETSCSTACKSIZE+1; i++) { ierr = baru();CHKERRQ(ierr); }
  for (PetscInt i = 0; i < PETSCSTACKSIZE+1; i++) { ierr = foo();CHKERRQ(ierr); }
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

  test:

TEST*/
