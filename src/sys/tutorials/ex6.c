
static char help[] = "Example of using PetscLikely() and PetscUnlikely().\n\n";

#include <petscsys.h>

int main(int argc,char **argv)
{
  PetscBool      flg = PETSC_TRUE;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc,&argv,(char*)0,help));
  if (PetscLikely(flg)) {
    /* do something */
  }
  if (PetscUnlikely(flg)) {
    /* do something */
  }
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   test:

TEST*/
