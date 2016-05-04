
static char help[] = "Example of using PetscLikely() and PetscUnlikely().\n\n";

/*T
   Concepts: optimization^likely
   Concepts: optimization^unlikely
   Processors: n
T*/

#include <petscsys.h>

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **argv)
{
  PetscBool      flg = PETSC_TRUE;
  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  if (PetscLikely(flg)) {
    /* do something */
  }
  if (PetscUnlikely(flg)) {
    /* do something */
  }
  ierr = PetscFinalize();
  return ierr;
}

