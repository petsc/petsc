
static char help[] = "Example of using PetscLikely() and PetscUnlikely().\n\n";

/*T
   Concepts: optimization, likely, unlikely
   Processors: n
T*/

#include "petscsys.h"

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **argv)
{
  PetscTruth flg = PETSC_TRUE;

  PetscInitialize(&argc,&argv,(char *)0,help);

  if (PetscLikely(flg)) {
    /* do something */
  }

  if (PetscUnlikely(flg)) {
    /* do something */
  }
  PetscFinalize();
  return 0;
}
 
