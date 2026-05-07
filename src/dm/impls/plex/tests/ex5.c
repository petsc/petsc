static char help[] = "Moved test to ex69.c\n\n";

#include <petscdmplex.h>

int main(int argc, char **argv)
{
  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST
 TEST*/
