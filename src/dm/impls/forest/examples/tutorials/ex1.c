static char help[] = "Create and view a forest mesh\n\n";

#include <petscdmforest.h>

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc, char **argv)
{
  DM             dm;
  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc, &argv, NULL, help);CHKERRQ(ierr);
  ierr = DMForestCreate(PETSC_COMM_WORLD, &dm);CHKERRQ(ierr);
#if 0
  ierr = DMForestSetFromOptions(dm);CHKERRQ(ierr);
  ierr = DMForestSetUp(dm);CHKERRQ(ierr);
#endif
  ierr = DMDestroy(&dm);CHKERRQ(ierr);
  return 0;
}
