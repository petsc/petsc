static char help[] = "Create and view a forest mesh\n\n";

#include <petscdmforest.h>

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc, char **argv)
{
  DM             dm;
  char           typeString[256] = {'\0'};
  PetscViewer    viewer = NULL;
  PetscBool      flg;
  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc, &argv, NULL, help);CHKERRQ(ierr);
  ierr = DMCreate(PETSC_COMM_WORLD, &dm);CHKERRQ(ierr);
  ierr = PetscStrncpy(typeString,DMFOREST,256);CHKERRQ(ierr);
  ierr = PetscOptionsGetString(NULL,"-dm_type",typeString,256,NULL);CHKERRQ(ierr);
  ierr = DMSetType(dm,(DMType) typeString);CHKERRQ(ierr);
  ierr = DMSetFromOptions(dm);CHKERRQ(ierr);
  ierr = DMSetUp(dm);CHKERRQ(ierr);
  ierr = PetscOptionsGetViewer(PETSC_COMM_WORLD,NULL,"-dm_view",&viewer,NULL,&flg);CHKERRQ(ierr);
  if (flg) {
    ierr = DMView(dm,viewer);CHKERRQ(ierr);
  }
  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
  ierr = DMDestroy(&dm);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return 0;
}
