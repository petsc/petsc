static char help[] = "Tests DMClone() with DMComposite\n\n";

#include <petscdmcomposite.h>
#include <petscdmda.h>

int main(int argc,char **argv)
{
  DM             newdm, dm, dm1,dm2;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = PetscInitialize(&argc, &argv, 0, help); if (ierr) return ierr;
  CHKERRQ(DMCompositeCreate(PETSC_COMM_WORLD, &dm));
  CHKERRQ(DMDACreate1d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, 100, 1, 1, NULL, &dm1));
  CHKERRQ(DMDACreate1d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, 100, 1, 1, NULL, &dm2));
  CHKERRQ(DMSetUp(dm1));
  CHKERRQ(DMSetUp(dm2));
  CHKERRQ(DMCompositeAddDM(dm, dm1));
  CHKERRQ(DMCompositeAddDM(dm, dm2));
  CHKERRQ(DMDestroy(&dm1));
  CHKERRQ(DMDestroy(&dm2));
  CHKERRQ(DMSetFromOptions(dm));
  CHKERRQ(DMSetUp(dm));
  CHKERRQ(DMClone(dm, &newdm));
  CHKERRQ(DMDestroy(&dm));
  CHKERRQ(DMDestroy(&newdm));
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

  test:
    suffix: 0

TEST*/
