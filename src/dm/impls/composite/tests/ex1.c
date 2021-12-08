static char help[] = "Tests DMClone() with DMComposite\n\n";

#include <petscdmcomposite.h>
#include <petscdmda.h>

int main(int argc,char **argv)
{
  DM             newdm, dm, dm1,dm2;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = PetscInitialize(&argc, &argv, 0, help); if (ierr) return ierr;
  ierr = DMCompositeCreate(PETSC_COMM_WORLD, &dm);CHKERRQ(ierr);
  ierr = DMDACreate1d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, 100, 1, 1, NULL, &dm1);CHKERRQ(ierr);
  ierr = DMDACreate1d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, 100, 1, 1, NULL, &dm2);CHKERRQ(ierr);
  ierr = DMSetUp(dm1);CHKERRQ(ierr);
  ierr = DMSetUp(dm2);CHKERRQ(ierr);
  ierr = DMCompositeAddDM(dm, dm1);CHKERRQ(ierr);
  ierr = DMCompositeAddDM(dm, dm2);CHKERRQ(ierr);
  ierr = DMDestroy(&dm1);CHKERRQ(ierr);
  ierr = DMDestroy(&dm2);CHKERRQ(ierr);
  ierr = DMSetFromOptions(dm);CHKERRQ(ierr);
  ierr = DMSetUp(dm);CHKERRQ(ierr);
  ierr = DMClone(dm, &newdm);CHKERRQ(ierr);
  ierr = DMDestroy(&dm);CHKERRQ(ierr);
  ierr = DMDestroy(&newdm);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

  test:
    suffix: 0

TEST*/
