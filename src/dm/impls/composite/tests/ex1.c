static char help[] = "Tests DMClone() with DMComposite\n\n";

#include <petscdmcomposite.h>
#include <petscdmda.h>

int main(int argc,char **argv)
{
  DM             newdm, dm, dm1,dm2;

  PetscFunctionBeginUser;
  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, 0, help));
  PetscCall(DMCompositeCreate(PETSC_COMM_WORLD, &dm));
  PetscCall(DMDACreate1d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, 100, 1, 1, NULL, &dm1));
  PetscCall(DMDACreate1d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, 100, 1, 1, NULL, &dm2));
  PetscCall(DMSetUp(dm1));
  PetscCall(DMSetUp(dm2));
  PetscCall(DMCompositeAddDM(dm, dm1));
  PetscCall(DMCompositeAddDM(dm, dm2));
  PetscCall(DMDestroy(&dm1));
  PetscCall(DMDestroy(&dm2));
  PetscCall(DMSetFromOptions(dm));
  PetscCall(DMSetUp(dm));
  PetscCall(DMClone(dm, &newdm));
  PetscCall(DMDestroy(&dm));
  PetscCall(DMDestroy(&newdm));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

  test:
    suffix: 0

TEST*/
