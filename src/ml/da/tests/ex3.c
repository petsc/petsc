static char help[] = "Tests LETKF-specific functionality: localization radius set/get.\n\n";
#include <petscda.h>

int main(int argc, char **argv)
{
  PetscDA   da;
  PetscReal radius = 3.14, radius_check;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));

  /* Create the LETKF DA object */
  PetscCall(PetscDACreate(PETSC_COMM_WORLD, &da));
  PetscCall(PetscDASetSizes(da, 10, 10));
  PetscCall(PetscDASetType(da, PETSCDALETKF));
  PetscCall(PetscDAEnsembleSetSize(da, 5));
  PetscCall(PetscDASetFromOptions(da));
  PetscCall(PetscDASetUp(da));

  /* Test localization radius set/get round-trip. Exact == compare is intentional:
     the setter stores the value verbatim and the getter returns it with no arithmetic. */
  PetscCall(PetscDALETKFSetLocalizationRadius(da, radius));
  PetscCall(PetscDALETKFGetLocalizationRadius(da, &radius_check));
  PetscCheck(radius_check == radius, PETSC_COMM_WORLD, PETSC_ERR_PLIB, "SetLocalizationRadius/GetLocalizationRadius round-trip failed: set %g, got %g", (double)radius, (double)radius_check);
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Localization radius set/get: %g\n", (double)radius_check));

  PetscCall(PetscDADestroy(&da));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

  test:
    suffix: 1
    requires: !complex

TEST*/
