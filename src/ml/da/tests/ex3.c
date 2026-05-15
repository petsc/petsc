static char help[] = "Tests LETKF-specific functionality: localization radius and type set/get.\n\n";
#include <petscda.h>

int main(int argc, char **argv)
{
  PetscDA                      da;
  PetscReal                    radius = 3.14, radius_check;
  PetscDALETKFLocalizationType type_check;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));

  /* Create the LETKF DA object */
  PetscCall(PetscDACreate(PETSC_COMM_WORLD, &da));
  PetscCall(PetscDASetSizes(da, 10, 10));
  PetscCall(PetscDAEnsembleSetSize(da, 5));
  PetscCall(PetscDASetFromOptions(da));
  PetscCall(PetscDASetUp(da));

  /* Test localization radius set/get round-trip. Exact == compare is intentional:
     the setter stores the value verbatim and the getter returns it with no arithmetic. */
  PetscCall(PetscDALETKFSetLocalizationRadius(da, radius));
  PetscCall(PetscDALETKFGetLocalizationRadius(da, &radius_check));
  PetscCheck(radius_check == radius, PETSC_COMM_WORLD, PETSC_ERR_PLIB, "SetLocalizationRadius/GetLocalizationRadius round-trip failed: set %g, got %g", (double)radius, (double)radius_check);
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Localization radius set/get: %g\n", (double)radius_check));

  /* Test localization type set/get round-trip across every enum value, and verify that
     changing the type leaves the previously-set radius untouched. Loop in PetscInt to keep
     compilers that treat the example as C++ (e.g. Apple clang) from rejecting enum++. */
  for (PetscInt ti = PETSCDA_LETKF_LOC_NONE; ti < PETSCDA_LETKF_LOC_NUM_TYPES; ++ti) {
    PetscDALETKFLocalizationType t = (PetscDALETKFLocalizationType)ti;

    PetscCall(PetscDALETKFSetLocalizationType(da, t));
    PetscCall(PetscDALETKFGetLocalizationType(da, &type_check));
    PetscCheck(type_check == t, PETSC_COMM_WORLD, PETSC_ERR_PLIB, "SetLocalizationType/GetLocalizationType round-trip failed at type %d: got %d", (int)t, (int)type_check);
    PetscCall(PetscDALETKFGetLocalizationRadius(da, &radius_check));
    PetscCheck(radius_check == radius, PETSC_COMM_WORLD, PETSC_ERR_PLIB, "Setting localization type %d clobbered radius: expected %g, got %g", (int)t, (double)radius, (double)radius_check);
  }
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Localization type set/get round-trip: ok\n"));

  PetscCall(PetscDADestroy(&da));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

  test:
    suffix: 1
    requires: !complex

TEST*/
