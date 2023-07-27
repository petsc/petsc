static char help[] = "Benchmark DMPlexInterpolate.\n\n";

#include <petscdmplex.h>

int main(int argc, char **argv)
{
  DM            dm, dm2;
  PetscLogStage stage;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  PetscCall(PetscLogStageRegister("Interpolate", &stage));

  PetscCall(DMCreate(PETSC_COMM_WORLD, &dm));
  PetscCall(DMSetType(dm, DMPLEX));
  PetscCall(DMSetFromOptions(dm));
  PetscCall(DMViewFromOptions(dm, NULL, "-init_dm_view"));

  PetscCall(DMPlexUninterpolate(dm, &dm2));
  PetscCall(DMDestroy(&dm));
  dm = dm2;
  PetscCall(DMViewFromOptions(dm, NULL, "-unint_dm_view"));

  PetscCall(PetscLogStagePush(stage));
  PetscCall(DMPlexInterpolate(dm, &dm2));
  PetscCall(PetscLogStagePop());

  PetscCall(DMViewFromOptions(dm2, NULL, "-interp_dm_view"));

  PetscCall(DMDestroy(&dm2));
  PetscCall(DMDestroy(&dm));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

  test:
    suffix: 0
    args: -dm_plex_simplex 0

TEST*/
