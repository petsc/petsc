static char help[] = "Test global numbering\n\n";

#include <petscdmplex.h>
#include <petscsf.h>

int main(int argc, char **argv)
{
  DM      dm;
  IS      point_numbering;
  PetscSF point_sf;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  PetscCall(DMCreate(PETSC_COMM_WORLD, &dm));
  PetscCall(DMSetType(dm, DMPLEX));
  PetscCall(DMSetFromOptions(dm));
  PetscCall(DMViewFromOptions(dm, NULL, "-dm_view"));

  PetscCall(DMPlexCreatePointNumbering(dm, &point_numbering));
  PetscCall(ISViewFromOptions(point_numbering, NULL, "-point_numbering_view"));
  PetscCall(ISDestroy(&point_numbering));

  PetscCall(DMGetPointSF(dm, &point_sf));
  PetscCall(PetscSFViewFromOptions(point_sf, NULL, "-point_sf_view"));

  PetscCall(DMDestroy(&dm));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

  test:
    nsize: 2
    args: -dm_plex_simplex 0 -dm_plex_box_faces 2,2 -dm_view -point_numbering_view -petscpartitioner_type simple
TEST*/
