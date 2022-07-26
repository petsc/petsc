static char help[] = "Tests mesh reordering\n\n";

#include <petscdmplex.h>

int main(int argc, char **argv)
{
  DM             dm;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL,help));
  PetscCall(DMCreate(PETSC_COMM_WORLD, &dm));
  PetscCall(DMSetType(dm, DMPLEX));
  PetscCall(DMSetFromOptions(dm));
  PetscCall(DMViewFromOptions(dm, NULL, "-dm_view"));
  PetscCall(DMDestroy(&dm));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

  testset:
    requires: triangle
    args: -dm_plex_box_faces 3,3 -dm_plex_reorder rcm -dm_view ::ascii_info_detail

    test:
      suffix: 0

    test:
      suffix: 1
      nsize: 2
      args: -petscpartitioner_type simple

  testset:
    args: -dm_plex_simplex 0 -dm_plex_box_faces 4,4 -dm_plex_reorder rcm -dm_view ::ascii_info_detail

    test:
      suffix: 2

    test:
      suffix: 3
      nsize: 2
      args: -petscpartitioner_type simple

TEST*/
