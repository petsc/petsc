static char help[] = "Tests mesh reordering\n\n";

#include <petscdmplex.h>

int main(int argc, char **argv)
{
  DM             dm;

  CHKERRQ(PetscInitialize(&argc, &argv, NULL,help));
  CHKERRQ(DMCreate(PETSC_COMM_WORLD, &dm));
  CHKERRQ(DMSetType(dm, DMPLEX));
  CHKERRQ(DMSetFromOptions(dm));
  CHKERRQ(DMViewFromOptions(dm, NULL, "-dm_view"));
  CHKERRQ(DMDestroy(&dm));
  CHKERRQ(PetscFinalize());
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
