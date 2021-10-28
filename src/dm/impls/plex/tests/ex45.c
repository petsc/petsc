static char help[] = "Tests mesh reordering\n\n";

#include <petscdmplex.h>

int main(int argc, char **argv)
{
  DM             dm;
  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc, &argv, NULL,help);if (ierr) return ierr;
  ierr = DMCreate(PETSC_COMM_WORLD, &dm);CHKERRQ(ierr);
  ierr = DMSetType(dm, DMPLEX);CHKERRQ(ierr);
  ierr = DMSetFromOptions(dm);CHKERRQ(ierr);
  ierr = DMViewFromOptions(dm, NULL, "-dm_view");CHKERRQ(ierr);
  ierr = DMDestroy(&dm);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
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
      args: -dm_distribute -petscpartitioner_type simple

  testset:
    args: -dm_plex_simplex 0 -dm_plex_box_faces 4,4 -dm_plex_reorder rcm -dm_view ::ascii_info_detail

    test:
      suffix: 2

    test:
      suffix: 3
      nsize: 2
      args: -dm_distribute -petscpartitioner_type simple

TEST*/
