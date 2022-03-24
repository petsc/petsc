const char help[] = "Test DMPlex implementation of DMAdaptLabel().\n\n";

#include <petscdm.h>
#include <petscdmplex.h>

int main(int argc, char **argv)
{
  DM             dm, dmAdapt;
  DMLabel        adaptLabel;
  PetscInt       cStart, cEnd;

  CHKERRQ(PetscInitialize(&argc, &argv, NULL, help));
  CHKERRQ(DMCreate(PETSC_COMM_WORLD, &dm));
  CHKERRQ(DMSetType(dm, DMPLEX));
  CHKERRQ(DMSetFromOptions(dm));
  CHKERRQ(PetscObjectSetName((PetscObject) dm, "Pre Adaptation Mesh"));
  CHKERRQ(DMViewFromOptions(dm, NULL, "-pre_adapt_dm_view"));

  CHKERRQ(DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd));
  CHKERRQ(DMLabelCreate(PETSC_COMM_SELF, "adapt", &adaptLabel));
  CHKERRQ(DMLabelSetDefaultValue(adaptLabel, DM_ADAPT_COARSEN));
  if (cEnd > cStart) CHKERRQ(DMLabelSetValue(adaptLabel, cStart, DM_ADAPT_REFINE));
  CHKERRQ(DMAdaptLabel(dm, adaptLabel, &dmAdapt));
  CHKERRQ(PetscObjectSetName((PetscObject) dmAdapt, "Post Adaptation Mesh"));
  CHKERRQ(DMViewFromOptions(dmAdapt, NULL, "-post_adapt_dm_view"));
  CHKERRQ(DMDestroy(&dmAdapt));
  CHKERRQ(DMLabelDestroy(&adaptLabel));
  CHKERRQ(DMDestroy(&dm));
  CHKERRQ(PetscFinalize());
  return 0;
}

/*TEST

  test:
    suffix: 2d
    requires: triangle !single
    args: -dm_plex_box_faces 3,3 -dm_coord_space 0 -pre_adapt_dm_view ascii::ascii_info -post_adapt_dm_view ascii::ascii_info
  test:
    suffix: 3d_tetgen
    requires: tetgen
    args: -dm_plex_dim 3 -dm_plex_box_faces 3,3,3 -dm_coord_space 0 -pre_adapt_dm_view ascii::ascii_info -post_adapt_dm_view ascii::ascii_info
  test:
    suffix: 3d_ctetgen
    requires: ctetgen !complex !single
    args: -dm_plex_dim 3 -dm_plex_box_faces 3,3,3 -dm_coord_space 0 -pre_adapt_dm_view ascii::ascii_info -post_adapt_dm_view ascii::ascii_info

TEST*/
