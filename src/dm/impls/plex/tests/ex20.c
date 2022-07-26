const char help[] = "Test DMPlex implementation of DMAdaptLabel().\n\n";

#include <petscdm.h>
#include <petscdmplex.h>

int main(int argc, char **argv)
{
  DM             dm, dmAdapt;
  DMLabel        adaptLabel;
  PetscInt       cStart, cEnd;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  PetscCall(DMCreate(PETSC_COMM_WORLD, &dm));
  PetscCall(DMSetType(dm, DMPLEX));
  PetscCall(DMSetFromOptions(dm));
  PetscCall(PetscObjectSetName((PetscObject) dm, "Pre Adaptation Mesh"));
  PetscCall(DMViewFromOptions(dm, NULL, "-pre_adapt_dm_view"));

  PetscCall(DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd));
  PetscCall(DMLabelCreate(PETSC_COMM_SELF, "adapt", &adaptLabel));
  PetscCall(DMLabelSetDefaultValue(adaptLabel, DM_ADAPT_COARSEN));
  if (cEnd > cStart) PetscCall(DMLabelSetValue(adaptLabel, cStart, DM_ADAPT_REFINE));
  PetscCall(DMAdaptLabel(dm, adaptLabel, &dmAdapt));
  PetscCall(PetscObjectSetName((PetscObject) dmAdapt, "Post Adaptation Mesh"));
  PetscCall(DMViewFromOptions(dmAdapt, NULL, "-post_adapt_dm_view"));
  PetscCall(DMDestroy(&dmAdapt));
  PetscCall(DMLabelDestroy(&adaptLabel));
  PetscCall(DMDestroy(&dm));
  PetscCall(PetscFinalize());
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
