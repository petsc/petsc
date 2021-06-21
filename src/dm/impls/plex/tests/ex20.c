const char help[] = "Test DMPlex implementation of DMAdaptLabel().\n\n";

#include <petscdm.h>
#include <petscdmplex.h>

int main(int argc, char **argv)
{
  DM             dm, dmAdapt;
  DMLabel        adaptLabel;
  PetscInt       cStart, cEnd;
  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc, &argv, NULL, help);if (ierr) return ierr;
  ierr = DMCreate(PETSC_COMM_WORLD, &dm);CHKERRQ(ierr);
  ierr = DMSetType(dm, DMPLEX);CHKERRQ(ierr);
  ierr = DMSetFromOptions(dm);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) dm, "Pre Adaptation Mesh");CHKERRQ(ierr);
  ierr = DMViewFromOptions(dm, NULL, "-pre_adapt_dm_view");CHKERRQ(ierr);

  ierr = DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd);CHKERRQ(ierr);
  ierr = DMLabelCreate(PETSC_COMM_SELF, "adapt", &adaptLabel);CHKERRQ(ierr);
  ierr = DMLabelSetDefaultValue(adaptLabel, DM_ADAPT_COARSEN);CHKERRQ(ierr);
  if (cEnd > cStart) {ierr = DMLabelSetValue(adaptLabel, cStart, DM_ADAPT_REFINE);CHKERRQ(ierr);}
  ierr = DMAdaptLabel(dm, adaptLabel, &dmAdapt);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) dmAdapt, "Post Adaptation Mesh");CHKERRQ(ierr);
  ierr = DMViewFromOptions(dmAdapt, NULL, "-post_adapt_dm_view");CHKERRQ(ierr);
  ierr = DMDestroy(&dmAdapt);CHKERRQ(ierr);
  ierr = DMLabelDestroy(&adaptLabel);CHKERRQ(ierr);
  ierr = DMDestroy(&dm);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

  test:
    suffix: 2d
    requires: triangle !single
    args: -dm_plex_box_faces 3,3 -dm_coord_space 0 -pre_adapt_dm_view ascii::ascii_info_detail -post_adapt_dm_view ascii::ascii_info_detail
  test:
    suffix: 3d_tetgen
    requires: tetgen complex
    args: -dm_plex_dim 3 -dm_plex_box_faces 3,3,3 -dm_coord_space 0 -pre_adapt_dm_view ascii::ascii_info_detail -post_adapt_dm_view ascii::ascii_info_detail
  test:
    suffix: 3d_ctetgen
    requires: ctetgen !complex !single
    args: -dm_plex_dim 3 -dm_plex_box_faces 3,3,3 -dm_coord_space 0 -pre_adapt_dm_view ascii::ascii_info_detail -post_adapt_dm_view ascii::ascii_info_detail

TEST*/
