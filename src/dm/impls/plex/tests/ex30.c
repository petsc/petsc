const char help[] = "Test memory allocation in DMPlex refinement.\n\n";

#include <petsc.h>

int main(int argc, char **argv)
{
  DM             dm;
  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc, &argv, NULL, help);if (ierr) return ierr;
  ierr = DMCreate(PETSC_COMM_WORLD, &dm);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) dm, "BaryDM");CHKERRQ(ierr);
  ierr = DMSetType(dm, DMPLEX);CHKERRQ(ierr);
  ierr = DMSetFromOptions(dm);CHKERRQ(ierr);
  ierr = DMViewFromOptions(dm, NULL, "-dm_view");CHKERRQ(ierr);
  //ierr = DMPlexSetRefinementUniform(dm, PETSC_TRUE);CHKERRQ(ierr);
  //ierr = DMRefine(dm, comm, &rdm);CHKERRQ(ierr);
  //ierr = DMPlexConvertOldOrientations_Internal(dm);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) dm, "RefinedDM");CHKERRQ(ierr);
  ierr = PetscObjectSetOptionsPrefix((PetscObject) dm, "ref_");CHKERRQ(ierr);
  ierr = DMSetFromOptions(dm);CHKERRQ(ierr);
  ierr = DMViewFromOptions(dm, NULL, "-dm_view");CHKERRQ(ierr);
  ierr = DMDestroy(&dm);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

  test:
    requires: hdf5 double !complex !define(PETSC_USE_64BIT_INDICES)
    args: -dm_plex_filename ${wPETSC_DIR}/share/petsc/datafiles/meshes/barycentricallyrefinedcube.h5 -dm_view ascii::ASCII_INFO_DETAIL -ref_dm_refine 1 -ref_dm_view ascii::ASCII_INFO_DETAIL

TEST*/
