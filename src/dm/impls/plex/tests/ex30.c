const char help[] = "Test memory allocation in DMPlex refinement.\n\n";

#include <petsc.h>

int main(int argc, char **argv)
{
  DM             dm;
  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc, &argv, NULL, help);if (ierr) return ierr;
  CHKERRQ(DMCreate(PETSC_COMM_WORLD, &dm));
  CHKERRQ(PetscObjectSetName((PetscObject) dm, "BaryDM"));
  CHKERRQ(DMSetType(dm, DMPLEX));
  CHKERRQ(DMSetFromOptions(dm));
  CHKERRQ(DMViewFromOptions(dm, NULL, "-dm_view"));
  //CHKERRQ(DMPlexSetRefinementUniform(dm, PETSC_TRUE));
  //CHKERRQ(DMRefine(dm, comm, &rdm));
  //CHKERRQ(DMPlexConvertOldOrientations_Internal(dm));
  CHKERRQ(PetscObjectSetName((PetscObject) dm, "RefinedDM"));
  CHKERRQ(PetscObjectSetOptionsPrefix((PetscObject) dm, "ref_"));
  CHKERRQ(DMSetFromOptions(dm));
  CHKERRQ(DMViewFromOptions(dm, NULL, "-dm_view"));
  CHKERRQ(DMDestroy(&dm));
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

  test:
    requires: hdf5 double !complex !defined(PETSC_USE_64BIT_INDICES)
    args: -dm_plex_filename ${wPETSC_DIR}/share/petsc/datafiles/meshes/barycentricallyrefinedcube.h5 -dm_view ascii::ASCII_INFO_DETAIL -ref_dm_refine 1 -ref_dm_view ascii::ASCII_INFO_DETAIL

TEST*/
