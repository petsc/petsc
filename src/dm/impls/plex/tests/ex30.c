const char help[] = "Test memory allocation in DMPlex refinement.\n\n";

#include <petsc.h>

int main(int argc, char **argv)
{
  PetscErrorCode ierr;
  DM             dm, rdm;
  PetscViewer    vwr;
  PetscBool      flg;
  char           datafile[PETSC_MAX_PATH_LEN];
  MPI_Comm       comm;

  ierr = PetscInitialize(&argc, &argv, NULL, help);if (ierr) return ierr;
  comm = PETSC_COMM_WORLD;
  ierr = PetscViewerCreate(comm, &vwr);CHKERRQ(ierr);
  ierr = PetscViewerSetType(vwr, PETSCVIEWERHDF5);CHKERRQ(ierr);
  ierr = PetscViewerFileSetMode(vwr, FILE_MODE_READ);CHKERRQ(ierr);
  ierr = PetscOptionsGetString(NULL, NULL, "-f", datafile, sizeof(datafile), &flg);CHKERRQ(ierr);
  if (!flg) SETERRQ(comm, PETSC_ERR_ARG_WRONG, "Must provide meshfile");
  ierr = PetscViewerFileSetName(vwr, datafile);CHKERRQ(ierr);
  ierr = DMCreate(comm, &dm);CHKERRQ(ierr);
  ierr = DMSetType(dm, DMPLEX);CHKERRQ(ierr);
  ierr = DMLoad(dm, vwr);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&vwr);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject)dm, "BaryDM");CHKERRQ(ierr);
  ierr = DMViewFromOptions(dm, NULL, "-dm_view");CHKERRQ(ierr);
  ierr = DMPlexSetRefinementUniform(dm, PETSC_TRUE);CHKERRQ(ierr);
  ierr = DMRefine(dm, comm, &rdm);CHKERRQ(ierr);
  ierr = DMDestroy(&dm);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject)rdm, "RefinedDM");CHKERRQ(ierr);
  ierr = DMViewFromOptions(rdm, NULL, "-refined_dm_view");CHKERRQ(ierr);
  ierr = DMDestroy(&rdm);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

  test:
    requires: hdf5 double !complex !define(PETSC_USE_64BIT_INDICES)
    args: -f ${wPETSC_DIR}/share/petsc/datafiles/meshes/barycentricallyrefinedcube.h5 -dm_view ascii::ASCII_INFO_DETAIL -refined_dm_view ascii::ASCII_INFO_DETAIL

TEST*/
