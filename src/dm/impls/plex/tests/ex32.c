static char help[] = "Tests for periodic mesh output\n\n";

#include <petscdmplex.h>

PetscErrorCode CheckMesh(DM dm)
{
  PetscReal      detJ, J[9], refVol = 1.0;
  PetscReal      vol;
  PetscInt       dim, depth, d, cStart, cEnd, c;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);
  ierr = DMPlexGetDepth(dm, &depth);CHKERRQ(ierr);
  for (d = 0; d < dim; ++d) {
    refVol *= 2.0;
  }
  ierr = DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd);CHKERRQ(ierr);
  for (c = cStart; c < cEnd; ++c) {
    ierr = DMPlexComputeCellGeometryFEM(dm, c, NULL, NULL, J, NULL, &detJ);CHKERRQ(ierr);
    PetscCheckFalse(detJ <= 0.0,PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Mesh cell %d is inverted, |J| = %g", c, detJ);
    if (depth > 1) {
      ierr = DMPlexComputeCellGeometryFVM(dm, c, &vol, NULL, NULL);CHKERRQ(ierr);
      PetscCheckFalse(vol <= 0.0,PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Mesh cell %d is inverted, vol = %g", c, vol);
    }
  }
  PetscFunctionReturn(0);
}

PetscErrorCode CreateMesh(MPI_Comm comm, DM *dm)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMCreate(comm, dm);CHKERRQ(ierr);
  ierr = DMSetType(*dm, DMPLEX);CHKERRQ(ierr);
  ierr = DMSetFromOptions(*dm);CHKERRQ(ierr);
  ierr = DMViewFromOptions(*dm, NULL, "-dm_view");CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

int main(int argc, char **argv)
{
  DM             dm;
  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc, &argv, NULL,help);if (ierr) return ierr;
  ierr = CreateMesh(PETSC_COMM_WORLD, &dm);CHKERRQ(ierr);
  ierr = CheckMesh(dm);CHKERRQ(ierr);
  ierr = DMDestroy(&dm);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

  test:
    suffix: 0
    args: -dm_plex_simplex 0 -dm_plex_box_faces 3,1,0 -dm_plex_box_bd periodic,none -dm_view ::ascii_info_detail
  test:
    suffix: 1
    nsize: 2
    args: -dm_plex_simplex 0 -dm_plex_box_faces 3,1,0 -dm_plex_box_bd periodic,none -petscpartitioner_type simple -dm_view ::ascii_info_detail
  test:
    suffix: 2
    nsize: 2
    args: -dm_plex_simplex 0 -dm_plex_box_faces 6,2,0 -dm_plex_box_bd periodic,none -petscpartitioner_type simple -dm_view ::ascii_info_detail
  test:
    suffix: 3
    nsize: 4
    args: -dm_plex_simplex 0 -dm_plex_box_faces 6,2,0 -dm_plex_box_bd periodic,none -petscpartitioner_type simple -dm_view ::ascii_info_detail
  test:
    suffix: 4
    nsize: 2
    args: -dm_plex_simplex 0 -dm_plex_box_faces 3,1,0 -dm_plex_box_bd periodic,none -dm_plex_periodic_cut -petscpartitioner_type simple -dm_view ::ascii_info_detail
  test:
    suffix: 5
    nsize: 2
    args: -dm_plex_simplex 0 -dm_plex_box_faces 6,2,0 -dm_plex_box_bd periodic,none -dm_plex_periodic_cut -petscpartitioner_type simple -dm_view ::ascii_info_detail
  test:
    suffix: 6
    nsize: 4
    args: -dm_plex_simplex 0 -dm_plex_box_faces 6,2,0 -dm_plex_box_bd periodic,none -dm_plex_periodic_cut -petscpartitioner_type simple -dm_view ::ascii_info_detail

TEST*/
