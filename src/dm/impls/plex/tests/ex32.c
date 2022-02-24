static char help[] = "Tests for periodic mesh output\n\n";

#include <petscdmplex.h>

PetscErrorCode CheckMesh(DM dm)
{
  PetscReal detJ, J[9];
  PetscReal vol;
  PetscInt  dim, depth, cStart, cEnd, c;

  PetscFunctionBegin;
  CHKERRQ(DMGetDimension(dm, &dim));
  CHKERRQ(DMPlexGetDepth(dm, &depth));
  CHKERRQ(DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd));
  for (c = cStart; c < cEnd; ++c) {
    CHKERRQ(DMPlexComputeCellGeometryFEM(dm, c, NULL, NULL, J, NULL, &detJ));
    PetscCheck(detJ > 0.0,PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Mesh cell %" PetscInt_FMT " is inverted, |J| = %g", c, detJ);
    if (depth > 1) {
      CHKERRQ(DMPlexComputeCellGeometryFVM(dm, c, &vol, NULL, NULL));
      PetscCheck(vol > 0.0,PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Mesh cell %" PetscInt_FMT " is inverted, vol = %g", c, vol);
    }
  }
  PetscFunctionReturn(0);
}

PetscErrorCode CreateMesh(MPI_Comm comm, DM *dm)
{
  PetscFunctionBegin;
  CHKERRQ(DMCreate(comm, dm));
  CHKERRQ(DMSetType(*dm, DMPLEX));
  CHKERRQ(DMSetFromOptions(*dm));
  CHKERRQ(DMViewFromOptions(*dm, NULL, "-dm_view"));
  PetscFunctionReturn(0);
}

int main(int argc, char **argv)
{
  DM             dm;
  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc, &argv, NULL,help);if (ierr) return ierr;
  CHKERRQ(CreateMesh(PETSC_COMM_WORLD, &dm));
  CHKERRQ(CheckMesh(dm));
  CHKERRQ(DMDestroy(&dm));
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
