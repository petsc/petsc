static char help[] = "Tests for periodic mesh output\n\n";

#include <petscdmplex.h>

PetscErrorCode CheckMesh(DM dm)
{
  PetscReal detJ, J[9];
  PetscReal vol;
  PetscInt  dim, depth, cStart, cEnd, c;

  PetscFunctionBegin;
  PetscCall(DMGetDimension(dm, &dim));
  PetscCall(DMPlexGetDepth(dm, &depth));
  PetscCall(DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd));
  for (c = cStart; c < cEnd; ++c) {
    PetscCall(DMPlexComputeCellGeometryFEM(dm, c, NULL, NULL, J, NULL, &detJ));
    PetscCheck(detJ > 0.0, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Mesh cell %" PetscInt_FMT " is inverted, |J| = %g", c, (double)detJ);
    if (depth > 1) {
      PetscCall(DMPlexComputeCellGeometryFVM(dm, c, &vol, NULL, NULL));
      PetscCheck(vol > 0.0, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Mesh cell %" PetscInt_FMT " is inverted, vol = %g", c, (double)vol);
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode CreateMesh(MPI_Comm comm, DM *dm)
{
  PetscFunctionBegin;
  PetscCall(DMCreate(comm, dm));
  PetscCall(DMSetType(*dm, DMPLEX));
  PetscCall(DMSetFromOptions(*dm));
  PetscCall(DMViewFromOptions(*dm, NULL, "-dm_view"));
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv)
{
  DM dm;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  PetscCall(CreateMesh(PETSC_COMM_WORLD, &dm));
  PetscCall(CheckMesh(dm));
  PetscCall(DMDestroy(&dm));
  PetscCall(PetscFinalize());
  return 0;
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
  # This checks that the SF with extra root for periodic cut still checks
  test:
    suffix: 5_hdf5
    requires: hdf5
    nsize: 2
    args: -dm_plex_simplex 0 -dm_plex_box_faces 6,2,0 -dm_plex_box_bd periodic,none -dm_plex_periodic_cut -petscpartitioner_type simple -dm_view hdf5:mesh.h5
  test:
    suffix: 6
    nsize: 4
    args: -dm_plex_simplex 0 -dm_plex_box_faces 6,2,0 -dm_plex_box_bd periodic,none -dm_plex_periodic_cut -petscpartitioner_type simple -dm_view ::ascii_info_detail

TEST*/
