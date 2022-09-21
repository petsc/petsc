static char help[] = "Tests for point location\n\n";

#include <petscsf.h>
#include <petscdmplex.h>

static PetscErrorCode CreateMesh(MPI_Comm comm, DM *dm)
{
  PetscFunctionBeginUser;
  PetscCall(DMCreate(comm, dm));
  PetscCall(DMSetType(*dm, DMPLEX));
  PetscCall(DMSetFromOptions(*dm));
  PetscCall(DMViewFromOptions(*dm, NULL, "-dm_view"));
  PetscFunctionReturn(0);
}

static PetscErrorCode TestLocation(DM dm)
{
  Vec                points;
  PetscSF            cellSF = NULL;
  const PetscSFNode *cells;
  PetscScalar       *a;
  PetscInt           cdim, n;
  PetscInt           cStart, cEnd, c;

  PetscFunctionBeginUser;
  PetscCall(DMGetCoordinateDim(dm, &cdim));
  PetscCall(DMGetCoordinatesLocalSetUp(dm));
  PetscCall(DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd));
  /* Locate all centroids */
  PetscCall(VecCreateSeq(PETSC_COMM_SELF, (cEnd - cStart) * cdim, &points));
  PetscCall(VecSetBlockSize(points, cdim));
  PetscCall(VecGetArray(points, &a));
  for (c = cStart; c < cEnd; ++c) {
    PetscReal centroid[3];
    PetscInt  off = (c - cStart) * cdim, d;

    PetscCall(DMPlexComputeCellGeometryFVM(dm, c, NULL, centroid, NULL));
    for (d = 0; d < cdim; ++d) a[off + d] = centroid[d];
  }
  PetscCall(VecRestoreArray(points, &a));
  PetscCall(DMLocatePoints(dm, points, DM_POINTLOCATION_NONE, &cellSF));
  PetscCall(VecDestroy(&points));
  PetscCall(PetscSFGetGraph(cellSF, NULL, &n, NULL, &cells));
  if (n != (cEnd - cStart)) {
    for (c = 0; c < n; ++c) {
      if (cells[c].index != c + cStart) PetscCall(PetscPrintf(PETSC_COMM_SELF, "Could not locate centroid of cell %" PetscInt_FMT ", error %" PetscInt_FMT "\n", c + cStart, cells[c].index));
    }
    SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Located %" PetscInt_FMT " points instead of %" PetscInt_FMT, n, cEnd - cStart);
  }
  for (c = cStart; c < cEnd; ++c) PetscCheck(cells[c - cStart].index == c, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Could not locate centroid of cell %" PetscInt_FMT ", instead found %" PetscInt_FMT, c, cells[c - cStart].index);
  PetscCall(PetscSFDestroy(&cellSF));
  PetscFunctionReturn(0);
}

int main(int argc, char **argv)
{
  DM dm;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  PetscCall(CreateMesh(PETSC_COMM_WORLD, &dm));
  PetscCall(TestLocation(dm));
  PetscCall(DMDestroy(&dm));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

  testset:
    args: -dm_plex_dim 1 -dm_plex_box_faces 10

    test:
      suffix: seg

    test:
      suffix: seg_hash
      args: -dm_refine 2 -dm_plex_hash_location

  testset:
    args: -dm_plex_box_faces 5,5

    test:
      suffix: tri
      requires: triangle

    test:
      suffix: tri_hash
      requires: triangle
      args: -dm_refine 2 -dm_plex_hash_location

    test:
      suffix: quad
      args: -dm_plex_simplex 0

    test:
      suffix: quad_hash
      args: -dm_plex_simplex 0 -dm_refine 2 -dm_plex_hash_location

  testset:
    args: -dm_plex_dim 3 -dm_plex_box_faces 3,3,3

    test:
      suffix: tet
      requires: ctetgen

    test:
      suffix: tet_hash
      requires: ctetgen
      args: -dm_refine 1 -dm_plex_hash_location

    test:
      suffix: hex
      args: -dm_plex_simplex 0

    test:
      suffix: hex_hash
      args: -dm_plex_simplex 0 -dm_refine 1 -dm_plex_hash_location

TEST*/
