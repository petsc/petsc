static char help[] = "Tests for point location\n\n";

#include <petscsf.h>
#include <petscdmplex.h>

static PetscErrorCode CreateMesh(MPI_Comm comm, DM *dm)
{
  PetscFunctionBeginUser;
  CHKERRQ(DMCreate(comm, dm));
  CHKERRQ(DMSetType(*dm, DMPLEX));
  CHKERRQ(DMSetFromOptions(*dm));
  CHKERRQ(DMViewFromOptions(*dm, NULL, "-dm_view"));
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
  CHKERRQ(DMGetCoordinateDim(dm, &cdim));
  CHKERRQ(DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd));
  /* Locate all centroids */
  CHKERRQ(VecCreateSeq(PETSC_COMM_SELF, (cEnd - cStart)*cdim, &points));
  CHKERRQ(VecSetBlockSize(points, cdim));
  CHKERRQ(VecGetArray(points, &a));
  for (c = cStart; c < cEnd; ++c) {
    PetscReal          centroid[3];
    PetscInt           off = (c - cStart)*cdim, d;

    CHKERRQ(DMPlexComputeCellGeometryFVM(dm, c, NULL, centroid, NULL));
    for (d = 0; d < cdim; ++d) a[off+d] = centroid[d];
  }
  CHKERRQ(VecRestoreArray(points, &a));
  CHKERRQ(DMLocatePoints(dm, points, DM_POINTLOCATION_NONE, &cellSF));
  CHKERRQ(VecDestroy(&points));
  CHKERRQ(PetscSFGetGraph(cellSF, NULL, &n, NULL, &cells));
  if (n != (cEnd - cStart)) {
    for (c = 0; c < n; ++c) {
      if (cells[c].index != c+cStart) CHKERRQ(PetscPrintf(PETSC_COMM_SELF, "Could not locate centroid of cell %D, error %D\n", c+cStart, cells[c].index));
    }
    SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Located %D points instead of %D", n, cEnd - cStart);
  }
  for (c = cStart; c < cEnd; ++c) {
    PetscCheckFalse(cells[c - cStart].index != c,PETSC_COMM_SELF, PETSC_ERR_PLIB, "Could not locate centroid of cell %D, instead found %D", c, cells[c - cStart].index);
  }
  CHKERRQ(PetscSFDestroy(&cellSF));
  PetscFunctionReturn(0);
}

int main(int argc, char **argv)
{
  DM             dm;

  CHKERRQ(PetscInitialize(&argc, &argv, NULL, help));
  CHKERRQ(CreateMesh(PETSC_COMM_WORLD, &dm));
  CHKERRQ(TestLocation(dm));
  CHKERRQ(DMDestroy(&dm));
  CHKERRQ(PetscFinalize());
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
