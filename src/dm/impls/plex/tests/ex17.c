static char help[] = "Tests for point location\n\n";

#include <petscsf.h>
#include <petscdmplex.h>

static PetscErrorCode CreateMesh(MPI_Comm comm, DM *dm)
{
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = DMCreate(comm, dm);CHKERRQ(ierr);
  ierr = DMSetType(*dm, DMPLEX);CHKERRQ(ierr);
  ierr = DMSetFromOptions(*dm);CHKERRQ(ierr);
  ierr = DMViewFromOptions(*dm, NULL, "-dm_view");CHKERRQ(ierr);
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
  PetscErrorCode     ierr;

  PetscFunctionBeginUser;
  ierr = DMGetCoordinateDim(dm, &cdim);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd);CHKERRQ(ierr);
  /* Locate all centroids */
  ierr = VecCreateSeq(PETSC_COMM_SELF, (cEnd - cStart)*cdim, &points);CHKERRQ(ierr);
  ierr = VecSetBlockSize(points, cdim);CHKERRQ(ierr);
  ierr = VecGetArray(points, &a);CHKERRQ(ierr);
  for (c = cStart; c < cEnd; ++c) {
    PetscReal          centroid[3];
    PetscInt           off = (c - cStart)*cdim, d;

    ierr = DMPlexComputeCellGeometryFVM(dm, c, NULL, centroid, NULL);CHKERRQ(ierr);
    for (d = 0; d < cdim; ++d) a[off+d] = centroid[d];
  }
  ierr = VecRestoreArray(points, &a);CHKERRQ(ierr);
  ierr = DMLocatePoints(dm, points, DM_POINTLOCATION_NONE, &cellSF);CHKERRQ(ierr);
  ierr = VecDestroy(&points);CHKERRQ(ierr);
  ierr = PetscSFGetGraph(cellSF, NULL, &n, NULL, &cells);CHKERRQ(ierr);
  if (n != (cEnd - cStart)) {
    for (c = 0; c < n; ++c) {
      if (cells[c].index != c+cStart) {ierr = PetscPrintf(PETSC_COMM_SELF, "Could not locate centroid of cell %D, error %D\n", c+cStart, cells[c].index);CHKERRQ(ierr);}
    }
    SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Located %D points instead of %D", n, cEnd - cStart);
  }
  for (c = cStart; c < cEnd; ++c) {
    if (cells[c - cStart].index != c) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Could not locate centroid of cell %D, instead found %D", c, cells[c - cStart].index);
  }
  ierr = PetscSFDestroy(&cellSF);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

int main(int argc, char **argv)
{
  DM             dm;
  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc, &argv, NULL, help);if (ierr) return ierr;
  ierr = CreateMesh(PETSC_COMM_WORLD, &dm);CHKERRQ(ierr);
  ierr = TestLocation(dm);CHKERRQ(ierr);
  ierr = DMDestroy(&dm);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

  test:
    suffix: seg
    args: -dm_plex_dim 1 -dm_plex_box_faces 10

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
