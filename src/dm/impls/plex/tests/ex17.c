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
  PetscInt       dim;
  PetscInt       cStart, cEnd, c;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd);CHKERRQ(ierr);
  /* Locate all centroids */
  for (c = cStart; c < cEnd; ++c) {
    Vec                v;
    PetscSF            cellSF = NULL;
    const PetscSFNode *cells;
    PetscScalar       *a;
    PetscReal          centroid[3];
    PetscInt           n, d;

    ierr = DMPlexComputeCellGeometryFVM(dm, c, NULL, centroid, NULL);CHKERRQ(ierr);
    ierr = VecCreateSeq(PETSC_COMM_SELF, dim, &v);CHKERRQ(ierr);
    ierr = VecSetBlockSize(v, dim);CHKERRQ(ierr);
    ierr = VecGetArray(v, &a);CHKERRQ(ierr);
    for (d = 0; d < dim; ++d) a[d] = centroid[d];
    ierr = VecRestoreArray(v, &a);CHKERRQ(ierr);
    ierr = DMLocatePoints(dm, v, DM_POINTLOCATION_NONE, &cellSF);CHKERRQ(ierr);
    ierr = VecDestroy(&v);CHKERRQ(ierr);
    ierr = PetscSFGetGraph(cellSF,NULL,&n,NULL,&cells);CHKERRQ(ierr);
    if (n              != 1) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Cell %D: Found %d cells instead of 1", c, n);
    if (cells[0].index != c) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Could not locate centroid of cell %D, instead found %D", c, cells[0].index);
    ierr = PetscSFDestroy(&cellSF);CHKERRQ(ierr);
  }
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

  test:
    suffix: tri
    requires: triangle
    args: -dm_coord_space 0 -dm_plex_box_faces 5,5

  test:
    suffix: quad
    args: -dm_plex_simplex 0 -dm_plex_box_faces 5,5

  test:
    suffix: tet
    requires: ctetgen
    args: -dm_plex_dim 3 -dm_plex_box_faces 3,3,3

  test:
    suffix: hex
    args: -dm_plex_dim 3 -dm_plex_simplex 0 -dm_plex_box_faces 3,3,3

TEST*/
