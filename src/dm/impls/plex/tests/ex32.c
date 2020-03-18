static char help[] = "Tests for periodic mesh output\n\n";

#include <petscdmplex.h>

typedef struct {
  DM        dm;
  PetscInt  dim;                          /* The topological mesh dimension */
  PetscBool cellSimplex;                  /* Use simplices or hexes */
  PetscInt  faces[3];                     /* Faces per direction */
  PetscBool isPeriodic;                   /* Flag for periodic mesh */
  DMBoundaryType periodicity[3];          /* Periodicity per direction */
  char      filename[PETSC_MAX_PATH_LEN]; /* Import mesh from file */
} AppCtx;

PetscErrorCode ProcessOptions(MPI_Comm comm, AppCtx *options)
{
  PetscInt       n;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  options->dim            = 2;
  options->cellSimplex    = PETSC_TRUE;
  options->faces[0]       = 1;
  options->faces[1]       = 1;
  options->faces[2]       = 1;
  options->periodicity[0] = DM_BOUNDARY_NONE;
  options->periodicity[1] = DM_BOUNDARY_NONE;
  options->periodicity[2] = DM_BOUNDARY_NONE;
  options->filename[0]    = '\0';

  ierr = PetscOptionsBegin(comm, "", "Meshing Interpolation Test Options", "DMPLEX");CHKERRQ(ierr);
  ierr = PetscOptionsRangeInt("-dim", "The topological mesh dimension", "ex32.c", options->dim, &options->dim, NULL,1,3);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-cell_simplex", "Use simplices if true, otherwise hexes", "ex32.c", options->cellSimplex, &options->cellSimplex, NULL);CHKERRQ(ierr);
  n    = 3;
  ierr = PetscOptionsIntArray("-faces", "Faces per direction", "ex32.c", options->faces, &n, NULL);CHKERRQ(ierr);
  n    = 3;
  ierr = PetscOptionsEnumArray("-periodicity", "Periodicity per direction", "ex32.c", DMBoundaryTypes, (PetscEnum *) options->periodicity, &n, &options->isPeriodic);CHKERRQ(ierr);
  ierr = PetscOptionsString("-filename", "The mesh file", "ex32.c", options->filename, options->filename, PETSC_MAX_PATH_LEN, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();
  PetscFunctionReturn(0);
}

PetscErrorCode CheckMesh(DM dm, AppCtx *user)
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
    if (detJ <= 0.0) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Mesh cell %d is inverted, |J| = %g", c, detJ);
    if (depth > 1) {
      ierr = DMPlexComputeCellGeometryFVM(dm, c, &vol, NULL, NULL);CHKERRQ(ierr);
      if (vol <= 0.0) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Mesh cell %d is inverted, vol = %g", c, vol);
    }
  }
  PetscFunctionReturn(0);
}

PetscErrorCode CompareCones(DM dm, DM idm)
{
  PetscInt       cStart, cEnd, c, vStart, vEnd, v;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd);CHKERRQ(ierr);
  ierr = DMPlexGetDepthStratum(dm, 0, &vStart, &vEnd);CHKERRQ(ierr);
  for (c = cStart; c < cEnd; ++c) {
    const PetscInt *cone;
    PetscInt       *points = NULL, numPoints, p, numVertices = 0, coneSize;

    ierr = DMPlexGetCone(dm, c, &cone);CHKERRQ(ierr);
    ierr = DMPlexGetConeSize(dm, c, &coneSize);CHKERRQ(ierr);
    ierr = DMPlexGetTransitiveClosure(idm, c, PETSC_TRUE, &numPoints, &points);CHKERRQ(ierr);
    for (p = 0; p < numPoints*2; p += 2) {
      const PetscInt point = points[p];
      if ((point >= vStart) && (point < vEnd)) points[numVertices++] = point;
    }
    if (numVertices != coneSize) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "In cell %d, cone size %d != %d vertices in closure", c, coneSize, numVertices);
    for (v = 0; v < numVertices; ++v) {
      if (cone[v] != points[v]) SETERRQ4(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "In cell %d, cone point %d is %d != %d vertex in closure", c, v, cone[v], points[v]);
    }
    ierr = DMPlexRestoreTransitiveClosure(idm, c, PETSC_TRUE, &numPoints, &points);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode CreateMesh(MPI_Comm comm, AppCtx *user, DM *dm)
{
  PetscInt       dim         = user->dim;
  PetscBool      cellSimplex = user->cellSimplex;
  const char    *filename    = user->filename;
  size_t         len;
  PetscMPIInt    rank;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MPI_Comm_rank(comm, &rank);CHKERRQ(ierr);
  ierr = PetscStrlen(filename, &len);CHKERRQ(ierr);
  if (len) {
    ierr = DMPlexCreateFromFile(comm, filename, PETSC_FALSE, dm);CHKERRQ(ierr);
    ierr = DMGetDimension(*dm, &dim);CHKERRQ(ierr);
  } else {
    PetscReal L[3] = {1.0, 1.0, 1.0};
    PetscReal maxCell[3];
    PetscInt  d;

    for (d = 0; d < dim; ++d) {maxCell[d] = (1.0/user->faces[d])*1.1;}
    ierr = DMPlexCreateBoxMesh(comm, dim, cellSimplex, user->faces, NULL, NULL, user->periodicity, PETSC_TRUE, dm);CHKERRQ(ierr);
    ierr = DMSetPeriodicity(*dm, user->isPeriodic, maxCell, L, user->periodicity);CHKERRQ(ierr);
  }
  {
    DM               pdm = NULL;
    PetscPartitioner part;

    ierr = DMPlexGetPartitioner(*dm, &part);CHKERRQ(ierr);
    ierr = PetscPartitionerSetFromOptions(part);CHKERRQ(ierr);
    /* Distribute mesh over processes */
    ierr = DMPlexDistribute(*dm, 0, NULL, &pdm);CHKERRQ(ierr);
    if (pdm) {
      ierr = DMDestroy(dm);CHKERRQ(ierr);
      *dm  = pdm;
    }
  }
  ierr = DMLocalizeCoordinates(*dm);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) *dm, "Mesh");CHKERRQ(ierr);
  ierr = DMViewFromOptions(*dm, NULL, "-dm_view");CHKERRQ(ierr);
  user->dm = *dm;
  PetscFunctionReturn(0);
}

int main(int argc, char **argv)
{
  AppCtx         user;                 /* user-defined work context */
  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc, &argv, NULL,help);if (ierr) return ierr;
  ierr = ProcessOptions(PETSC_COMM_WORLD, &user);CHKERRQ(ierr);
  ierr = CreateMesh(PETSC_COMM_WORLD, &user, &user.dm);CHKERRQ(ierr);
  ierr = CheckMesh(user.dm, &user);CHKERRQ(ierr);
  ierr = DMDestroy(&user.dm);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

  test:
    suffix: 0
    args: -dim 2 -cell_simplex 0 -faces 3,1,0 -periodicity periodic,none,none -dm_view ::ascii_info_detail
  test:
    suffix: 1
    nsize: 2
    args: -dim 2 -cell_simplex 0 -faces 3,1,0 -periodicity periodic,none,none -petscpartitioner_type simple -dm_view ::ascii_info_detail
  test:
    suffix: 2
    nsize: 2
    args: -dim 2 -cell_simplex 0 -faces 6,2,0 -periodicity periodic,none,none -petscpartitioner_type simple -dm_view ::ascii_info_detail
  test:
    suffix: 3
    nsize: 4
    args: -dim 2 -cell_simplex 0 -faces 6,2,0 -periodicity periodic,none,none -petscpartitioner_type simple -dm_view ::ascii_info_detail
  test:
    suffix: 4
    nsize: 2
    args: -dim 2 -cell_simplex 0 -faces 3,1,0 -periodicity periodic,none,none -dm_plex_periodic_cut -petscpartitioner_type simple -dm_view ::ascii_info_detail
  test:
    suffix: 5
    nsize: 2
    args: -dim 2 -cell_simplex 0 -faces 6,2,0 -periodicity periodic,none,none -dm_plex_periodic_cut -petscpartitioner_type simple -dm_view ::ascii_info_detail
  test:
    suffix: 6
    nsize: 4
    args: -dim 2 -cell_simplex 0 -faces 6,2,0 -periodicity periodic,none,none -dm_plex_periodic_cut -petscpartitioner_type simple -dm_view ::ascii_info_detail

TEST*/
