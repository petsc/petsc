static char help[] = "Interpolation Tests for Plex\n\n";

#include <petscsnes.h>
#include <petscdmplex.h>
#include <petscdmda.h>
#include <petscds.h>

typedef enum {CENTROID, GRID, GRID_REPLICATED} PointType;

typedef struct {
  PetscInt      dim;                          /* The topological mesh dimension */
  PetscBool     cellSimplex;                  /* Use simplices or hexes */
  char          filename[PETSC_MAX_PATH_LEN]; /* Import mesh from file */
  PointType     pointType;                    /* Point generation mechanism */
} AppCtx;

static PetscInt Nc = 3;

static PetscErrorCode linear(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nf, PetscScalar *u, void *ctx)
{
  PetscInt d, c;

  for (c = 0; c < Nc; ++c) {
    u[c] = 0.0;
    for (d = 0; d < dim; ++d) u[c] += x[d];
  }
  return 0;
}

static PetscErrorCode ProcessOptions(MPI_Comm comm, AppCtx *options)
{
  const char    *pointTypes[3] = {"centroid", "grid", "grid_replicated"};
  PetscInt       pt;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  options->dim           = 3;
  options->cellSimplex   = PETSC_TRUE;
  options->filename[0]   = '\0';
  options->pointType     = CENTROID;

  ierr = PetscOptionsBegin(comm, "", "Interpolation Options", "DMPLEX");CHKERRQ(ierr);
  ierr = PetscOptionsInt("-dim", "The topological mesh dimension", "ex2.c", options->dim, &options->dim, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-cell_simplex", "Use simplices if true, otherwise hexes", "ex2.c", options->cellSimplex, &options->cellSimplex, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsString("-filename", "The mesh file", "ex2.c", options->filename, options->filename, sizeof(options->filename), NULL);CHKERRQ(ierr);
  pt   = options->pointType;
  ierr = PetscOptionsEList("-point_type", "The point type", "ex2.c", pointTypes, 3, pointTypes[options->pointType], &pt, NULL);CHKERRQ(ierr);
  options->pointType = (PointType) pt;
  ierr = PetscOptionsEnd();

  PetscFunctionReturn(0);
}

static PetscErrorCode CreateMesh(MPI_Comm comm, AppCtx *ctx, DM *dm)
{
  PetscInt       dim         = ctx->dim;
  PetscBool      cellSimplex = ctx->cellSimplex;
  const char    *filename    = ctx->filename;
  const PetscInt cells[3]    = {1, 1, 1};
  size_t         len;
  PetscMPIInt    rank, size;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MPI_Comm_rank(comm, &rank);CHKERRMPI(ierr);
  ierr = MPI_Comm_size(comm, &size);CHKERRMPI(ierr);
  ierr = PetscStrlen(filename, &len);CHKERRQ(ierr);
  if (len) {ierr = DMPlexCreateFromFile(comm, filename, PETSC_TRUE, dm);CHKERRQ(ierr);}
  else     {ierr = DMPlexCreateBoxMesh(comm, dim, cellSimplex, cells, NULL, NULL, NULL, PETSC_TRUE, dm);CHKERRQ(ierr);}
  {
    DM               distributedMesh = NULL;
    PetscPartitioner part;

    ierr = DMPlexGetPartitioner(*dm, &part);CHKERRQ(ierr);
    ierr = PetscPartitionerSetFromOptions(part);CHKERRQ(ierr);

    /* Distribute mesh over processes */
    ierr = DMPlexDistribute(*dm, 0, NULL, &distributedMesh);CHKERRQ(ierr);
    if (distributedMesh) {
      ierr = DMDestroy(dm);CHKERRQ(ierr);
      *dm  = distributedMesh;
    }
  }
  ierr = PetscObjectSetName((PetscObject) *dm, "Mesh");CHKERRQ(ierr);
  ierr = DMSetFromOptions(*dm);CHKERRQ(ierr);
  ierr = DMViewFromOptions(*dm, NULL, "-dm_view");CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode CreatePoints_Centroid(DM dm, PetscInt *Np, PetscReal **pcoords, PetscBool *pointsAllProcs, AppCtx *ctx)
{
  PetscSection   coordSection;
  Vec            coordsLocal;
  PetscInt       spaceDim, p;
  PetscMPIInt    rank;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD, &rank);CHKERRMPI(ierr);
  ierr = DMGetCoordinatesLocal(dm, &coordsLocal);CHKERRQ(ierr);
  ierr = DMGetCoordinateSection(dm, &coordSection);CHKERRQ(ierr);
  ierr = DMGetCoordinateDim(dm, &spaceDim);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm, 0, NULL, Np);CHKERRQ(ierr);
  ierr = PetscCalloc1(*Np * spaceDim, pcoords);CHKERRQ(ierr);
  for (p = 0; p < *Np; ++p) {
    PetscScalar *coords = NULL;
    PetscInt     size, num, n, d;

    ierr = DMPlexVecGetClosure(dm, coordSection, coordsLocal, p, &size, &coords);CHKERRQ(ierr);
    num  = size/spaceDim;
    for (n = 0; n < num; ++n) {
      for (d = 0; d < spaceDim; ++d) (*pcoords)[p*spaceDim+d] += PetscRealPart(coords[n*spaceDim+d]) / num;
    }
    ierr = PetscSynchronizedPrintf(PETSC_COMM_WORLD, "[%d]Point %D (", rank, p);CHKERRQ(ierr);
    for (d = 0; d < spaceDim; ++d) {
      ierr = PetscSynchronizedPrintf(PETSC_COMM_WORLD, "%g", (double)(*pcoords)[p*spaceDim+d]);CHKERRQ(ierr);
      if (d < spaceDim-1) {ierr = PetscSynchronizedPrintf(PETSC_COMM_WORLD, ", ");CHKERRQ(ierr);}
    }
    ierr = PetscSynchronizedPrintf(PETSC_COMM_WORLD, ")\n");CHKERRQ(ierr);
    ierr = DMPlexVecRestoreClosure(dm, coordSection, coordsLocal, p, &num, &coords);CHKERRQ(ierr);
  }
  ierr = PetscSynchronizedFlush(PETSC_COMM_WORLD, NULL);CHKERRQ(ierr);
  *pointsAllProcs = PETSC_FALSE;
  PetscFunctionReturn(0);
}

static PetscErrorCode CreatePoints_Grid(DM dm, PetscInt *Np, PetscReal **pcoords, PetscBool *pointsAllProcs, AppCtx *ctx)
{
  DM             da;
  DMDALocalInfo  info;
  PetscInt       N = 3, n = 0, spaceDim, i, j, k, *ind, d;
  PetscReal      *h;
  PetscMPIInt    rank;
  PetscErrorCode ierr;

  ierr = MPI_Comm_rank(PETSC_COMM_WORLD, &rank);CHKERRMPI(ierr);
  ierr = DMGetCoordinateDim(dm, &spaceDim);CHKERRQ(ierr);
  ierr = PetscCalloc1(spaceDim,&ind);CHKERRQ(ierr);
  ierr = PetscCalloc1(spaceDim,&h);CHKERRQ(ierr);
  h[0] = 1.0/(N-1); h[1] = 1.0/(N-1); h[2] = 1.0/(N-1);
  ierr = DMDACreate(PetscObjectComm((PetscObject) dm), &da);CHKERRQ(ierr);
  ierr = DMSetDimension(da, ctx->dim);CHKERRQ(ierr);
  ierr = DMDASetSizes(da, N, N, N);CHKERRQ(ierr);
  ierr = DMDASetDof(da, 1);CHKERRQ(ierr);
  ierr = DMDASetStencilWidth(da, 1);CHKERRQ(ierr);
  ierr = DMSetUp(da);CHKERRQ(ierr);
  ierr = DMDASetUniformCoordinates(da, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0);CHKERRQ(ierr);
  ierr = DMDAGetLocalInfo(da, &info);CHKERRQ(ierr);
  *Np  = info.xm * info.ym * info.zm;
  ierr = PetscCalloc1(*Np * spaceDim, pcoords);CHKERRQ(ierr);
  for (k = info.zs; k < info.zs + info.zm; ++k) {
    ind[2] = k;
    for (j = info.ys; j < info.ys + info.ym; ++j) {
      ind[1] = j;
      for (i = info.xs; i < info.xs + info.xm; ++i, ++n) {
        ind[0] = i;

        for (d = 0; d < spaceDim; ++d) (*pcoords)[n*spaceDim+d] = ind[d]*h[d];
        ierr = PetscSynchronizedPrintf(PETSC_COMM_WORLD, "[%d]Point %D (", rank, n);CHKERRQ(ierr);
        for (d = 0; d < spaceDim; ++d) {
          ierr = PetscSynchronizedPrintf(PETSC_COMM_WORLD, "%g", (double)(*pcoords)[n*spaceDim+d]);CHKERRQ(ierr);
          if (d < spaceDim-1) {ierr = PetscSynchronizedPrintf(PETSC_COMM_WORLD, ", ");CHKERRQ(ierr);}
        }
        ierr = PetscSynchronizedPrintf(PETSC_COMM_WORLD, ")\n");CHKERRQ(ierr);
      }
    }
  }
  ierr = DMDestroy(&da);CHKERRQ(ierr);
  ierr = PetscSynchronizedFlush(PETSC_COMM_WORLD, NULL);CHKERRQ(ierr);
  ierr = PetscFree(ind);CHKERRQ(ierr);
  ierr = PetscFree(h);CHKERRQ(ierr);
  *pointsAllProcs = PETSC_FALSE;
  PetscFunctionReturn(0);
}

static PetscErrorCode CreatePoints_GridReplicated(DM dm, PetscInt *Np, PetscReal **pcoords, PetscBool *pointsAllProcs, AppCtx *ctx)
{
  PetscInt       N = 3, n = 0, spaceDim, i, j, k, *ind, d;
  PetscReal      *h;
  PetscMPIInt    rank;
  PetscErrorCode ierr;

  ierr = MPI_Comm_rank(PETSC_COMM_WORLD, &rank);CHKERRMPI(ierr);
  ierr = DMGetCoordinateDim(dm, &spaceDim);CHKERRQ(ierr);
  ierr = PetscCalloc1(spaceDim,&ind);CHKERRQ(ierr);
  ierr = PetscCalloc1(spaceDim,&h);CHKERRQ(ierr);
  h[0] = 1.0/(N-1); h[1] = 1.0/(N-1); h[2] = 1.0/(N-1);
  *Np  = N * (ctx->dim > 1 ? N : 1) * (ctx->dim > 2 ? N : 1);
  ierr = PetscCalloc1(*Np * spaceDim, pcoords);CHKERRQ(ierr);
  for (k = 0; k < N; ++k) {
    ind[2] = k;
    for (j = 0; j < N; ++j) {
      ind[1] = j;
      for (i = 0; i < N; ++i, ++n) {
        ind[0] = i;

        for (d = 0; d < spaceDim; ++d) (*pcoords)[n*spaceDim+d] = ind[d]*h[d];
        ierr = PetscSynchronizedPrintf(PETSC_COMM_WORLD, "[%d]Point %D (", rank, n);CHKERRQ(ierr);
        for (d = 0; d < spaceDim; ++d) {
          ierr = PetscSynchronizedPrintf(PETSC_COMM_WORLD, "%g", (double)(*pcoords)[n*spaceDim+d]);CHKERRQ(ierr);
          if (d < spaceDim-1) {ierr = PetscSynchronizedPrintf(PETSC_COMM_WORLD, ", ");CHKERRQ(ierr);}
        }
        ierr = PetscSynchronizedPrintf(PETSC_COMM_WORLD, ")\n");CHKERRQ(ierr);
      }
    }
  }
  ierr = PetscSynchronizedFlush(PETSC_COMM_WORLD, NULL);CHKERRQ(ierr);
  *pointsAllProcs = PETSC_TRUE;
  ierr = PetscFree(ind);CHKERRQ(ierr);
  ierr = PetscFree(h);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode CreatePoints(DM dm, PetscInt *Np, PetscReal **pcoords, PetscBool *pointsAllProcs, AppCtx *ctx)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  *pointsAllProcs = PETSC_FALSE;
  switch (ctx->pointType) {
  case CENTROID:        ierr = CreatePoints_Centroid(dm, Np, pcoords, pointsAllProcs, ctx);CHKERRQ(ierr);break;
  case GRID:            ierr = CreatePoints_Grid(dm, Np, pcoords, pointsAllProcs, ctx);CHKERRQ(ierr);break;
  case GRID_REPLICATED: ierr = CreatePoints_GridReplicated(dm, Np, pcoords, pointsAllProcs, ctx);CHKERRQ(ierr);break;
  default: SETERRQ1(PetscObjectComm((PetscObject) dm), PETSC_ERR_ARG_WRONG, "Invalid point generation type %d", (int) ctx->pointType);
  }
  PetscFunctionReturn(0);
}

int main(int argc, char **argv)
{
  AppCtx              ctx;
  PetscErrorCode   (**funcs)(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nf, PetscScalar *u, void *ctx);
  DM                  dm;
  PetscFE             fe;
  DMInterpolationInfo interpolator;
  Vec                 lu, fieldVals;
  PetscScalar        *vals;
  const PetscScalar  *ivals, *vcoords;
  PetscReal          *pcoords;
  PetscBool           pointsAllProcs=PETSC_TRUE;
  PetscInt            spaceDim, c, Np, p;
  PetscMPIInt         rank, size;
  PetscViewer         selfviewer;
  PetscErrorCode      ierr;

  ierr = PetscInitialize(&argc, &argv, NULL,help);if (ierr) return ierr;
  ierr = ProcessOptions(PETSC_COMM_WORLD, &ctx);CHKERRQ(ierr);
  ierr = CreateMesh(PETSC_COMM_WORLD, &ctx, &dm);CHKERRQ(ierr);
  ierr = DMGetCoordinateDim(dm, &spaceDim);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD, &rank);CHKERRMPI(ierr);
  ierr = MPI_Comm_size(PETSC_COMM_WORLD, &size);CHKERRMPI(ierr);
  /* Create points */
  ierr = CreatePoints(dm, &Np, &pcoords, &pointsAllProcs, &ctx);CHKERRQ(ierr);
  /* Create interpolator */
  ierr = DMInterpolationCreate(PETSC_COMM_WORLD, &interpolator);CHKERRQ(ierr);
  ierr = DMInterpolationSetDim(interpolator, spaceDim);CHKERRQ(ierr);
  ierr = DMInterpolationAddPoints(interpolator, Np, pcoords);CHKERRQ(ierr);
  ierr = DMInterpolationSetUp(interpolator, dm, pointsAllProcs, PETSC_FALSE);CHKERRQ(ierr);
  /* Check locations */
  for (c = 0; c < interpolator->n; ++c) {
    ierr = PetscSynchronizedPrintf(PETSC_COMM_WORLD, "[%d]Point %D is in Cell %D\n", rank, c, interpolator->cells[c]);CHKERRQ(ierr);
  }
  ierr = PetscSynchronizedFlush(PETSC_COMM_WORLD, NULL);CHKERRQ(ierr);
  ierr = VecView(interpolator->coords, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  /* Setup Discretization */
  ierr = PetscFECreateDefault(PetscObjectComm((PetscObject) dm), ctx.dim, Nc, ctx.cellSimplex, NULL, -1, &fe);CHKERRQ(ierr);
  ierr = DMSetField(dm, 0, NULL, (PetscObject) fe);CHKERRQ(ierr);
  ierr = DMCreateDS(dm);CHKERRQ(ierr);
  ierr = PetscFEDestroy(&fe);CHKERRQ(ierr);
  /* Create function */
  ierr = PetscCalloc2(Nc, &funcs, Nc, &vals);CHKERRQ(ierr);
  for (c = 0; c < Nc; ++c) funcs[c] = linear;
  ierr = DMGetLocalVector(dm, &lu);CHKERRQ(ierr);
  ierr = DMProjectFunctionLocal(dm, 0.0, funcs, NULL, INSERT_ALL_VALUES, lu);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPushSynchronized(PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = PetscViewerGetSubViewer(PETSC_VIEWER_STDOUT_WORLD,PETSC_COMM_SELF,&selfviewer);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(selfviewer, "[%d]solution\n", rank);CHKERRQ(ierr);
  ierr = VecView(lu,selfviewer);CHKERRQ(ierr);
  ierr = PetscViewerRestoreSubViewer(PETSC_VIEWER_STDOUT_WORLD,PETSC_COMM_SELF,&selfviewer);CHKERRQ(ierr);
  ierr = PetscViewerFlush(PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPopSynchronized(PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  /* Check interpolant */
  ierr = VecCreateSeq(PETSC_COMM_SELF, interpolator->n * Nc, &fieldVals);CHKERRQ(ierr);
  ierr = DMInterpolationSetDof(interpolator, Nc);CHKERRQ(ierr);
  ierr = DMInterpolationEvaluate(interpolator, dm, lu, fieldVals);CHKERRQ(ierr);
  for (p = 0; p < size; ++p) {
    if (p == rank) {
      ierr = PetscPrintf(PETSC_COMM_SELF, "[%d]Field values\n", rank);CHKERRQ(ierr);
      ierr = VecView(fieldVals, PETSC_VIEWER_STDOUT_SELF);CHKERRQ(ierr);
    }
    ierr = PetscBarrier((PetscObject) dm);CHKERRQ(ierr);
  }
  ierr = VecGetArrayRead(interpolator->coords, &vcoords);CHKERRQ(ierr);
  ierr = VecGetArrayRead(fieldVals, &ivals);CHKERRQ(ierr);
  for (p = 0; p < interpolator->n; ++p) {
    for (c = 0; c < Nc; ++c) {
#if defined(PETSC_USE_COMPLEX)
      PetscReal vcoordsReal[3];
      PetscInt  i;

      for (i = 0; i < spaceDim; i++) vcoordsReal[i] = PetscRealPart(vcoords[p * spaceDim + i]);
#else
      const PetscReal *vcoordsReal = &vcoords[p*spaceDim];
#endif
      (*funcs[c])(ctx.dim, 0.0, vcoordsReal, 1, vals, NULL);
      if (PetscAbsScalar(ivals[p*Nc+c] - vals[c]) > PETSC_SQRT_MACHINE_EPSILON)
        SETERRQ4(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Invalid interpolated value %g != %g (%D, %D)", (double) PetscRealPart(ivals[p*Nc+c]), (double) PetscRealPart(vals[c]), p, c);
    }
  }
  ierr = VecRestoreArrayRead(interpolator->coords, &vcoords);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(fieldVals, &ivals);CHKERRQ(ierr);
  /* Cleanup */
  ierr = PetscFree(pcoords);CHKERRQ(ierr);
  ierr = PetscFree2(funcs, vals);CHKERRQ(ierr);
  ierr = VecDestroy(&fieldVals);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(dm, &lu);CHKERRQ(ierr);
  ierr = DMInterpolationDestroy(&interpolator);CHKERRQ(ierr);
  ierr = DMDestroy(&dm);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

  test:
    suffix: 0
    requires: ctetgen
    args: -petscspace_degree 1
  test:
    suffix: 1
    requires: ctetgen
    args: -petscspace_degree 1 -dm_refine 2
  test:
    suffix: 2
    requires: ctetgen
    nsize: 2
    args: -petscspace_degree 1 -petscpartitioner_type simple
  test:
    suffix: 3
    requires: ctetgen
    nsize: 2
    args: -petscspace_degree 1 -dm_refine 2 -petscpartitioner_type simple
  test:
    suffix: 4
    requires: ctetgen
    nsize: 5
    args: -petscspace_degree 1 -petscpartitioner_type simple
  test:
    suffix: 5
    requires: ctetgen
    nsize: 5
    args: -petscspace_degree 1 -dm_refine 2 -petscpartitioner_type simple
  test:
    suffix: 6
    requires: ctetgen
    args: -petscspace_degree 1 -point_type grid
  test:
    suffix: 7
    requires: ctetgen
    args: -petscspace_degree 1 -dm_refine 2 -point_type grid
  test:
    suffix: 8
    requires: ctetgen
    nsize: 2
    args: -petscspace_degree 1 -point_type grid -petscpartitioner_type simple
  test:
    suffix: 9
    requires: ctetgen
    args: -petscspace_degree 1 -point_type grid_replicated
  test:
    suffix: 10
    requires: ctetgen
    nsize: 2
    args: -petscspace_degree 1 -point_type grid_replicated -petscpartitioner_type simple
  test:
    suffix: 11
    requires: ctetgen
    nsize: 2
    args: -petscspace_degree 1 -dm_refine 2 -point_type grid_replicated -petscpartitioner_type simple

TEST*/
