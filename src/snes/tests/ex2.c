static char help[] = "Interpolation Tests for Plex\n\n";

#include <petscsnes.h>
#include <petscdmplex.h>
#include <petscdmda.h>
#include <petscds.h>

typedef enum {CENTROID, GRID, GRID_REPLICATED} PointType;

typedef struct {
  PointType pointType; /* Point generation mechanism */
} AppCtx;

static PetscErrorCode linear(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nc, PetscScalar *u, void *ctx)
{
  PetscInt d, c;

  PetscCheckFalse(Nc != 3,PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Something is wrong: %D", Nc);
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
  options->pointType = CENTROID;

  ierr = PetscOptionsBegin(comm, "", "Interpolation Options", "DMPLEX");CHKERRQ(ierr);
  pt   = options->pointType;
  CHKERRQ(PetscOptionsEList("-point_type", "The point type", "ex2.c", pointTypes, 3, pointTypes[options->pointType], &pt, NULL));
  options->pointType = (PointType) pt;
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode CreateMesh(MPI_Comm comm, AppCtx *ctx, DM *dm)
{
  PetscFunctionBegin;
  CHKERRQ(DMCreate(comm, dm));
  CHKERRQ(DMSetType(*dm, DMPLEX));
  CHKERRQ(DMSetFromOptions(*dm));
  CHKERRQ(DMViewFromOptions(*dm, NULL, "-dm_view"));
  PetscFunctionReturn(0);
}

static PetscErrorCode CreatePoints_Centroid(DM dm, PetscInt *Np, PetscReal **pcoords, PetscBool *pointsAllProcs, AppCtx *ctx)
{
  PetscSection   coordSection;
  Vec            coordsLocal;
  PetscInt       spaceDim, p;
  PetscMPIInt    rank;

  PetscFunctionBegin;
  CHKERRMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &rank));
  CHKERRQ(DMGetCoordinatesLocal(dm, &coordsLocal));
  CHKERRQ(DMGetCoordinateSection(dm, &coordSection));
  CHKERRQ(DMGetCoordinateDim(dm, &spaceDim));
  CHKERRQ(DMPlexGetHeightStratum(dm, 0, NULL, Np));
  CHKERRQ(PetscCalloc1(*Np * spaceDim, pcoords));
  for (p = 0; p < *Np; ++p) {
    PetscScalar *coords = NULL;
    PetscInt     size, num, n, d;

    CHKERRQ(DMPlexVecGetClosure(dm, coordSection, coordsLocal, p, &size, &coords));
    num  = size/spaceDim;
    for (n = 0; n < num; ++n) {
      for (d = 0; d < spaceDim; ++d) (*pcoords)[p*spaceDim+d] += PetscRealPart(coords[n*spaceDim+d]) / num;
    }
    CHKERRQ(PetscSynchronizedPrintf(PETSC_COMM_WORLD, "[%d]Point %D (", rank, p));
    for (d = 0; d < spaceDim; ++d) {
      CHKERRQ(PetscSynchronizedPrintf(PETSC_COMM_WORLD, "%g", (double)(*pcoords)[p*spaceDim+d]));
      if (d < spaceDim-1) CHKERRQ(PetscSynchronizedPrintf(PETSC_COMM_WORLD, ", "));
    }
    CHKERRQ(PetscSynchronizedPrintf(PETSC_COMM_WORLD, ")\n"));
    CHKERRQ(DMPlexVecRestoreClosure(dm, coordSection, coordsLocal, p, &num, &coords));
  }
  CHKERRQ(PetscSynchronizedFlush(PETSC_COMM_WORLD, NULL));
  *pointsAllProcs = PETSC_FALSE;
  PetscFunctionReturn(0);
}

static PetscErrorCode CreatePoints_Grid(DM dm, PetscInt *Np, PetscReal **pcoords, PetscBool *pointsAllProcs, AppCtx *ctx)
{
  DM             da;
  DMDALocalInfo  info;
  PetscInt       N = 3, n = 0, dim, spaceDim, i, j, k, *ind, d;
  PetscReal      *h;
  PetscMPIInt    rank;

  PetscFunctionBegin;
  CHKERRMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &rank));
  CHKERRQ(DMGetDimension(dm, &dim));
  CHKERRQ(DMGetCoordinateDim(dm, &spaceDim));
  CHKERRQ(PetscCalloc1(spaceDim,&ind));
  CHKERRQ(PetscCalloc1(spaceDim,&h));
  h[0] = 1.0/(N-1); h[1] = 1.0/(N-1); h[2] = 1.0/(N-1);
  CHKERRQ(DMDACreate(PetscObjectComm((PetscObject) dm), &da));
  CHKERRQ(DMSetDimension(da, dim));
  CHKERRQ(DMDASetSizes(da, N, N, N));
  CHKERRQ(DMDASetDof(da, 1));
  CHKERRQ(DMDASetStencilWidth(da, 1));
  CHKERRQ(DMSetUp(da));
  CHKERRQ(DMDASetUniformCoordinates(da, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0));
  CHKERRQ(DMDAGetLocalInfo(da, &info));
  *Np  = info.xm * info.ym * info.zm;
  CHKERRQ(PetscCalloc1(*Np * spaceDim, pcoords));
  for (k = info.zs; k < info.zs + info.zm; ++k) {
    ind[2] = k;
    for (j = info.ys; j < info.ys + info.ym; ++j) {
      ind[1] = j;
      for (i = info.xs; i < info.xs + info.xm; ++i, ++n) {
        ind[0] = i;

        for (d = 0; d < spaceDim; ++d) (*pcoords)[n*spaceDim+d] = ind[d]*h[d];
        CHKERRQ(PetscSynchronizedPrintf(PETSC_COMM_WORLD, "[%d]Point %D (", rank, n));
        for (d = 0; d < spaceDim; ++d) {
          CHKERRQ(PetscSynchronizedPrintf(PETSC_COMM_WORLD, "%g", (double)(*pcoords)[n*spaceDim+d]));
          if (d < spaceDim-1) CHKERRQ(PetscSynchronizedPrintf(PETSC_COMM_WORLD, ", "));
        }
        CHKERRQ(PetscSynchronizedPrintf(PETSC_COMM_WORLD, ")\n"));
      }
    }
  }
  CHKERRQ(DMDestroy(&da));
  CHKERRQ(PetscSynchronizedFlush(PETSC_COMM_WORLD, NULL));
  CHKERRQ(PetscFree(ind));
  CHKERRQ(PetscFree(h));
  *pointsAllProcs = PETSC_FALSE;
  PetscFunctionReturn(0);
}

static PetscErrorCode CreatePoints_GridReplicated(DM dm, PetscInt *Np, PetscReal **pcoords, PetscBool *pointsAllProcs, AppCtx *ctx)
{
  PetscInt       N = 3, n = 0, dim, spaceDim, i, j, k, *ind, d;
  PetscReal      *h;
  PetscMPIInt    rank;

  PetscFunctionBeginUser;
  CHKERRQ(DMGetDimension(dm, &dim));
  CHKERRMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &rank));
  CHKERRQ(DMGetCoordinateDim(dm, &spaceDim));
  CHKERRQ(PetscCalloc1(spaceDim,&ind));
  CHKERRQ(PetscCalloc1(spaceDim,&h));
  h[0] = 1.0/(N-1); h[1] = 1.0/(N-1); h[2] = 1.0/(N-1);
  *Np  = N * (dim > 1 ? N : 1) * (dim > 2 ? N : 1);
  CHKERRQ(PetscCalloc1(*Np * spaceDim, pcoords));
  for (k = 0; k < N; ++k) {
    ind[2] = k;
    for (j = 0; j < N; ++j) {
      ind[1] = j;
      for (i = 0; i < N; ++i, ++n) {
        ind[0] = i;

        for (d = 0; d < spaceDim; ++d) (*pcoords)[n*spaceDim+d] = ind[d]*h[d];
        CHKERRQ(PetscSynchronizedPrintf(PETSC_COMM_WORLD, "[%d]Point %D (", rank, n));
        for (d = 0; d < spaceDim; ++d) {
          CHKERRQ(PetscSynchronizedPrintf(PETSC_COMM_WORLD, "%g", (double)(*pcoords)[n*spaceDim+d]));
          if (d < spaceDim-1) CHKERRQ(PetscSynchronizedPrintf(PETSC_COMM_WORLD, ", "));
        }
        CHKERRQ(PetscSynchronizedPrintf(PETSC_COMM_WORLD, ")\n"));
      }
    }
  }
  CHKERRQ(PetscSynchronizedFlush(PETSC_COMM_WORLD, NULL));
  *pointsAllProcs = PETSC_TRUE;
  CHKERRQ(PetscFree(ind));
  CHKERRQ(PetscFree(h));
  PetscFunctionReturn(0);
}

static PetscErrorCode CreatePoints(DM dm, PetscInt *Np, PetscReal **pcoords, PetscBool *pointsAllProcs, AppCtx *ctx)
{
  PetscFunctionBegin;
  *pointsAllProcs = PETSC_FALSE;
  switch (ctx->pointType) {
  case CENTROID:        CHKERRQ(CreatePoints_Centroid(dm, Np, pcoords, pointsAllProcs, ctx));break;
  case GRID:            CHKERRQ(CreatePoints_Grid(dm, Np, pcoords, pointsAllProcs, ctx));break;
  case GRID_REPLICATED: CHKERRQ(CreatePoints_GridReplicated(dm, Np, pcoords, pointsAllProcs, ctx));break;
  default: SETERRQ(PetscObjectComm((PetscObject) dm), PETSC_ERR_ARG_WRONG, "Invalid point generation type %d", (int) ctx->pointType);
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
  PetscBool           simplex, pointsAllProcs=PETSC_TRUE;
  PetscInt            dim, spaceDim, Nc, c, Np, p;
  PetscMPIInt         rank, size;
  PetscViewer         selfviewer;
  PetscErrorCode      ierr;

  ierr = PetscInitialize(&argc, &argv, NULL,help);if (ierr) return ierr;
  CHKERRQ(ProcessOptions(PETSC_COMM_WORLD, &ctx));
  CHKERRQ(CreateMesh(PETSC_COMM_WORLD, &ctx, &dm));
  CHKERRQ(DMGetDimension(dm, &dim));
  CHKERRQ(DMGetCoordinateDim(dm, &spaceDim));
  CHKERRMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &rank));
  CHKERRMPI(MPI_Comm_size(PETSC_COMM_WORLD, &size));
  /* Create points */
  CHKERRQ(CreatePoints(dm, &Np, &pcoords, &pointsAllProcs, &ctx));
  /* Create interpolator */
  CHKERRQ(DMInterpolationCreate(PETSC_COMM_WORLD, &interpolator));
  CHKERRQ(DMInterpolationSetDim(interpolator, spaceDim));
  CHKERRQ(DMInterpolationAddPoints(interpolator, Np, pcoords));
  CHKERRQ(DMInterpolationSetUp(interpolator, dm, pointsAllProcs, PETSC_FALSE));
  /* Check locations */
  for (c = 0; c < interpolator->n; ++c) {
    CHKERRQ(PetscSynchronizedPrintf(PETSC_COMM_WORLD, "[%d]Point %D is in Cell %D\n", rank, c, interpolator->cells[c]));
  }
  CHKERRQ(PetscSynchronizedFlush(PETSC_COMM_WORLD, NULL));
  CHKERRQ(VecView(interpolator->coords, PETSC_VIEWER_STDOUT_WORLD));
  /* Setup Discretization */
  Nc   = dim;
  CHKERRQ(DMPlexIsSimplex(dm, &simplex));
  CHKERRQ(PetscFECreateDefault(PetscObjectComm((PetscObject) dm), dim, Nc, simplex, NULL, -1, &fe));
  CHKERRQ(DMSetField(dm, 0, NULL, (PetscObject) fe));
  CHKERRQ(DMCreateDS(dm));
  CHKERRQ(PetscFEDestroy(&fe));
  /* Create function */
  CHKERRQ(PetscCalloc2(Nc, &funcs, Nc, &vals));
  for (c = 0; c < Nc; ++c) funcs[c] = linear;
  CHKERRQ(DMGetLocalVector(dm, &lu));
  CHKERRQ(DMProjectFunctionLocal(dm, 0.0, funcs, NULL, INSERT_ALL_VALUES, lu));
  CHKERRQ(PetscViewerASCIIPushSynchronized(PETSC_VIEWER_STDOUT_WORLD));
  CHKERRQ(PetscViewerGetSubViewer(PETSC_VIEWER_STDOUT_WORLD,PETSC_COMM_SELF,&selfviewer));
  CHKERRQ(PetscViewerASCIIPrintf(selfviewer, "[%d]solution\n", rank));
  CHKERRQ(VecView(lu,selfviewer));
  CHKERRQ(PetscViewerRestoreSubViewer(PETSC_VIEWER_STDOUT_WORLD,PETSC_COMM_SELF,&selfviewer));
  CHKERRQ(PetscViewerFlush(PETSC_VIEWER_STDOUT_WORLD));
  CHKERRQ(PetscViewerASCIIPopSynchronized(PETSC_VIEWER_STDOUT_WORLD));
  /* Check interpolant */
  CHKERRQ(VecCreateSeq(PETSC_COMM_SELF, interpolator->n * Nc, &fieldVals));
  CHKERRQ(DMInterpolationSetDof(interpolator, Nc));
  CHKERRQ(DMInterpolationEvaluate(interpolator, dm, lu, fieldVals));
  for (p = 0; p < size; ++p) {
    if (p == rank) {
      CHKERRQ(PetscPrintf(PETSC_COMM_SELF, "[%d]Field values\n", rank));
      CHKERRQ(VecView(fieldVals, PETSC_VIEWER_STDOUT_SELF));
    }
    CHKERRQ(PetscBarrier((PetscObject) dm));
  }
  CHKERRQ(VecGetArrayRead(interpolator->coords, &vcoords));
  CHKERRQ(VecGetArrayRead(fieldVals, &ivals));
  for (p = 0; p < interpolator->n; ++p) {
    for (c = 0; c < Nc; ++c) {
#if defined(PETSC_USE_COMPLEX)
      PetscReal vcoordsReal[3];
      PetscInt  i;

      for (i = 0; i < spaceDim; i++) vcoordsReal[i] = PetscRealPart(vcoords[p * spaceDim + i]);
#else
      const PetscReal *vcoordsReal = &vcoords[p*spaceDim];
#endif
      (*funcs[c])(dim, 0.0, vcoordsReal, Nc, vals, NULL);
      if (PetscAbsScalar(ivals[p*Nc+c] - vals[c]) > PETSC_SQRT_MACHINE_EPSILON)
        SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Invalid interpolated value %g != %g (%D, %D)", (double) PetscRealPart(ivals[p*Nc+c]), (double) PetscRealPart(vals[c]), p, c);
    }
  }
  CHKERRQ(VecRestoreArrayRead(interpolator->coords, &vcoords));
  CHKERRQ(VecRestoreArrayRead(fieldVals, &ivals));
  /* Cleanup */
  CHKERRQ(PetscFree(pcoords));
  CHKERRQ(PetscFree2(funcs, vals));
  CHKERRQ(VecDestroy(&fieldVals));
  CHKERRQ(DMRestoreLocalVector(dm, &lu));
  CHKERRQ(DMInterpolationDestroy(&interpolator));
  CHKERRQ(DMDestroy(&dm));
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

  testset:
    requires: ctetgen
    args: -dm_plex_dim 3 -petscspace_degree 1

    test:
      suffix: 0
    test:
      suffix: 1
      args: -dm_refine 2
    test:
      suffix: 2
      nsize: 2
      args: -petscpartitioner_type simple
    test:
      suffix: 3
      nsize: 2
      args: -dm_refine 2 -petscpartitioner_type simple
    test:
      suffix: 4
      nsize: 5
      args: -petscpartitioner_type simple
    test:
      suffix: 5
      nsize: 5
      args: -dm_refine 2 -petscpartitioner_type simple
    test:
      suffix: 6
      args: -point_type grid
    test:
      suffix: 7
      args: -dm_refine 2 -point_type grid
    test:
      suffix: 8
      nsize: 2
      args: -petscpartitioner_type simple -point_type grid
    test:
      suffix: 9
      args: -point_type grid_replicated
    test:
      suffix: 10
      nsize: 2
      args: -petscpartitioner_type simple -point_type grid_replicated
    test:
      suffix: 11
      nsize: 2
      args: -dm_refine 2 -petscpartitioner_type simple -point_type grid_replicated

TEST*/
