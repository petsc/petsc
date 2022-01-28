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

  PetscAssertFalse(Nc != 3,PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Something is wrong: %D", Nc);
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
  ierr = PetscOptionsEList("-point_type", "The point type", "ex2.c", pointTypes, 3, pointTypes[options->pointType], &pt, NULL);CHKERRQ(ierr);
  options->pointType = (PointType) pt;
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode CreateMesh(MPI_Comm comm, AppCtx *ctx, DM *dm)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMCreate(comm, dm);CHKERRQ(ierr);
  ierr = DMSetType(*dm, DMPLEX);CHKERRQ(ierr);
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
  PetscInt       N = 3, n = 0, dim, spaceDim, i, j, k, *ind, d;
  PetscReal      *h;
  PetscMPIInt    rank;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD, &rank);CHKERRMPI(ierr);
  ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);
  ierr = DMGetCoordinateDim(dm, &spaceDim);CHKERRQ(ierr);
  ierr = PetscCalloc1(spaceDim,&ind);CHKERRQ(ierr);
  ierr = PetscCalloc1(spaceDim,&h);CHKERRQ(ierr);
  h[0] = 1.0/(N-1); h[1] = 1.0/(N-1); h[2] = 1.0/(N-1);
  ierr = DMDACreate(PetscObjectComm((PetscObject) dm), &da);CHKERRQ(ierr);
  ierr = DMSetDimension(da, dim);CHKERRQ(ierr);
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
  PetscInt       N = 3, n = 0, dim, spaceDim, i, j, k, *ind, d;
  PetscReal      *h;
  PetscMPIInt    rank;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD, &rank);CHKERRMPI(ierr);
  ierr = DMGetCoordinateDim(dm, &spaceDim);CHKERRQ(ierr);
  ierr = PetscCalloc1(spaceDim,&ind);CHKERRQ(ierr);
  ierr = PetscCalloc1(spaceDim,&h);CHKERRQ(ierr);
  h[0] = 1.0/(N-1); h[1] = 1.0/(N-1); h[2] = 1.0/(N-1);
  *Np  = N * (dim > 1 ? N : 1) * (dim > 2 ? N : 1);
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
  ierr = ProcessOptions(PETSC_COMM_WORLD, &ctx);CHKERRQ(ierr);
  ierr = CreateMesh(PETSC_COMM_WORLD, &ctx, &dm);CHKERRQ(ierr);
  ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);
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
  Nc   = dim;
  ierr = DMPlexIsSimplex(dm, &simplex);CHKERRQ(ierr);
  ierr = PetscFECreateDefault(PetscObjectComm((PetscObject) dm), dim, Nc, simplex, NULL, -1, &fe);CHKERRQ(ierr);
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
      (*funcs[c])(dim, 0.0, vcoordsReal, Nc, vals, NULL);
      if (PetscAbsScalar(ivals[p*Nc+c] - vals[c]) > PETSC_SQRT_MACHINE_EPSILON)
        SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Invalid interpolated value %g != %g (%D, %D)", (double) PetscRealPart(ivals[p*Nc+c]), (double) PetscRealPart(vals[c]), p, c);
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
      args: -dm_distribute -petscpartitioner_type simple
    test:
      suffix: 3
      nsize: 2
      args: -dm_refine 2 -dm_distribute -petscpartitioner_type simple
    test:
      suffix: 4
      nsize: 5
      args: -dm_distribute -petscpartitioner_type simple
    test:
      suffix: 5
      nsize: 5
      args: -dm_refine 2 -dm_distribute -petscpartitioner_type simple
    test:
      suffix: 6
      args: -point_type grid
    test:
      suffix: 7
      args: -dm_refine 2 -point_type grid
    test:
      suffix: 8
      nsize: 2
      args: -dm_distribute -petscpartitioner_type simple -point_type grid
    test:
      suffix: 9
      args: -point_type grid_replicated
    test:
      suffix: 10
      nsize: 2
      args: -dm_distribute -petscpartitioner_type simple -point_type grid_replicated
    test:
      suffix: 11
      nsize: 2
      args: -dm_refine 2 -dm_distribute -petscpartitioner_type simple -point_type grid_replicated

TEST*/
