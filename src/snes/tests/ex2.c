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

  PetscCheck(Nc == 3,PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Something is wrong: %" PetscInt_FMT, Nc);
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

  PetscFunctionBegin;
  options->pointType = CENTROID;
  PetscOptionsBegin(comm, "", "Interpolation Options", "DMPLEX");
  pt   = options->pointType;
  PetscCall(PetscOptionsEList("-point_type", "The point type", "ex2.c", pointTypes, 3, pointTypes[options->pointType], &pt, NULL));
  options->pointType = (PointType) pt;
  PetscOptionsEnd();
  PetscFunctionReturn(0);
}

static PetscErrorCode CreateMesh(MPI_Comm comm, AppCtx *ctx, DM *dm)
{
  PetscFunctionBegin;
  PetscCall(DMCreate(comm, dm));
  PetscCall(DMSetType(*dm, DMPLEX));
  PetscCall(DMSetFromOptions(*dm));
  PetscCall(DMViewFromOptions(*dm, NULL, "-dm_view"));
  PetscFunctionReturn(0);
}

static PetscErrorCode CreatePoints_Centroid(DM dm, PetscInt *Np, PetscReal **pcoords, PetscBool *pointsAllProcs, AppCtx *ctx)
{
  PetscSection   coordSection;
  Vec            coordsLocal;
  PetscInt       spaceDim, p;
  PetscMPIInt    rank;

  PetscFunctionBegin;
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &rank));
  PetscCall(DMGetCoordinatesLocal(dm, &coordsLocal));
  PetscCall(DMGetCoordinateSection(dm, &coordSection));
  PetscCall(DMGetCoordinateDim(dm, &spaceDim));
  PetscCall(DMPlexGetHeightStratum(dm, 0, NULL, Np));
  PetscCall(PetscCalloc1(*Np * spaceDim, pcoords));
  for (p = 0; p < *Np; ++p) {
    PetscScalar *coords = NULL;
    PetscInt     size, num, n, d;

    PetscCall(DMPlexVecGetClosure(dm, coordSection, coordsLocal, p, &size, &coords));
    num  = size/spaceDim;
    for (n = 0; n < num; ++n) {
      for (d = 0; d < spaceDim; ++d) (*pcoords)[p*spaceDim+d] += PetscRealPart(coords[n*spaceDim+d]) / num;
    }
    PetscCall(PetscSynchronizedPrintf(PETSC_COMM_WORLD, "[%d]Point %" PetscInt_FMT " (", rank, p));
    for (d = 0; d < spaceDim; ++d) {
      PetscCall(PetscSynchronizedPrintf(PETSC_COMM_WORLD, "%g", (double)(*pcoords)[p*spaceDim+d]));
      if (d < spaceDim-1) PetscCall(PetscSynchronizedPrintf(PETSC_COMM_WORLD, ", "));
    }
    PetscCall(PetscSynchronizedPrintf(PETSC_COMM_WORLD, ")\n"));
    PetscCall(DMPlexVecRestoreClosure(dm, coordSection, coordsLocal, p, &num, &coords));
  }
  PetscCall(PetscSynchronizedFlush(PETSC_COMM_WORLD, NULL));
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
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &rank));
  PetscCall(DMGetDimension(dm, &dim));
  PetscCall(DMGetCoordinateDim(dm, &spaceDim));
  PetscCall(PetscCalloc1(spaceDim,&ind));
  PetscCall(PetscCalloc1(spaceDim,&h));
  h[0] = 1.0/(N-1); h[1] = 1.0/(N-1); h[2] = 1.0/(N-1);
  PetscCall(DMDACreate(PetscObjectComm((PetscObject) dm), &da));
  PetscCall(DMSetDimension(da, dim));
  PetscCall(DMDASetSizes(da, N, N, N));
  PetscCall(DMDASetDof(da, 1));
  PetscCall(DMDASetStencilWidth(da, 1));
  PetscCall(DMSetUp(da));
  PetscCall(DMDASetUniformCoordinates(da, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0));
  PetscCall(DMDAGetLocalInfo(da, &info));
  *Np  = info.xm * info.ym * info.zm;
  PetscCall(PetscCalloc1(*Np * spaceDim, pcoords));
  for (k = info.zs; k < info.zs + info.zm; ++k) {
    ind[2] = k;
    for (j = info.ys; j < info.ys + info.ym; ++j) {
      ind[1] = j;
      for (i = info.xs; i < info.xs + info.xm; ++i, ++n) {
        ind[0] = i;

        for (d = 0; d < spaceDim; ++d) (*pcoords)[n*spaceDim+d] = ind[d]*h[d];
        PetscCall(PetscSynchronizedPrintf(PETSC_COMM_WORLD, "[%d]Point %" PetscInt_FMT " (", rank, n));
        for (d = 0; d < spaceDim; ++d) {
          PetscCall(PetscSynchronizedPrintf(PETSC_COMM_WORLD, "%g", (double)(*pcoords)[n*spaceDim+d]));
          if (d < spaceDim-1) PetscCall(PetscSynchronizedPrintf(PETSC_COMM_WORLD, ", "));
        }
        PetscCall(PetscSynchronizedPrintf(PETSC_COMM_WORLD, ")\n"));
      }
    }
  }
  PetscCall(DMDestroy(&da));
  PetscCall(PetscSynchronizedFlush(PETSC_COMM_WORLD, NULL));
  PetscCall(PetscFree(ind));
  PetscCall(PetscFree(h));
  *pointsAllProcs = PETSC_FALSE;
  PetscFunctionReturn(0);
}

static PetscErrorCode CreatePoints_GridReplicated(DM dm, PetscInt *Np, PetscReal **pcoords, PetscBool *pointsAllProcs, AppCtx *ctx)
{
  PetscInt       N = 3, n = 0, dim, spaceDim, i, j, k, *ind, d;
  PetscReal      *h;
  PetscMPIInt    rank;

  PetscFunctionBeginUser;
  PetscCall(DMGetDimension(dm, &dim));
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &rank));
  PetscCall(DMGetCoordinateDim(dm, &spaceDim));
  PetscCall(PetscCalloc1(spaceDim,&ind));
  PetscCall(PetscCalloc1(spaceDim,&h));
  h[0] = 1.0/(N-1); h[1] = 1.0/(N-1); h[2] = 1.0/(N-1);
  *Np  = N * (dim > 1 ? N : 1) * (dim > 2 ? N : 1);
  PetscCall(PetscCalloc1(*Np * spaceDim, pcoords));
  for (k = 0; k < N; ++k) {
    ind[2] = k;
    for (j = 0; j < N; ++j) {
      ind[1] = j;
      for (i = 0; i < N; ++i, ++n) {
        ind[0] = i;

        for (d = 0; d < spaceDim; ++d) (*pcoords)[n*spaceDim+d] = ind[d]*h[d];
        PetscCall(PetscSynchronizedPrintf(PETSC_COMM_WORLD, "[%d]Point %" PetscInt_FMT " (", rank, n));
        for (d = 0; d < spaceDim; ++d) {
          PetscCall(PetscSynchronizedPrintf(PETSC_COMM_WORLD, "%g", (double)(*pcoords)[n*spaceDim+d]));
          if (d < spaceDim-1) PetscCall(PetscSynchronizedPrintf(PETSC_COMM_WORLD, ", "));
        }
        PetscCall(PetscSynchronizedPrintf(PETSC_COMM_WORLD, ")\n"));
      }
    }
  }
  PetscCall(PetscSynchronizedFlush(PETSC_COMM_WORLD, NULL));
  *pointsAllProcs = PETSC_TRUE;
  PetscCall(PetscFree(ind));
  PetscCall(PetscFree(h));
  PetscFunctionReturn(0);
}

static PetscErrorCode CreatePoints(DM dm, PetscInt *Np, PetscReal **pcoords, PetscBool *pointsAllProcs, AppCtx *ctx)
{
  PetscFunctionBegin;
  *pointsAllProcs = PETSC_FALSE;
  switch (ctx->pointType) {
  case CENTROID:        PetscCall(CreatePoints_Centroid(dm, Np, pcoords, pointsAllProcs, ctx));break;
  case GRID:            PetscCall(CreatePoints_Grid(dm, Np, pcoords, pointsAllProcs, ctx));break;
  case GRID_REPLICATED: PetscCall(CreatePoints_GridReplicated(dm, Np, pcoords, pointsAllProcs, ctx));break;
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

  PetscCall(PetscInitialize(&argc, &argv, NULL,help));
  PetscCall(ProcessOptions(PETSC_COMM_WORLD, &ctx));
  PetscCall(CreateMesh(PETSC_COMM_WORLD, &ctx, &dm));
  PetscCall(DMGetDimension(dm, &dim));
  PetscCall(DMGetCoordinateDim(dm, &spaceDim));
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &rank));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &size));
  /* Create points */
  PetscCall(CreatePoints(dm, &Np, &pcoords, &pointsAllProcs, &ctx));
  /* Create interpolator */
  PetscCall(DMInterpolationCreate(PETSC_COMM_WORLD, &interpolator));
  PetscCall(DMInterpolationSetDim(interpolator, spaceDim));
  PetscCall(DMInterpolationAddPoints(interpolator, Np, pcoords));
  PetscCall(DMInterpolationSetUp(interpolator, dm, pointsAllProcs, PETSC_FALSE));
  /* Check locations */
  for (c = 0; c < interpolator->n; ++c) {
    PetscCall(PetscSynchronizedPrintf(PETSC_COMM_WORLD, "[%d]Point %" PetscInt_FMT " is in Cell %" PetscInt_FMT "\n", rank, c, interpolator->cells[c]));
  }
  PetscCall(PetscSynchronizedFlush(PETSC_COMM_WORLD, NULL));
  PetscCall(VecView(interpolator->coords, PETSC_VIEWER_STDOUT_WORLD));
  /* Setup Discretization */
  Nc   = dim;
  PetscCall(DMPlexIsSimplex(dm, &simplex));
  PetscCall(PetscFECreateDefault(PetscObjectComm((PetscObject) dm), dim, Nc, simplex, NULL, -1, &fe));
  PetscCall(DMSetField(dm, 0, NULL, (PetscObject) fe));
  PetscCall(DMCreateDS(dm));
  PetscCall(PetscFEDestroy(&fe));
  /* Create function */
  PetscCall(PetscCalloc2(Nc, &funcs, Nc, &vals));
  for (c = 0; c < Nc; ++c) funcs[c] = linear;
  PetscCall(DMGetLocalVector(dm, &lu));
  PetscCall(DMProjectFunctionLocal(dm, 0.0, funcs, NULL, INSERT_ALL_VALUES, lu));
  PetscCall(PetscViewerASCIIPushSynchronized(PETSC_VIEWER_STDOUT_WORLD));
  PetscCall(PetscViewerGetSubViewer(PETSC_VIEWER_STDOUT_WORLD,PETSC_COMM_SELF,&selfviewer));
  PetscCall(PetscViewerASCIIPrintf(selfviewer, "[%d]solution\n", rank));
  PetscCall(VecView(lu,selfviewer));
  PetscCall(PetscViewerRestoreSubViewer(PETSC_VIEWER_STDOUT_WORLD,PETSC_COMM_SELF,&selfviewer));
  PetscCall(PetscViewerFlush(PETSC_VIEWER_STDOUT_WORLD));
  PetscCall(PetscViewerASCIIPopSynchronized(PETSC_VIEWER_STDOUT_WORLD));
  /* Check interpolant */
  PetscCall(VecCreateSeq(PETSC_COMM_SELF, interpolator->n * Nc, &fieldVals));
  PetscCall(DMInterpolationSetDof(interpolator, Nc));
  PetscCall(DMInterpolationEvaluate(interpolator, dm, lu, fieldVals));
  for (p = 0; p < size; ++p) {
    if (p == rank) {
      PetscCall(PetscPrintf(PETSC_COMM_SELF, "[%d]Field values\n", rank));
      PetscCall(VecView(fieldVals, PETSC_VIEWER_STDOUT_SELF));
    }
    PetscCall(PetscBarrier((PetscObject) dm));
  }
  PetscCall(VecGetArrayRead(interpolator->coords, &vcoords));
  PetscCall(VecGetArrayRead(fieldVals, &ivals));
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
        SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Invalid interpolated value %g != %g (%" PetscInt_FMT ", %" PetscInt_FMT ")", (double) PetscRealPart(ivals[p*Nc+c]), (double) PetscRealPart(vals[c]), p, c);
    }
  }
  PetscCall(VecRestoreArrayRead(interpolator->coords, &vcoords));
  PetscCall(VecRestoreArrayRead(fieldVals, &ivals));
  /* Cleanup */
  PetscCall(PetscFree(pcoords));
  PetscCall(PetscFree2(funcs, vals));
  PetscCall(VecDestroy(&fieldVals));
  PetscCall(DMRestoreLocalVector(dm, &lu));
  PetscCall(DMInterpolationDestroy(&interpolator));
  PetscCall(DMDestroy(&dm));
  PetscCall(PetscFinalize());
  return 0;
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
