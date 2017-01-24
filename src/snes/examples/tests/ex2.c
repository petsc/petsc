static char help[] = "Interpolation Tests for Plex\n\n";

#include <petscsnes.h>
#include <petscdmplex.h>
#include <petscds.h>

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

static PetscErrorCode quadratic(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nf, PetscScalar *u, void *ctx)
{
  PetscInt d, c;

  for (c = 0; c < Nc; ++c) {
    u[c] = 0.0;
    for (d = 0; d < dim; ++d) u[c] += x[d]*x[d];
  }
  return 0;
}

int main(int argc, char **argv)
{
  PetscErrorCode   (**funcs)(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nf, PetscScalar *u, void *ctx);
  DM                  dm;
  PetscDS             prob;
  PetscFE             fe;
  DMInterpolationInfo interpolator;
  PetscSection        coordSection;
  Vec                 coordsLocal, lu, fieldVals;
  PetscScalar        *vals;
  const PetscScalar  *ivals, *vcoords;
  PetscReal          *pcoords;
  PetscBool           simplex = PETSC_TRUE, pointsAllProcs = PETSC_FALSE;
  PetscInt            dim = 3, spaceDim, c, Np, p;
  PetscMPIInt         rank, numProcs;
  PetscErrorCode      ierr;

  ierr = PetscInitialize(&argc, &argv, NULL,help);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD, &rank);CHKERRQ(ierr);
  ierr = MPI_Comm_size(PETSC_COMM_WORLD, &numProcs);CHKERRQ(ierr);
  /* Create mesh */
  if (simplex) {
    ierr = DMPlexCreateBoxMesh(PETSC_COMM_WORLD, dim, 1, PETSC_TRUE, &dm);CHKERRQ(ierr);
  } else {
    PetscInt cells[3] = {1, 1, 1};
    ierr = DMPlexCreateHexBoxMesh(PETSC_COMM_WORLD, dim, cells, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, &dm);CHKERRQ(ierr);
  }
  {
    DM pdm = NULL;

    ierr = DMPlexDistribute(dm, 0, NULL, &pdm);CHKERRQ(ierr);
    if (pdm) {
      ierr = DMDestroy(&dm);CHKERRQ(ierr);
      dm   = pdm;
    }
  }
  ierr = DMSetFromOptions(dm);CHKERRQ(ierr);
  ierr = DMViewFromOptions(dm, NULL, "-dm_view");CHKERRQ(ierr);
  /* Create points */
  ierr = DMGetCoordinatesLocal(dm, &coordsLocal);CHKERRQ(ierr);
  ierr = DMGetCoordinateSection(dm, &coordSection);CHKERRQ(ierr);
  ierr = DMGetCoordinateDim(dm, &spaceDim);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm, 0, NULL, &Np);CHKERRQ(ierr);
  ierr = PetscCalloc3(Np * spaceDim, &pcoords, Nc, &funcs, Nc, &vals);CHKERRQ(ierr);
  for (p = 0; p < Np; ++p) {
    PetscScalar *coords = NULL;
    PetscInt     size, num, n, d;

    ierr = DMPlexVecGetClosure(dm, coordSection, coordsLocal, p, &size, &coords);CHKERRQ(ierr);
    num  = size/spaceDim;
    for (n = 0; n < num; ++n) {
      for (d = 0; d < spaceDim; ++d) pcoords[p*spaceDim+d] += coords[n*spaceDim+d] / num;
    }
    ierr = PetscSynchronizedPrintf(PETSC_COMM_WORLD, "[%d]Point %D (", rank, p);CHKERRQ(ierr);
    for (d = 0; d < spaceDim; ++d) {
      ierr = PetscSynchronizedPrintf(PETSC_COMM_WORLD, "%g", pcoords[p*spaceDim+d]);CHKERRQ(ierr);
      if (d < spaceDim-1) {ierr = PetscSynchronizedPrintf(PETSC_COMM_WORLD, ", ");CHKERRQ(ierr);}
    }
    ierr = PetscSynchronizedPrintf(PETSC_COMM_WORLD, ")\n");CHKERRQ(ierr);
    ierr = DMPlexVecRestoreClosure(dm, coordSection, coordsLocal, p, &num, &coords);CHKERRQ(ierr);
  }
  ierr = PetscSynchronizedFlush(PETSC_COMM_WORLD, NULL);CHKERRQ(ierr);
  /* Create interpolator */
  ierr = DMInterpolationCreate(PETSC_COMM_WORLD, &interpolator);CHKERRQ(ierr);
  ierr = DMInterpolationSetDim(interpolator, spaceDim);CHKERRQ(ierr);
  ierr = DMInterpolationAddPoints(interpolator, Np, pcoords);CHKERRQ(ierr);
  ierr = DMInterpolationSetUp(interpolator, dm, pointsAllProcs);CHKERRQ(ierr);
  /* Check locations */
  for (c = 0; c < interpolator->n; ++c) {
    ierr = PetscSynchronizedPrintf(PETSC_COMM_WORLD, "[%d]Point %D is in Cell %D\n", rank, c, interpolator->cells[c]);CHKERRQ(ierr);
  }
  ierr = PetscSynchronizedFlush(PETSC_COMM_WORLD, NULL);CHKERRQ(ierr);
  ierr = VecView(interpolator->coords, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  /* Setup Discretization */
  ierr = PetscFECreateDefault(dm, dim, Nc, simplex, NULL, -1, &fe);CHKERRQ(ierr);
  ierr = DMGetDS(dm, &prob);CHKERRQ(ierr);
  ierr = PetscDSSetDiscretization(prob, 0, (PetscObject) fe);CHKERRQ(ierr);
  ierr = PetscFEDestroy(&fe);CHKERRQ(ierr);
  /* Create function */
  for (c = 0; c < Nc; ++c) funcs[c] = linear;
  ierr = DMGetLocalVector(dm, &lu);CHKERRQ(ierr);
  ierr = DMProjectFunctionLocal(dm, 0.0, funcs, NULL, INSERT_ALL_VALUES, lu);CHKERRQ(ierr);
  for (p = 0; p < numProcs; ++p) {
    if (p == rank) {
      ierr = PetscPrintf(PETSC_COMM_SELF, "[%d]solution\n", rank);CHKERRQ(ierr);
      ierr = VecView(lu, PETSC_VIEWER_STDOUT_SELF);CHKERRQ(ierr);
    }
    ierr = PetscBarrier((PetscObject) dm);CHKERRQ(ierr);
  }
  /* Check interpolant */
  ierr = VecCreateSeq(PETSC_COMM_SELF, interpolator->n * Nc, &fieldVals);CHKERRQ(ierr);
  ierr = DMInterpolationSetDof(interpolator, Nc);CHKERRQ(ierr);
  ierr = DMInterpolationEvaluate(interpolator, dm, lu, fieldVals);CHKERRQ(ierr);
  for (p = 0; p < numProcs; ++p) {
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
      (*funcs[c])(dim, 0.0, &vcoords[p*spaceDim], 1, vals, NULL);
      if (PetscAbsScalar(ivals[p*Nc+c] - vals[c]) > PETSC_SQRT_MACHINE_EPSILON)
        SETERRQ4(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Invalid interpolated value %g != %g (%D, %D)", (double) ivals[p*Nc+c], (double) vals[c], p, c);
    }
  }
  ierr = VecRestoreArrayRead(interpolator->coords, &vcoords);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(fieldVals, &ivals);CHKERRQ(ierr);
  /* Cleanup */
  ierr = PetscFree3(pcoords, funcs, vals);CHKERRQ(ierr);
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
    args: -petscspace_order 1
  test:
    suffix: 1
    args: -petscspace_order 1 -dm_refine 2
  test:
    suffix: 2
    nsize: 2
    args: -petscspace_order 1
  test:
    suffix: 3
    nsize: 2
    args: -petscspace_order 1 -dm_refine 2
  test:
    suffix: 4
    nsize: 5
    args: -petscspace_order 1
  test:
    suffix: 5
    nsize: 5
    args: -petscspace_order 1 -dm_refine 2

TEST*/
