static char help[] = "Particle Basis Landau Example using nonlinear solve + Implicit Midpoint-like time stepping.";

/* TODO

1) SNES is sensitive to epsilon (but not to h). Should we do continuation in it?

2) Put this timestepper in library, maybe by changing DG

3) Add monitor to visualize distributions

*/

/* References
  [1] https://arxiv.org/abs/1910.03080v2
*/

#include <petscdmplex.h>
#include <petscdmswarm.h>
#include <petscts.h>
#include <petscviewer.h>
#include <petscmath.h>

typedef struct {
  /* Velocity space grid and functions */
  PetscInt  N;         /* The number of partices per spatial cell */
  PetscReal L;         /* Velocity space is [-L, L]^d */
  PetscReal h;         /* Spacing for grid 2L / N^{1/d} */
  PetscReal epsilon;   /* gaussian regularization parameter */
  PetscReal momentTol; /* Tolerance for checking moment conservation */
} AppCtx;

static PetscErrorCode ProcessOptions(MPI_Comm comm, AppCtx *options)
{
  PetscFunctionBeginUser;
  options->N         = 1;
  options->momentTol = 100.0 * PETSC_MACHINE_EPSILON;
  options->L         = 1.0;
  options->h         = -1.0;
  options->epsilon   = -1.0;

  PetscOptionsBegin(comm, "", "Collision Options", "DMPLEX");
  PetscCall(PetscOptionsInt("-N", "Number of particles per spatial cell", "ex27.c", options->N, &options->N, NULL));
  PetscCall(PetscOptionsReal("-L", "Velocity-space extent", "ex27.c", options->L, &options->L, NULL));
  PetscCall(PetscOptionsReal("-h", "Velocity-space resolution", "ex27.c", options->h, &options->h, NULL));
  PetscCall(PetscOptionsReal("-epsilon", "Mollifier regularization parameter", "ex27.c", options->epsilon, &options->epsilon, NULL));
  PetscOptionsEnd();
  PetscFunctionReturn(0);
}

static PetscErrorCode CreateMesh(MPI_Comm comm, DM *dm, AppCtx *user)
{
  PetscFunctionBeginUser;
  PetscCall(DMCreate(comm, dm));
  PetscCall(DMSetType(*dm, DMPLEX));
  PetscCall(DMSetFromOptions(*dm));
  PetscCall(DMViewFromOptions(*dm, NULL, "-dm_view"));
  PetscFunctionReturn(0);
}

static PetscErrorCode SetInitialCoordinates(DM sw)
{
  AppCtx        *user;
  PetscRandom    rnd, rndv;
  DM             dm;
  DMPolytopeType ct;
  PetscBool      simplex;
  PetscReal     *centroid, *coords, *velocity, *xi0, *v0, *J, *invJ, detJ, *vals;
  PetscInt       dim, d, cStart, cEnd, c, Np, p;

  PetscFunctionBeginUser;
  /* Randomization for coordinates */
  PetscCall(PetscRandomCreate(PetscObjectComm((PetscObject)sw), &rnd));
  PetscCall(PetscRandomSetInterval(rnd, -1.0, 1.0));
  PetscCall(PetscRandomSetFromOptions(rnd));
  PetscCall(PetscRandomCreate(PetscObjectComm((PetscObject)sw), &rndv));
  PetscCall(PetscRandomSetInterval(rndv, -1., 1.));
  PetscCall(PetscRandomSetFromOptions(rndv));
  PetscCall(DMGetApplicationContext(sw, &user));
  Np = user->N;
  PetscCall(DMGetDimension(sw, &dim));
  PetscCall(DMSwarmGetCellDM(sw, &dm));
  PetscCall(DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd));
  PetscCall(DMPlexGetCellType(dm, cStart, &ct));
  simplex = DMPolytopeTypeGetNumVertices(ct) == DMPolytopeTypeGetDim(ct) + 1 ? PETSC_TRUE : PETSC_FALSE;
  PetscCall(PetscMalloc5(dim, &centroid, dim, &xi0, dim, &v0, dim * dim, &J, dim * dim, &invJ));
  for (d = 0; d < dim; ++d) xi0[d] = -1.0;
  PetscCall(DMSwarmGetField(sw, DMSwarmPICField_coor, NULL, NULL, (void **)&coords));
  PetscCall(DMSwarmGetField(sw, "velocity", NULL, NULL, (void **)&velocity));
  PetscCall(DMSwarmGetField(sw, "w_q", NULL, NULL, (void **)&vals));
  for (c = cStart; c < cEnd; ++c) {
    if (Np == 1) {
      PetscCall(DMPlexComputeCellGeometryFVM(dm, c, NULL, centroid, NULL));
      for (d = 0; d < dim; ++d) coords[c * dim + d] = centroid[d];
      vals[c] = 1.0;
    } else {
      PetscCall(DMPlexComputeCellGeometryFEM(dm, c, NULL, v0, J, invJ, &detJ)); /* affine */
      for (p = 0; p < Np; ++p) {
        const PetscInt n   = c * Np + p;
        PetscReal      sum = 0.0, refcoords[3];

        for (d = 0; d < dim; ++d) {
          PetscCall(PetscRandomGetValueReal(rnd, &refcoords[d]));
          sum += refcoords[d];
        }
        if (simplex && sum > 0.0)
          for (d = 0; d < dim; ++d) refcoords[d] -= PetscSqrtReal(dim) * sum;
        vals[n] = 1.0;
        PetscCall(DMPlexReferenceToCoordinates(dm, c, 1, refcoords, &coords[n * dim]));
      }
    }
  }
  /* Random velocity IC */
  for (c = cStart; c < cEnd; ++c) {
    for (p = 0; p < Np; ++p) {
      for (d = 0; d < dim; ++d) {
        PetscReal v_val;

        PetscCall(PetscRandomGetValueReal(rndv, &v_val));
        velocity[p * dim + d] = v_val;
      }
    }
  }
  PetscCall(DMSwarmRestoreField(sw, DMSwarmPICField_coor, NULL, NULL, (void **)&coords));
  PetscCall(DMSwarmRestoreField(sw, "velocity", NULL, NULL, (void **)&velocity));
  PetscCall(DMSwarmRestoreField(sw, "w_q", NULL, NULL, (void **)&vals));
  PetscCall(PetscFree5(centroid, xi0, v0, J, invJ));
  PetscCall(PetscRandomDestroy(&rnd));
  PetscCall(PetscRandomDestroy(&rndv));
  PetscFunctionReturn(0);
}

/* Get velocities from swarm and place in solution vector */
static PetscErrorCode SetInitialConditions(DM dmSw, Vec u)
{
  DM           dm;
  AppCtx      *user;
  PetscReal   *velocity;
  PetscScalar *initialConditions;
  PetscInt     dim, d, cStart, cEnd, c, Np, p, n;

  PetscFunctionBeginUser;
  PetscCall(VecGetLocalSize(u, &n));
  PetscCall(DMGetApplicationContext(dmSw, &user));
  Np = user->N;
  PetscCall(DMSwarmGetCellDM(dmSw, &dm));
  PetscCall(DMGetDimension(dm, &dim));
  PetscCall(DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd));
  PetscCall(DMSwarmGetField(dmSw, "velocity", NULL, NULL, (void **)&velocity));
  PetscCall(VecGetArray(u, &initialConditions));
  for (c = cStart; c < cEnd; ++c) {
    for (p = 0; p < Np; ++p) {
      const PetscInt n = c * Np + p;
      for (d = 0; d < dim; d++) initialConditions[n * dim + d] = velocity[n * dim + d];
    }
  }
  PetscCall(VecRestoreArray(u, &initialConditions));
  PetscCall(DMSwarmRestoreField(dmSw, "velocity", NULL, NULL, (void **)&velocity));
  PetscFunctionReturn(0);
}

static PetscErrorCode CreateParticles(DM dm, DM *sw, AppCtx *user)
{
  PetscInt *cellid;
  PetscInt  dim, cStart, cEnd, c, Np = user->N, p;
  PetscBool view = PETSC_FALSE;

  PetscFunctionBeginUser;
  PetscCall(DMGetDimension(dm, &dim));
  PetscCall(DMCreate(PetscObjectComm((PetscObject)dm), sw));
  PetscCall(DMSetType(*sw, DMSWARM));
  PetscCall(DMSetDimension(*sw, dim));
  /* h = 2L/n and N = n^d */
  if (user->h < 0.) user->h = 2. * user->L / PetscPowReal(user->N, 1. / dim);
  /* From Section 4 in [1], \epsilon = 0.64 h^.98 */
  if (user->epsilon < 0.) user->epsilon = 0.64 * pow(user->h, 1.98);
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-param_view", &view, NULL));
  if (view) PetscCall(PetscPrintf(PETSC_COMM_SELF, "N: %" PetscInt_FMT " L: %g h: %g eps: %g\n", user->N, (double)user->L, (double)user->h, (double)user->epsilon));
  PetscCall(DMSwarmSetType(*sw, DMSWARM_PIC));
  PetscCall(DMSwarmSetCellDM(*sw, dm));
  PetscCall(DMSwarmRegisterPetscDatatypeField(*sw, "velocity", dim, PETSC_REAL));
  PetscCall(DMSwarmRegisterPetscDatatypeField(*sw, "w_q", 1, PETSC_SCALAR));
  PetscCall(DMSwarmFinalizeFieldRegister(*sw));
  PetscCall(DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd));
  PetscCall(DMSwarmSetLocalSizes(*sw, (cEnd - cStart) * Np, 0));
  PetscCall(DMSetFromOptions(*sw));
  PetscCall(DMSwarmGetField(*sw, DMSwarmPICField_cellid, NULL, NULL, (void **)&cellid));
  for (c = cStart; c < cEnd; ++c) {
    for (p = 0; p < Np; ++p) {
      const PetscInt n = c * Np + p;
      cellid[n]        = c;
    }
  }
  PetscCall(DMSwarmRestoreField(*sw, DMSwarmPICField_cellid, NULL, NULL, (void **)&cellid));
  PetscCall(PetscObjectSetName((PetscObject)*sw, "Particles"));
  PetscCall(DMViewFromOptions(*sw, NULL, "-sw_view"));
  PetscFunctionReturn(0);
}

/* Internal dmplex function, same as found in dmpleximpl.h */
static void DMPlex_WaxpyD_Internal(PetscInt dim, PetscReal a, const PetscReal *x, const PetscReal *y, PetscReal *w)
{
  PetscInt d;

  for (d = 0; d < dim; ++d) w[d] = a * x[d] + y[d];
}

/* Internal dmplex function, same as found in dmpleximpl.h */
static PetscReal DMPlex_DotD_Internal(PetscInt dim, const PetscScalar *x, const PetscReal *y)
{
  PetscReal sum = 0.0;
  PetscInt  d;

  for (d = 0; d < dim; ++d) sum += PetscRealPart(x[d]) * y[d];
  return sum;
}

/* Internal dmplex function, same as found in dmpleximpl.h */
static void DMPlex_MultAdd2DReal_Internal(const PetscReal A[], PetscInt ldx, const PetscScalar x[], PetscScalar y[])
{
  PetscScalar z[2];
  z[0] = x[0];
  z[1] = x[ldx];
  y[0] += A[0] * z[0] + A[1] * z[1];
  y[ldx] += A[2] * z[0] + A[3] * z[1];
  (void)PetscLogFlops(6.0);
}

/* Internal dmplex function, same as found in dmpleximpl.h to avoid private includes. */
static void DMPlex_MultAdd3DReal_Internal(const PetscReal A[], PetscInt ldx, const PetscScalar x[], PetscScalar y[])
{
  PetscScalar z[3];
  z[0] = x[0];
  z[1] = x[ldx];
  z[2] = x[ldx * 2];
  y[0] += A[0] * z[0] + A[1] * z[1] + A[2] * z[2];
  y[ldx] += A[3] * z[0] + A[4] * z[1] + A[5] * z[2];
  y[ldx * 2] += A[6] * z[0] + A[7] * z[1] + A[8] * z[2];
  (void)PetscLogFlops(15.0);
}

/*
  Gaussian - The Gaussian function G(x)

  Input Parameters:
+  dim   - The number of dimensions, or size of x
.  mu    - The mean, or center
.  sigma - The standard deviation, or width
-  x     - The evaluation point of the function

  Output Parameter:
. ret - The value G(x)
*/
static PetscReal Gaussian(PetscInt dim, const PetscReal mu[], PetscReal sigma, const PetscReal x[])
{
  PetscReal arg = 0.0;
  PetscInt  d;

  for (d = 0; d < dim; ++d) arg += PetscSqr(x[d] - mu[d]);
  return PetscPowReal(2.0 * PETSC_PI * sigma, -dim / 2.0) * PetscExpReal(-arg / (2.0 * sigma));
}

/*
  ComputeGradS - Compute grad_v dS_eps/df

  Input Parameters:
+ dim      - The dimension
. Np       - The number of particles
. vp       - The velocity v_p of the particle at which we evaluate
. velocity - The velocity field for all particles
. epsilon  - The regularization strength

  Output Parameter:
. integral - The output grad_v dS_eps/df (v_p)

  Note:
  This comes from (3.6) in [1], and we are computing
$   \nabla_v S_p = \grad \psi_\epsilon(v_p - v) log \sum_q \psi_\epsilon(v - v_q)
  which is discretized by using a one-point quadrature in each box l at its center v^c_l
$   \sum_l h^d \nabla\psi_\epsilon(v_p - v^c_l) \log\left( \sum_q w_q \psi_\epsilon(v^c_l - v_q) \right)
  where h^d is the volume of each box.
*/
static PetscErrorCode ComputeGradS(PetscInt dim, PetscInt Np, const PetscReal vp[], const PetscReal velocity[], PetscReal integral[], AppCtx *ctx)
{
  PetscReal vc_l[3], L = ctx->L, h = ctx->h, epsilon = ctx->epsilon, init = 0.5 * h - L;
  PetscInt  nx = roundf(2. * L / h);
  PetscInt  ny = dim > 1 ? nx : 1;
  PetscInt  nz = dim > 2 ? nx : 1;
  PetscInt  i, j, k, d, q, dbg = 0;

  PetscFunctionBeginHot;
  for (d = 0; d < dim; ++d) integral[d] = 0.0;
  for (k = 0, vc_l[2] = init; k < nz; ++k, vc_l[2] += h) {
    for (j = 0, vc_l[1] = init; j < ny; ++j, vc_l[1] += h) {
      for (i = 0, vc_l[0] = init; i < nx; ++i, vc_l[0] += h) {
        PetscReal sum = 0.0;

        if (dbg) PetscCall(PetscPrintf(PETSC_COMM_SELF, "(%" PetscInt_FMT " %" PetscInt_FMT ") vc_l: %g %g\n", i, j, (double)vc_l[0], (double)vc_l[1]));
        /* \log \sum_k \psi(v - v_k)  */
        for (q = 0; q < Np; ++q) sum += Gaussian(dim, &velocity[q * dim], epsilon, vc_l);
        sum = PetscLogReal(sum);
        for (d = 0; d < dim; ++d) integral[d] += (-1. / (epsilon)) * PetscAbsReal(vp[d] - vc_l[d]) * (Gaussian(dim, vp, epsilon, vc_l)) * sum;
      }
    }
  }
  PetscFunctionReturn(0);
}

/* Q = 1/|xi| (I - xi xi^T / |xi|^2), xi = vp - vq */
static PetscErrorCode QCompute(PetscInt dim, const PetscReal vp[], const PetscReal vq[], PetscReal Q[])
{
  PetscReal xi[3], xi2, xi3, mag;
  PetscInt  d, e;

  PetscFunctionBeginHot;
  DMPlex_WaxpyD_Internal(dim, -1.0, vq, vp, xi);
  xi2 = DMPlex_DotD_Internal(dim, xi, xi);
  mag = PetscSqrtReal(xi2);
  xi3 = xi2 * mag;
  for (d = 0; d < dim; ++d) {
    for (e = 0; e < dim; ++e) Q[d * dim + e] = -xi[d] * xi[e] / xi3;
    Q[d * dim + d] += 1. / mag;
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode RHSFunctionParticles(TS ts, PetscReal t, Vec U, Vec R, void *ctx)
{
  AppCtx            *user = (AppCtx *)ctx;
  PetscInt           dbg  = 0;
  DM                 sw;  /* Particles */
  Vec                sol; /* Solution vector at current time */
  const PetscScalar *u;   /* input solution vector */
  PetscScalar       *r;
  PetscReal         *velocity;
  PetscInt           dim, Np, p, q;

  PetscFunctionBeginUser;
  PetscCall(VecZeroEntries(R));
  PetscCall(TSGetDM(ts, &sw));
  PetscCall(DMGetDimension(sw, &dim));
  PetscCall(VecGetLocalSize(U, &Np));
  PetscCall(TSGetSolution(ts, &sol));
  PetscCall(VecGetArray(sol, &velocity));
  PetscCall(VecGetArray(R, &r));
  PetscCall(VecGetArrayRead(U, &u));
  Np /= dim;
  if (dbg) PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Part  ppr     x        y\n"));
  for (p = 0; p < Np; ++p) {
    PetscReal gradS_p[3] = {0., 0., 0.};

    PetscCall(ComputeGradS(dim, Np, &velocity[p * dim], velocity, gradS_p, user));
    for (q = 0; q < Np; ++q) {
      PetscReal gradS_q[3] = {0., 0., 0.}, GammaS[3] = {0., 0., 0.}, Q[9];

      if (q == p) continue;
      PetscCall(ComputeGradS(dim, Np, &velocity[q * dim], velocity, gradS_q, user));
      DMPlex_WaxpyD_Internal(dim, -1.0, gradS_q, gradS_p, GammaS);
      PetscCall(QCompute(dim, &u[p * dim], &u[q * dim], Q));
      switch (dim) {
      case 2:
        DMPlex_MultAdd2DReal_Internal(Q, 1, GammaS, &r[p * dim]);
        break;
      case 3:
        DMPlex_MultAdd3DReal_Internal(Q, 1, GammaS, &r[p * dim]);
        break;
      default:
        SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Do not support dimension %" PetscInt_FMT, dim);
      }
    }
    if (dbg) PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Final %4" PetscInt_FMT " %10.8lf %10.8lf\n", p, r[p * dim + 0], r[p * dim + 1]));
  }
  PetscCall(VecRestoreArrayRead(U, &u));
  PetscCall(VecRestoreArray(R, &r));
  PetscCall(VecRestoreArray(sol, &velocity));
  PetscCall(VecViewFromOptions(R, NULL, "-residual_view"));
  PetscFunctionReturn(0);
}

/*
 TS Post Step Function. Copy the solution back into the swarm for migration. We may also need to reform
 the solution vector in cases of particle migration, but we forgo that here since there is no velocity space grid
 to migrate between.
*/
static PetscErrorCode UpdateSwarm(TS ts)
{
  PetscInt           idx, n;
  const PetscScalar *u;
  PetscScalar       *velocity;
  DM                 sw;
  Vec                sol;

  PetscFunctionBeginUser;
  PetscCall(TSGetDM(ts, &sw));
  PetscCall(DMSwarmGetField(sw, "velocity", NULL, NULL, (void **)&velocity));
  PetscCall(TSGetSolution(ts, &sol));
  PetscCall(VecGetArrayRead(sol, &u));
  PetscCall(VecGetLocalSize(sol, &n));
  for (idx = 0; idx < n; ++idx) velocity[idx] = u[idx];
  PetscCall(VecRestoreArrayRead(sol, &u));
  PetscCall(DMSwarmRestoreField(sw, "velocity", NULL, NULL, (void **)&velocity));
  PetscFunctionReturn(0);
}

static PetscErrorCode InitializeSolve(TS ts, Vec u)
{
  DM      dm;
  AppCtx *user;

  PetscFunctionBeginUser;
  PetscCall(TSGetDM(ts, &dm));
  PetscCall(DMGetApplicationContext(dm, &user));
  PetscCall(SetInitialCoordinates(dm));
  PetscCall(SetInitialConditions(dm, u));
  PetscFunctionReturn(0);
}

int main(int argc, char **argv)
{
  TS       ts;     /* nonlinear solver */
  DM       dm, sw; /* Velocity space mesh and Particle Swarm */
  Vec      u, v;   /* problem vector */
  MPI_Comm comm;
  AppCtx   user;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  comm = PETSC_COMM_WORLD;
  PetscCall(ProcessOptions(comm, &user));
  /* Initialize objects and set initial conditions */
  PetscCall(CreateMesh(comm, &dm, &user));
  PetscCall(CreateParticles(dm, &sw, &user));
  PetscCall(DMSetApplicationContext(sw, &user));
  PetscCall(DMSwarmVectorDefineField(sw, "velocity"));
  PetscCall(TSCreate(comm, &ts));
  PetscCall(TSSetDM(ts, sw));
  PetscCall(TSSetMaxTime(ts, 10.0));
  PetscCall(TSSetTimeStep(ts, 0.1));
  PetscCall(TSSetMaxSteps(ts, 1));
  PetscCall(TSSetExactFinalTime(ts, TS_EXACTFINALTIME_MATCHSTEP));
  PetscCall(TSSetRHSFunction(ts, NULL, RHSFunctionParticles, &user));
  PetscCall(TSSetFromOptions(ts));
  PetscCall(TSSetComputeInitialCondition(ts, InitializeSolve));
  PetscCall(DMSwarmCreateGlobalVectorFromField(sw, "velocity", &v));
  PetscCall(VecDuplicate(v, &u));
  PetscCall(DMSwarmDestroyGlobalVectorFromField(sw, "velocity", &v));
  PetscCall(TSComputeInitialCondition(ts, u));
  PetscCall(TSSetPostStep(ts, UpdateSwarm));
  PetscCall(TSSolve(ts, u));
  PetscCall(VecDestroy(&u));
  PetscCall(TSDestroy(&ts));
  PetscCall(DMDestroy(&sw));
  PetscCall(DMDestroy(&dm));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST
   build:
     requires: triangle !single !complex
   test:
     suffix: midpoint
     args: -N 3 -dm_plex_dim 2 -dm_plex_simplex 0 -dm_plex_box_faces 1,1 -dm_plex_box_lower -1,-1 -dm_plex_box_upper 1,1 -dm_view \
           -ts_type theta -ts_theta_theta 0.5 -ts_dmswarm_monitor_moments -ts_monitor_frequency 1 -snes_fd
TEST*/
