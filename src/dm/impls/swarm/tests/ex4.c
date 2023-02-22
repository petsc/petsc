static char help[] = "Testing integrators on the simple harmonic oscillator\n";

/*
  In order to check long time behavior, you can give

     -ts_max_steps 10000 -ts_max_time 10.0 -output_step 1000

  To look at the long time behavior for different integrators, we can use the nergy monitor. Below we see that Euler is almost 100 times worse at conserving energy than the symplectic integrator of the same order.

    make -f ./gmakefile test search="dm_impls_swarm_tests-ex4_bsi_1"
      EXTRA_OPTIONS="-ts_max_steps 10000 -ts_max_time 10.0 -output_step 1000 -ts_type euler" | grep error

      energy/exact energy 398.236 / 396.608 (0.4104399231)
      energy/exact energy 1579.52 / 1573.06 (0.4104399231)
      energy/exact energy 397.421 / 396.608 (0.2050098479)
      energy/exact energy 1576.29 / 1573.06 (0.2050098479)
      energy/exact energy 397.014 / 396.608 (0.1024524454)
      energy/exact energy 1574.68 / 1573.06 (0.1024524454)

    make -f ./gmakefile test search="dm_impls_swarm_tests-ex4_bsi_1"
      EXTRA_OPTIONS="-ts_max_steps 10000 -ts_max_time 10.0 -output_step 1000" | grep error

      energy/exact energy 396.579 / 396.608 (0.0074080434)
      energy/exact energy 1572.95 / 1573.06 (0.0074080434)
      energy/exact energy 396.593 / 396.608 (0.0037040885)
      energy/exact energy 1573.01 / 1573.06 (0.0037040886)
      energy/exact energy 396.601 / 396.608 (0.0018520613)
      energy/exact energy 1573.04 / 1573.06 (0.0018520613)

  We can look at third order integrators in the same way, but we need to use more steps.

    make -f ./gmakefile test search="dm_impls_swarm_tests-ex4_bsi_1"
      EXTRA_OPTIONS="-ts_max_steps 1000000 -ts_max_time 1000.0 -output_step 100000 -ts_type rk -ts_adapt_type none" | grep error

      energy/exact energy 396.608 / 396.608 (0.0000013981)
      energy/exact energy 1573.06 / 1573.06 (0.0000013981)
      energy/exact energy 396.608 / 396.608 (0.0000001747)
      energy/exact energy 1573.06 / 1573.06 (0.0000001748)
      energy/exact energy 396.608 / 396.608 (0.0000000218)
      energy/exact energy 1573.06 / 1573.06 (0.0000000218)

    make -f ./gmakefile test search="dm_impls_swarm_tests-ex4_bsi_3"
      EXTRA_OPTIONS="-ts_max_steps 1000000 -ts_max_time 1000.0 -output_step 100000 -ts_adapt_type none" | grep error

      energy/exact energy 396.608 / 396.608 (0.0000000007)
      energy/exact energy 1573.06 / 1573.06 (0.0000000007)
      energy/exact energy 396.608 / 396.608 (0.0000000001)
      energy/exact energy 1573.06 / 1573.06 (0.0000000001)
      energy/exact energy 396.608 / 396.608 (0.0000000000)
      energy/exact energy 1573.06 / 1573.06 (0.0000000000)
*/

#include <petscts.h>
#include <petscdmplex.h>
#include <petscdmswarm.h>
#include <petsc/private/dmpleximpl.h> /* For norm and dot */

typedef struct {
  PetscReal omega; /* Oscillation frequency omega */
  PetscBool error; /* Flag for printing the error */
  PetscInt  ostep; /* print the energy at each ostep time steps */
} AppCtx;

static PetscErrorCode ProcessOptions(MPI_Comm comm, AppCtx *options)
{
  PetscFunctionBeginUser;
  options->omega = 64.0;
  options->error = PETSC_FALSE;
  options->ostep = 100;

  PetscOptionsBegin(comm, "", "Harmonic Oscillator Options", "DMSWARM");
  PetscCall(PetscOptionsReal("-omega", "Oscillator frequency", "ex4.c", options->omega, &options->omega, NULL));
  PetscCall(PetscOptionsBool("-error", "Flag to print the error", "ex4.c", options->error, &options->error, NULL));
  PetscCall(PetscOptionsInt("-output_step", "Number of time steps between output", "ex4.c", options->ostep, &options->ostep, NULL));
  PetscOptionsEnd();
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode CreateMesh(MPI_Comm comm, AppCtx *user, DM *dm)
{
  PetscFunctionBeginUser;
  PetscCall(DMCreate(comm, dm));
  PetscCall(DMSetType(*dm, DMPLEX));
  PetscCall(DMSetFromOptions(*dm));
  PetscCall(DMViewFromOptions(*dm, NULL, "-dm_view"));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode CreateSwarm(DM dm, AppCtx *user, DM *sw)
{
  PetscReal v0[1] = {0.}; // Initialize velocities to 0
  PetscInt  dim;

  PetscFunctionBeginUser;
  PetscCall(DMGetDimension(dm, &dim));
  PetscCall(DMCreate(PetscObjectComm((PetscObject)dm), sw));
  PetscCall(DMSetType(*sw, DMSWARM));
  PetscCall(DMSetDimension(*sw, dim));
  PetscCall(DMSwarmSetType(*sw, DMSWARM_PIC));
  PetscCall(DMSwarmSetCellDM(*sw, dm));
  PetscCall(DMSwarmRegisterPetscDatatypeField(*sw, "w_q", 1, PETSC_SCALAR));
  PetscCall(DMSwarmRegisterPetscDatatypeField(*sw, "velocity", dim, PETSC_REAL));
  PetscCall(DMSwarmRegisterPetscDatatypeField(*sw, "species", 1, PETSC_INT));
  PetscCall(DMSwarmRegisterPetscDatatypeField(*sw, "initCoordinates", dim, PETSC_REAL));
  PetscCall(DMSwarmFinalizeFieldRegister(*sw));
  PetscCall(DMSwarmComputeLocalSizeFromOptions(*sw));
  PetscCall(DMSwarmInitializeCoordinates(*sw));
  PetscCall(DMSwarmInitializeVelocitiesFromOptions(*sw, v0));
  PetscCall(DMSetFromOptions(*sw));
  PetscCall(DMSetApplicationContext(*sw, user));
  PetscCall(PetscObjectSetName((PetscObject)*sw, "Particles"));
  PetscCall(DMViewFromOptions(*sw, NULL, "-sw_view"));
  {
    Vec gc, gc0;

    PetscCall(DMSwarmCreateGlobalVectorFromField(*sw, DMSwarmPICField_coor, &gc));
    PetscCall(DMSwarmCreateGlobalVectorFromField(*sw, "initCoordinates", &gc0));
    PetscCall(VecCopy(gc, gc0));
    PetscCall(DMSwarmDestroyGlobalVectorFromField(*sw, DMSwarmPICField_coor, &gc));
    PetscCall(DMSwarmDestroyGlobalVectorFromField(*sw, "initCoordinates", &gc0));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode RHSFunction(TS ts, PetscReal t, Vec U, Vec G, void *ctx)
{
  const PetscReal    omega = ((AppCtx *)ctx)->omega;
  DM                 sw;
  const PetscScalar *u;
  PetscScalar       *g;
  PetscInt           dim, d, Np, p;

  PetscFunctionBeginUser;
  PetscCall(TSGetDM(ts, &sw));
  PetscCall(DMGetDimension(sw, &dim));
  PetscCall(VecGetLocalSize(U, &Np));
  PetscCall(VecGetArray(G, &g));
  PetscCall(VecGetArrayRead(U, &u));
  Np /= 2 * dim;
  for (p = 0; p < Np; ++p) {
    for (d = 0; d < dim; ++d) {
      g[(p * 2 + 0) * dim + d] = u[(p * 2 + 1) * dim + d];
      g[(p * 2 + 1) * dim + d] = -PetscSqr(omega) * u[(p * 2 + 0) * dim + d];
    }
  }
  PetscCall(VecRestoreArrayRead(U, &u));
  PetscCall(VecRestoreArray(G, &g));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* J_{ij} = dF_i/dx_j
   J_p = (  0   1)
         (-w^2  0)
*/
static PetscErrorCode RHSJacobian(TS ts, PetscReal t, Vec U, Mat J, Mat P, void *ctx)
{
  PetscScalar vals[4] = {0., 1., -PetscSqr(((AppCtx *)ctx)->omega), 0.};
  DM          sw;
  PetscInt    dim, d, Np, p, rStart;

  PetscFunctionBeginUser;
  PetscCall(TSGetDM(ts, &sw));
  PetscCall(DMGetDimension(sw, &dim));
  PetscCall(VecGetLocalSize(U, &Np));
  PetscCall(MatGetOwnershipRange(J, &rStart, NULL));
  Np /= 2 * dim;
  for (p = 0; p < Np; ++p) {
    for (d = 0; d < dim; ++d) {
      const PetscInt rows[2] = {(p * 2 + 0) * dim + d + rStart, (p * 2 + 1) * dim + d + rStart};
      PetscCall(MatSetValues(J, 2, rows, 2, rows, vals, INSERT_VALUES));
    }
  }
  PetscCall(MatAssemblyBegin(J, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(J, MAT_FINAL_ASSEMBLY));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode RHSFunctionX(TS ts, PetscReal t, Vec V, Vec Xres, void *ctx)
{
  const PetscScalar *v;
  PetscScalar       *xres;
  PetscInt           Np, p;

  PetscFunctionBeginUser;
  PetscCall(VecGetLocalSize(Xres, &Np));
  PetscCall(VecGetArrayRead(V, &v));
  PetscCall(VecGetArray(Xres, &xres));
  for (p = 0; p < Np; ++p) xres[p] = v[p];
  PetscCall(VecRestoreArrayRead(V, &v));
  PetscCall(VecRestoreArray(Xres, &xres));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode RHSFunctionV(TS ts, PetscReal t, Vec X, Vec Vres, void *ctx)
{
  const PetscReal    omega = ((AppCtx *)ctx)->omega;
  const PetscScalar *x;
  PetscScalar       *vres;
  PetscInt           Np, p;

  PetscFunctionBeginUser;
  PetscCall(VecGetArray(Vres, &vres));
  PetscCall(VecGetArrayRead(X, &x));
  PetscCall(VecGetLocalSize(Vres, &Np));
  for (p = 0; p < Np; ++p) vres[p] = -PetscSqr(omega) * x[p];
  PetscCall(VecRestoreArrayRead(X, &x));
  PetscCall(VecRestoreArray(Vres, &vres));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode RHSJacobianS(TS ts, PetscReal t, Vec U, Mat S, void *ctx)
{
  PetscScalar vals[4] = {0., 1., -1., 0.};
  DM          sw;
  PetscInt    dim, d, Np, p, rStart;

  PetscFunctionBeginUser;
  PetscCall(TSGetDM(ts, &sw));
  PetscCall(DMGetDimension(sw, &dim));
  PetscCall(VecGetLocalSize(U, &Np));
  PetscCall(MatGetOwnershipRange(S, &rStart, NULL));
  Np /= 2 * dim;
  for (p = 0; p < Np; ++p) {
    for (d = 0; d < dim; ++d) {
      const PetscInt rows[2] = {(p * 2 + 0) * dim + d + rStart, (p * 2 + 1) * dim + d + rStart};
      PetscCall(MatSetValues(S, 2, rows, 2, rows, vals, INSERT_VALUES));
    }
  }
  PetscCall(MatAssemblyBegin(S, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(S, MAT_FINAL_ASSEMBLY));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode RHSObjectiveF(TS ts, PetscReal t, Vec U, PetscScalar *F, void *ctx)
{
  const PetscReal    omega = ((AppCtx *)ctx)->omega;
  DM                 sw;
  const PetscScalar *u;
  PetscInt           dim, Np, p;

  PetscFunctionBeginUser;
  PetscCall(TSGetDM(ts, &sw));
  PetscCall(DMGetDimension(sw, &dim));
  PetscCall(VecGetArrayRead(U, &u));
  PetscCall(VecGetLocalSize(U, &Np));
  Np /= 2 * dim;
  for (p = 0; p < Np; ++p) {
    const PetscReal x2 = DMPlex_DotRealD_Internal(dim, &u[(p * 2 + 0) * dim], &u[(p * 2 + 0) * dim]);
    const PetscReal v2 = DMPlex_DotRealD_Internal(dim, &u[(p * 2 + 1) * dim], &u[(p * 2 + 1) * dim]);
    const PetscReal E  = 0.5 * (v2 + PetscSqr(omega) * x2);

    *F += E;
  }
  PetscCall(VecRestoreArrayRead(U, &u));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* dF/dx = omega^2 x   dF/dv = v */
PetscErrorCode RHSFunctionG(TS ts, PetscReal t, Vec U, Vec G, void *ctx)
{
  const PetscReal    omega = ((AppCtx *)ctx)->omega;
  DM                 sw;
  const PetscScalar *u;
  PetscScalar       *g;
  PetscInt           dim, d, Np, p;

  PetscFunctionBeginUser;
  PetscCall(TSGetDM(ts, &sw));
  PetscCall(DMGetDimension(sw, &dim));
  PetscCall(VecGetArray(G, &g));
  PetscCall(VecGetArrayRead(U, &u));
  PetscCall(VecGetLocalSize(U, &Np));
  Np /= 2 * dim;
  for (p = 0; p < Np; ++p) {
    for (d = 0; d < dim; ++d) {
      g[(p * 2 + 0) * dim + d] = PetscSqr(omega) * u[(p * 2 + 0) * dim + d];
      g[(p * 2 + 1) * dim + d] = u[(p * 2 + 1) * dim + d];
    }
  }
  PetscCall(VecRestoreArrayRead(U, &u));
  PetscCall(VecRestoreArray(G, &g));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode CreateSolution(TS ts)
{
  DM       sw;
  Vec      u;
  PetscInt dim, Np;

  PetscFunctionBegin;
  PetscCall(TSGetDM(ts, &sw));
  PetscCall(DMGetDimension(sw, &dim));
  PetscCall(DMSwarmGetLocalSize(sw, &Np));
  PetscCall(VecCreate(PETSC_COMM_WORLD, &u));
  PetscCall(VecSetBlockSize(u, dim));
  PetscCall(VecSetSizes(u, 2 * Np * dim, PETSC_DECIDE));
  PetscCall(VecSetUp(u));
  PetscCall(TSSetSolution(ts, u));
  PetscCall(VecDestroy(&u));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode SetProblem(TS ts)
{
  AppCtx *user;
  DM      sw;

  PetscFunctionBegin;
  PetscCall(TSGetDM(ts, &sw));
  PetscCall(DMGetApplicationContext(sw, (void **)&user));
  // Define unified system for (X, V)
  {
    Mat      J;
    PetscInt dim, Np;

    PetscCall(DMGetDimension(sw, &dim));
    PetscCall(DMSwarmGetLocalSize(sw, &Np));
    PetscCall(MatCreate(PETSC_COMM_WORLD, &J));
    PetscCall(MatSetSizes(J, 2 * Np * dim, 2 * Np * dim, PETSC_DECIDE, PETSC_DECIDE));
    PetscCall(MatSetBlockSize(J, 2 * dim));
    PetscCall(MatSetFromOptions(J));
    PetscCall(MatSetUp(J));
    PetscCall(TSSetRHSFunction(ts, NULL, RHSFunction, user));
    PetscCall(TSSetRHSJacobian(ts, J, J, RHSJacobian, user));
    PetscCall(MatDestroy(&J));
  }
  // Define split system for X and V
  {
    IS              isx, isv, istmp;
    const PetscInt *idx;
    PetscInt        dim, Np;

    PetscCall(DMGetDimension(sw, &dim));
    PetscCall(DMSwarmGetLocalSize(sw, &Np));
    PetscCall(ISCreateStride(PETSC_COMM_SELF, Np, 0, 2, &istmp));
    PetscCall(ISGetIndices(istmp, &idx));
    PetscCall(ISCreateBlock(PETSC_COMM_SELF, dim, Np, idx, PETSC_COPY_VALUES, &isx));
    PetscCall(ISRestoreIndices(istmp, &idx));
    PetscCall(ISDestroy(&istmp));
    PetscCall(ISCreateStride(PETSC_COMM_SELF, Np, 1, 2, &istmp));
    PetscCall(ISGetIndices(istmp, &idx));
    PetscCall(ISCreateBlock(PETSC_COMM_SELF, dim, Np, idx, PETSC_COPY_VALUES, &isv));
    PetscCall(ISRestoreIndices(istmp, &idx));
    PetscCall(ISDestroy(&istmp));
    PetscCall(TSRHSSplitSetIS(ts, "position", isx));
    PetscCall(TSRHSSplitSetIS(ts, "momentum", isv));
    PetscCall(ISDestroy(&isx));
    PetscCall(ISDestroy(&isv));
    PetscCall(TSRHSSplitSetRHSFunction(ts, "position", NULL, RHSFunctionX, user));
    PetscCall(TSRHSSplitSetRHSFunction(ts, "momentum", NULL, RHSFunctionV, user));
  }
  // Define symplectic formulation U_t = S . G, where G = grad F
  {
    PetscCall(TSDiscGradSetFormulation(ts, RHSJacobianS, RHSObjectiveF, RHSFunctionG, user));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode InitializeSolve(TS ts, Vec u)
{
  DM      sw;
  Vec     gc, gc0;
  IS      isx, isv;
  AppCtx *user;

  PetscFunctionBeginUser;
  PetscCall(TSGetDM(ts, &sw));
  PetscCall(DMGetApplicationContext(sw, &user));
  {
    PetscReal v0[1] = {1.};

    PetscCall(DMSwarmInitializeCoordinates(sw));
    PetscCall(DMSwarmInitializeVelocitiesFromOptions(sw, v0));
    PetscCall(SetProblem(ts));
  }
  PetscCall(TSRHSSplitGetIS(ts, "position", &isx));
  PetscCall(TSRHSSplitGetIS(ts, "momentum", &isv));
  PetscCall(DMSwarmCreateGlobalVectorFromField(sw, DMSwarmPICField_coor, &gc));
  PetscCall(DMSwarmCreateGlobalVectorFromField(sw, "initCoordinates", &gc0));
  PetscCall(VecCopy(gc, gc0));
  PetscCall(VecISCopy(u, isx, SCATTER_FORWARD, gc));
  PetscCall(DMSwarmDestroyGlobalVectorFromField(sw, DMSwarmPICField_coor, &gc));
  PetscCall(DMSwarmDestroyGlobalVectorFromField(sw, "initCoordinates", &gc0));
  PetscCall(VecISSet(u, isv, 0.));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode ComputeError(TS ts, Vec U, Vec E)
{
  MPI_Comm           comm;
  DM                 sw;
  AppCtx            *user;
  const PetscScalar *u;
  const PetscReal   *coords;
  PetscScalar       *e;
  PetscReal          t;
  PetscInt           dim, d, Np, p;

  PetscFunctionBeginUser;
  PetscCall(PetscObjectGetComm((PetscObject)ts, &comm));
  PetscCall(TSGetDM(ts, &sw));
  PetscCall(DMGetApplicationContext(sw, &user));
  PetscCall(DMGetDimension(sw, &dim));
  PetscCall(TSGetSolveTime(ts, &t));
  PetscCall(VecGetArray(E, &e));
  PetscCall(VecGetArrayRead(U, &u));
  PetscCall(VecGetLocalSize(U, &Np));
  PetscCall(DMSwarmGetField(sw, "initCoordinates", NULL, NULL, (void **)&coords));
  Np /= 2 * dim;
  for (p = 0; p < Np; ++p) {
    const PetscReal omega = user->omega;
    const PetscReal ct    = PetscCosReal(omega * t);
    const PetscReal st    = PetscSinReal(omega * t);
    const PetscReal x0    = DMPlex_NormD_Internal(dim, &coords[p * dim]);
    const PetscReal ex    = x0 * ct;
    const PetscReal ev    = -x0 * omega * st;
    const PetscReal x     = DMPlex_DotRealD_Internal(dim, &u[(p * 2 + 0) * dim], &coords[p * dim]) / x0;
    const PetscReal v     = DMPlex_DotRealD_Internal(dim, &u[(p * 2 + 1) * dim], &coords[p * dim]) / x0;

    if (user->error) {
      const PetscReal en   = 0.5 * (v * v + PetscSqr(omega) * x * x);
      const PetscReal exen = 0.5 * PetscSqr(omega * x0);
      PetscCall(PetscPrintf(comm, "p%" PetscInt_FMT " error [%.2g %.2g] sol [%.6lf %.6lf] exact [%.6lf %.6lf] energy/exact energy %g / %g (%.10lf%%)\n", p, (double)PetscAbsReal(x - ex), (double)PetscAbsReal(v - ev), (double)x, (double)v, (double)ex, (double)ev, (double)en, (double)exen, (double)(PetscAbsReal(exen - en) * 100. / exen)));
    }
    for (d = 0; d < dim; ++d) {
      e[(p * 2 + 0) * dim + d] = u[(p * 2 + 0) * dim + d] - coords[p * dim + d] * ct;
      e[(p * 2 + 1) * dim + d] = u[(p * 2 + 1) * dim + d] + coords[p * dim + d] * omega * st;
    }
  }
  PetscCall(DMSwarmRestoreField(sw, "initCoordinates", NULL, NULL, (void **)&coords));
  PetscCall(VecRestoreArrayRead(U, &u));
  PetscCall(VecRestoreArray(E, &e));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode EnergyMonitor(TS ts, PetscInt step, PetscReal t, Vec U, void *ctx)
{
  const PetscReal    omega = ((AppCtx *)ctx)->omega;
  const PetscInt     ostep = ((AppCtx *)ctx)->ostep;
  DM                 sw;
  const PetscScalar *u;
  PetscReal          dt;
  PetscInt           dim, Np, p;
  MPI_Comm           comm;

  PetscFunctionBeginUser;
  if (step % ostep == 0) {
    PetscCall(PetscObjectGetComm((PetscObject)ts, &comm));
    PetscCall(TSGetDM(ts, &sw));
    PetscCall(TSGetTimeStep(ts, &dt));
    PetscCall(DMGetDimension(sw, &dim));
    PetscCall(VecGetArrayRead(U, &u));
    PetscCall(VecGetLocalSize(U, &Np));
    Np /= 2 * dim;
    if (!step) PetscCall(PetscPrintf(comm, "Time     Step Part     Energy Mod Energy\n"));
    for (p = 0; p < Np; ++p) {
      const PetscReal x2 = DMPlex_DotRealD_Internal(dim, &u[(p * 2 + 0) * dim], &u[(p * 2 + 0) * dim]);
      const PetscReal v2 = DMPlex_DotRealD_Internal(dim, &u[(p * 2 + 1) * dim], &u[(p * 2 + 1) * dim]);
      const PetscReal E  = 0.5 * (v2 + PetscSqr(omega) * x2);
      const PetscReal mE = 0.5 * (v2 + PetscSqr(omega) * x2 - PetscSqr(omega) * dt * PetscSqrtReal(x2 * v2));

      PetscCall(PetscPrintf(comm, "%.6lf %4" PetscInt_FMT " %4" PetscInt_FMT " %10.4lf %10.4lf\n", (double)t, step, p, (double)E, (double)mE));
    }
    PetscCall(VecRestoreArrayRead(U, &u));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv)
{
  DM     dm, sw;
  TS     ts;
  Vec    u;
  AppCtx user;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  PetscCall(ProcessOptions(PETSC_COMM_WORLD, &user));
  PetscCall(CreateMesh(PETSC_COMM_WORLD, &user, &dm));
  PetscCall(CreateSwarm(dm, &user, &sw));
  PetscCall(DMSetApplicationContext(sw, &user));

  PetscCall(TSCreate(PETSC_COMM_WORLD, &ts));
  PetscCall(TSSetProblemType(ts, TS_NONLINEAR));
  PetscCall(TSSetDM(ts, sw));
  PetscCall(TSSetMaxTime(ts, 0.1));
  PetscCall(TSSetTimeStep(ts, 0.00001));
  PetscCall(TSSetMaxSteps(ts, 100));
  PetscCall(TSSetExactFinalTime(ts, TS_EXACTFINALTIME_MATCHSTEP));
  PetscCall(TSMonitorSet(ts, EnergyMonitor, &user, NULL));
  PetscCall(TSSetFromOptions(ts));
  PetscCall(TSSetComputeInitialCondition(ts, InitializeSolve));
  PetscCall(TSSetComputeExactError(ts, ComputeError));

  PetscCall(CreateSolution(ts));
  PetscCall(TSGetSolution(ts, &u));
  PetscCall(TSComputeInitialCondition(ts, u));
  PetscCall(TSSolve(ts, NULL));

  PetscCall(TSDestroy(&ts));
  PetscCall(DMDestroy(&sw));
  PetscCall(DMDestroy(&dm));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   build:
     requires: triangle !single !complex

   testset:
     args: -dm_plex_dim 1 -dm_plex_box_faces 1 -dm_plex_box_lower -1 -dm_plex_box_upper 1 \
           -dm_swarm_num_particles 2 -dm_swarm_coordinate_density constant \
           -ts_type basicsymplectic -ts_convergence_estimate -convest_num_refine 2 \
           -dm_view -output_step 50 -error
     test:
       suffix: bsi_1
       args: -ts_basicsymplectic_type 1
     test:
       suffix: bsi_2
       args: -ts_basicsymplectic_type 2
     test:
       suffix: bsi_3
       args: -ts_basicsymplectic_type 3
     test:
       suffix: bsi_4
       args: -ts_basicsymplectic_type 4 -ts_dt 0.0001

   testset:
     args: -dm_plex_dim 2 -dm_plex_simplex 0 -dm_plex_box_faces 1,1 -dm_plex_box_lower -1,-1 -dm_plex_box_upper 1,1 \
           -dm_swarm_num_particles 2 -dm_swarm_coordinate_density constant \
           -ts_type basicsymplectic -ts_convergence_estimate -convest_num_refine 2 \
           -dm_view -output_step 50 -error
     test:
       suffix: bsi_2d_1
       args: -ts_basicsymplectic_type 1
     test:
       suffix: bsi_2d_2
       args: -ts_basicsymplectic_type 2
     test:
       suffix: bsi_2d_3
       args: -ts_basicsymplectic_type 3
     test:
       suffix: bsi_2d_4
       args: -ts_basicsymplectic_type 4 -ts_dt 0.0001

   testset:
     args: -dm_plex_dim 3 -dm_plex_simplex 0 -dm_plex_box_faces 1,1 -dm_plex_box_lower -1,-1,-1 -dm_plex_box_upper 1,1,1 \
           -dm_swarm_num_particles 2 -dm_swarm_coordinate_density constant \
           -ts_type basicsymplectic -ts_convergence_estimate -convest_num_refine 2 \
           -dm_view -output_step 50 -error
     test:
       suffix: bsi_3d_1
       args: -ts_basicsymplectic_type 1
     test:
       suffix: bsi_3d_2
       args: -ts_basicsymplectic_type 2
     test:
       suffix: bsi_3d_3
       args: -ts_basicsymplectic_type 3
     test:
       suffix: bsi_3d_4
       args: -ts_basicsymplectic_type 4 -ts_dt 0.0001

   testset:
     args: -dm_swarm_num_particles 2 -dm_swarm_coordinate_density constant \
           -ts_type theta -ts_theta_theta 0.5 -ts_convergence_estimate -convest_num_refine 2 \
             -mat_type baij -ksp_error_if_not_converged -pc_type lu \
           -dm_view -output_step 50 -error
     test:
       suffix: im_1d
       args: -dm_plex_dim 1 -dm_plex_box_faces 1 -dm_plex_box_lower -1 -dm_plex_box_upper 1
     test:
       suffix: im_2d
       args: -dm_plex_dim 2 -dm_plex_simplex 0 -dm_plex_box_faces 1,1 -dm_plex_box_lower -1,-1 -dm_plex_box_upper 1,1
     test:
       suffix: im_3d
       args: -dm_plex_dim 3 -dm_plex_simplex 0 -dm_plex_box_faces 1,1,1 -dm_plex_box_lower -1,-1,-1 -dm_plex_box_upper 1,1,1

   testset:
     args: -dm_swarm_num_particles 2 -dm_swarm_coordinate_density constant \
           -ts_type discgrad -ts_discgrad_gonzalez -ts_convergence_estimate -convest_num_refine 2 \
             -mat_type baij -ksp_error_if_not_converged -pc_type lu \
           -dm_view -output_step 50 -error
     test:
       suffix: dg_1d
       args: -dm_plex_dim 1 -dm_plex_box_faces 1 -dm_plex_box_lower -1 -dm_plex_box_upper 1
     test:
       suffix: dg_2d
       args: -dm_plex_dim 2 -dm_plex_simplex 0 -dm_plex_box_faces 1,1 -dm_plex_box_lower -1,-1 -dm_plex_box_upper 1,1
     test:
       suffix: dg_3d
       args: -dm_plex_dim 3 -dm_plex_simplex 0 -dm_plex_box_faces 1,1,1 -dm_plex_box_lower -1,-1,-1 -dm_plex_box_upper 1,1,1

TEST*/
