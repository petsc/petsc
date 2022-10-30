static char help[] = "Vlasov example of central orbits\n";

/*
  To visualize the orbit, we can used

    -ts_monitor_sp_swarm -ts_monitor_sp_swarm_retain -1 -ts_monitor_sp_swarm_phase 0 -draw_size 500,500

  and we probably want it to run fast and not check convergence

    -convest_num_refine 0 -ts_dt 0.01 -ts_max_steps 100 -ts_max_time 100 -output_step 10
*/

#include <petscts.h>
#include <petscdmplex.h>
#include <petscdmswarm.h>
#include <petsc/private/dmpleximpl.h> /* For norm and dot */

PETSC_EXTERN PetscErrorCode circleSingleX(PetscInt, PetscReal, const PetscReal[], PetscInt, PetscScalar[], void *);
PETSC_EXTERN PetscErrorCode circleSingleV(PetscInt, PetscReal, const PetscReal[], PetscInt, PetscScalar[], void *);
PETSC_EXTERN PetscErrorCode circleMultipleX(PetscInt, PetscReal, const PetscReal[], PetscInt, PetscScalar[], void *);
PETSC_EXTERN PetscErrorCode circleMultipleV(PetscInt, PetscReal, const PetscReal[], PetscInt, PetscScalar[], void *);

typedef struct {
  PetscBool error; /* Flag for printing the error */
  PetscInt  ostep; /* print the energy at each ostep time steps */
} AppCtx;

static PetscErrorCode ProcessOptions(MPI_Comm comm, AppCtx *options)
{
  PetscFunctionBeginUser;
  options->error = PETSC_FALSE;
  options->ostep = 100;

  PetscOptionsBegin(comm, "", "Central Orbit Options", "DMSWARM");
  PetscCall(PetscOptionsBool("-error", "Flag to print the error", "ex5.c", options->error, &options->error, NULL));
  PetscCall(PetscOptionsInt("-output_step", "Number of time steps between output", "ex5.c", options->ostep, &options->ostep, PETSC_NULL));
  PetscOptionsEnd();
  PetscFunctionReturn(0);
}

static PetscErrorCode CreateMesh(MPI_Comm comm, AppCtx *user, DM *dm)
{
  PetscFunctionBeginUser;
  PetscCall(DMCreate(comm, dm));
  PetscCall(DMSetType(*dm, DMPLEX));
  PetscCall(DMSetFromOptions(*dm));
  PetscCall(DMViewFromOptions(*dm, NULL, "-dm_view"));
  PetscFunctionReturn(0);
}

static PetscErrorCode CreateSwarm(DM dm, AppCtx *user, DM *sw)
{
  PetscReal v0[1] = {1.};
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
  PetscCall(DMSwarmRegisterPetscDatatypeField(*sw, "initVelocity", dim, PETSC_REAL));
  PetscCall(DMSwarmFinalizeFieldRegister(*sw));
  PetscCall(DMSwarmComputeLocalSizeFromOptions(*sw));
  PetscCall(DMSwarmInitializeCoordinates(*sw));
  PetscCall(DMSwarmInitializeVelocitiesFromOptions(*sw, v0));
  PetscCall(DMSetFromOptions(*sw));
  PetscCall(DMSetApplicationContext(*sw, user));
  PetscCall(PetscObjectSetName((PetscObject)*sw, "Particles"));
  PetscCall(DMViewFromOptions(*sw, NULL, "-sw_view"));
  {
    Vec gc, gc0, gv, gv0;

    PetscCall(DMSwarmCreateGlobalVectorFromField(*sw, DMSwarmPICField_coor, &gc));
    PetscCall(DMSwarmCreateGlobalVectorFromField(*sw, "initCoordinates", &gc0));
    PetscCall(VecCopy(gc, gc0));
    PetscCall(DMSwarmDestroyGlobalVectorFromField(*sw, DMSwarmPICField_coor, &gc));
    PetscCall(DMSwarmDestroyGlobalVectorFromField(*sw, "initCoordinates", &gc0));
    PetscCall(DMSwarmCreateGlobalVectorFromField(*sw, "velocity", &gv));
    PetscCall(DMSwarmCreateGlobalVectorFromField(*sw, "initVelocity", &gv0));
    PetscCall(VecCopy(gv, gv0));
    PetscCall(DMSwarmDestroyGlobalVectorFromField(*sw, "velocity", &gv));
    PetscCall(DMSwarmDestroyGlobalVectorFromField(*sw, "initVelocity", &gv0));
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode RHSFunction(TS ts, PetscReal t, Vec U, Vec G, void *ctx)
{
  DM                 sw;
  const PetscReal   *coords, *vel;
  const PetscScalar *u;
  PetscScalar       *g;
  PetscInt           dim, d, Np, p;

  PetscFunctionBeginUser;
  PetscCall(TSGetDM(ts, &sw));
  PetscCall(DMGetDimension(sw, &dim));
  PetscCall(DMSwarmGetField(sw, "initCoordinates", NULL, NULL, (void **)&coords));
  PetscCall(DMSwarmGetField(sw, "initVelocity", NULL, NULL, (void **)&vel));
  PetscCall(VecGetLocalSize(U, &Np));
  PetscCall(VecGetArrayRead(U, &u));
  PetscCall(VecGetArray(G, &g));
  Np /= 2 * dim;
  for (p = 0; p < Np; ++p) {
    const PetscReal x0    = coords[p * dim + 0];
    const PetscReal vy0   = vel[p * dim + 1];
    const PetscReal omega = vy0 / x0;

    for (d = 0; d < dim; ++d) {
      g[(p * 2 + 0) * dim + d] = u[(p * 2 + 1) * dim + d];
      g[(p * 2 + 1) * dim + d] = -PetscSqr(omega) * u[(p * 2 + 0) * dim + d];
    }
  }
  PetscCall(DMSwarmRestoreField(sw, "initCoordinates", NULL, NULL, (void **)&coords));
  PetscCall(DMSwarmRestoreField(sw, "initVelocity", NULL, NULL, (void **)&vel));
  PetscCall(VecRestoreArrayRead(U, &u));
  PetscCall(VecRestoreArray(G, &g));
  PetscFunctionReturn(0);
}

/* J_{ij} = dF_i/dx_j
   J_p = (  0   1)
         (-w^2  0)
*/
static PetscErrorCode RHSJacobian(TS ts, PetscReal t, Vec U, Mat J, Mat P, void *ctx)
{
  DM               sw;
  const PetscReal *coords, *vel;
  PetscInt         dim, d, Np, p, rStart;

  PetscFunctionBeginUser;
  PetscCall(TSGetDM(ts, &sw));
  PetscCall(DMGetDimension(sw, &dim));
  PetscCall(VecGetLocalSize(U, &Np));
  PetscCall(MatGetOwnershipRange(J, &rStart, NULL));
  PetscCall(DMSwarmGetField(sw, "initCoordinates", NULL, NULL, (void **)&coords));
  PetscCall(DMSwarmGetField(sw, "initVelocity", NULL, NULL, (void **)&vel));
  Np /= 2 * dim;
  for (p = 0; p < Np; ++p) {
    const PetscReal x0      = coords[p * dim + 0];
    const PetscReal vy0     = vel[p * dim + 1];
    const PetscReal omega   = vy0 / x0;
    PetscScalar     vals[4] = {0., 1., -PetscSqr(omega), 0.};

    for (d = 0; d < dim; ++d) {
      const PetscInt rows[2] = {(p * 2 + 0) * dim + d + rStart, (p * 2 + 1) * dim + d + rStart};
      PetscCall(MatSetValues(J, 2, rows, 2, rows, vals, INSERT_VALUES));
    }
  }
  PetscCall(DMSwarmRestoreField(sw, "initCoordinates", NULL, NULL, (void **)&coords));
  PetscCall(DMSwarmRestoreField(sw, "initVelocity", NULL, NULL, (void **)&vel));
  PetscCall(MatAssemblyBegin(J, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(J, MAT_FINAL_ASSEMBLY));
  PetscFunctionReturn(0);
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
  PetscFunctionReturn(0);
}

static PetscErrorCode RHSFunctionV(TS ts, PetscReal t, Vec X, Vec Vres, void *ctx)
{
  DM                 sw;
  const PetscScalar *x;
  const PetscReal   *coords, *vel;
  PetscScalar       *vres;
  PetscInt           Np, p, dim, d;

  PetscFunctionBeginUser;
  PetscCall(TSGetDM(ts, &sw));
  PetscCall(DMGetDimension(sw, &dim));
  PetscCall(DMSwarmGetField(sw, "initCoordinates", NULL, NULL, (void **)&coords));
  PetscCall(DMSwarmGetField(sw, "initVelocity", NULL, NULL, (void **)&vel));
  PetscCall(VecGetLocalSize(Vres, &Np));
  PetscCall(VecGetArrayRead(X, &x));
  PetscCall(VecGetArray(Vres, &vres));
  PetscCheck(dim == 2, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Dimension must be 2");
  Np /= dim;
  for (p = 0; p < Np; ++p) {
    const PetscReal x0    = coords[p * dim + 0];
    const PetscReal vy0   = vel[p * dim + 1];
    const PetscReal omega = vy0 / x0;

    for (d = 0; d < dim; ++d) vres[p * dim + d] = -PetscSqr(omega) * x[p * dim + d];
  }
  PetscCall(VecRestoreArrayRead(X, &x));
  PetscCall(VecRestoreArray(Vres, &vres));
  PetscCall(DMSwarmRestoreField(sw, "initCoordinates", NULL, NULL, (void **)&coords));
  PetscCall(DMSwarmRestoreField(sw, "initVelocity", NULL, NULL, (void **)&vel));
  PetscFunctionReturn(0);
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
  PetscFunctionReturn(0);
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
    PetscCall(TSSetRHSFunction(ts, NULL, RHSFunction, &user));
    PetscCall(TSSetRHSJacobian(ts, J, J, RHSJacobian, &user));
    PetscCall(MatDestroy(&J));
  }
  // Define split system for X and V
  {
    Vec             u;
    IS              isx, isv, istmp;
    const PetscInt *idx;
    PetscInt        dim, Np, rstart;

    PetscCall(TSGetSolution(ts, &u));
    PetscCall(DMGetDimension(sw, &dim));
    PetscCall(DMSwarmGetLocalSize(sw, &Np));
    PetscCall(VecGetOwnershipRange(u, &rstart, NULL));
    PetscCall(ISCreateStride(PETSC_COMM_WORLD, Np, (rstart / dim) + 0, 2, &istmp));
    PetscCall(ISGetIndices(istmp, &idx));
    PetscCall(ISCreateBlock(PETSC_COMM_WORLD, dim, Np, idx, PETSC_COPY_VALUES, &isx));
    PetscCall(ISRestoreIndices(istmp, &idx));
    PetscCall(ISDestroy(&istmp));
    PetscCall(ISCreateStride(PETSC_COMM_WORLD, Np, (rstart / dim) + 1, 2, &istmp));
    PetscCall(ISGetIndices(istmp, &idx));
    PetscCall(ISCreateBlock(PETSC_COMM_WORLD, dim, Np, idx, PETSC_COPY_VALUES, &isv));
    PetscCall(ISRestoreIndices(istmp, &idx));
    PetscCall(ISDestroy(&istmp));
    PetscCall(TSRHSSplitSetIS(ts, "position", isx));
    PetscCall(TSRHSSplitSetIS(ts, "momentum", isv));
    PetscCall(ISDestroy(&isx));
    PetscCall(ISDestroy(&isv));
    PetscCall(TSRHSSplitSetRHSFunction(ts, "position", NULL, RHSFunctionX, &user));
    PetscCall(TSRHSSplitSetRHSFunction(ts, "momentum", NULL, RHSFunctionV, &user));
  }
  // Define symplectic formulation U_t = S . G, where G = grad F
  {
    //PetscCall(TSDiscGradSetFormulation(ts, RHSJacobianS, RHSObjectiveF, RHSFunctionG, &user));
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode DMSwarmTSRedistribute(TS ts)
{
  DM        sw;
  Vec       u;
  PetscReal t, maxt, dt;
  PetscInt  n, maxn;

  PetscFunctionBegin;
  PetscCall(TSGetDM(ts, &sw));
  PetscCall(TSGetTime(ts, &t));
  PetscCall(TSGetMaxTime(ts, &maxt));
  PetscCall(TSGetTimeStep(ts, &dt));
  PetscCall(TSGetStepNumber(ts, &n));
  PetscCall(TSGetMaxSteps(ts, &maxn));

  PetscCall(TSReset(ts));
  PetscCall(TSSetDM(ts, sw));
  /* TODO Check whether TS was set from options */
  PetscCall(TSSetFromOptions(ts));
  PetscCall(TSSetTime(ts, t));
  PetscCall(TSSetMaxTime(ts, maxt));
  PetscCall(TSSetTimeStep(ts, dt));
  PetscCall(TSSetStepNumber(ts, n));
  PetscCall(TSSetMaxSteps(ts, maxn));

  PetscCall(CreateSolution(ts));
  PetscCall(SetProblem(ts));
  PetscCall(TSGetSolution(ts, &u));
  PetscFunctionReturn(0);
}

PetscErrorCode circleSingleX(PetscInt dim, PetscReal time, const PetscReal dummy[], PetscInt p, PetscScalar x[], void *ctx)
{
  x[0] = p + 1.;
  x[1] = 0.;
  return 0;
}

PetscErrorCode circleSingleV(PetscInt dim, PetscReal time, const PetscReal dummy[], PetscInt p, PetscScalar v[], void *ctx)
{
  v[0] = 0.;
  v[1] = PetscSqrtReal(1000. / (p + 1.));
  return 0;
}

/* Put 5 particles into each circle */
PetscErrorCode circleMultipleX(PetscInt dim, PetscReal time, const PetscReal dummy[], PetscInt p, PetscScalar x[], void *ctx)
{
  const PetscInt  n   = 5;
  const PetscReal r0  = (p / n) + 1.;
  const PetscReal th0 = (2. * PETSC_PI * (p % n)) / n;

  x[0] = r0 * PetscCosReal(th0);
  x[1] = r0 * PetscSinReal(th0);
  return 0;
}

/* Put 5 particles into each circle */
PetscErrorCode circleMultipleV(PetscInt dim, PetscReal time, const PetscReal dummy[], PetscInt p, PetscScalar v[], void *ctx)
{
  const PetscInt  n     = 5;
  const PetscReal r0    = (p / n) + 1.;
  const PetscReal th0   = (2. * PETSC_PI * (p % n)) / n;
  const PetscReal omega = PetscSqrtReal(1000. / r0) / r0;

  v[0] = -r0 * omega * PetscSinReal(th0);
  v[1] = r0 * omega * PetscCosReal(th0);
  return 0;
}

/*
  InitializeSolveAndSwarm - Set the solution values to the swarm coordinates and velocities, and also possibly set the initial values.

  Input Parameters:
+ ts         - The TS
- useInitial - Flag to also set the initial conditions to the current coordinates and velocities and setup the problem

  Output Parameters:
. u - The initialized solution vector

  Level: advanced

.seealso: InitializeSolve()
*/
static PetscErrorCode InitializeSolveAndSwarm(TS ts, PetscBool useInitial)
{
  DM      sw;
  Vec     u, gc, gv, gc0, gv0;
  IS      isx, isv;
  AppCtx *user;

  PetscFunctionBeginUser;
  PetscCall(TSGetDM(ts, &sw));
  PetscCall(DMGetApplicationContext(sw, &user));
  if (useInitial) {
    PetscReal v0[1] = {1.};

    PetscCall(DMSwarmInitializeCoordinates(sw));
    PetscCall(DMSwarmInitializeVelocitiesFromOptions(sw, v0));
    PetscCall(DMSwarmMigrate(sw, PETSC_TRUE));
    PetscCall(DMSwarmTSRedistribute(ts));
  }
  PetscCall(TSGetSolution(ts, &u));
  PetscCall(TSRHSSplitGetIS(ts, "position", &isx));
  PetscCall(TSRHSSplitGetIS(ts, "momentum", &isv));
  PetscCall(DMSwarmCreateGlobalVectorFromField(sw, DMSwarmPICField_coor, &gc));
  PetscCall(DMSwarmCreateGlobalVectorFromField(sw, "initCoordinates", &gc0));
  if (useInitial) PetscCall(VecCopy(gc, gc0));
  PetscCall(VecISCopy(u, isx, SCATTER_FORWARD, gc));
  PetscCall(DMSwarmDestroyGlobalVectorFromField(sw, DMSwarmPICField_coor, &gc));
  PetscCall(DMSwarmDestroyGlobalVectorFromField(sw, "initCoordinates", &gc0));
  PetscCall(DMSwarmCreateGlobalVectorFromField(sw, "velocity", &gv));
  PetscCall(DMSwarmCreateGlobalVectorFromField(sw, "initVelocity", &gv0));
  if (useInitial) PetscCall(VecCopy(gv, gv0));
  PetscCall(VecISCopy(u, isv, SCATTER_FORWARD, gv));
  PetscCall(DMSwarmDestroyGlobalVectorFromField(sw, "velocity", &gv));
  PetscCall(DMSwarmDestroyGlobalVectorFromField(sw, "initVelocity", &gv0));
  PetscFunctionReturn(0);
}

static PetscErrorCode InitializeSolve(TS ts, Vec u)
{
  PetscFunctionBegin;
  PetscCall(TSSetSolution(ts, u));
  PetscCall(InitializeSolveAndSwarm(ts, PETSC_TRUE));
  PetscFunctionReturn(0);
}

static PetscErrorCode ComputeError(TS ts, Vec U, Vec E)
{
  MPI_Comm           comm;
  DM                 sw;
  AppCtx            *user;
  const PetscScalar *u;
  const PetscReal   *coords, *vel;
  PetscScalar       *e;
  PetscReal          t;
  PetscInt           dim, Np, p;

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
  PetscCall(DMSwarmGetField(sw, "initVelocity", NULL, NULL, (void **)&vel));
  Np /= 2 * dim;
  for (p = 0; p < Np; ++p) {
    /* TODO generalize initial conditions and project into plane instead of assuming x-y */
    const PetscReal    r0    = DMPlex_NormD_Internal(dim, &coords[p * dim]);
    const PetscReal    th0   = PetscAtan2Real(coords[p * dim + 1], coords[p * dim + 0]);
    const PetscReal    v0    = DMPlex_NormD_Internal(dim, &vel[p * dim]);
    const PetscReal    omega = v0 / r0;
    const PetscReal    ct    = PetscCosReal(omega * t + th0);
    const PetscReal    st    = PetscSinReal(omega * t + th0);
    const PetscScalar *x     = &u[(p * 2 + 0) * dim];
    const PetscScalar *v     = &u[(p * 2 + 1) * dim];
    const PetscReal    xe[3] = {r0 * ct, r0 * st, 0.0};
    const PetscReal    ve[3] = {-v0 * st, v0 * ct, 0.0};
    PetscInt           d;

    for (d = 0; d < dim; ++d) {
      e[(p * 2 + 0) * dim + d] = x[d] - xe[d];
      e[(p * 2 + 1) * dim + d] = v[d] - ve[d];
    }
    if (user->error) {
      const PetscReal en   = 0.5 * DMPlex_DotRealD_Internal(dim, v, v);
      const PetscReal exen = 0.5 * PetscSqr(v0);
      PetscCall(PetscPrintf(comm, "t %.4g: p%" PetscInt_FMT " error [%.2g %.2g] sol [(%.6lf %.6lf) (%.6lf %.6lf)] exact [(%.6lf %.6lf) (%.6lf %.6lf)] energy/exact energy %g / %g (%.10lf%%)\n", (double)t, p, (double)DMPlex_NormD_Internal(dim, &e[(p * 2 + 0) * dim]), (double)DMPlex_NormD_Internal(dim, &e[(p * 2 + 1) * dim]), (double)x[0], (double)x[1], (double)v[0], (double)v[1], (double)xe[0], (double)xe[1], (double)ve[0], (double)ve[1], (double)en, (double)exen, (double)(PetscAbsReal(exen - en) * 100. / exen)));
    }
  }
  PetscCall(DMSwarmRestoreField(sw, "initCoordinates", NULL, NULL, (void **)&coords));
  PetscCall(DMSwarmRestoreField(sw, "initVelocity", NULL, NULL, (void **)&vel));
  PetscCall(VecRestoreArrayRead(U, &u));
  PetscCall(VecRestoreArray(E, &e));
  PetscFunctionReturn(0);
}

static PetscErrorCode EnergyMonitor(TS ts, PetscInt step, PetscReal t, Vec U, void *ctx)
{
  const PetscInt     ostep = ((AppCtx *)ctx)->ostep;
  DM                 sw;
  const PetscScalar *u;
  PetscInt           dim, Np, p;

  PetscFunctionBeginUser;
  if (step % ostep == 0) {
    PetscCall(TSGetDM(ts, &sw));
    PetscCall(DMGetDimension(sw, &dim));
    PetscCall(VecGetArrayRead(U, &u));
    PetscCall(VecGetLocalSize(U, &Np));
    Np /= 2 * dim;
    if (!step) PetscCall(PetscPrintf(PetscObjectComm((PetscObject)ts), "Time     Step Part     Energy\n"));
    for (p = 0; p < Np; ++p) {
      const PetscReal v2 = DMPlex_DotRealD_Internal(dim, &u[(p * 2 + 1) * dim], &u[(p * 2 + 1) * dim]);

      PetscCall(PetscSynchronizedPrintf(PetscObjectComm((PetscObject)ts), "%.6lf %4" PetscInt_FMT " %4" PetscInt_FMT " %10.4lf\n", (double)t, step, p, (double)(0.5 * v2)));
    }
    PetscCall(PetscSynchronizedFlush(PetscObjectComm((PetscObject)ts), NULL));
    PetscCall(VecRestoreArrayRead(U, &u));
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode MigrateParticles(TS ts)
{
  DM sw;

  PetscFunctionBeginUser;
  PetscCall(TSGetDM(ts, &sw));
  PetscCall(DMViewFromOptions(sw, NULL, "-migrate_view_pre"));
  {
    Vec u, gc, gv;
    IS  isx, isv;

    PetscCall(TSGetSolution(ts, &u));
    PetscCall(TSRHSSplitGetIS(ts, "position", &isx));
    PetscCall(TSRHSSplitGetIS(ts, "momentum", &isv));
    PetscCall(DMSwarmCreateGlobalVectorFromField(sw, DMSwarmPICField_coor, &gc));
    PetscCall(VecISCopy(u, isx, SCATTER_REVERSE, gc));
    PetscCall(DMSwarmDestroyGlobalVectorFromField(sw, DMSwarmPICField_coor, &gc));
    PetscCall(DMSwarmCreateGlobalVectorFromField(sw, "velocity", &gv));
    PetscCall(VecISCopy(u, isv, SCATTER_REVERSE, gv));
    PetscCall(DMSwarmDestroyGlobalVectorFromField(sw, "velocity", &gv));
  }
  PetscCall(DMSwarmMigrate(sw, PETSC_TRUE));
  PetscCall(DMSwarmTSRedistribute(ts));
  PetscCall(InitializeSolveAndSwarm(ts, PETSC_FALSE));
  //PetscCall(VecViewFromOptions(u, NULL, "-sol_migrate_view"));
  PetscFunctionReturn(0);
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
  PetscCall(TSSetPostStep(ts, MigrateParticles));

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
     requires: double !complex

   testset:
     requires: defined(PETSC_HAVE_EXECUTABLE_EXPORT)
     args: -dm_plex_dim 2 -dm_plex_simplex 0 -dm_plex_box_faces 1,1 -dm_plex_box_lower -5,-5 -dm_plex_box_upper 5,5 \
           -dm_swarm_num_particles 2 -dm_swarm_coordinate_function circleSingleX -dm_swarm_velocity_function circleSingleV \
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
     requires: defined(PETSC_HAVE_EXECUTABLE_EXPORT)
     args: -dm_swarm_num_particles 2 -dm_swarm_coordinate_function circleSingleX -dm_swarm_velocity_function circleSingleV \
           -ts_type theta -ts_theta_theta 0.5 -ts_convergence_estimate -convest_num_refine 2 \
             -mat_type baij -ksp_error_if_not_converged -pc_type lu \
           -dm_view -output_step 50 -error
     test:
       suffix: im_2d_0
       args: -dm_plex_dim 2 -dm_plex_simplex 0 -dm_plex_box_faces 1,1 -dm_plex_box_lower -5,-5 -dm_plex_box_upper 5,5

   testset:
     requires: defined(PETSC_HAVE_EXECUTABLE_EXPORT)
     args: -dm_plex_dim 2 -dm_plex_simplex 0 -dm_plex_box_faces 10,10 -dm_plex_box_lower -5,-5 -dm_plex_box_upper 5,5 -petscpartitioner_type simple \
           -dm_swarm_num_particles 2 -dm_swarm_coordinate_function circleSingleX -dm_swarm_velocity_function circleSingleV \
           -ts_type basicsymplectic -ts_convergence_estimate -convest_num_refine 2 \
           -dm_view -output_step 50
     test:
       suffix: bsi_2d_mesh_1
       args: -ts_basicsymplectic_type 1 -error
     test:
       suffix: bsi_2d_mesh_1_par_2
       nsize: 2
       args: -ts_basicsymplectic_type 1
     test:
       suffix: bsi_2d_mesh_1_par_3
       nsize: 3
       args: -ts_basicsymplectic_type 1
     test:
       suffix: bsi_2d_mesh_1_par_4
       nsize: 4
       args: -ts_basicsymplectic_type 1 -dm_swarm_num_particles 0,0,2,0

   testset:
     requires: defined(PETSC_HAVE_EXECUTABLE_EXPORT)
     args: -dm_plex_dim 2 -dm_plex_simplex 0 -dm_plex_box_faces 10,10 -dm_plex_box_lower -5,-5 -dm_plex_box_upper 5,5 \
           -dm_swarm_num_particles 10 -dm_swarm_coordinate_function circleMultipleX -dm_swarm_velocity_function circleMultipleV \
           -ts_convergence_estimate -convest_num_refine 2 \
           -dm_view -output_step 50 -error
     test:
       suffix: bsi_2d_multiple_1
       args: -ts_type basicsymplectic -ts_basicsymplectic_type 1
     test:
       suffix: bsi_2d_multiple_2
       args: -ts_type basicsymplectic -ts_basicsymplectic_type 2
     test:
       suffix: bsi_2d_multiple_3
       args: -ts_type basicsymplectic -ts_basicsymplectic_type 3 -ts_dt 0.001
     test:
       suffix: im_2d_multiple_0
       args: -ts_type theta -ts_theta_theta 0.5 \
               -mat_type baij -ksp_error_if_not_converged -pc_type lu

TEST*/
