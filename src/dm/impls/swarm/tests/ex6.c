static char help[] = "Vlasov example of many particles orbiting around a several massive points.\n";

#include <petscdmplex.h>
#include <petsc/private/dmpleximpl.h>  /* For norm */
#include <petsc/private/petscfeimpl.h> /* For CoordinatesRefToReal() */
#include <petscdmswarm.h>
#include <petscts.h>

typedef struct {
  PetscInt  dim;                          /* The topological mesh dimension */
  PetscInt  particlesPerCircle;           /* The number of partices per circle */
  PetscReal momentTol;                    /* Tolerance for checking moment conservation */
  PetscBool monitor;                      /* Flag for using the TS monitor */
  PetscBool error;                        /* Flag for printing the error */
  PetscInt  ostep;                        /* print the energy at each ostep time steps */
  PetscReal center[6];                    /* Centers of the two orbits */
  PetscReal radius[2];                    /* Radii of the two orbits */
} AppCtx;

static PetscErrorCode ProcessOptions(MPI_Comm comm, AppCtx *options)
{
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  options->monitor            = PETSC_FALSE;
  options->error              = PETSC_FALSE;
  options->particlesPerCircle = 1;
  options->momentTol          = 100.0*PETSC_MACHINE_EPSILON;
  options->ostep              = 100;

  ierr = PetscOptionsBegin(comm, "", "Vlasov Options", "DMPLEX");PetscCall(ierr);
  PetscCall(PetscOptionsInt("-output_step", "Number of time steps between output", "ex4.c", options->ostep, &options->ostep, PETSC_NULL));
  PetscCall(PetscOptionsBool("-monitor", "Flag to use the TS monitor", "ex4.c", options->monitor, &options->monitor, NULL));
  PetscCall(PetscOptionsBool("-error", "Flag to print the error", "ex4.c", options->error, &options->error, NULL));
  PetscCall(PetscOptionsInt("-particles_per_circle", "Number of particles per circle", "ex4.c", options->particlesPerCircle, &options->particlesPerCircle, NULL));
  ierr = PetscOptionsEnd();PetscCall(ierr);

  PetscFunctionReturn(0);
}

static PetscErrorCode CreateMesh(MPI_Comm comm, DM *dm, AppCtx *user)
{
  PetscFunctionBeginUser;
  PetscCall(DMCreate(comm, dm));
  PetscCall(DMSetType(*dm, DMPLEX));
  PetscCall(DMSetFromOptions(*dm));
  PetscCall(DMViewFromOptions(*dm, NULL, "-dm_view"));
  PetscCall(DMGetDimension(*dm, &user->dim));
  PetscFunctionReturn(0);
}

static PetscErrorCode orbit(AppCtx *ctx, PetscInt c, PetscInt p, PetscReal t, PetscReal x[], PetscReal v[])
{
  const PetscInt  Np    = ctx->particlesPerCircle;
  const PetscReal r     = ctx->radius[c];
  const PetscReal omega = PetscSqrtReal(1000./r)/r;
  const PetscReal t0    = (2.*PETSC_PI*p)/(Np*omega);
  const PetscInt  dim   = 2;

  PetscFunctionBeginUser;
  if (x) {
    x[0] = r*PetscCosReal(omega*(t + t0)) + ctx->center[c*dim + 0];
    x[1] = r*PetscSinReal(omega*(t + t0)) + ctx->center[c*dim + 1];
  }
  if (v) {
    v[0] = -r*omega*PetscSinReal(omega*(t + t0));
    v[1] =  r*omega*PetscCosReal(omega*(t + t0));
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode force(AppCtx *ctx, PetscInt c, const PetscReal x[], PetscReal force[])
{
  const PetscReal r     = ctx->radius[c];
  const PetscReal omega = PetscSqrtReal(1000./r)/r;
  const PetscInt  dim   = 2;
  PetscInt        d;

  PetscFunctionBeginUser;
  for (d = 0; d < dim; ++d) force[d] = -PetscSqr(omega)*(x[d] - ctx->center[c*dim + d]);
  PetscFunctionReturn(0);
}

static PetscReal energy(AppCtx *ctx, PetscInt c)
{
  const PetscReal r     = ctx->radius[c];
  const PetscReal omega = PetscSqrtReal(1000./r)/r;

  return 0.5 * omega * r;
}

static PetscErrorCode SetInitialCoordinates(DM dmSw)
{
  DM                dm;
  AppCtx            *ctx;
  Vec               coordinates;
  PetscSF           cellSF = NULL;
  PetscReal         *coords;
  PetscInt          *cellid;
  const PetscInt    *found;
  const PetscSFNode *cells;
  PetscInt          dim, d, c, Np, p;
  PetscMPIInt       rank;

  PetscFunctionBeginUser;
  PetscCall(DMGetApplicationContext(dmSw, &ctx));
  Np   = ctx->particlesPerCircle;
  PetscCall(DMSwarmGetCellDM(dmSw, &dm));
  PetscCall(DMGetDimension(dm, &dim));
  PetscCall(DMSwarmGetField(dmSw, DMSwarmPICField_coor, NULL, NULL, (void **) &coords));
  for (c = 0; c < 2; ++c) {
    for (d = 0; d < dim; ++d) ctx->center[c*dim+d] = (!c && !d) ? 3.0 : 0.0;
    ctx->radius[c] = 3.*c+1.;
    for (p = 0; p < Np; ++p) {
      const PetscInt n = c*Np + p;

      PetscCall(orbit(ctx, c, p, 0.0, &coords[n*dim], NULL));
    }
  }
  PetscCall(DMSwarmRestoreField(dmSw, DMSwarmPICField_coor, NULL, NULL, (void **) &coords));

  PetscCall(DMSwarmCreateGlobalVectorFromField(dmSw, DMSwarmPICField_coor, &coordinates));
  PetscCall(DMLocatePoints(dm, coordinates, DM_POINTLOCATION_NONE, &cellSF));
  PetscCall(DMSwarmDestroyGlobalVectorFromField(dmSw, DMSwarmPICField_coor, &coordinates));

  PetscCallMPI(MPI_Comm_rank(PetscObjectComm((PetscObject) dmSw), &rank));
  PetscCall(DMSwarmGetField(dmSw, DMSwarmPICField_cellid, NULL, NULL, (void **) &cellid));
  PetscCall(PetscSFGetGraph(cellSF, NULL, &Np, &found, &cells));
  for (p = 0; p < Np; ++p) {
    const PetscInt part = found ? found[p] : p;

    PetscCheck(cells[p].rank == rank,PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Point %D not found in the mesh", part);
    cellid[part] = cells[p].index;
  }
  PetscCall(DMSwarmRestoreField(dmSw, DMSwarmPICField_cellid, NULL, NULL, (void **) &cellid));
  PetscCall(PetscSFDestroy(&cellSF));
  PetscFunctionReturn(0);
}

static PetscErrorCode SetInitialConditions(DM dmSw, Vec u)
{
  DM             dm;
  AppCtx        *ctx;
  PetscScalar   *initialConditions;
  PetscInt       dim, cStart, cEnd, c, Np, p;

  PetscFunctionBeginUser;
  PetscCall(DMGetApplicationContext(dmSw, &ctx));
  Np   = ctx->particlesPerCircle;
  PetscCall(DMSwarmGetCellDM(dmSw, &dm));
  PetscCall(DMGetDimension(dm, &dim));
  PetscCall(DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd));
  PetscCall(VecGetArray(u, &initialConditions));
  for (c = 0; c < 2; ++c) {
    for (p = 0; p < Np; ++p) {
      const PetscInt n = c*Np + p;

      PetscCall(orbit(ctx, c, p, 0.0, &initialConditions[(n*2 + 0)*dim], &initialConditions[(n*2 + 1)*dim]));
    }
  }
  PetscCall(VecRestoreArray(u, &initialConditions));
  PetscFunctionReturn(0);
}

static PetscErrorCode CreateParticles(DM dm, DM *sw, AppCtx *user)
{
  PetscInt       dim, Np = user->particlesPerCircle;

  PetscFunctionBeginUser;
  PetscCall(DMGetDimension(dm, &dim));
  PetscCall(DMCreate(PetscObjectComm((PetscObject) dm), sw));
  PetscCall(DMSetType(*sw, DMSWARM));
  PetscCall(DMSetDimension(*sw, dim));

  PetscCall(DMSwarmSetType(*sw, DMSWARM_PIC));
  PetscCall(DMSwarmSetCellDM(*sw, dm));
  PetscCall(DMSwarmRegisterPetscDatatypeField(*sw, "kinematics", 2*dim, PETSC_REAL));
  PetscCall(DMSwarmFinalizeFieldRegister(*sw));
  PetscCall(DMSwarmSetLocalSizes(*sw, 2 * Np, 0));
  PetscCall(DMSetFromOptions(*sw));
  PetscCall(PetscObjectSetName((PetscObject) *sw, "Particles"));
  PetscCall(DMViewFromOptions(*sw, NULL, "-sw_view"));
  PetscFunctionReturn(0);
}

/* Create particle RHS Functions for gravity with G = 1 for simplification */
static PetscErrorCode RHSFunction1(TS ts, PetscReal t, Vec V, Vec Xres, void *ctx)
{
  const PetscScalar *v;
  PetscScalar       *xres;
  PetscInt          Np, p, dim, d;

  PetscFunctionBeginUser;
  /* The DM is not currently pushed down to the splits */
  dim  = ((AppCtx *) ctx)->dim;
  PetscCall(VecGetLocalSize(Xres, &Np));
  Np  /= dim;
  PetscCall(VecGetArray(Xres, &xres));
  PetscCall(VecGetArrayRead(V, &v));
  for (p = 0; p < Np; ++p) {
     for (d = 0; d < dim; ++d) {
       xres[p*dim+d] = v[p*dim+d];
     }
  }
  PetscCall(VecRestoreArrayRead(V,& v));
  PetscCall(VecRestoreArray(Xres, &xres));
  PetscFunctionReturn(0);
}

static PetscErrorCode RHSFunction2(TS ts, PetscReal t, Vec X, Vec Vres, void *user)
{
  AppCtx           *ctx = (AppCtx *) user;
  const PetscScalar *x;
  PetscScalar       *vres;
  PetscInt          Np, p, dim;

  PetscFunctionBeginUser;
  /* The DM is not currently pushed down to the splits */
  dim  = ctx->dim;
  PetscCall(VecGetLocalSize(Vres, &Np));
  Np  /= dim;
  PetscCall(VecGetArray(Vres, &vres));
  PetscCall(VecGetArrayRead(X, &x));
  for (p = 0; p < Np; ++p) {
    const PetscInt c = p / ctx->particlesPerCircle;

    PetscCall(force(ctx, c, &x[p*dim], &vres[p*dim]));
  }
  PetscCall(VecRestoreArray(Vres, &vres));
  PetscCall(VecRestoreArrayRead(X, &x));
  PetscFunctionReturn(0);
}

static PetscErrorCode RHSFunctionParticles(TS ts, PetscReal t , Vec U, Vec R, void *user)
{
  AppCtx           *ctx = (AppCtx *) user;
  DM                dm;
  const PetscScalar *u;
  PetscScalar       *r;
  PetscInt          Np, p, dim, d;

  PetscFunctionBeginUser;
  PetscCall(TSGetDM(ts, &dm));
  PetscCall(DMGetDimension(dm, &dim));
  PetscCall(VecGetLocalSize(U, &Np));
  Np  /= 2*dim;
  PetscCall(VecGetArrayRead(U, &u));
  PetscCall(VecGetArray(R, &r));
  for (p = 0; p < Np; ++p) {
    const PetscInt c = p / ctx->particlesPerCircle;

    for (d = 0; d < dim; ++d) r[(p*2 + 0)*dim + d] = u[(p*2 + 1)*dim + d];
    PetscCall(force(ctx, c, &u[(p*2 + 0)*dim], &r[(p*2 + 1)*dim]));
  }
  PetscCall(VecRestoreArrayRead(U, &u));
  PetscCall(VecRestoreArray(R, &r));
  PetscFunctionReturn(0);
}

static PetscErrorCode InitializeSolve(TS ts, Vec u)
{
  DM             dm;
  AppCtx        *user;

  PetscFunctionBeginUser;
  PetscCall(TSGetDM(ts, &dm));
  PetscCall(DMGetApplicationContext(dm, &user));
  PetscCall(SetInitialCoordinates(dm));
  PetscCall(SetInitialConditions(dm, u));
  PetscFunctionReturn(0);
}

static PetscErrorCode ComputeError(TS ts, Vec U, Vec E)
{
  MPI_Comm           comm;
  DM                 sdm;
  AppCtx            *ctx;
  const PetscScalar *u, *coords;
  PetscScalar       *e;
  PetscReal          t;
  PetscInt           dim, Np, p, c;

  PetscFunctionBeginUser;
  PetscCall(PetscObjectGetComm((PetscObject) ts, &comm));
  PetscCall(TSGetDM(ts, &sdm));
  PetscCall(DMGetApplicationContext(sdm, &ctx));
  PetscCall(DMGetDimension(sdm, &dim));
  PetscCall(TSGetSolveTime(ts, &t));
  PetscCall(VecGetArray(E, &e));
  PetscCall(VecGetArrayRead(U, &u));
  PetscCall(VecGetLocalSize(U, &Np));
  Np  /= 2*dim*2;
  PetscCall(DMSwarmGetField(sdm, DMSwarmPICField_coor, NULL, NULL, (void **) &coords));
  for (c = 0; c < 2; ++c) {
    for (p = 0; p < Np; ++p) {
      const PetscInt     n = c*Np + p;
      const PetscScalar *x = &u[(n*2+0)*dim];
      const PetscScalar *v = &u[(n*2+1)*dim];
      PetscReal          xe[3], ve[3];
      PetscInt           d;

      PetscCall(orbit(ctx, c, p, t, xe, ve));
      for (d = 0; d < dim; ++d) {
        e[(p*2+0)*dim+d] = x[d] - xe[d];
        e[(p*2+1)*dim+d] = v[d] - ve[d];
      }
      if (ctx->error) PetscCall(PetscPrintf(comm, "p%D error [%.2g %.2g] sol [(%.6lf %.6lf) (%.6lf %.6lf)] exact [(%.6lf %.6lf) (%.6lf %.6lf)] energy/exact energy %g / %g\n", p, (double) DMPlex_NormD_Internal(dim, &e[(p*2+0)*dim]), (double) DMPlex_NormD_Internal(dim, &e[(p*2+1)*dim]), (double) x[0], (double) x[1], (double) v[0], (double) v[1], (double) xe[0], (double) xe[1], (double) ve[0], (double) ve[1], 0.5*DMPlex_NormD_Internal(dim, v), (double) energy(ctx, c)));
    }
  }
  PetscCall(DMSwarmRestoreField(sdm, DMSwarmPICField_coor, NULL, NULL, (void **) &coords));
  PetscCall(VecRestoreArrayRead(U, &u));
  PetscCall(VecRestoreArray(E, &e));
  PetscFunctionReturn(0);
}

int main(int argc, char **argv)
{
  TS                 ts;
  TSConvergedReason  reason;
  DM                 dm, sw;
  Vec                u;
  IS                 is1, is2;
  PetscInt          *idx1, *idx2;
  MPI_Comm           comm;
  AppCtx             user;
  const PetscScalar *endVals;
  PetscReal          ftime   = .1;
  PetscInt           locSize, dim, d, Np, p, c, steps;

  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  comm = PETSC_COMM_WORLD;
  PetscCall(ProcessOptions(comm, &user));

  PetscCall(CreateMesh(comm, &dm, &user));
  PetscCall(CreateParticles(dm, &sw, &user));
  PetscCall(DMSetApplicationContext(sw, &user));

  PetscCall(TSCreate(comm, &ts));
  PetscCall(TSSetType(ts, TSBASICSYMPLECTIC));
  PetscCall(TSSetDM(ts, sw));
  PetscCall(TSSetMaxTime(ts, ftime));
  PetscCall(TSSetTimeStep(ts, 0.0001));
  PetscCall(TSSetMaxSteps(ts, 10));
  PetscCall(TSSetExactFinalTime(ts, TS_EXACTFINALTIME_MATCHSTEP));
  PetscCall(TSSetTime(ts, 0.0));
  PetscCall(TSSetRHSFunction(ts, NULL, RHSFunctionParticles, &user));

  PetscCall(DMSwarmCreateGlobalVectorFromField(sw, "kinematics", &u));
  PetscCall(DMGetDimension(sw, &dim));
  PetscCall(VecGetLocalSize(u, &locSize));
  Np   = locSize/(2*dim);
  PetscCall(PetscMalloc1(locSize/2, &idx1));
  PetscCall(PetscMalloc1(locSize/2, &idx2));
  for (p = 0; p < Np; ++p) {
    for (d = 0; d < dim; ++d) {
      idx1[p*dim+d] = (p*2+0)*dim + d;
      idx2[p*dim+d] = (p*2+1)*dim + d;
    }
  }
  PetscCall(ISCreateGeneral(comm, locSize/2, idx1, PETSC_OWN_POINTER, &is1));
  PetscCall(ISCreateGeneral(comm, locSize/2, idx2, PETSC_OWN_POINTER, &is2));
  PetscCall(TSRHSSplitSetIS(ts, "position", is1));
  PetscCall(TSRHSSplitSetIS(ts, "momentum", is2));
  PetscCall(ISDestroy(&is1));
  PetscCall(ISDestroy(&is2));
  PetscCall(TSRHSSplitSetRHSFunction(ts,"position",NULL,RHSFunction1,&user));
  PetscCall(TSRHSSplitSetRHSFunction(ts,"momentum",NULL,RHSFunction2,&user));

  PetscCall(TSSetFromOptions(ts));
  PetscCall(TSSetComputeInitialCondition(ts, InitializeSolve));
  PetscCall(TSSetComputeExactError(ts, ComputeError));
  PetscCall(TSComputeInitialCondition(ts, u));
  PetscCall(VecViewFromOptions(u, NULL, "-init_view"));
  PetscCall(TSSolve(ts, u));
  PetscCall(TSGetSolveTime(ts, &ftime));
  PetscCall(TSGetConvergedReason(ts, &reason));
  PetscCall(TSGetStepNumber(ts, &steps));
  PetscCall(PetscPrintf(comm,"%s at time %g after %D steps\n", TSConvergedReasons[reason], (double) ftime, steps));

  PetscCall(VecGetArrayRead(u, &endVals));
  for (c = 0; c < 2; ++c) {
    for (p = 0; p < Np/2; ++p) {
      const PetscInt  n    = c*(Np/2) + p;
      const PetscReal norm = DMPlex_NormD_Internal(dim, &endVals[(n*2 + 1)*dim]);
      PetscCall(PetscPrintf(comm, "Particle %D initial Energy: %g  Final Energy: %g\n", p, (double) (0.5*(1000./(3*c+1.))), (double) 0.5*PetscSqr(norm)));
    }
  }
  PetscCall(VecRestoreArrayRead(u, &endVals));
  PetscCall(DMSwarmDestroyGlobalVectorFromField(sw, "kinematics", &u));
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
     suffix: bsi1
     args: -dm_plex_box_faces 1,1 -dm_view -sw_view -particles_per_circle 5 -ts_basicsymplectic_type 1 -ts_max_time 0.1 -ts_dt 0.001 -ts_monitor_sp_swarm -ts_convergence_estimate -convest_num_refine 2
   test:
     suffix: bsi2
     args: -dm_plex_box_faces 1,1 -dm_view -sw_view -particles_per_circle 5 -ts_basicsymplectic_type 2 -ts_max_time 0.1 -ts_dt 0.001 -ts_monitor_sp_swarm -ts_convergence_estimate -convest_num_refine 2
   test:
     suffix: euler
     args: -dm_plex_box_faces 1,1 -dm_view -sw_view -particles_per_circle 5 -ts_type euler -ts_max_time 0.1 -ts_dt 0.001 -ts_monitor_sp_swarm -ts_convergence_estimate -convest_num_refine 2

TEST*/
