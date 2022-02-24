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

  ierr = PetscOptionsBegin(comm, "", "Vlasov Options", "DMPLEX");CHKERRQ(ierr);
  CHKERRQ(PetscOptionsInt("-output_step", "Number of time steps between output", "ex4.c", options->ostep, &options->ostep, PETSC_NULL));
  CHKERRQ(PetscOptionsBool("-monitor", "Flag to use the TS monitor", "ex4.c", options->monitor, &options->monitor, NULL));
  CHKERRQ(PetscOptionsBool("-error", "Flag to print the error", "ex4.c", options->error, &options->error, NULL));
  CHKERRQ(PetscOptionsInt("-particles_per_circle", "Number of particles per circle", "ex4.c", options->particlesPerCircle, &options->particlesPerCircle, NULL));
  ierr = PetscOptionsEnd();CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

static PetscErrorCode CreateMesh(MPI_Comm comm, DM *dm, AppCtx *user)
{
  PetscFunctionBeginUser;
  CHKERRQ(DMCreate(comm, dm));
  CHKERRQ(DMSetType(*dm, DMPLEX));
  CHKERRQ(DMSetFromOptions(*dm));
  CHKERRQ(DMViewFromOptions(*dm, NULL, "-dm_view"));
  CHKERRQ(DMGetDimension(*dm, &user->dim));
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
  CHKERRQ(DMGetApplicationContext(dmSw, &ctx));
  Np   = ctx->particlesPerCircle;
  CHKERRQ(DMSwarmGetCellDM(dmSw, &dm));
  CHKERRQ(DMGetDimension(dm, &dim));
  CHKERRQ(DMSwarmGetField(dmSw, DMSwarmPICField_coor, NULL, NULL, (void **) &coords));
  for (c = 0; c < 2; ++c) {
    for (d = 0; d < dim; ++d) ctx->center[c*dim+d] = (!c && !d) ? 3.0 : 0.0;
    ctx->radius[c] = 3.*c+1.;
    for (p = 0; p < Np; ++p) {
      const PetscInt n = c*Np + p;

      CHKERRQ(orbit(ctx, c, p, 0.0, &coords[n*dim], NULL));
    }
  }
  CHKERRQ(DMSwarmRestoreField(dmSw, DMSwarmPICField_coor, NULL, NULL, (void **) &coords));

  CHKERRQ(DMSwarmCreateGlobalVectorFromField(dmSw, DMSwarmPICField_coor, &coordinates));
  CHKERRQ(DMLocatePoints(dm, coordinates, DM_POINTLOCATION_NONE, &cellSF));
  CHKERRQ(DMSwarmDestroyGlobalVectorFromField(dmSw, DMSwarmPICField_coor, &coordinates));

  CHKERRMPI(MPI_Comm_rank(PetscObjectComm((PetscObject) dmSw), &rank));
  CHKERRQ(DMSwarmGetField(dmSw, DMSwarmPICField_cellid, NULL, NULL, (void **) &cellid));
  CHKERRQ(PetscSFGetGraph(cellSF, NULL, &Np, &found, &cells));
  for (p = 0; p < Np; ++p) {
    const PetscInt part = found ? found[p] : p;

    PetscCheckFalse(cells[p].rank != rank,PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Point %D not found in the mesh", part);
    cellid[part] = cells[p].index;
  }
  CHKERRQ(DMSwarmRestoreField(dmSw, DMSwarmPICField_cellid, NULL, NULL, (void **) &cellid));
  CHKERRQ(PetscSFDestroy(&cellSF));
  PetscFunctionReturn(0);
}

static PetscErrorCode SetInitialConditions(DM dmSw, Vec u)
{
  DM             dm;
  AppCtx        *ctx;
  PetscScalar   *initialConditions;
  PetscInt       dim, cStart, cEnd, c, Np, p;

  PetscFunctionBeginUser;
  CHKERRQ(DMGetApplicationContext(dmSw, &ctx));
  Np   = ctx->particlesPerCircle;
  CHKERRQ(DMSwarmGetCellDM(dmSw, &dm));
  CHKERRQ(DMGetDimension(dm, &dim));
  CHKERRQ(DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd));
  CHKERRQ(VecGetArray(u, &initialConditions));
  for (c = 0; c < 2; ++c) {
    for (p = 0; p < Np; ++p) {
      const PetscInt n = c*Np + p;

      CHKERRQ(orbit(ctx, c, p, 0.0, &initialConditions[(n*2 + 0)*dim], &initialConditions[(n*2 + 1)*dim]));
    }
  }
  CHKERRQ(VecRestoreArray(u, &initialConditions));
  PetscFunctionReturn(0);
}

static PetscErrorCode CreateParticles(DM dm, DM *sw, AppCtx *user)
{
  PetscInt       dim, Np = user->particlesPerCircle;

  PetscFunctionBeginUser;
  CHKERRQ(DMGetDimension(dm, &dim));
  CHKERRQ(DMCreate(PetscObjectComm((PetscObject) dm), sw));
  CHKERRQ(DMSetType(*sw, DMSWARM));
  CHKERRQ(DMSetDimension(*sw, dim));

  CHKERRQ(DMSwarmSetType(*sw, DMSWARM_PIC));
  CHKERRQ(DMSwarmSetCellDM(*sw, dm));
  CHKERRQ(DMSwarmRegisterPetscDatatypeField(*sw, "kinematics", 2*dim, PETSC_REAL));
  CHKERRQ(DMSwarmFinalizeFieldRegister(*sw));
  CHKERRQ(DMSwarmSetLocalSizes(*sw, 2 * Np, 0));
  CHKERRQ(DMSetFromOptions(*sw));
  CHKERRQ(PetscObjectSetName((PetscObject) *sw, "Particles"));
  CHKERRQ(DMViewFromOptions(*sw, NULL, "-sw_view"));
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
  CHKERRQ(VecGetLocalSize(Xres, &Np));
  Np  /= dim;
  CHKERRQ(VecGetArray(Xres, &xres));
  CHKERRQ(VecGetArrayRead(V, &v));
  for (p = 0; p < Np; ++p) {
     for (d = 0; d < dim; ++d) {
       xres[p*dim+d] = v[p*dim+d];
     }
  }
  CHKERRQ(VecRestoreArrayRead(V,& v));
  CHKERRQ(VecRestoreArray(Xres, &xres));
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
  CHKERRQ(VecGetLocalSize(Vres, &Np));
  Np  /= dim;
  CHKERRQ(VecGetArray(Vres, &vres));
  CHKERRQ(VecGetArrayRead(X, &x));
  for (p = 0; p < Np; ++p) {
    const PetscInt c = p / ctx->particlesPerCircle;

    CHKERRQ(force(ctx, c, &x[p*dim], &vres[p*dim]));
  }
  CHKERRQ(VecRestoreArray(Vres, &vres));
  CHKERRQ(VecRestoreArrayRead(X, &x));
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
  CHKERRQ(TSGetDM(ts, &dm));
  CHKERRQ(DMGetDimension(dm, &dim));
  CHKERRQ(VecGetLocalSize(U, &Np));
  Np  /= 2*dim;
  CHKERRQ(VecGetArrayRead(U, &u));
  CHKERRQ(VecGetArray(R, &r));
  for (p = 0; p < Np; ++p) {
    const PetscInt c = p / ctx->particlesPerCircle;

    for (d = 0; d < dim; ++d) r[(p*2 + 0)*dim + d] = u[(p*2 + 1)*dim + d];
    CHKERRQ(force(ctx, c, &u[(p*2 + 0)*dim], &r[(p*2 + 1)*dim]));
  }
  CHKERRQ(VecRestoreArrayRead(U, &u));
  CHKERRQ(VecRestoreArray(R, &r));
  PetscFunctionReturn(0);
}

static PetscErrorCode InitializeSolve(TS ts, Vec u)
{
  DM             dm;
  AppCtx        *user;

  PetscFunctionBeginUser;
  CHKERRQ(TSGetDM(ts, &dm));
  CHKERRQ(DMGetApplicationContext(dm, &user));
  CHKERRQ(SetInitialCoordinates(dm));
  CHKERRQ(SetInitialConditions(dm, u));
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
  CHKERRQ(PetscObjectGetComm((PetscObject) ts, &comm));
  CHKERRQ(TSGetDM(ts, &sdm));
  CHKERRQ(DMGetApplicationContext(sdm, &ctx));
  CHKERRQ(DMGetDimension(sdm, &dim));
  CHKERRQ(TSGetSolveTime(ts, &t));
  CHKERRQ(VecGetArray(E, &e));
  CHKERRQ(VecGetArrayRead(U, &u));
  CHKERRQ(VecGetLocalSize(U, &Np));
  Np  /= 2*dim*2;
  CHKERRQ(DMSwarmGetField(sdm, DMSwarmPICField_coor, NULL, NULL, (void **) &coords));
  for (c = 0; c < 2; ++c) {
    for (p = 0; p < Np; ++p) {
      const PetscInt     n = c*Np + p;
      const PetscScalar *x = &u[(n*2+0)*dim];
      const PetscScalar *v = &u[(n*2+1)*dim];
      PetscReal          xe[3], ve[3];
      PetscInt           d;

      CHKERRQ(orbit(ctx, c, p, t, xe, ve));
      for (d = 0; d < dim; ++d) {
        e[(p*2+0)*dim+d] = x[d] - xe[d];
        e[(p*2+1)*dim+d] = v[d] - ve[d];
      }
      if (ctx->error) CHKERRQ(PetscPrintf(comm, "p%D error [%.2g %.2g] sol [(%.6lf %.6lf) (%.6lf %.6lf)] exact [(%.6lf %.6lf) (%.6lf %.6lf)] energy/exact energy %g / %g\n", p, (double) DMPlex_NormD_Internal(dim, &e[(p*2+0)*dim]), (double) DMPlex_NormD_Internal(dim, &e[(p*2+1)*dim]), (double) x[0], (double) x[1], (double) v[0], (double) v[1], (double) xe[0], (double) xe[1], (double) ve[0], (double) ve[1], 0.5*DMPlex_NormD_Internal(dim, v), (double) energy(ctx, c)));
    }
  }
  CHKERRQ(DMSwarmRestoreField(sdm, DMSwarmPICField_coor, NULL, NULL, (void **) &coords));
  CHKERRQ(VecRestoreArrayRead(U, &u));
  CHKERRQ(VecRestoreArray(E, &e));
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
  PetscErrorCode     ierr;

  ierr = PetscInitialize(&argc, &argv, NULL, help);if (ierr) return ierr;
  comm = PETSC_COMM_WORLD;
  CHKERRQ(ProcessOptions(comm, &user));

  CHKERRQ(CreateMesh(comm, &dm, &user));
  CHKERRQ(CreateParticles(dm, &sw, &user));
  CHKERRQ(DMSetApplicationContext(sw, &user));

  CHKERRQ(TSCreate(comm, &ts));
  CHKERRQ(TSSetType(ts, TSBASICSYMPLECTIC));
  CHKERRQ(TSSetDM(ts, sw));
  CHKERRQ(TSSetMaxTime(ts, ftime));
  CHKERRQ(TSSetTimeStep(ts, 0.0001));
  CHKERRQ(TSSetMaxSteps(ts, 10));
  CHKERRQ(TSSetExactFinalTime(ts, TS_EXACTFINALTIME_MATCHSTEP));
  CHKERRQ(TSSetTime(ts, 0.0));
  CHKERRQ(TSSetRHSFunction(ts, NULL, RHSFunctionParticles, &user));

  CHKERRQ(DMSwarmCreateGlobalVectorFromField(sw, "kinematics", &u));
  CHKERRQ(DMGetDimension(sw, &dim));
  CHKERRQ(VecGetLocalSize(u, &locSize));
  Np   = locSize/(2*dim);
  CHKERRQ(PetscMalloc1(locSize/2, &idx1));
  CHKERRQ(PetscMalloc1(locSize/2, &idx2));
  for (p = 0; p < Np; ++p) {
    for (d = 0; d < dim; ++d) {
      idx1[p*dim+d] = (p*2+0)*dim + d;
      idx2[p*dim+d] = (p*2+1)*dim + d;
    }
  }
  CHKERRQ(ISCreateGeneral(comm, locSize/2, idx1, PETSC_OWN_POINTER, &is1));
  CHKERRQ(ISCreateGeneral(comm, locSize/2, idx2, PETSC_OWN_POINTER, &is2));
  CHKERRQ(TSRHSSplitSetIS(ts, "position", is1));
  CHKERRQ(TSRHSSplitSetIS(ts, "momentum", is2));
  CHKERRQ(ISDestroy(&is1));
  CHKERRQ(ISDestroy(&is2));
  CHKERRQ(TSRHSSplitSetRHSFunction(ts,"position",NULL,RHSFunction1,&user));
  CHKERRQ(TSRHSSplitSetRHSFunction(ts,"momentum",NULL,RHSFunction2,&user));

  CHKERRQ(TSSetFromOptions(ts));
  CHKERRQ(TSSetComputeInitialCondition(ts, InitializeSolve));
  CHKERRQ(TSSetComputeExactError(ts, ComputeError));
  CHKERRQ(TSComputeInitialCondition(ts, u));
  CHKERRQ(VecViewFromOptions(u, NULL, "-init_view"));
  CHKERRQ(TSSolve(ts, u));
  CHKERRQ(TSGetSolveTime(ts, &ftime));
  CHKERRQ(TSGetConvergedReason(ts, &reason));
  CHKERRQ(TSGetStepNumber(ts, &steps));
  CHKERRQ(PetscPrintf(comm,"%s at time %g after %D steps\n", TSConvergedReasons[reason], (double) ftime, steps));

  CHKERRQ(VecGetArrayRead(u, &endVals));
  for (c = 0; c < 2; ++c) {
    for (p = 0; p < Np/2; ++p) {
      const PetscInt  n    = c*(Np/2) + p;
      const PetscReal norm = DMPlex_NormD_Internal(dim, &endVals[(n*2 + 1)*dim]);
      CHKERRQ(PetscPrintf(comm, "Particle %D initial Energy: %g  Final Energy: %g\n", p, (double) (0.5*(1000./(3*c+1.))), (double) 0.5*PetscSqr(norm)));
    }
  }
  CHKERRQ(VecRestoreArrayRead(u, &endVals));
  CHKERRQ(DMSwarmDestroyGlobalVectorFromField(sw, "kinematics", &u));
  CHKERRQ(TSDestroy(&ts));
  CHKERRQ(DMDestroy(&sw));
  CHKERRQ(DMDestroy(&dm));
  ierr = PetscFinalize();
  return ierr;
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
