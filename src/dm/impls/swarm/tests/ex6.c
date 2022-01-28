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
  ierr = PetscOptionsInt("-output_step", "Number of time steps between output", "ex4.c", options->ostep, &options->ostep, PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-monitor", "Flag to use the TS monitor", "ex4.c", options->monitor, &options->monitor, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-error", "Flag to print the error", "ex4.c", options->error, &options->error, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-particles_per_circle", "Number of particles per circle", "ex4.c", options->particlesPerCircle, &options->particlesPerCircle, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

static PetscErrorCode CreateMesh(MPI_Comm comm, DM *dm, AppCtx *user)
{
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = DMCreate(comm, dm);CHKERRQ(ierr);
  ierr = DMSetType(*dm, DMPLEX);CHKERRQ(ierr);
  ierr = DMSetFromOptions(*dm);CHKERRQ(ierr);
  ierr = DMViewFromOptions(*dm, NULL, "-dm_view");CHKERRQ(ierr);
  ierr = DMGetDimension(*dm, &user->dim);CHKERRQ(ierr);
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
  PetscErrorCode    ierr;

  PetscFunctionBeginUser;
  ierr = DMGetApplicationContext(dmSw, &ctx);CHKERRQ(ierr);
  Np   = ctx->particlesPerCircle;
  ierr = DMSwarmGetCellDM(dmSw, &dm);CHKERRQ(ierr);
  ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);
  ierr = DMSwarmGetField(dmSw, DMSwarmPICField_coor, NULL, NULL, (void **) &coords);CHKERRQ(ierr);
  for (c = 0; c < 2; ++c) {
    for (d = 0; d < dim; ++d) ctx->center[c*dim+d] = (!c && !d) ? 3.0 : 0.0;
    ctx->radius[c] = 3.*c+1.;
    for (p = 0; p < Np; ++p) {
      const PetscInt n = c*Np + p;

      ierr = orbit(ctx, c, p, 0.0, &coords[n*dim], NULL);CHKERRQ(ierr);
    }
  }
  ierr = DMSwarmRestoreField(dmSw, DMSwarmPICField_coor, NULL, NULL, (void **) &coords);CHKERRQ(ierr);

  ierr = DMSwarmCreateGlobalVectorFromField(dmSw, DMSwarmPICField_coor, &coordinates);CHKERRQ(ierr);
  ierr = DMLocatePoints(dm, coordinates, DM_POINTLOCATION_NONE, &cellSF);CHKERRQ(ierr);
  ierr = DMSwarmDestroyGlobalVectorFromField(dmSw, DMSwarmPICField_coor, &coordinates);CHKERRQ(ierr);

  ierr = MPI_Comm_rank(PetscObjectComm((PetscObject) dmSw), &rank);CHKERRMPI(ierr);
  ierr = DMSwarmGetField(dmSw, DMSwarmPICField_cellid, NULL, NULL, (void **) &cellid);CHKERRQ(ierr);
  ierr = PetscSFGetGraph(cellSF, NULL, &Np, &found, &cells);CHKERRQ(ierr);
  for (p = 0; p < Np; ++p) {
    const PetscInt part = found ? found[p] : p;

    if (cells[p].rank != rank) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Point %D not found in the mesh", part);
    cellid[part] = cells[p].index;
  }
  ierr = DMSwarmRestoreField(dmSw, DMSwarmPICField_cellid, NULL, NULL, (void **) &cellid);CHKERRQ(ierr);
  ierr = PetscSFDestroy(&cellSF);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode SetInitialConditions(DM dmSw, Vec u)
{
  DM             dm;
  AppCtx        *ctx;
  PetscScalar   *initialConditions;
  PetscInt       dim, cStart, cEnd, c, Np, p;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = DMGetApplicationContext(dmSw, &ctx);CHKERRQ(ierr);
  Np   = ctx->particlesPerCircle;
  ierr = DMSwarmGetCellDM(dmSw, &dm);CHKERRQ(ierr);
  ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd);CHKERRQ(ierr);
  ierr = VecGetArray(u, &initialConditions);CHKERRQ(ierr);
  for (c = 0; c < 2; ++c) {
    for (p = 0; p < Np; ++p) {
      const PetscInt n = c*Np + p;

      ierr = orbit(ctx, c, p, 0.0, &initialConditions[(n*2 + 0)*dim], &initialConditions[(n*2 + 1)*dim]);CHKERRQ(ierr);
    }
  }
  ierr = VecRestoreArray(u, &initialConditions);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode CreateParticles(DM dm, DM *sw, AppCtx *user)
{
  PetscInt       dim, Np = user->particlesPerCircle;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);
  ierr = DMCreate(PetscObjectComm((PetscObject) dm), sw);CHKERRQ(ierr);
  ierr = DMSetType(*sw, DMSWARM);CHKERRQ(ierr);
  ierr = DMSetDimension(*sw, dim);CHKERRQ(ierr);

  ierr = DMSwarmSetType(*sw, DMSWARM_PIC);CHKERRQ(ierr);
  ierr = DMSwarmSetCellDM(*sw, dm);CHKERRQ(ierr);
  ierr = DMSwarmRegisterPetscDatatypeField(*sw, "kinematics", 2*dim, PETSC_REAL);CHKERRQ(ierr);
  ierr = DMSwarmFinalizeFieldRegister(*sw);CHKERRQ(ierr);
  ierr = DMSwarmSetLocalSizes(*sw, 2 * Np, 0);CHKERRQ(ierr);
  ierr = DMSetFromOptions(*sw);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) *sw, "Particles");CHKERRQ(ierr);
  ierr = DMViewFromOptions(*sw, NULL, "-sw_view");CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* Create particle RHS Functions for gravity with G = 1 for simplification */
static PetscErrorCode RHSFunction1(TS ts, PetscReal t, Vec V, Vec Xres, void *ctx)
{
  const PetscScalar *v;
  PetscScalar       *xres;
  PetscInt          Np, p, dim, d;
  PetscErrorCode    ierr;

  PetscFunctionBeginUser;
  /* The DM is not currently pushed down to the splits */
  dim  = ((AppCtx *) ctx)->dim;
  ierr = VecGetLocalSize(Xres, &Np);CHKERRQ(ierr);
  Np  /= dim;
  ierr = VecGetArray(Xres, &xres);CHKERRQ(ierr);
  ierr = VecGetArrayRead(V, &v);CHKERRQ(ierr);
  for (p = 0; p < Np; ++p) {
     for (d = 0; d < dim; ++d) {
       xres[p*dim+d] = v[p*dim+d];
     }
  }
  ierr = VecRestoreArrayRead(V,& v);CHKERRQ(ierr);
  ierr = VecRestoreArray(Xres, &xres);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode RHSFunction2(TS ts, PetscReal t, Vec X, Vec Vres, void *user)
{
  AppCtx           *ctx = (AppCtx *) user;
  const PetscScalar *x;
  PetscScalar       *vres;
  PetscInt          Np, p, dim;
  PetscErrorCode    ierr;

  PetscFunctionBeginUser;
  /* The DM is not currently pushed down to the splits */
  dim  = ctx->dim;
  ierr = VecGetLocalSize(Vres, &Np);CHKERRQ(ierr);
  Np  /= dim;
  ierr = VecGetArray(Vres, &vres);CHKERRQ(ierr);
  ierr = VecGetArrayRead(X, &x);CHKERRQ(ierr);
  for (p = 0; p < Np; ++p) {
    const PetscInt c = p / ctx->particlesPerCircle;

    ierr = force(ctx, c, &x[p*dim], &vres[p*dim]);CHKERRQ(ierr);
  }
  ierr = VecRestoreArray(Vres, &vres);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(X, &x);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode RHSFunctionParticles(TS ts, PetscReal t , Vec U, Vec R, void *user)
{
  AppCtx           *ctx = (AppCtx *) user;
  DM                dm;
  const PetscScalar *u;
  PetscScalar       *r;
  PetscInt          Np, p, dim, d;
  PetscErrorCode    ierr;

  PetscFunctionBeginUser;
  ierr = TSGetDM(ts, &dm);CHKERRQ(ierr);
  ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);
  ierr = VecGetLocalSize(U, &Np);CHKERRQ(ierr);
  Np  /= 2*dim;
  ierr = VecGetArrayRead(U, &u);CHKERRQ(ierr);
  ierr = VecGetArray(R, &r);CHKERRQ(ierr);
  for (p = 0; p < Np; ++p) {
    const PetscInt c = p / ctx->particlesPerCircle;

    for (d = 0; d < dim; ++d) r[(p*2 + 0)*dim + d] = u[(p*2 + 1)*dim + d];
    ierr = force(ctx, c, &u[(p*2 + 0)*dim], &r[(p*2 + 1)*dim]);CHKERRQ(ierr);
  }
  ierr = VecRestoreArrayRead(U, &u);CHKERRQ(ierr);
  ierr = VecRestoreArray(R, &r);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode InitializeSolve(TS ts, Vec u)
{
  DM             dm;
  AppCtx        *user;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = TSGetDM(ts, &dm);CHKERRQ(ierr);
  ierr = DMGetApplicationContext(dm, &user);CHKERRQ(ierr);
  ierr = SetInitialCoordinates(dm);CHKERRQ(ierr);
  ierr = SetInitialConditions(dm, u);CHKERRQ(ierr);
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
  PetscErrorCode     ierr;

  PetscFunctionBeginUser;
  ierr = PetscObjectGetComm((PetscObject) ts, &comm);CHKERRQ(ierr);
  ierr = TSGetDM(ts, &sdm);CHKERRQ(ierr);
  ierr = DMGetApplicationContext(sdm, &ctx);CHKERRQ(ierr);
  ierr = DMGetDimension(sdm, &dim);CHKERRQ(ierr);
  ierr = TSGetSolveTime(ts, &t);CHKERRQ(ierr);
  ierr = VecGetArray(E, &e);CHKERRQ(ierr);
  ierr = VecGetArrayRead(U, &u);CHKERRQ(ierr);
  ierr = VecGetLocalSize(U, &Np);CHKERRQ(ierr);
  Np  /= 2*dim*2;
  ierr = DMSwarmGetField(sdm, DMSwarmPICField_coor, NULL, NULL, (void **) &coords);CHKERRQ(ierr);
  for (c = 0; c < 2; ++c) {
    for (p = 0; p < Np; ++p) {
      const PetscInt     n = c*Np + p;
      const PetscScalar *x = &u[(n*2+0)*dim];
      const PetscScalar *v = &u[(n*2+1)*dim];
      PetscReal          xe[3], ve[3];
      PetscInt           d;

      ierr = orbit(ctx, c, p, t, xe, ve);CHKERRQ(ierr);
      for (d = 0; d < dim; ++d) {
        e[(p*2+0)*dim+d] = x[d] - xe[d];
        e[(p*2+1)*dim+d] = v[d] - ve[d];
      }
      if (ctx->error) {ierr = PetscPrintf(comm, "p%D error [%.2g %.2g] sol [(%.6lf %.6lf) (%.6lf %.6lf)] exact [(%.6lf %.6lf) (%.6lf %.6lf)] energy/exact energy %g / %g\n", p, (double) DMPlex_NormD_Internal(dim, &e[(p*2+0)*dim]), (double) DMPlex_NormD_Internal(dim, &e[(p*2+1)*dim]), (double) x[0], (double) x[1], (double) v[0], (double) v[1], (double) xe[0], (double) xe[1], (double) ve[0], (double) ve[1], 0.5*DMPlex_NormD_Internal(dim, v), (double) energy(ctx, c));}
    }
  }
  ierr = DMSwarmRestoreField(sdm, DMSwarmPICField_coor, NULL, NULL, (void **) &coords);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(U, &u);CHKERRQ(ierr);
  ierr = VecRestoreArray(E, &e);CHKERRQ(ierr);
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
  ierr = ProcessOptions(comm, &user);CHKERRQ(ierr);

  ierr = CreateMesh(comm, &dm, &user);CHKERRQ(ierr);
  ierr = CreateParticles(dm, &sw, &user);CHKERRQ(ierr);
  ierr = DMSetApplicationContext(sw, &user);CHKERRQ(ierr);

  ierr = TSCreate(comm, &ts);CHKERRQ(ierr);
  ierr = TSSetType(ts, TSBASICSYMPLECTIC);CHKERRQ(ierr);
  ierr = TSSetDM(ts, sw);CHKERRQ(ierr);
  ierr = TSSetMaxTime(ts, ftime);CHKERRQ(ierr);
  ierr = TSSetTimeStep(ts, 0.0001);CHKERRQ(ierr);
  ierr = TSSetMaxSteps(ts, 10);CHKERRQ(ierr);
  ierr = TSSetExactFinalTime(ts, TS_EXACTFINALTIME_MATCHSTEP);CHKERRQ(ierr);
  ierr = TSSetTime(ts, 0.0);CHKERRQ(ierr);
  ierr = TSSetRHSFunction(ts, NULL, RHSFunctionParticles, &user);CHKERRQ(ierr);

  ierr = DMSwarmCreateGlobalVectorFromField(sw, "kinematics", &u);CHKERRQ(ierr);
  ierr = DMGetDimension(sw, &dim);CHKERRQ(ierr);
  ierr = VecGetLocalSize(u, &locSize);CHKERRQ(ierr);
  Np   = locSize/(2*dim);
  ierr = PetscMalloc1(locSize/2, &idx1);CHKERRQ(ierr);
  ierr = PetscMalloc1(locSize/2, &idx2);CHKERRQ(ierr);
  for (p = 0; p < Np; ++p) {
    for (d = 0; d < dim; ++d) {
      idx1[p*dim+d] = (p*2+0)*dim + d;
      idx2[p*dim+d] = (p*2+1)*dim + d;
    }
  }
  ierr = ISCreateGeneral(comm, locSize/2, idx1, PETSC_OWN_POINTER, &is1);CHKERRQ(ierr);
  ierr = ISCreateGeneral(comm, locSize/2, idx2, PETSC_OWN_POINTER, &is2);CHKERRQ(ierr);
  ierr = TSRHSSplitSetIS(ts, "position", is1);CHKERRQ(ierr);
  ierr = TSRHSSplitSetIS(ts, "momentum", is2);CHKERRQ(ierr);
  ierr = ISDestroy(&is1);CHKERRQ(ierr);
  ierr = ISDestroy(&is2);CHKERRQ(ierr);
  ierr = TSRHSSplitSetRHSFunction(ts,"position",NULL,RHSFunction1,&user);CHKERRQ(ierr);
  ierr = TSRHSSplitSetRHSFunction(ts,"momentum",NULL,RHSFunction2,&user);CHKERRQ(ierr);

  ierr = TSSetFromOptions(ts);CHKERRQ(ierr);
  ierr = TSSetComputeInitialCondition(ts, InitializeSolve);CHKERRQ(ierr);
  ierr = TSSetComputeExactError(ts, ComputeError);CHKERRQ(ierr);
  ierr = TSComputeInitialCondition(ts, u);CHKERRQ(ierr);
  ierr = VecViewFromOptions(u, NULL, "-init_view");CHKERRQ(ierr);
  ierr = TSSolve(ts, u);CHKERRQ(ierr);
  ierr = TSGetSolveTime(ts, &ftime);CHKERRQ(ierr);
  ierr = TSGetConvergedReason(ts, &reason);CHKERRQ(ierr);
  ierr = TSGetStepNumber(ts, &steps);CHKERRQ(ierr);
  ierr = PetscPrintf(comm,"%s at time %g after %D steps\n", TSConvergedReasons[reason], (double) ftime, steps);CHKERRQ(ierr);

  ierr = VecGetArrayRead(u, &endVals);CHKERRQ(ierr);
  for (c = 0; c < 2; ++c) {
    for (p = 0; p < Np/2; ++p) {
      const PetscInt  n    = c*(Np/2) + p;
      const PetscReal norm = DMPlex_NormD_Internal(dim, &endVals[(n*2 + 1)*dim]);
      ierr = PetscPrintf(comm, "Particle %D initial Energy: %g  Final Energy: %g\n", p, (double) (0.5*(1000./(3*c+1.))), (double) 0.5*PetscSqr(norm));CHKERRQ(ierr);
    }
  }
  ierr = VecRestoreArrayRead(u, &endVals);CHKERRQ(ierr);
  ierr = DMSwarmDestroyGlobalVectorFromField(sw, "kinematics", &u);CHKERRQ(ierr);
  ierr = TSDestroy(&ts);CHKERRQ(ierr);
  ierr = DMDestroy(&sw);CHKERRQ(ierr);
  ierr = DMDestroy(&dm);CHKERRQ(ierr);
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
