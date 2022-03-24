static char help[] = "Vlasov example of particles orbiting around a central massive point.\n";

#include <petscdmplex.h>
#include <petsc/private/dmpleximpl.h>  /* For norm */
#include <petsc/private/petscfeimpl.h> /* For CoordinatesRefToReal() */
#include <petscdmswarm.h>
#include <petscts.h>

typedef struct {
  PetscInt  dim;
  PetscInt  particlesPerCell; /* The number of partices per cell */
  PetscReal momentTol;        /* Tolerance for checking moment conservation */
  PetscBool monitor;          /* Flag for using the TS monitor */
  PetscBool error;            /* Flag for printing the error */
  PetscInt  ostep;            /* print the energy at each ostep time steps */
} AppCtx;

static PetscErrorCode ProcessOptions(MPI_Comm comm, AppCtx *options)
{
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  options->monitor          = PETSC_FALSE;
  options->error            = PETSC_FALSE;
  options->particlesPerCell = 1;
  options->momentTol        = 100.0*PETSC_MACHINE_EPSILON;
  options->ostep            = 100;

  ierr = PetscOptionsBegin(comm, "", "Vlasov Options", "DMPLEX");CHKERRQ(ierr);
  CHKERRQ(PetscOptionsInt("-output_step", "Number of time steps between output", "ex4.c", options->ostep, &options->ostep, PETSC_NULL));
  CHKERRQ(PetscOptionsBool("-monitor", "Flag to use the TS monitor", "ex4.c", options->monitor, &options->monitor, NULL));
  CHKERRQ(PetscOptionsBool("-error", "Flag to print the error", "ex4.c", options->error, &options->error, NULL));
  CHKERRQ(PetscOptionsInt("-particles_per_cell", "Number of particles per cell", "ex4.c", options->particlesPerCell, &options->particlesPerCell, NULL));
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

static PetscErrorCode SetInitialCoordinates(DM dmSw)
{
  DM             dm;
  AppCtx        *user;
  PetscRandom    rnd;
  PetscBool      simplex;
  PetscReal     *centroid, *coords, *xi0, *v0, *J, *invJ, detJ;
  PetscInt       dim, d, cStart, cEnd, c, Np, p;

  PetscFunctionBeginUser;
  CHKERRQ(PetscRandomCreate(PetscObjectComm((PetscObject) dmSw), &rnd));
  CHKERRQ(PetscRandomSetInterval(rnd, -1.0, 1.0));
  CHKERRQ(PetscRandomSetFromOptions(rnd));

  CHKERRQ(DMGetApplicationContext(dmSw, &user));
  Np   = user->particlesPerCell;
  CHKERRQ(DMSwarmGetCellDM(dmSw, &dm));
  CHKERRQ(DMGetDimension(dm, &dim));
  CHKERRQ(DMPlexIsSimplex(dm, &simplex));
  CHKERRQ(DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd));
  CHKERRQ(PetscMalloc5(dim, &centroid, dim, &xi0, dim, &v0, dim*dim, &J, dim*dim, &invJ));
  for (d = 0; d < dim; ++d) xi0[d] = -1.0;
  CHKERRQ(DMSwarmGetField(dmSw, DMSwarmPICField_coor, NULL, NULL, (void **) &coords));
  for (c = cStart; c < cEnd; ++c) {
    if (Np == 1) {
      CHKERRQ(DMPlexComputeCellGeometryFVM(dm, c, NULL, centroid, NULL));
      for (d = 0; d < dim; ++d) coords[c*dim+d] = centroid[d];
    } else {
      CHKERRQ(DMPlexComputeCellGeometryFEM(dm, c, NULL, v0, J, invJ, &detJ)); /* affine */
      for (p = 0; p < Np; ++p) {
        const PetscInt n   = c*Np + p;
        PetscReal      sum = 0.0, refcoords[3];

        for (d = 0; d < dim; ++d) {
          CHKERRQ(PetscRandomGetValueReal(rnd, &refcoords[d]));
          sum += refcoords[d];
        }
        if (simplex && sum > 0.0) for (d = 0; d < dim; ++d) refcoords[d] -= PetscSqrtReal(dim)*sum;
        CoordinatesRefToReal(dim, dim, xi0, v0, J, refcoords, &coords[n*dim]);
      }
    }
  }
  CHKERRQ(DMSwarmRestoreField(dmSw, DMSwarmPICField_coor, NULL, NULL, (void **) &coords));
  CHKERRQ(PetscFree5(centroid, xi0, v0, J, invJ));
  CHKERRQ(PetscRandomDestroy(&rnd));
  PetscFunctionReturn(0);
}

static PetscErrorCode SetInitialConditions(DM dmSw, Vec u)
{
  DM             dm;
  AppCtx         *user;
  PetscScalar    *initialConditions;
  PetscInt       dim, cStart, cEnd, c, Np, p;

  PetscFunctionBeginUser;
  CHKERRQ(DMGetApplicationContext(dmSw, &user));
  Np   = user->particlesPerCell;
  CHKERRQ(DMSwarmGetCellDM(dmSw, &dm));
  CHKERRQ(DMGetDimension(dm, &dim));
  CHKERRQ(DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd));
  CHKERRQ(VecGetArray(u, &initialConditions));
  for (c = cStart; c < cEnd; ++c) {
    for (p = 0; p < Np; ++p) {
      const PetscInt n = c*Np + p;

      initialConditions[(n*2 + 0)*dim + 0] = n+1;
      initialConditions[(n*2 + 0)*dim + 1] = 0;
      initialConditions[(n*2 + 1)*dim + 0] = 0;
      initialConditions[(n*2 + 1)*dim + 1] = PetscSqrtReal(1000./(n+1.));
    }
  }
  CHKERRQ(VecRestoreArray(u, &initialConditions));
  PetscFunctionReturn(0);
}

static PetscErrorCode CreateParticles(DM dm, DM *sw, AppCtx *user)
{
  PetscInt      *cellid;
  PetscInt       dim, cStart, cEnd, c, Np = user->particlesPerCell, p;

  PetscFunctionBeginUser;
  CHKERRQ(DMGetDimension(dm, &dim));
  CHKERRQ(DMCreate(PetscObjectComm((PetscObject) dm), sw));
  CHKERRQ(DMSetType(*sw, DMSWARM));
  CHKERRQ(DMSetDimension(*sw, dim));

  CHKERRQ(DMSwarmSetType(*sw, DMSWARM_PIC));
  CHKERRQ(DMSwarmSetCellDM(*sw, dm));
  CHKERRQ(DMSwarmRegisterPetscDatatypeField(*sw, "kinematics", 2*dim, PETSC_REAL));
  CHKERRQ(DMSwarmFinalizeFieldRegister(*sw));
  CHKERRQ(DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd));
  CHKERRQ(DMSwarmSetLocalSizes(*sw, (cEnd - cStart) * Np, 0));
  CHKERRQ(DMSetFromOptions(*sw));
  CHKERRQ(DMSwarmGetField(*sw, DMSwarmPICField_cellid, NULL, NULL, (void **) &cellid));
  for (c = cStart; c < cEnd; ++c) {
    for (p = 0; p < Np; ++p) {
      const PetscInt n = c*Np + p;

      cellid[n] = c;
    }
  }
  CHKERRQ(DMSwarmRestoreField(*sw, DMSwarmPICField_cellid, NULL, NULL, (void **) &cellid));
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

static PetscErrorCode RHSFunction2(TS ts, PetscReal t, Vec X, Vec Vres, void *ctx)
{
  const PetscScalar *x;
  PetscScalar       *vres;
  PetscInt          Np, p, dim, d;

  PetscFunctionBeginUser;
  /* The DM is not currently pushed down to the splits */
  dim  = ((AppCtx *) ctx)->dim;
  CHKERRQ(VecGetLocalSize(Vres, &Np));
  Np  /= dim;
  CHKERRQ(VecGetArray(Vres, &vres));
  CHKERRQ(VecGetArrayRead(X, &x));
  for (p = 0; p < Np; ++p) {
    const PetscScalar rsqr = DMPlex_NormD_Internal(dim, &x[p*dim]);

    for (d = 0; d < dim; ++d) {
      vres[p*dim+d] = -(1000./(p+1.)) * x[p*dim+d]/PetscSqr(rsqr);
    }
  }
  CHKERRQ(VecRestoreArray(Vres, &vres));
  CHKERRQ(VecRestoreArrayRead(X, &x));
  PetscFunctionReturn(0);
}

static PetscErrorCode RHSFunctionParticles(TS ts, PetscReal t , Vec U, Vec R, void *ctx)
{
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
    const PetscScalar rsqr = DMPlex_NormD_Internal(dim, &u[p*2*dim]);

    for (d = 0; d < dim; ++d) {
        r[p*2*dim+d]   = u[p*2*dim+d+2];
        r[p*2*dim+d+2] = -(1000./(1.+p)) * u[p*2*dim+d]/PetscSqr(rsqr);
    }
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
  MPI_Comm          comm;
  DM                sdm;
  AppCtx            *user;
  const PetscScalar *u, *coords;
  PetscScalar       *e;
  PetscReal         t;
  PetscInt          dim, Np, p;

  PetscFunctionBeginUser;
  CHKERRQ(PetscObjectGetComm((PetscObject) ts, &comm));
  CHKERRQ(TSGetDM(ts, &sdm));
  CHKERRQ(DMGetApplicationContext(sdm, &user));
  CHKERRQ(DMGetDimension(sdm, &dim));
  CHKERRQ(TSGetSolveTime(ts, &t));
  CHKERRQ(VecGetArray(E, &e));
  CHKERRQ(VecGetArrayRead(U, &u));
  CHKERRQ(VecGetLocalSize(U, &Np));
  Np  /= 2*dim;
  CHKERRQ(DMSwarmGetField(sdm, DMSwarmPICField_coor, NULL, NULL, (void **) &coords));
  for (p = 0; p < Np; ++p) {
    const PetscScalar *x     = &u[(p*2+0)*dim];
    const PetscScalar *v     = &u[(p*2+1)*dim];
    const PetscReal   x0    = p+1.;
    const PetscReal   omega = PetscSqrtReal(1000./(p+1.))/x0;
    const PetscReal   xe[3] = { x0*PetscCosReal(omega*t),       x0*PetscSinReal(omega*t),       0.0};
    const PetscReal   ve[3] = {-x0*omega*PetscSinReal(omega*t), x0*omega*PetscCosReal(omega*t), 0.0};
    PetscInt          d;

    for (d = 0; d < dim; ++d) {
      e[(p*2+0)*dim+d] = x[d] - xe[d];
      e[(p*2+1)*dim+d] = v[d] - ve[d];
    }
    if (user->error) CHKERRQ(PetscPrintf(comm, "t %.4g: p%D error [%.2g %.2g] sol [(%.6lf %.6lf) (%.6lf %.6lf)] exact [(%.6lf %.6lf) (%.6lf %.6lf)] energy/exact energy %g / %g\n", t, p, (double) DMPlex_NormD_Internal(dim, &e[(p*2+0)*dim]), (double) DMPlex_NormD_Internal(dim, &e[(p*2+1)*dim]), (double) x[0], (double) x[1], (double) v[0], (double) v[1], (double) xe[0], (double) xe[1], (double) ve[0], (double) ve[1], 0.5*DMPlex_NormD_Internal(dim, v), (double) (0.5*(1000./(p+1)))));
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
  PetscInt           locSize, dim, d, Np, p, steps;

  CHKERRQ(PetscInitialize(&argc, &argv, NULL, help));
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
  for (p = 0; p < Np; ++p) {
    const PetscReal norm = DMPlex_NormD_Internal(dim, &endVals[(p*2 + 1)*dim]);
    CHKERRQ(PetscPrintf(comm, "Particle %D initial Energy: %g  Final Energy: %g\n", p, (double) (0.5*(1000./(p+1))), (double) 0.5*PetscSqr(norm)));
  }
  CHKERRQ(VecRestoreArrayRead(u, &endVals));
  CHKERRQ(DMSwarmDestroyGlobalVectorFromField(sw, "kinematics", &u));
  CHKERRQ(TSDestroy(&ts));
  CHKERRQ(DMDestroy(&sw));
  CHKERRQ(DMDestroy(&dm));
  CHKERRQ(PetscFinalize());
  return 0;
}

/*TEST

   build:
     requires: triangle !single !complex
   test:
     suffix: bsi1
     args: -dm_plex_box_faces 1,1 -dm_plex_box_lower -5,-5 -dm_plex_box_upper 5,5 \
           -ts_basicsymplectic_type 1 -ts_max_time 0.1 -ts_convergence_estimate -convest_num_refine 2 \
           -dm_view -sw_view -ts_monitor_sp_swarm -ts_monitor_sp_swarm_retain -ts_monitor_sp_swarm_phase 0
   test:
     suffix: bsi2
     args: -dm_plex_box_faces 1,1 -dm_plex_box_lower -5,-5 -dm_plex_box_upper 5,5 \
           -ts_basicsymplectic_type 2 -ts_max_time 0.1 -ts_convergence_estimate -convest_num_refine 2 \
           -dm_view -sw_view -ts_monitor_sp_swarm -ts_monitor_sp_swarm_retain -ts_monitor_sp_swarm_phase 0
   test:
     suffix: euler
     args: -dm_plex_box_faces 1,1 -dm_plex_box_lower -5,-5 -dm_plex_box_upper 5,5 \
           -ts_type euler -ts_max_time 0.1 -ts_convergence_estimate -convest_num_refine 2 \
           -dm_view -sw_view -ts_monitor_sp_swarm -ts_monitor_sp_swarm_retain -ts_monitor_sp_swarm_phase 0

TEST*/
