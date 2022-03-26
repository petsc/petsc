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

  ierr = PetscOptionsBegin(comm, "", "Vlasov Options", "DMPLEX");PetscCall(ierr);
  PetscCall(PetscOptionsInt("-output_step", "Number of time steps between output", "ex4.c", options->ostep, &options->ostep, PETSC_NULL));
  PetscCall(PetscOptionsBool("-monitor", "Flag to use the TS monitor", "ex4.c", options->monitor, &options->monitor, NULL));
  PetscCall(PetscOptionsBool("-error", "Flag to print the error", "ex4.c", options->error, &options->error, NULL));
  PetscCall(PetscOptionsInt("-particles_per_cell", "Number of particles per cell", "ex4.c", options->particlesPerCell, &options->particlesPerCell, NULL));
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

static PetscErrorCode SetInitialCoordinates(DM dmSw)
{
  DM             dm;
  AppCtx        *user;
  PetscRandom    rnd;
  PetscBool      simplex;
  PetscReal     *centroid, *coords, *xi0, *v0, *J, *invJ, detJ;
  PetscInt       dim, d, cStart, cEnd, c, Np, p;

  PetscFunctionBeginUser;
  PetscCall(PetscRandomCreate(PetscObjectComm((PetscObject) dmSw), &rnd));
  PetscCall(PetscRandomSetInterval(rnd, -1.0, 1.0));
  PetscCall(PetscRandomSetFromOptions(rnd));

  PetscCall(DMGetApplicationContext(dmSw, &user));
  Np   = user->particlesPerCell;
  PetscCall(DMSwarmGetCellDM(dmSw, &dm));
  PetscCall(DMGetDimension(dm, &dim));
  PetscCall(DMPlexIsSimplex(dm, &simplex));
  PetscCall(DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd));
  PetscCall(PetscMalloc5(dim, &centroid, dim, &xi0, dim, &v0, dim*dim, &J, dim*dim, &invJ));
  for (d = 0; d < dim; ++d) xi0[d] = -1.0;
  PetscCall(DMSwarmGetField(dmSw, DMSwarmPICField_coor, NULL, NULL, (void **) &coords));
  for (c = cStart; c < cEnd; ++c) {
    if (Np == 1) {
      PetscCall(DMPlexComputeCellGeometryFVM(dm, c, NULL, centroid, NULL));
      for (d = 0; d < dim; ++d) coords[c*dim+d] = centroid[d];
    } else {
      PetscCall(DMPlexComputeCellGeometryFEM(dm, c, NULL, v0, J, invJ, &detJ)); /* affine */
      for (p = 0; p < Np; ++p) {
        const PetscInt n   = c*Np + p;
        PetscReal      sum = 0.0, refcoords[3];

        for (d = 0; d < dim; ++d) {
          PetscCall(PetscRandomGetValueReal(rnd, &refcoords[d]));
          sum += refcoords[d];
        }
        if (simplex && sum > 0.0) for (d = 0; d < dim; ++d) refcoords[d] -= PetscSqrtReal(dim)*sum;
        CoordinatesRefToReal(dim, dim, xi0, v0, J, refcoords, &coords[n*dim]);
      }
    }
  }
  PetscCall(DMSwarmRestoreField(dmSw, DMSwarmPICField_coor, NULL, NULL, (void **) &coords));
  PetscCall(PetscFree5(centroid, xi0, v0, J, invJ));
  PetscCall(PetscRandomDestroy(&rnd));
  PetscFunctionReturn(0);
}

static PetscErrorCode SetInitialConditions(DM dmSw, Vec u)
{
  DM             dm;
  AppCtx         *user;
  PetscScalar    *initialConditions;
  PetscInt       dim, cStart, cEnd, c, Np, p;

  PetscFunctionBeginUser;
  PetscCall(DMGetApplicationContext(dmSw, &user));
  Np   = user->particlesPerCell;
  PetscCall(DMSwarmGetCellDM(dmSw, &dm));
  PetscCall(DMGetDimension(dm, &dim));
  PetscCall(DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd));
  PetscCall(VecGetArray(u, &initialConditions));
  for (c = cStart; c < cEnd; ++c) {
    for (p = 0; p < Np; ++p) {
      const PetscInt n = c*Np + p;

      initialConditions[(n*2 + 0)*dim + 0] = n+1;
      initialConditions[(n*2 + 0)*dim + 1] = 0;
      initialConditions[(n*2 + 1)*dim + 0] = 0;
      initialConditions[(n*2 + 1)*dim + 1] = PetscSqrtReal(1000./(n+1.));
    }
  }
  PetscCall(VecRestoreArray(u, &initialConditions));
  PetscFunctionReturn(0);
}

static PetscErrorCode CreateParticles(DM dm, DM *sw, AppCtx *user)
{
  PetscInt      *cellid;
  PetscInt       dim, cStart, cEnd, c, Np = user->particlesPerCell, p;

  PetscFunctionBeginUser;
  PetscCall(DMGetDimension(dm, &dim));
  PetscCall(DMCreate(PetscObjectComm((PetscObject) dm), sw));
  PetscCall(DMSetType(*sw, DMSWARM));
  PetscCall(DMSetDimension(*sw, dim));

  PetscCall(DMSwarmSetType(*sw, DMSWARM_PIC));
  PetscCall(DMSwarmSetCellDM(*sw, dm));
  PetscCall(DMSwarmRegisterPetscDatatypeField(*sw, "kinematics", 2*dim, PETSC_REAL));
  PetscCall(DMSwarmFinalizeFieldRegister(*sw));
  PetscCall(DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd));
  PetscCall(DMSwarmSetLocalSizes(*sw, (cEnd - cStart) * Np, 0));
  PetscCall(DMSetFromOptions(*sw));
  PetscCall(DMSwarmGetField(*sw, DMSwarmPICField_cellid, NULL, NULL, (void **) &cellid));
  for (c = cStart; c < cEnd; ++c) {
    for (p = 0; p < Np; ++p) {
      const PetscInt n = c*Np + p;

      cellid[n] = c;
    }
  }
  PetscCall(DMSwarmRestoreField(*sw, DMSwarmPICField_cellid, NULL, NULL, (void **) &cellid));
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

static PetscErrorCode RHSFunction2(TS ts, PetscReal t, Vec X, Vec Vres, void *ctx)
{
  const PetscScalar *x;
  PetscScalar       *vres;
  PetscInt          Np, p, dim, d;

  PetscFunctionBeginUser;
  /* The DM is not currently pushed down to the splits */
  dim  = ((AppCtx *) ctx)->dim;
  PetscCall(VecGetLocalSize(Vres, &Np));
  Np  /= dim;
  PetscCall(VecGetArray(Vres, &vres));
  PetscCall(VecGetArrayRead(X, &x));
  for (p = 0; p < Np; ++p) {
    const PetscScalar rsqr = DMPlex_NormD_Internal(dim, &x[p*dim]);

    for (d = 0; d < dim; ++d) {
      vres[p*dim+d] = -(1000./(p+1.)) * x[p*dim+d]/PetscSqr(rsqr);
    }
  }
  PetscCall(VecRestoreArray(Vres, &vres));
  PetscCall(VecRestoreArrayRead(X, &x));
  PetscFunctionReturn(0);
}

static PetscErrorCode RHSFunctionParticles(TS ts, PetscReal t , Vec U, Vec R, void *ctx)
{
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
    const PetscScalar rsqr = DMPlex_NormD_Internal(dim, &u[p*2*dim]);

    for (d = 0; d < dim; ++d) {
        r[p*2*dim+d]   = u[p*2*dim+d+2];
        r[p*2*dim+d+2] = -(1000./(1.+p)) * u[p*2*dim+d]/PetscSqr(rsqr);
    }
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
  MPI_Comm          comm;
  DM                sdm;
  AppCtx            *user;
  const PetscScalar *u, *coords;
  PetscScalar       *e;
  PetscReal         t;
  PetscInt          dim, Np, p;

  PetscFunctionBeginUser;
  PetscCall(PetscObjectGetComm((PetscObject) ts, &comm));
  PetscCall(TSGetDM(ts, &sdm));
  PetscCall(DMGetApplicationContext(sdm, &user));
  PetscCall(DMGetDimension(sdm, &dim));
  PetscCall(TSGetSolveTime(ts, &t));
  PetscCall(VecGetArray(E, &e));
  PetscCall(VecGetArrayRead(U, &u));
  PetscCall(VecGetLocalSize(U, &Np));
  Np  /= 2*dim;
  PetscCall(DMSwarmGetField(sdm, DMSwarmPICField_coor, NULL, NULL, (void **) &coords));
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
    if (user->error) PetscCall(PetscPrintf(comm, "t %.4g: p%D error [%.2g %.2g] sol [(%.6lf %.6lf) (%.6lf %.6lf)] exact [(%.6lf %.6lf) (%.6lf %.6lf)] energy/exact energy %g / %g\n", t, p, (double) DMPlex_NormD_Internal(dim, &e[(p*2+0)*dim]), (double) DMPlex_NormD_Internal(dim, &e[(p*2+1)*dim]), (double) x[0], (double) x[1], (double) v[0], (double) v[1], (double) xe[0], (double) xe[1], (double) ve[0], (double) ve[1], 0.5*DMPlex_NormD_Internal(dim, v), (double) (0.5*(1000./(p+1)))));
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
  PetscInt           locSize, dim, d, Np, p, steps;

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
  for (p = 0; p < Np; ++p) {
    const PetscReal norm = DMPlex_NormD_Internal(dim, &endVals[(p*2 + 1)*dim]);
    PetscCall(PetscPrintf(comm, "Particle %D initial Energy: %g  Final Energy: %g\n", p, (double) (0.5*(1000./(p+1))), (double) 0.5*PetscSqr(norm)));
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
