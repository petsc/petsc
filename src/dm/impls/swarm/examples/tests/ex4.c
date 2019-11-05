static char help[] = "Example of simple hamiltonian system (harmonic oscillator) with particles and a basic symplectic integrator\n";

#include <petscdmplex.h>
#include <petsc/private/dmpleximpl.h>  /* For norm */
#include <petsc/private/petscfeimpl.h> /* Fpr CoordinatesRefToReal() */
#include <petscdmswarm.h>
#include <petscts.h>

typedef struct {
  PetscInt  dim;                          /* The topological mesh dimension */
  PetscBool simplex;                      /* Flag for simplices or tensor cells */
  char      filename[PETSC_MAX_PATH_LEN]; /* Name of the mesh filename if any */
  PetscReal omega;                        /* Oscillation frequency omega */
  PetscInt  particlesPerCell;             /* The number of partices per cell */
  PetscReal momentTol;                    /* Tolerance for checking moment conservation */
  PetscBool monitor;                      /* Flag for using the TS monitor */
  PetscBool error;                        /* Flag for printing the error */
  PetscInt  ostep;                        /* print the energy at each ostep time steps */
} AppCtx;

static PetscErrorCode ProcessOptions(MPI_Comm comm, AppCtx *options)
{
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  options->dim              = 2;
  options->simplex          = PETSC_TRUE;
  options->monitor          = PETSC_FALSE;
  options->error            = PETSC_FALSE;
  options->particlesPerCell = 1;
  options->momentTol        = 100.0*PETSC_MACHINE_EPSILON;
  options->omega            = 64.0;
  options->ostep            = 100;

  ierr = PetscStrcpy(options->filename, "");CHKERRQ(ierr);

  ierr = PetscOptionsBegin(comm, "", "Harmonic Oscillator Options", "DMPLEX");CHKERRQ(ierr);
  ierr = PetscOptionsInt("-output_step", "Number of time steps between output", "ex4.c", options->ostep, &options->ostep, PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-dim", "The topological mesh dimension", "ex4.c", options->dim, &options->dim, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-monitor", "Flag to use the TS monitor", "ex4.c", options->monitor, &options->monitor, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-error", "Flag to print the error", "ex4.c", options->error, &options->error, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-simplex", "The flag for simplices or tensor cells", "ex4.c", options->simplex, &options->simplex, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsString("-mesh", "Name of the mesh filename if any", "ex4.c", options->filename, options->filename, PETSC_MAX_PATH_LEN, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-particles_per_cell", "Number of particles per cell", "ex4.c", options->particlesPerCell, &options->particlesPerCell, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-omega", "Oscillator frequency", "ex4.c", options->omega, &options->omega, PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

static PetscErrorCode CreateMesh(MPI_Comm comm, DM *dm, AppCtx *user)
{
  PetscBool      flg;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = PetscStrcmp(user->filename, "", &flg);CHKERRQ(ierr);
  if (flg) {
    ierr = DMPlexCreateBoxMesh(comm, user->dim, user->simplex, NULL, NULL, NULL, NULL, PETSC_TRUE, dm);CHKERRQ(ierr);
  } else {
    ierr = DMPlexCreateFromFile(comm, user->filename, PETSC_TRUE, dm);CHKERRQ(ierr);
    ierr = DMGetDimension(*dm, &user->dim);CHKERRQ(ierr);
  }
  {
    DM distributedMesh = NULL;

    ierr = DMPlexDistribute(*dm, 0, NULL, &distributedMesh);CHKERRQ(ierr);
    if (distributedMesh) {
      ierr = DMDestroy(dm);CHKERRQ(ierr);
      *dm  = distributedMesh;
    }
  }
  ierr = DMLocalizeCoordinates(*dm);CHKERRQ(ierr); /* needed for periodic */
  ierr = DMSetFromOptions(*dm);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) *dm, "Mesh");CHKERRQ(ierr);
  ierr = DMViewFromOptions(*dm, NULL, "-dm_view");CHKERRQ(ierr);
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
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = PetscRandomCreate(PetscObjectComm((PetscObject) dmSw), &rnd);CHKERRQ(ierr);
  ierr = PetscRandomSetInterval(rnd, -1.0, 1.0);CHKERRQ(ierr);
  ierr = PetscRandomSetFromOptions(rnd);CHKERRQ(ierr);

  ierr = DMGetApplicationContext(dmSw, (void **) &user);CHKERRQ(ierr);
  simplex = user->simplex;
  Np   = user->particlesPerCell;
  ierr = DMSwarmGetCellDM(dmSw, &dm);CHKERRQ(ierr);
  ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd);CHKERRQ(ierr);
  ierr = PetscMalloc5(dim, &centroid, dim, &xi0, dim, &v0, dim*dim, &J, dim*dim, &invJ);CHKERRQ(ierr);
  for (d = 0; d < dim; ++d) xi0[d] = -1.0;
  ierr = DMSwarmGetField(dmSw, DMSwarmPICField_coor, NULL, NULL, (void **) &coords);CHKERRQ(ierr);
  for (c = cStart; c < cEnd; ++c) {
    if (Np == 1) {
      ierr = DMPlexComputeCellGeometryFVM(dm, c, NULL, centroid, NULL);CHKERRQ(ierr);
      for (d = 0; d < dim; ++d) coords[c*dim+d] = centroid[d];
    } else {
      ierr = DMPlexComputeCellGeometryFEM(dm, c, NULL, v0, J, invJ, &detJ);CHKERRQ(ierr); /* affine */
      for (p = 0; p < Np; ++p) {
        const PetscInt n   = c*Np + p;
        PetscReal      sum = 0.0, refcoords[3];

        for (d = 0; d < dim; ++d) {
          ierr = PetscRandomGetValueReal(rnd, &refcoords[d]);CHKERRQ(ierr);
          sum += refcoords[d];
        }
        if (simplex && sum > 0.0) for (d = 0; d < dim; ++d) refcoords[d] -= PetscSqrtReal(dim)*sum;
        CoordinatesRefToReal(dim, dim, xi0, v0, J, refcoords, &coords[n*dim]);
      }
    }
  }
  ierr = DMSwarmRestoreField(dmSw, DMSwarmPICField_coor, NULL, NULL, (void **) &coords);CHKERRQ(ierr);
  ierr = PetscFree5(centroid, xi0, v0, J, invJ);CHKERRQ(ierr);
  ierr = PetscRandomDestroy(&rnd);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode SetInitialConditions(DM dmSw, Vec u)
{
  DM             dm;
  AppCtx        *user;
  PetscReal     *coords;
  PetscScalar   *initialConditions;
  PetscInt       dim, cStart, cEnd, c, Np, p;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = DMGetApplicationContext(dmSw, (void **) &user);CHKERRQ(ierr);
  Np   = user->particlesPerCell;
  ierr = DMSwarmGetCellDM(dmSw, &dm);CHKERRQ(ierr);
  ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd);CHKERRQ(ierr);
  ierr = DMSwarmGetField(dmSw, DMSwarmPICField_coor, NULL, NULL, (void **) &coords);CHKERRQ(ierr);
  ierr = VecGetArray(u, &initialConditions);CHKERRQ(ierr);
  for (c = cStart; c < cEnd; ++c) {
    for (p = 0; p < Np; ++p) {
      const PetscInt n = c*Np + p;

      initialConditions[n*2+0] = DMPlex_NormD_Internal(dim, &coords[n*dim]);
      initialConditions[n*2+1] = 0.0;
    }
  }
  ierr = VecRestoreArray(u, &initialConditions);CHKERRQ(ierr);
  ierr = DMSwarmRestoreField(dmSw, DMSwarmPICField_coor, NULL, NULL, (void **) &coords);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode CreateParticles(DM dm, DM *sw, AppCtx *user)
{
  PetscInt      *cellid;
  PetscInt       dim, cStart, cEnd, c, Np = user->particlesPerCell, p;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);
  ierr = DMCreate(PetscObjectComm((PetscObject) dm), sw);CHKERRQ(ierr);
  ierr = DMSetType(*sw, DMSWARM);CHKERRQ(ierr);
  ierr = DMSetDimension(*sw, dim);CHKERRQ(ierr);

  ierr = DMSwarmSetType(*sw, DMSWARM_PIC);CHKERRQ(ierr);
  ierr = DMSwarmSetCellDM(*sw, dm);CHKERRQ(ierr);
  ierr = DMSwarmRegisterPetscDatatypeField(*sw, "kinematics", 2, PETSC_REAL);CHKERRQ(ierr);
  ierr = DMSwarmFinalizeFieldRegister(*sw);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd);CHKERRQ(ierr);
  ierr = DMSwarmSetLocalSizes(*sw, (cEnd - cStart) * Np, 0);CHKERRQ(ierr);
  ierr = DMSetFromOptions(*sw);CHKERRQ(ierr);
  ierr = DMSwarmGetField(*sw, DMSwarmPICField_cellid, NULL, NULL, (void **) &cellid);CHKERRQ(ierr);
  for (c = cStart; c < cEnd; ++c) {
    for (p = 0; p < Np; ++p) {
      const PetscInt n = c*Np + p;

      cellid[n] = c;
    }
  }
  ierr = DMSwarmRestoreField(*sw, DMSwarmPICField_cellid, NULL, NULL, (void **) &cellid);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) *sw, "Particles");CHKERRQ(ierr);
  ierr = DMViewFromOptions(*sw, NULL, "-sw_view");CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* Create particle RHS Functions */
static PetscErrorCode RHSFunction1(TS ts, PetscReal t, Vec V, Vec Xres, void *ctx)
{
  const PetscScalar *v;
  PetscScalar       *xres;
  PetscInt           Np, p;
  PetscErrorCode     ierr;

  PetscFunctionBeginUser;
  ierr = VecGetArray(Xres, &xres);CHKERRQ(ierr);
  ierr = VecGetArrayRead(V, &v);CHKERRQ(ierr);
  ierr = VecGetLocalSize(Xres, &Np);CHKERRQ(ierr);
  for (p = 0; p < Np; ++p) {
     xres[p] = v[p];
  }
  ierr = VecRestoreArrayRead(V, &v);CHKERRQ(ierr);
  ierr = VecRestoreArray(Xres, &xres);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode RHSFunction2(TS ts, PetscReal t, Vec X, Vec Vres, void *ctx)
{
  AppCtx            *user = (AppCtx *)ctx;
  const PetscScalar *x;
  PetscScalar       *vres;
  PetscInt           Np, p;
  PetscErrorCode     ierr;

  PetscFunctionBeginUser;
  ierr = VecGetArray(Vres, &vres);CHKERRQ(ierr);
  ierr = VecGetArrayRead(X, &x);CHKERRQ(ierr);
  ierr = VecGetLocalSize(Vres, &Np);CHKERRQ(ierr);
  for (p = 0; p < Np; ++p) {
    vres[p] = -PetscSqr(user->omega)*x[p];
  }
  ierr = VecRestoreArrayRead(X, &x);CHKERRQ(ierr);
  ierr = VecRestoreArray(Vres, &vres);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode RHSFunctionParticles(TS ts, PetscReal t, Vec U, Vec R, void *ctx)
{
  AppCtx            *user = (AppCtx *) ctx;
  DM                 dm;
  const PetscScalar *u;
  PetscScalar       *r;
  PetscInt           Np, p;
  PetscErrorCode     ierr;

  PetscFunctionBeginUser;
  ierr = TSGetDM(ts, &dm);CHKERRQ(ierr);
  ierr = VecGetArray(R, &r);CHKERRQ(ierr);
  ierr = VecGetArrayRead(U, &u);CHKERRQ(ierr);
  ierr = VecGetLocalSize(U, &Np);CHKERRQ(ierr);
  Np  /= 2;
  for (p = 0; p < Np; ++p) {
    r[p*2+0] = u[p*2+1];
    r[p*2+1] = -PetscSqr(user->omega)*u[p*2+0];
  }
  ierr = VecRestoreArrayRead(U, &u);CHKERRQ(ierr);
  ierr = VecRestoreArray(R, &r);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode Monitor(TS ts, PetscInt step, PetscReal t, Vec U, void *ctx)
{
  AppCtx            *user  = (AppCtx *) ctx;
  const PetscReal    omega = user->omega;
  const PetscScalar *u;
  MPI_Comm           comm;
  PetscReal          dt;
  PetscInt           Np, p;
  PetscErrorCode     ierr;

  PetscFunctionBeginUser;
  if (step%user->ostep == 0) {
    ierr = PetscObjectGetComm((PetscObject) ts, &comm);CHKERRQ(ierr);
    if (!step) {ierr = PetscPrintf(comm, "Time     Step Part     Energy Mod Energy\n");CHKERRQ(ierr);}
    ierr = TSGetTimeStep(ts, &dt);CHKERRQ(ierr);
    ierr = VecGetArrayRead(U, &u);CHKERRQ(ierr);
    ierr = VecGetLocalSize(U, &Np);CHKERRQ(ierr);
    Np /= 2;
    for (p = 0; p < Np; ++p) {
      const PetscReal x  = PetscRealPart(u[p*2+0]);
      const PetscReal v  = PetscRealPart(u[p*2+1]);
      const PetscReal E  = 0.5*(v*v + PetscSqr(omega)*x*x);
      const PetscReal mE = 0.5*(v*v + PetscSqr(omega)*x*x - PetscSqr(omega)*dt*x*v);

      ierr = PetscPrintf(comm, "%.6lf %4D %4D %10.4lf %10.4lf\n", t, step, p, (double) E, (double) mE);CHKERRQ(ierr);
    }
    ierr = VecRestoreArrayRead(U, &u);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode InitializeSolve(TS ts, Vec u)
{
  DM             dm;
  AppCtx        *user;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = TSGetDM(ts, &dm);CHKERRQ(ierr);
  ierr = DMGetApplicationContext(dm, (void **) &user);CHKERRQ(ierr);
  ierr = SetInitialCoordinates(dm);CHKERRQ(ierr);
  ierr = SetInitialConditions(dm, u);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode ComputeError(TS ts, Vec U, Vec E)
{
  MPI_Comm           comm;
  DM                 sdm;
  AppCtx            *user;
  const PetscScalar *u, *coords;
  PetscScalar       *e;
  PetscReal          t, omega;
  PetscInt           dim, Np, p;
  PetscErrorCode     ierr;

  PetscFunctionBeginUser;
  ierr = PetscObjectGetComm((PetscObject) ts, &comm);CHKERRQ(ierr);
  ierr = TSGetDM(ts, &sdm);CHKERRQ(ierr);
  ierr = DMGetApplicationContext(sdm, (void **) &user);CHKERRQ(ierr);
  omega = user->omega;
  ierr = DMGetDimension(sdm, &dim);CHKERRQ(ierr);
  ierr = TSGetSolveTime(ts, &t);CHKERRQ(ierr);
  ierr = VecGetArray(E, &e);CHKERRQ(ierr);
  ierr = VecGetArrayRead(U, &u);CHKERRQ(ierr);
  ierr = VecGetLocalSize(U, &Np);CHKERRQ(ierr);
  Np  /= 2;
  ierr = DMSwarmGetField(sdm, DMSwarmPICField_coor, NULL, NULL, (void **) &coords);CHKERRQ(ierr);
  for (p = 0; p < Np; ++p) {
    const PetscReal x  = PetscRealPart(u[p*2+0]);
    const PetscReal v  = PetscRealPart(u[p*2+1]);
    const PetscReal x0 = DMPlex_NormD_Internal(dim, &coords[p*dim]);
    const PetscReal ex =  x0*PetscCosReal(omega*t);
    const PetscReal ev = -x0*omega*PetscSinReal(omega*t);

    if (user->error) {ierr = PetscPrintf(comm, "p%D error [%.2g %.2g] sol [%.6lf %.6lf] exact [%.6lf %.6lf] energy/exact energy %g / %g\n", p, (double) PetscAbsReal(x-ex), (double) PetscAbsReal(v-ev), (double) x, (double) v, (double) ex, (double) ev, 0.5*(v*v + PetscSqr(omega)*x*x), (double) 0.5*PetscSqr(omega*x0));}
    e[p*2+0] = x - ex;
    e[p*2+1] = v - ev;
  }
  ierr = DMSwarmRestoreField(sdm, DMSwarmPICField_coor, NULL, NULL, (void **) &coords);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(U, &u);CHKERRQ(ierr);
  ierr = VecRestoreArray(E, &e);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

int main(int argc,char **argv)
{
  TS             ts;     /* nonlinear solver */
  DM             dm, sw; /* Mesh and particle managers */
  Vec            u;      /* swarm vector */
  IS             is1, is2;
  PetscInt       n;
  MPI_Comm       comm;
  AppCtx         user;
  PetscErrorCode ierr;


  ierr = PetscInitialize(&argc, &argv, NULL, help);if (ierr) return ierr;
  comm = PETSC_COMM_WORLD;
  ierr = ProcessOptions(comm, &user);CHKERRQ(ierr);

  ierr = CreateMesh(comm, &dm, &user);CHKERRQ(ierr);
  ierr = CreateParticles(dm, &sw, &user);CHKERRQ(ierr);
  ierr = DMSetApplicationContext(sw, &user);CHKERRQ(ierr);

  ierr = TSCreate(comm, &ts);CHKERRQ(ierr);
  ierr = TSSetType(ts, TSBASICSYMPLECTIC);CHKERRQ(ierr);
  ierr = TSSetDM(ts, sw);CHKERRQ(ierr);
  ierr = TSSetMaxTime(ts, 0.1);CHKERRQ(ierr);
  ierr = TSSetTimeStep(ts, 0.00001);CHKERRQ(ierr);
  ierr = TSSetMaxSteps(ts, 100);CHKERRQ(ierr);
  ierr = TSSetExactFinalTime(ts, TS_EXACTFINALTIME_MATCHSTEP);CHKERRQ(ierr);
  if (user.monitor) {ierr = TSMonitorSet(ts, Monitor, &user, NULL);CHKERRQ(ierr);}
  ierr = TSSetRHSFunction(ts, NULL, RHSFunctionParticles, &user);CHKERRQ(ierr);

  ierr = DMSwarmCreateGlobalVectorFromField(sw, "kinematics", &u);CHKERRQ(ierr);
  ierr = VecGetLocalSize(u, &n);CHKERRQ(ierr);
  ierr = ISCreateStride(comm, n/2, 0, 2, &is1);CHKERRQ(ierr);
  ierr = ISCreateStride(comm, n/2, 1, 2, &is2);CHKERRQ(ierr);
  ierr = TSRHSSplitSetIS(ts, "position", is1);CHKERRQ(ierr);
  ierr = TSRHSSplitSetIS(ts, "momentum", is2);CHKERRQ(ierr);
  ierr = ISDestroy(&is1);CHKERRQ(ierr);
  ierr = ISDestroy(&is2);CHKERRQ(ierr);
  ierr = TSRHSSplitSetRHSFunction(ts, "position", NULL, RHSFunction1, &user);CHKERRQ(ierr);
  ierr = TSRHSSplitSetRHSFunction(ts, "momentum", NULL, RHSFunction2, &user);CHKERRQ(ierr);
  ierr = TSSetComputeInitialCondition(ts, InitializeSolve);CHKERRQ(ierr);
  ierr = TSSetComputeExactError(ts, ComputeError);CHKERRQ(ierr);

  ierr = TSSetFromOptions(ts);CHKERRQ(ierr);
  ierr = TSComputeInitialCondition(ts, u);CHKERRQ(ierr);
  ierr = TSSolve(ts, u);CHKERRQ(ierr);
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
     suffix: 1
     args: -dm_plex_box_faces 1,1 -ts_basicsymplectic_type 1 -ts_convergence_estimate -convest_num_refine 2 -dm_view -sw_view -monitor -output_step 50 -error
   test:
     suffix: 2
     args: -dm_plex_box_faces 1,1 -ts_basicsymplectic_type 2 -ts_convergence_estimate -convest_num_refine 2 -dm_view -sw_view -monitor -output_step 50 -error
   test:
     suffix: 3
     args: -dm_plex_box_faces 1,1 -ts_basicsymplectic_type 3 -ts_convergence_estimate -convest_num_refine 2 -ts_dt 0.0001 -dm_view -sw_view -monitor -output_step 50 -error
   test:
     suffix: 4
     args: -dm_plex_box_faces 1,1 -ts_basicsymplectic_type 4 -ts_convergence_estimate -convest_num_refine 2 -ts_dt 0.0001 -dm_view -sw_view -monitor -output_step 50 -error

TEST*/
