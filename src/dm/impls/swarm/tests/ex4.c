static char help[] = "Comparing basic symplectic, theta and discrete gradients interators on a simple hamiltonian system (harmonic oscillator) with particles\n";

#include <petscdmplex.h>
#include <petsc/private/dmpleximpl.h>  /* For norm */
#include <petsc/private/petscfeimpl.h> /* For CoordinatesRefToReal() */
#include <petscdmswarm.h>
#include <petscts.h>

typedef struct {
  PetscInt  dim;                          /* The topological mesh dimension */
  PetscBool simplex;                      /* Flag for simplices or tensor cells */
  char      filename[PETSC_MAX_PATH_LEN]; /* Name of the mesh filename if any */
  PetscReal omega;                        /* Oscillation frequency omega */
  PetscInt  particlesPerCell;             /* The number of partices per cell */
  PetscInt  numberOfCells;                /* Number of cells in mesh */
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
  options->numberOfCells    = 2;
  options->momentTol        = 100.0*PETSC_MACHINE_EPSILON;
  options->omega            = 64.0;
  options->ostep            = 100;

  CHKERRQ(PetscStrcpy(options->filename, ""));

  ierr = PetscOptionsBegin(comm, "", "Harmonic Oscillator Options", "DMPLEX");CHKERRQ(ierr);
  CHKERRQ(PetscOptionsInt("-output_step", "Number of time steps between output", "ex4.c", options->ostep, &options->ostep, PETSC_NULL));
  CHKERRQ(PetscOptionsInt("-dim", "The topological mesh dimension", "ex4.c", options->dim, &options->dim, NULL));
  CHKERRQ(PetscOptionsBool("-monitor", "Flag to use the TS monitor", "ex4.c", options->monitor, &options->monitor, NULL));
  CHKERRQ(PetscOptionsBool("-error", "Flag to print the error", "ex4.c", options->error, &options->error, NULL));
  CHKERRQ(PetscOptionsBool("-simplex", "The flag for simplices or tensor cells", "ex4.c", options->simplex, &options->simplex, NULL));
  CHKERRQ(PetscOptionsString("-mesh", "Name of the mesh filename if any", "ex4.c", options->filename, options->filename, sizeof(options->filename), NULL));
  CHKERRQ(PetscOptionsInt("-particles_per_cell", "Number of particles per cell", "ex4.c", options->particlesPerCell, &options->particlesPerCell, NULL));
  CHKERRQ(PetscOptionsReal("-omega", "Oscillator frequency", "ex4.c", options->omega, &options->omega, PETSC_NULL));
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
  user->numberOfCells = cEnd - cStart;
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
  AppCtx        *user;
  PetscReal     *coords;
  PetscScalar   *initialConditions;
  PetscInt       dim, cStart, cEnd, c, Np, p;

  PetscFunctionBeginUser;
  CHKERRQ(DMGetApplicationContext(dmSw, &user));
  Np   = user->particlesPerCell;
  CHKERRQ(DMSwarmGetCellDM(dmSw, &dm));
  CHKERRQ(DMGetDimension(dm, &dim));
  CHKERRQ(DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd));
  CHKERRQ(DMSwarmGetField(dmSw, DMSwarmPICField_coor, NULL, NULL, (void **) &coords));
  CHKERRQ(VecGetArray(u, &initialConditions));
  for (c = cStart; c < cEnd; ++c) {
    for (p = 0; p < Np; ++p) {
      const PetscInt n = c*Np + p;
      initialConditions[n*2+0] = DMPlex_NormD_Internal(dim, &coords[n*dim]);
      initialConditions[n*2+1] = 0.0;
    }
  }
  CHKERRQ(VecRestoreArray(u, &initialConditions));
  CHKERRQ(DMSwarmRestoreField(dmSw, DMSwarmPICField_coor, NULL, NULL, (void **) &coords));
  PetscFunctionReturn(0);
}

static PetscErrorCode CreateParticles(DM dm, DM *sw, AppCtx *user)
{
  PetscInt      *cellid;
  PetscInt       dim, cStart, cEnd, c, Np, p;

  PetscFunctionBeginUser;
  CHKERRQ(DMGetDimension(dm, &dim));
  CHKERRQ(DMCreate(PetscObjectComm((PetscObject) dm), sw));
  CHKERRQ(DMSetType(*sw, DMSWARM));
  CHKERRQ(DMSetDimension(*sw, dim));
  Np = user->particlesPerCell;
  CHKERRQ(DMSwarmSetType(*sw, DMSWARM_PIC));
  CHKERRQ(DMSwarmSetCellDM(*sw, dm));
  CHKERRQ(DMSwarmRegisterPetscDatatypeField(*sw, "kinematics", 2, PETSC_REAL));
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

static PetscErrorCode Monitor(TS ts, PetscInt step, PetscReal t, Vec U, void *ctx)
{
  AppCtx            *user  = (AppCtx *) ctx;
  const PetscReal    omega = user->omega;
  const PetscScalar *u;
  MPI_Comm           comm;
  PetscReal          dt;
  PetscInt           Np, p;

  PetscFunctionBeginUser;
  if (step%user->ostep == 0) {
    CHKERRQ(PetscObjectGetComm((PetscObject) ts, &comm));
    if (!step) CHKERRQ(PetscPrintf(comm, "Time     Step Part     Energy Mod Energy\n"));
    CHKERRQ(TSGetTimeStep(ts, &dt));
    CHKERRQ(VecGetArrayRead(U, &u));
    CHKERRQ(VecGetLocalSize(U, &Np));
    Np /= 2;
    for (p = 0; p < Np; ++p) {
      const PetscReal x  = PetscRealPart(u[p*2+0]);
      const PetscReal v  = PetscRealPart(u[p*2+1]);
      const PetscReal E  = 0.5*(v*v + PetscSqr(omega)*x*x);
      const PetscReal mE = 0.5*(v*v + PetscSqr(omega)*x*x - PetscSqr(omega)*dt*x*v);
      CHKERRQ(PetscPrintf(comm, "%.6lf %4D %4D %10.4lf %10.4lf\n", t, step, p, (double) E, (double) mE));
    }
    CHKERRQ(VecRestoreArrayRead(U, &u));
  }
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
  AppCtx            *user;
  const PetscScalar *u, *coords;
  PetscScalar       *e;
  PetscReal          t, omega;
  PetscInt           dim, Np, p;

  PetscFunctionBeginUser;
  CHKERRQ(PetscObjectGetComm((PetscObject) ts, &comm));
  CHKERRQ(TSGetDM(ts, &sdm));
  CHKERRQ(DMGetApplicationContext(sdm, &user));
  omega = user->omega;
  CHKERRQ(DMGetDimension(sdm, &dim));
  CHKERRQ(TSGetSolveTime(ts, &t));
  CHKERRQ(VecGetArray(E, &e));
  CHKERRQ(VecGetArrayRead(U, &u));
  CHKERRQ(VecGetLocalSize(U, &Np));
  Np  /= 2;
  CHKERRQ(DMSwarmGetField(sdm, DMSwarmPICField_coor, NULL, NULL, (void **) &coords));
  for (p = 0; p < Np; ++p) {
    const PetscReal x  = PetscRealPart(u[p*2+0]);
    const PetscReal v  = PetscRealPart(u[p*2+1]);
    const PetscReal x0 = DMPlex_NormD_Internal(dim, &coords[p*dim]);
    const PetscReal ex =  x0*PetscCosReal(omega*t);
    const PetscReal ev = -x0*omega*PetscSinReal(omega*t);

    if (user->error) CHKERRQ(PetscPrintf(comm, "p%D error [%.2g %.2g] sol [%.6lf %.6lf] exact [%.6lf %.6lf] energy/exact energy %g / %g\n", p, (double) PetscAbsReal(x-ex), (double) PetscAbsReal(v-ev), (double) x, (double) v, (double) ex, (double) ev, 0.5*(v*v + PetscSqr(omega)*x*x), (double) 0.5*PetscSqr(omega*x0)));
    e[p*2+0] = x - ex;
    e[p*2+1] = v - ev;
  }
  CHKERRQ(DMSwarmRestoreField(sdm, DMSwarmPICField_coor, NULL, NULL, (void **) &coords));

  CHKERRQ(VecRestoreArrayRead(U, &u));
  CHKERRQ(VecRestoreArray(E, &e));
  PetscFunctionReturn(0);
}

/*---------------------Create particle RHS Functions--------------------------*/
static PetscErrorCode RHSFunction1(TS ts, PetscReal t, Vec V, Vec Xres, void *ctx)
{
  const PetscScalar *v;
  PetscScalar       *xres;
  PetscInt          Np, p;

  PetscFunctionBeginUser;
  CHKERRQ(VecGetArray(Xres, &xres));
  CHKERRQ(VecGetArrayRead(V, &v));
  CHKERRQ(VecGetLocalSize(Xres, &Np));
  for (p = 0; p < Np; ++p) {
     xres[p] = v[p];
  }
  CHKERRQ(VecRestoreArrayRead(V, &v));
  CHKERRQ(VecRestoreArray(Xres, &xres));
  PetscFunctionReturn(0);
}

static PetscErrorCode RHSFunction2(TS ts, PetscReal t, Vec X, Vec Vres, void *ctx)
{
  AppCtx            *user = (AppCtx *)ctx;
  const PetscScalar *x;
  PetscScalar       *vres;
  PetscInt          Np, p;

  PetscFunctionBeginUser;
  CHKERRQ(VecGetArray(Vres, &vres));
  CHKERRQ(VecGetArrayRead(X, &x));
  CHKERRQ(VecGetLocalSize(Vres, &Np));
  for (p = 0; p < Np; ++p) {
    vres[p] = -PetscSqr(user->omega)*x[p];
  }
  CHKERRQ(VecRestoreArrayRead(X, &x));
  CHKERRQ(VecRestoreArray(Vres, &vres));
  PetscFunctionReturn(0);
}

/*----------------------------------------------------------------------------*/

/*--------------------Define RHSFunction, RHSJacobian (Theta)-----------------*/
static PetscErrorCode RHSFunction(TS ts, PetscReal t, Vec U, Vec G, void *ctx)
{
  AppCtx            *user = (AppCtx *) ctx;
  DM                dm;
  const PetscScalar *u;
  PetscScalar       *g;
  PetscInt          Np, p;

  PetscFunctionBeginUser;
  CHKERRQ(TSGetDM(ts, &dm));
  CHKERRQ(VecGetArray(G, &g));
  CHKERRQ(VecGetArrayRead(U, &u));
  CHKERRQ(VecGetLocalSize(U, &Np));
  Np  /= 2;
  for (p = 0; p < Np; ++p) {
    g[p*2+0] = u[p*2+1];
    g[p*2+1] = -PetscSqr(user->omega)*u[p*2+0];
  }
  CHKERRQ(VecRestoreArrayRead(U, &u));
  CHKERRQ(VecRestoreArray(G, &g));
  PetscFunctionReturn(0);
}

/*Ji = dFi/dxj
J= (0    1)
   (-w^2 0)
*/
static PetscErrorCode RHSJacobian(TS ts, PetscReal t, Vec U , Mat J, Mat P, void *ctx)
{
  AppCtx             *user = (AppCtx *) ctx;
  PetscInt           Np;
  PetscInt           i, m, n;
  const PetscScalar *u;
  PetscScalar        vals[4] = {0., 1., -PetscSqr(user->omega), 0.};

  PetscFunctionBeginUser;
  CHKERRQ(VecGetArrayRead(U, &u));
  CHKERRQ(VecGetLocalSize(U, &Np));
  Np /= 2;
  CHKERRQ(MatGetOwnershipRange(J, &m, &n));
  for (i = 0; i < Np; ++i) {
    const PetscInt rows[2] = {2*i, 2*i+1};
    CHKERRQ(MatSetValues(J, 2, rows, 2, rows, vals, INSERT_VALUES));
  }
  CHKERRQ(MatAssemblyBegin(J,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(J,MAT_FINAL_ASSEMBLY));
  PetscFunctionReturn(0);

}

/*----------------------------------------------------------------------------*/

/*----------------Define S, F, G Functions (Discrete Gradients)---------------*/
/*
  u_t = S * gradF
    --or--
  u_t = S * G

  + Sfunc - constructor for the S matrix from the formulation
  . Ffunc - functional F from the formulation
  - Gfunc - constructor for the gradient of F from the formulation
*/

PetscErrorCode Sfunc(TS ts, PetscReal t, Vec U, Mat S, void *ctx)
{
  PetscInt           Np;
  PetscInt           i, m, n;
  const PetscScalar *u;
  PetscScalar        vals[4] = {0., 1., -1, 0.};

  PetscFunctionBeginUser;
  CHKERRQ(VecGetArrayRead(U, &u));
  CHKERRQ(VecGetLocalSize(U, &Np));
  Np /= 2;
  CHKERRQ(MatGetOwnershipRange(S, &m, &n));
  for (i = 0; i < Np; ++i) {
    const PetscInt rows[2] = {2*i, 2*i+1};
    CHKERRQ(MatSetValues(S, 2, rows, 2, rows, vals, INSERT_VALUES));
  }
  CHKERRQ(MatAssemblyBegin(S,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(S,MAT_FINAL_ASSEMBLY));
  PetscFunctionReturn(0);

}

PetscErrorCode Ffunc(TS ts, PetscReal t, Vec U, PetscScalar *F, void *ctx)
{
  AppCtx            *user = (AppCtx *) ctx;
  DM                 dm;
  const PetscScalar *u;
  PetscInt           Np;
  PetscInt           p;

  PetscFunctionBeginUser;
  CHKERRQ(TSGetDM(ts, &dm));

  /* Define F */
  CHKERRQ(VecGetArrayRead(U, &u));
  CHKERRQ(VecGetLocalSize(U, &Np));
  Np /= 2;
  for (p = 0; p < Np; ++p) {
    *F += 0.5*PetscSqr(user->omega)*PetscSqr(u[p*2+0]) + 0.5*PetscSqr(u[p*2+1]);
  }
  CHKERRQ(VecRestoreArrayRead(U, &u));
  PetscFunctionReturn(0);
}

PetscErrorCode gradFfunc(TS ts, PetscReal t, Vec U, Vec gradF, void *ctx)
{
  AppCtx            *user = (AppCtx *) ctx;
  DM                dm;
  const PetscScalar *u;
  PetscScalar       *g;
  PetscInt          Np;
  PetscInt          p;

  PetscFunctionBeginUser;
  CHKERRQ(TSGetDM(ts, &dm));
  CHKERRQ(VecGetArrayRead(U, &u));
  CHKERRQ(VecGetLocalSize(U, &Np));
  Np /= 2;
  /*Define gradF*/
  CHKERRQ(VecGetArray(gradF, &g));
  for (p = 0; p < Np; ++p) {
    g[p*2+0] = PetscSqr(user->omega)*u[p*2+0]; /*dF/dx*/
    g[p*2+1] = u[p*2+1]; /*dF/dv*/
  }
  CHKERRQ(VecRestoreArrayRead(U, &u));
  CHKERRQ(VecRestoreArray(gradF, &g));
  PetscFunctionReturn(0);
}

/*----------------------------------------------------------------------------*/

int main(int argc,char **argv)
{
  TS             ts;     /* nonlinear solver                 */
  DM             dm, sw; /* Mesh and particle managers       */
  Vec            u;      /* swarm vector                     */
  PetscInt       n;      /* swarm vector size                */
  IS             is1, is2;
  MPI_Comm       comm;
  Mat            J;      /* Jacobian matrix                  */
  AppCtx         user;

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Initialize program
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  CHKERRQ(PetscInitialize(&argc, &argv, NULL, help));
  comm = PETSC_COMM_WORLD;
  CHKERRQ(ProcessOptions(comm, &user));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create Particle-Mesh
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  CHKERRQ(CreateMesh(comm, &dm, &user));
  CHKERRQ(CreateParticles(dm, &sw, &user));
  CHKERRQ(DMSetApplicationContext(sw, &user));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Setup timestepping solver
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  CHKERRQ(TSCreate(comm, &ts));
  CHKERRQ(TSSetProblemType(ts,TS_NONLINEAR));

  CHKERRQ(TSSetDM(ts, sw));
  CHKERRQ(TSSetMaxTime(ts, 0.1));
  CHKERRQ(TSSetTimeStep(ts, 0.00001));
  CHKERRQ(TSSetMaxSteps(ts, 100));
  CHKERRQ(TSSetExactFinalTime(ts, TS_EXACTFINALTIME_MATCHSTEP));
  if (user.monitor) CHKERRQ(TSMonitorSet(ts, Monitor, &user, NULL));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Prepare to solve
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  CHKERRQ(DMSwarmCreateGlobalVectorFromField(sw, "kinematics", &u));
  CHKERRQ(VecGetLocalSize(u, &n));
  CHKERRQ(TSSetFromOptions(ts));
  CHKERRQ(TSSetComputeInitialCondition(ts, InitializeSolve));
  CHKERRQ(TSSetComputeExactError(ts, ComputeError));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Define function F(U, Udot , x , t) = G(U, x, t)
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  /* - - - - - - - Basic Symplectic - - - - - - - - - - - - - - - - - - - - - -*/
  CHKERRQ(ISCreateStride(comm, n/2, 0, 2, &is1));
  CHKERRQ(ISCreateStride(comm, n/2, 1, 2, &is2));
  CHKERRQ(TSRHSSplitSetIS(ts, "position", is1));
  CHKERRQ(TSRHSSplitSetIS(ts, "momentum", is2));
  CHKERRQ(ISDestroy(&is1));
  CHKERRQ(ISDestroy(&is2));
  CHKERRQ(TSRHSSplitSetRHSFunction(ts, "position", NULL, RHSFunction1, &user));
  CHKERRQ(TSRHSSplitSetRHSFunction(ts, "momentum", NULL, RHSFunction2, &user));

  /* - - - - - - - Theta (Implicit Midpoint) - - - - - - - - - - - - - - - - -*/

  CHKERRQ(TSSetRHSFunction(ts, NULL, RHSFunction, &user));

  CHKERRQ(MatCreate(PETSC_COMM_WORLD,&J));
  CHKERRQ(MatSetSizes(J,PETSC_DECIDE,PETSC_DECIDE,n,n));
  CHKERRQ(MatSetFromOptions(J));
  CHKERRQ(MatSetUp(J));
  CHKERRQ(MatAssemblyBegin(J,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(J,MAT_FINAL_ASSEMBLY));
  CHKERRQ(TSSetRHSJacobian(ts,J,J,RHSJacobian,&user));

  /* - - - - - - - Discrete Gradients - - - - - - - - - - - - - - - - - - - - */
  CHKERRQ(TSDiscGradSetFormulation(ts, Sfunc, Ffunc, gradFfunc, &user));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Solve
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  CHKERRQ(TSComputeInitialCondition(ts, u));
  CHKERRQ(TSSolve(ts, u));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Clean up workspace
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  CHKERRQ(MatDestroy(&J));
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
     suffix: 1
     args: -dm_plex_box_faces 1,1 -ts_type basicsymplectic -ts_basicsymplectic_type 1 -ts_convergence_estimate -convest_num_refine 2 -dm_view  -monitor -output_step 50 -error

   test:
     suffix: 2
     args: -dm_plex_box_faces 1,1 -ts_type basicsymplectic -ts_basicsymplectic_type 2 -ts_convergence_estimate -convest_num_refine 2 -dm_view  -monitor -output_step 50 -error

   test:
     suffix: 3
     args: -dm_plex_box_faces 1,1 -ts_type basicsymplectic -ts_basicsymplectic_type 3 -ts_convergence_estimate -convest_num_refine 2 -ts_dt 0.0001 -dm_view -monitor -output_step 50 -error

   test:
     suffix: 4
     args: -dm_plex_box_faces 1,1 -ts_type basicsymplectic -ts_basicsymplectic_type 4 -ts_convergence_estimate -convest_num_refine 2 -ts_dt 0.0001 -dm_view  -monitor -output_step 50 -error

   test:
     suffix: 5
     args: -dm_plex_box_faces 1,1 -ts_type theta -ts_theta_theta 0.5 -monitor -output_step 50 -error -ts_convergence_estimate -convest_num_refine 2 -dm_view

   test:
     suffix: 6
     args: -dm_plex_box_faces 1,1 -ts_type discgrad -monitor -output_step 50 -error -ts_convergence_estimate -convest_num_refine 2 -dm_view

   test:
     suffix: 7
     args: -dm_plex_box_faces 1,1 -ts_type discgrad -ts_discgrad_gonzalez -monitor -output_step 50 -error -ts_convergence_estimate -convest_num_refine 2 -dm_view

TEST*/
