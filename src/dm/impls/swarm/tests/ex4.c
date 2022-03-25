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

  PetscCall(PetscStrcpy(options->filename, ""));

  ierr = PetscOptionsBegin(comm, "", "Harmonic Oscillator Options", "DMPLEX");PetscCall(ierr);
  PetscCall(PetscOptionsInt("-output_step", "Number of time steps between output", "ex4.c", options->ostep, &options->ostep, PETSC_NULL));
  PetscCall(PetscOptionsInt("-dim", "The topological mesh dimension", "ex4.c", options->dim, &options->dim, NULL));
  PetscCall(PetscOptionsBool("-monitor", "Flag to use the TS monitor", "ex4.c", options->monitor, &options->monitor, NULL));
  PetscCall(PetscOptionsBool("-error", "Flag to print the error", "ex4.c", options->error, &options->error, NULL));
  PetscCall(PetscOptionsBool("-simplex", "The flag for simplices or tensor cells", "ex4.c", options->simplex, &options->simplex, NULL));
  PetscCall(PetscOptionsString("-mesh", "Name of the mesh filename if any", "ex4.c", options->filename, options->filename, sizeof(options->filename), NULL));
  PetscCall(PetscOptionsInt("-particles_per_cell", "Number of particles per cell", "ex4.c", options->particlesPerCell, &options->particlesPerCell, NULL));
  PetscCall(PetscOptionsReal("-omega", "Oscillator frequency", "ex4.c", options->omega, &options->omega, PETSC_NULL));
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
  user->numberOfCells = cEnd - cStart;
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
  AppCtx        *user;
  PetscReal     *coords;
  PetscScalar   *initialConditions;
  PetscInt       dim, cStart, cEnd, c, Np, p;

  PetscFunctionBeginUser;
  PetscCall(DMGetApplicationContext(dmSw, &user));
  Np   = user->particlesPerCell;
  PetscCall(DMSwarmGetCellDM(dmSw, &dm));
  PetscCall(DMGetDimension(dm, &dim));
  PetscCall(DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd));
  PetscCall(DMSwarmGetField(dmSw, DMSwarmPICField_coor, NULL, NULL, (void **) &coords));
  PetscCall(VecGetArray(u, &initialConditions));
  for (c = cStart; c < cEnd; ++c) {
    for (p = 0; p < Np; ++p) {
      const PetscInt n = c*Np + p;
      initialConditions[n*2+0] = DMPlex_NormD_Internal(dim, &coords[n*dim]);
      initialConditions[n*2+1] = 0.0;
    }
  }
  PetscCall(VecRestoreArray(u, &initialConditions));
  PetscCall(DMSwarmRestoreField(dmSw, DMSwarmPICField_coor, NULL, NULL, (void **) &coords));
  PetscFunctionReturn(0);
}

static PetscErrorCode CreateParticles(DM dm, DM *sw, AppCtx *user)
{
  PetscInt      *cellid;
  PetscInt       dim, cStart, cEnd, c, Np, p;

  PetscFunctionBeginUser;
  PetscCall(DMGetDimension(dm, &dim));
  PetscCall(DMCreate(PetscObjectComm((PetscObject) dm), sw));
  PetscCall(DMSetType(*sw, DMSWARM));
  PetscCall(DMSetDimension(*sw, dim));
  Np = user->particlesPerCell;
  PetscCall(DMSwarmSetType(*sw, DMSWARM_PIC));
  PetscCall(DMSwarmSetCellDM(*sw, dm));
  PetscCall(DMSwarmRegisterPetscDatatypeField(*sw, "kinematics", 2, PETSC_REAL));
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
    PetscCall(PetscObjectGetComm((PetscObject) ts, &comm));
    if (!step) PetscCall(PetscPrintf(comm, "Time     Step Part     Energy Mod Energy\n"));
    PetscCall(TSGetTimeStep(ts, &dt));
    PetscCall(VecGetArrayRead(U, &u));
    PetscCall(VecGetLocalSize(U, &Np));
    Np /= 2;
    for (p = 0; p < Np; ++p) {
      const PetscReal x  = PetscRealPart(u[p*2+0]);
      const PetscReal v  = PetscRealPart(u[p*2+1]);
      const PetscReal E  = 0.5*(v*v + PetscSqr(omega)*x*x);
      const PetscReal mE = 0.5*(v*v + PetscSqr(omega)*x*x - PetscSqr(omega)*dt*x*v);
      PetscCall(PetscPrintf(comm, "%.6lf %4D %4D %10.4lf %10.4lf\n", t, step, p, (double) E, (double) mE));
    }
    PetscCall(VecRestoreArrayRead(U, &u));
  }
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
  AppCtx            *user;
  const PetscScalar *u, *coords;
  PetscScalar       *e;
  PetscReal          t, omega;
  PetscInt           dim, Np, p;

  PetscFunctionBeginUser;
  PetscCall(PetscObjectGetComm((PetscObject) ts, &comm));
  PetscCall(TSGetDM(ts, &sdm));
  PetscCall(DMGetApplicationContext(sdm, &user));
  omega = user->omega;
  PetscCall(DMGetDimension(sdm, &dim));
  PetscCall(TSGetSolveTime(ts, &t));
  PetscCall(VecGetArray(E, &e));
  PetscCall(VecGetArrayRead(U, &u));
  PetscCall(VecGetLocalSize(U, &Np));
  Np  /= 2;
  PetscCall(DMSwarmGetField(sdm, DMSwarmPICField_coor, NULL, NULL, (void **) &coords));
  for (p = 0; p < Np; ++p) {
    const PetscReal x  = PetscRealPart(u[p*2+0]);
    const PetscReal v  = PetscRealPart(u[p*2+1]);
    const PetscReal x0 = DMPlex_NormD_Internal(dim, &coords[p*dim]);
    const PetscReal ex =  x0*PetscCosReal(omega*t);
    const PetscReal ev = -x0*omega*PetscSinReal(omega*t);

    if (user->error) PetscCall(PetscPrintf(comm, "p%D error [%.2g %.2g] sol [%.6lf %.6lf] exact [%.6lf %.6lf] energy/exact energy %g / %g\n", p, (double) PetscAbsReal(x-ex), (double) PetscAbsReal(v-ev), (double) x, (double) v, (double) ex, (double) ev, 0.5*(v*v + PetscSqr(omega)*x*x), (double) 0.5*PetscSqr(omega*x0)));
    e[p*2+0] = x - ex;
    e[p*2+1] = v - ev;
  }
  PetscCall(DMSwarmRestoreField(sdm, DMSwarmPICField_coor, NULL, NULL, (void **) &coords));

  PetscCall(VecRestoreArrayRead(U, &u));
  PetscCall(VecRestoreArray(E, &e));
  PetscFunctionReturn(0);
}

/*---------------------Create particle RHS Functions--------------------------*/
static PetscErrorCode RHSFunction1(TS ts, PetscReal t, Vec V, Vec Xres, void *ctx)
{
  const PetscScalar *v;
  PetscScalar       *xres;
  PetscInt          Np, p;

  PetscFunctionBeginUser;
  PetscCall(VecGetArray(Xres, &xres));
  PetscCall(VecGetArrayRead(V, &v));
  PetscCall(VecGetLocalSize(Xres, &Np));
  for (p = 0; p < Np; ++p) {
     xres[p] = v[p];
  }
  PetscCall(VecRestoreArrayRead(V, &v));
  PetscCall(VecRestoreArray(Xres, &xres));
  PetscFunctionReturn(0);
}

static PetscErrorCode RHSFunction2(TS ts, PetscReal t, Vec X, Vec Vres, void *ctx)
{
  AppCtx            *user = (AppCtx *)ctx;
  const PetscScalar *x;
  PetscScalar       *vres;
  PetscInt          Np, p;

  PetscFunctionBeginUser;
  PetscCall(VecGetArray(Vres, &vres));
  PetscCall(VecGetArrayRead(X, &x));
  PetscCall(VecGetLocalSize(Vres, &Np));
  for (p = 0; p < Np; ++p) {
    vres[p] = -PetscSqr(user->omega)*x[p];
  }
  PetscCall(VecRestoreArrayRead(X, &x));
  PetscCall(VecRestoreArray(Vres, &vres));
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
  PetscCall(TSGetDM(ts, &dm));
  PetscCall(VecGetArray(G, &g));
  PetscCall(VecGetArrayRead(U, &u));
  PetscCall(VecGetLocalSize(U, &Np));
  Np  /= 2;
  for (p = 0; p < Np; ++p) {
    g[p*2+0] = u[p*2+1];
    g[p*2+1] = -PetscSqr(user->omega)*u[p*2+0];
  }
  PetscCall(VecRestoreArrayRead(U, &u));
  PetscCall(VecRestoreArray(G, &g));
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
  PetscCall(VecGetArrayRead(U, &u));
  PetscCall(VecGetLocalSize(U, &Np));
  Np /= 2;
  PetscCall(MatGetOwnershipRange(J, &m, &n));
  for (i = 0; i < Np; ++i) {
    const PetscInt rows[2] = {2*i, 2*i+1};
    PetscCall(MatSetValues(J, 2, rows, 2, rows, vals, INSERT_VALUES));
  }
  PetscCall(MatAssemblyBegin(J,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(J,MAT_FINAL_ASSEMBLY));
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
  PetscCall(VecGetArrayRead(U, &u));
  PetscCall(VecGetLocalSize(U, &Np));
  Np /= 2;
  PetscCall(MatGetOwnershipRange(S, &m, &n));
  for (i = 0; i < Np; ++i) {
    const PetscInt rows[2] = {2*i, 2*i+1};
    PetscCall(MatSetValues(S, 2, rows, 2, rows, vals, INSERT_VALUES));
  }
  PetscCall(MatAssemblyBegin(S,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(S,MAT_FINAL_ASSEMBLY));
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
  PetscCall(TSGetDM(ts, &dm));

  /* Define F */
  PetscCall(VecGetArrayRead(U, &u));
  PetscCall(VecGetLocalSize(U, &Np));
  Np /= 2;
  for (p = 0; p < Np; ++p) {
    *F += 0.5*PetscSqr(user->omega)*PetscSqr(u[p*2+0]) + 0.5*PetscSqr(u[p*2+1]);
  }
  PetscCall(VecRestoreArrayRead(U, &u));
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
  PetscCall(TSGetDM(ts, &dm));
  PetscCall(VecGetArrayRead(U, &u));
  PetscCall(VecGetLocalSize(U, &Np));
  Np /= 2;
  /*Define gradF*/
  PetscCall(VecGetArray(gradF, &g));
  for (p = 0; p < Np; ++p) {
    g[p*2+0] = PetscSqr(user->omega)*u[p*2+0]; /*dF/dx*/
    g[p*2+1] = u[p*2+1]; /*dF/dv*/
  }
  PetscCall(VecRestoreArrayRead(U, &u));
  PetscCall(VecRestoreArray(gradF, &g));
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
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  comm = PETSC_COMM_WORLD;
  PetscCall(ProcessOptions(comm, &user));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create Particle-Mesh
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(CreateMesh(comm, &dm, &user));
  PetscCall(CreateParticles(dm, &sw, &user));
  PetscCall(DMSetApplicationContext(sw, &user));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Setup timestepping solver
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(TSCreate(comm, &ts));
  PetscCall(TSSetProblemType(ts,TS_NONLINEAR));

  PetscCall(TSSetDM(ts, sw));
  PetscCall(TSSetMaxTime(ts, 0.1));
  PetscCall(TSSetTimeStep(ts, 0.00001));
  PetscCall(TSSetMaxSteps(ts, 100));
  PetscCall(TSSetExactFinalTime(ts, TS_EXACTFINALTIME_MATCHSTEP));
  if (user.monitor) PetscCall(TSMonitorSet(ts, Monitor, &user, NULL));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Prepare to solve
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(DMSwarmCreateGlobalVectorFromField(sw, "kinematics", &u));
  PetscCall(VecGetLocalSize(u, &n));
  PetscCall(TSSetFromOptions(ts));
  PetscCall(TSSetComputeInitialCondition(ts, InitializeSolve));
  PetscCall(TSSetComputeExactError(ts, ComputeError));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Define function F(U, Udot , x , t) = G(U, x, t)
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  /* - - - - - - - Basic Symplectic - - - - - - - - - - - - - - - - - - - - - -*/
  PetscCall(ISCreateStride(comm, n/2, 0, 2, &is1));
  PetscCall(ISCreateStride(comm, n/2, 1, 2, &is2));
  PetscCall(TSRHSSplitSetIS(ts, "position", is1));
  PetscCall(TSRHSSplitSetIS(ts, "momentum", is2));
  PetscCall(ISDestroy(&is1));
  PetscCall(ISDestroy(&is2));
  PetscCall(TSRHSSplitSetRHSFunction(ts, "position", NULL, RHSFunction1, &user));
  PetscCall(TSRHSSplitSetRHSFunction(ts, "momentum", NULL, RHSFunction2, &user));

  /* - - - - - - - Theta (Implicit Midpoint) - - - - - - - - - - - - - - - - -*/

  PetscCall(TSSetRHSFunction(ts, NULL, RHSFunction, &user));

  PetscCall(MatCreate(PETSC_COMM_WORLD,&J));
  PetscCall(MatSetSizes(J,PETSC_DECIDE,PETSC_DECIDE,n,n));
  PetscCall(MatSetFromOptions(J));
  PetscCall(MatSetUp(J));
  PetscCall(MatAssemblyBegin(J,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(J,MAT_FINAL_ASSEMBLY));
  PetscCall(TSSetRHSJacobian(ts,J,J,RHSJacobian,&user));

  /* - - - - - - - Discrete Gradients - - - - - - - - - - - - - - - - - - - - */
  PetscCall(TSDiscGradSetFormulation(ts, Sfunc, Ffunc, gradFfunc, &user));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Solve
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(TSComputeInitialCondition(ts, u));
  PetscCall(TSSolve(ts, u));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Clean up workspace
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(MatDestroy(&J));
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
