static char help[] = "Stokes Problem in 2d and 3d with particles.\n\
We solve the Stokes problem in a rectangular\n\
domain, using a parallel unstructured mesh (DMPLEX) to discretize it and particles (DMSWARM).\n\n\n";

/*
The isoviscous Stokes problem, which we discretize using the finite
element method on an unstructured mesh. The weak form equations are

  < \nabla v, \nabla u + {\nabla u}^T > - < \nabla\cdot v, p > + < v, f > = 0
  < q, \nabla\cdot u >                                                    = 0

We start with homogeneous Dirichlet conditions.
*/

#include <petscdmplex.h>
#include <petscdmswarm.h>
#include <petscsnes.h>
#include <petscds.h>

typedef enum {NEUMANN, DIRICHLET} BCType;
typedef enum {RUN_FULL, RUN_TEST} RunType;

typedef struct {
  RunType   runType;           /* Whether to run tests, or solve the full problem */
  PetscBool showInitial, showSolution, showError;
  BCType    bcType;
} AppCtx;

PetscErrorCode zero_scalar(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nf, PetscScalar *u, void *ctx)
{
  u[0] = 0.0;
  return 0;
}
PetscErrorCode zero_vector(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nf, PetscScalar *u, void *ctx)
{
  PetscInt d;
  for (d = 0; d < dim; ++d) u[d] = 0.0;
  return 0;
}

PetscErrorCode linear_u_2d(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nf, PetscScalar *u, void *ctx)
{
  u[0] = x[0];
  u[1] = 0.0;
  return 0;
}

/*
  In 2D we use exact solution:

    u = x^2 + y^2
    v = 2 x^2 - 2xy
    p = x + y - 1
    f_x = f_y = 3

  so that

    -\Delta u + \nabla p + f = <-4, -4> + <1, 1> + <3, 3> = 0
    \nabla \cdot u           = 2x - 2x                    = 0
*/
PetscErrorCode quadratic_u_2d(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nf, PetscScalar *u, void *ctx)
{
  u[0] = x[0]*x[0] + x[1]*x[1];
  u[1] = 2.0*x[0]*x[0] - 2.0*x[0]*x[1];
  return 0;
}

PetscErrorCode linear_p_2d(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nf, PetscScalar *p, void *ctx)
{
  *p = x[0] + x[1] - 1.0;
  return 0;
}
PetscErrorCode constant_p(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nf, PetscScalar *p, void *ctx)
{
  *p = 1.0;
  return 0;
}

void f0_u(PetscInt dim, PetscInt Nf, PetscInt NfAux,
          const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
          const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
          PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f0[])
{
  PetscInt c;
  for (c = 0; c < dim; ++c) f0[c] = 3.0;
}

/* gradU[comp*dim+d] = {u_x, u_y, v_x, v_y} or {u_x, u_y, u_z, v_x, v_y, v_z, w_x, w_y, w_z}
   u[Ncomp]          = {p} */
void f1_u(PetscInt dim, PetscInt Nf, PetscInt NfAux,
          const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
          const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
          PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f1[])
{
  const PetscInt Ncomp = dim;
  PetscInt       comp, d;

  for (comp = 0; comp < Ncomp; ++comp) {
    for (d = 0; d < dim; ++d) {
      /* f1[comp*dim+d] = 0.5*(gradU[comp*dim+d] + gradU[d*dim+comp]); */
      f1[comp*dim+d] = u_x[comp*dim+d];
    }
    f1[comp*dim+comp] -= u[Ncomp];
  }
}

/* gradU[comp*dim+d] = {u_x, u_y, v_x, v_y} or {u_x, u_y, u_z, v_x, v_y, v_z, w_x, w_y, w_z} */
void f0_p(PetscInt dim, PetscInt Nf, PetscInt NfAux,
          const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
          const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
          PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f0[])
{
  PetscInt d;
  for (d = 0, f0[0] = 0.0; d < dim; ++d) f0[0] += u_x[d*dim+d];
}

void f1_p(PetscInt dim, PetscInt Nf, PetscInt NfAux,
          const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
          const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
          PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f1[])
{
  PetscInt d;
  for (d = 0; d < dim; ++d) f1[d] = 0.0;
}

/* < q, \nabla\cdot u >
   NcompI = 1, NcompJ = dim */
void g1_pu(PetscInt dim, PetscInt Nf, PetscInt NfAux,
           const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
           const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
           PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g1[])
{
  PetscInt d;
  for (d = 0; d < dim; ++d) g1[d*dim+d] = 1.0; /* \frac{\partial\phi^{u_d}}{\partial x_d} */
}

/* -< \nabla\cdot v, p >
    NcompI = dim, NcompJ = 1 */
void g2_up(PetscInt dim, PetscInt Nf, PetscInt NfAux,
           const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
           const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
           PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g2[])
{
  PetscInt d;
  for (d = 0; d < dim; ++d) g2[d*dim+d] = -1.0; /* \frac{\partial\psi^{u_d}}{\partial x_d} */
}

/* < \nabla v, \nabla u + {\nabla u}^T >
   This just gives \nabla u, give the perdiagonal for the transpose */
void g3_uu(PetscInt dim, PetscInt Nf, PetscInt NfAux,
           const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
           const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
           PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g3[])
{
  const PetscInt Ncomp = dim;
  PetscInt       compI, d;

  for (compI = 0; compI < Ncomp; ++compI) {
    for (d = 0; d < dim; ++d) {
      g3[((compI*Ncomp+compI)*dim+d)*dim+d] = 1.0;
    }
  }
}

/*
  In 3D we use exact solution:

    u = x^2 + y^2
    v = y^2 + z^2
    w = x^2 + y^2 - 2(x+y)z
    p = x + y + z - 3/2
    f_x = f_y = f_z = 3

  so that

    -\Delta u + \nabla p + f = <-4, -4, -4> + <1, 1, 1> + <3, 3, 3> = 0
    \nabla \cdot u           = 2x + 2y - 2(x + y)                   = 0
*/
PetscErrorCode quadratic_u_3d(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nf, PetscScalar *u, void *ctx)
{
  u[0] = x[0]*x[0] + x[1]*x[1];
  u[1] = x[1]*x[1] + x[2]*x[2];
  u[2] = x[0]*x[0] + x[1]*x[1] - 2.0*(x[0] + x[1])*x[2];
  return 0;
}

PetscErrorCode linear_p_3d(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nf, PetscScalar *p, void *ctx)
{
  *p = x[0] + x[1] + x[2] - 1.5;
  return 0;
}

PetscErrorCode ProcessOptions(MPI_Comm comm, AppCtx *options)
{
  const char    *bcTypes[2]  = {"neumann", "dirichlet"};
  const char    *runTypes[2] = {"full", "test"};
  PetscInt       bc, run;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  options->runType      = RUN_FULL;
  options->bcType       = DIRICHLET;
  options->showInitial  = PETSC_FALSE;
  options->showSolution = PETSC_TRUE;
  options->showError    = PETSC_FALSE;

  ierr = PetscOptionsBegin(comm, "", "Stokes Problem Options", "DMPLEX");PetscCall(ierr);
  run  = options->runType;
  PetscCall(PetscOptionsEList("-run_type", "The run type", "ex62.c", runTypes, 2, runTypes[options->runType], &run, NULL));
  options->runType = (RunType) run;
  bc   = options->bcType;
  PetscCall(PetscOptionsEList("-bc_type","Type of boundary condition","ex62.c",bcTypes,2,bcTypes[options->bcType],&bc,NULL));
  options->bcType = (BCType) bc;

  PetscCall(PetscOptionsBool("-show_initial", "Output the initial guess for verification", "ex62.c", options->showInitial, &options->showInitial, NULL));
  PetscCall(PetscOptionsBool("-show_solution", "Output the solution for verification", "ex62.c", options->showSolution, &options->showSolution, NULL));
  PetscCall(PetscOptionsBool("-show_error", "Output the error for verification", "ex62.c", options->showError, &options->showError, NULL));
  ierr = PetscOptionsEnd();PetscCall(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode DMVecViewLocal(DM dm, Vec v, PetscViewer viewer)
{
  Vec            lv;
  PetscInt       p;
  PetscMPIInt    rank, size;

  PetscFunctionBeginUser;
  PetscCallMPI(MPI_Comm_rank(PetscObjectComm((PetscObject)dm), &rank));
  PetscCallMPI(MPI_Comm_size(PetscObjectComm((PetscObject)dm), &size));
  PetscCall(DMGetLocalVector(dm, &lv));
  PetscCall(DMGlobalToLocalBegin(dm, v, INSERT_VALUES, lv));
  PetscCall(DMGlobalToLocalEnd(dm, v, INSERT_VALUES, lv));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Local function\n"));
  for (p = 0; p < size; ++p) {
    if (p == rank) PetscCall(VecView(lv, PETSC_VIEWER_STDOUT_SELF));
    PetscCall(PetscBarrier((PetscObject) dm));
  }
  PetscCall(DMRestoreLocalVector(dm, &lv));
  PetscFunctionReturn(0);
}

PetscErrorCode CreateMesh(MPI_Comm comm, AppCtx *user, DM *dm)
{
  PetscFunctionBeginUser;
  PetscCall(DMCreate(comm, dm));
  PetscCall(DMSetType(*dm, DMPLEX));
  PetscCall(DMSetFromOptions(*dm));
  PetscCall(DMViewFromOptions(*dm, NULL, "-dm_view"));
  PetscFunctionReturn(0);
}

PetscErrorCode SetupProblem(DM dm, AppCtx *user)
{
  PetscDS        prob;

  PetscFunctionBeginUser;
  PetscCall(DMGetDS(dm, &prob));
  PetscCall(PetscDSSetResidual(prob, 0, f0_u, f1_u));
  PetscCall(PetscDSSetResidual(prob, 1, f0_p, f1_p));
  PetscCall(PetscDSSetJacobian(prob, 0, 0, NULL, NULL,  NULL,  g3_uu));
  PetscCall(PetscDSSetJacobian(prob, 0, 1, NULL, NULL,  g2_up, NULL));
  PetscCall(PetscDSSetJacobian(prob, 1, 0, NULL, g1_pu, NULL,  NULL));
  switch (user->dim) {
  case 2:
    user->exactFuncs[0] = quadratic_u_2d;
    user->exactFuncs[1] = linear_p_2d;
    break;
  case 3:
    user->exactFuncs[0] = quadratic_u_3d;
    user->exactFuncs[1] = linear_p_3d;
    break;
  default:
    SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE, "Invalid dimension %d", user->dim);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode SetupDiscretization(DM dm, AppCtx *user)
{
  DM              cdm   = dm;
  const PetscInt  dim   = user->dim;
  const PetscInt  id    = 1;
  PetscFE         fe[2];
  PetscDS         ds;
  DMLabel         label;
  MPI_Comm        comm;

  PetscFunctionBeginUser;
  /* Create finite element */
  PetscCall(PetscObjectGetComm((PetscObject) dm, &comm));
  PetscCall(PetscFECreateDefault(comm, dim, dim, user->simplex, "vel_", PETSC_DEFAULT, &fe[0]));
  PetscCall(PetscObjectSetName((PetscObject) fe[0], "velocity"));
  PetscCall(PetscFECreateDefault(comm, dim, 1, user->simplex, "pres_", PETSC_DEFAULT, &fe[1]));
  PetscCall(PetscFECopyQuadrature(fe[0], fe[1]));
  PetscCall(PetscObjectSetName((PetscObject) fe[1], "pressure"));
  /* Set discretization and boundary conditions for each mesh */
  while (cdm) {
    PetscCall(DMGetDS(cdm, &ds));
    PetscCall(PetscDSSetDiscretization(ds, 0, (PetscObject) fe[0]));
    PetscCall(PetscDSSetDiscretization(ds, 1, (PetscObject) fe[1]));
    PetscCall(SetupProblem(cdm, user));
    if (user->bcType == NEUMANN) PetscCall(DMGetLabel(cdm, "boundary", &label));
    else                         PetscCall(DMGetLabel(cdm, "marker",   &label));
    PetscCall(DMAddBoundary(cdm, user->bcType == DIRICHLET ? DM_BC_ESSENTIAL : DM_BC_NATURAL, "wall", label, 1, &id, 0, 0, NULL, (void (*)()) user->exactFuncs[0], NULL, user, NULL));
    PetscCall(DMGetCoarseDM(cdm, &cdm));
  }
  PetscCall(PetscFEDestroy(&fe[0]));
  PetscCall(PetscFEDestroy(&fe[1]));
  PetscFunctionReturn(0);
}

PetscErrorCode CreatePressureNullSpace(DM dm, AppCtx *user, Vec *v, MatNullSpace *nullSpace)
{
  Vec              vec;
  PetscErrorCode (*funcs[2])(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nf, PetscScalar *u, void* ctx) = {zero_vector, constant_p};

  PetscFunctionBeginUser;
  PetscCall(DMGetGlobalVector(dm, &vec));
  PetscCall(DMProjectFunction(dm, 0.0, funcs, NULL, INSERT_ALL_VALUES, vec));
  PetscCall(VecNormalize(vec, NULL));
  if (user->debug) {
    PetscCall(PetscPrintf(PetscObjectComm((PetscObject)dm), "Pressure Null Space\n"));
    PetscCall(VecView(vec, PETSC_VIEWER_STDOUT_WORLD));
  }
  PetscCall(MatNullSpaceCreate(PetscObjectComm((PetscObject)dm), PETSC_FALSE, 1, &vec, nullSpace));
  if (v) {
    PetscCall(DMCreateGlobalVector(dm, v));
    PetscCall(VecCopy(vec, *v));
  }
  PetscCall(DMRestoreGlobalVector(dm, &vec));
  /* New style for field null spaces */
  {
    PetscObject  pressure;
    MatNullSpace nullSpacePres;

    PetscCall(DMGetField(dm, 1, &pressure));
    PetscCall(MatNullSpaceCreate(PetscObjectComm(pressure), PETSC_TRUE, 0, NULL, &nullSpacePres));
    PetscCall(PetscObjectCompose(pressure, "nullspace", (PetscObject) nullSpacePres));
    PetscCall(MatNullSpaceDestroy(&nullSpacePres));
  }
  PetscFunctionReturn(0);
}

int main(int argc, char **argv)
{
  SNES           snes;                 /* nonlinear solver */
  DM             dm, sdm;              /* problem definition */
  Vec            u,r;                  /* solution, residual vectors */
  Mat            A,J;                  /* Jacobian matrix */
  MatNullSpace   nullSpace;            /* May be necessary for pressure */
  AppCtx         user;                 /* user-defined work context */
  PetscInt       its;                  /* iterations for convergence */
  PetscReal      error         = 0.0;  /* L_2 error in the solution */
  PetscReal      ferrors[2];
  PetscReal     *coords, *viscosities;
  PetscInt      *materials;
  const PetscInt particlesPerCell = 1;
  PetscInt       cStart, cEnd, c, d, bs;

  PetscCall(PetscInitialize(&argc, &argv, NULL,help));
  PetscCall(ProcessOptions(PETSC_COMM_WORLD, &user));
  PetscCall(SNESCreate(PETSC_COMM_WORLD, &snes));
  PetscCall(CreateMesh(PETSC_COMM_WORLD, &user, &dm));
  PetscCall(SNESSetDM(snes, dm));
  PetscCall(DMSetApplicationContext(dm, &user));

  PetscCall(PetscMalloc(2 * sizeof(void (*)(const PetscReal[], PetscScalar *, void *)), &user.exactFuncs));
  PetscCall(SetupDiscretization(dm, &user));
  //PetscCall(DMPlexCreateClosureIndex(dm, NULL));

  /* Add a DMSWARM for particles */
  PetscCall(DMCreate(PETSC_COMM_WORLD, &sdm));
  PetscCall(DMSetType(sdm, DMSWARM));
  PetscCall(DMSetDimension(sdm, user.dim));
  PetscCall(DMSwarmSetCellDM(sdm, dm));

  /* Setup particle information */
  PetscCall(DMSwarmSetType(sdm, DMSWARM_PIC));
  PetscCall(DMSwarmRegisterPetscDatatypeField(sdm, "material", 1, PETSC_INT));
  PetscCall(DMSwarmRegisterPetscDatatypeField(sdm, "viscosity", 1, PETSC_REAL));
  PetscCall(DMSwarmFinalizeFieldRegister(sdm));

  /* Setup number of particles and coordinates */
  PetscCall(DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd));
  PetscCall(DMSwarmSetLocalSizes(sdm, particlesPerCell * (cEnd - cStart), 4));
  PetscCall(DMSwarmGetField(sdm, DMSwarmPICField_coor, &bs, NULL, (void **) &coords));
  PetscCall(DMSwarmGetField(sdm, "material", NULL, NULL, (void **) &materials));
  PetscCall(DMSwarmGetField(sdm, "viscosity", NULL, NULL, (void **) &viscosities));
  for (c = cStart; c < cEnd; ++c) {
    const PetscInt i = (c-cStart)*bs;
    PetscReal      centroid[3];

    PetscCall(DMPlexComputeCellGeometryFVM(dm, c, NULL, centroid, NULL));
    for (d = 0; d < user.dim; ++d) coords[i+d] = centroid[d];
    materials[c-cStart] = c % 4;
    viscosities[c-cStart] = 1.0e20 + 1e18*(cos(2*PETSC_PI*centroid[0])+1.0);
  }
  PetscCall(DMSwarmRestoreField(sdm, DMSwarmPICField_coor, &bs, NULL, (void **) &coords));
  PetscCall(DMSwarmRestoreField(sdm, "material", NULL, NULL, (void **) &materials));
  PetscCall(DMSwarmRestoreField(sdm, "viscosity", NULL, NULL, (void **) &viscosities));

  PetscCall(DMCreateGlobalVector(dm, &u));
  PetscCall(VecDuplicate(u, &r));

  PetscCall(DMSetMatType(dm,MATAIJ));
  PetscCall(DMCreateMatrix(dm, &J));
  A = J;
  PetscCall(CreatePressureNullSpace(dm, &user, NULL, &nullSpace));
  PetscCall(MatSetNullSpace(A, nullSpace));

  PetscCall(DMPlexSetSNESLocalFEM(dm,&user,&user,&user));
  PetscCall(SNESSetJacobian(snes, A, J, NULL, NULL));

  PetscCall(SNESSetFromOptions(snes));

  PetscCall(DMProjectFunction(dm, 0.0, user.exactFuncs, NULL, INSERT_ALL_VALUES, u));
  if (user.showInitial) PetscCall(DMVecViewLocal(dm, u, PETSC_VIEWER_STDOUT_SELF));
  if (user.runType == RUN_FULL) {
    PetscErrorCode (*initialGuess[2])(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nf, PetscScalar *u, void* ctx) = {zero_vector, zero_scalar};

    PetscCall(DMProjectFunction(dm, 0.0, initialGuess, NULL, INSERT_VALUES, u));
    if (user.debug) {
      PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Initial guess\n"));
      PetscCall(VecView(u, PETSC_VIEWER_STDOUT_WORLD));
    }
    PetscCall(SNESSolve(snes, NULL, u));
    PetscCall(SNESGetIterationNumber(snes, &its));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Number of SNES iterations = %D\n", its));
    PetscCall(DMComputeL2Diff(dm, 0.0, user.exactFuncs, NULL, u, &error));
    PetscCall(DMComputeL2FieldDiff(dm, 0.0, user.exactFuncs, NULL, u, ferrors));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "L_2 Error: %.3g [%.3g, %.3g]\n", (double)error, (double)ferrors[0], (double)ferrors[1]));
    if (user.showError) {
      Vec r;
      PetscCall(DMGetGlobalVector(dm, &r));
      PetscCall(DMProjectFunction(dm, 0.0, user.exactFuncs, NULL, INSERT_ALL_VALUES, r));
      PetscCall(VecAXPY(r, -1.0, u));
      PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Solution Error\n"));
      PetscCall(VecView(r, PETSC_VIEWER_STDOUT_WORLD));
      PetscCall(DMRestoreGlobalVector(dm, &r));
    }
    if (user.showSolution) {
      PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Solution\n"));
      PetscCall(VecChop(u, 3.0e-9));
      PetscCall(VecView(u, PETSC_VIEWER_STDOUT_WORLD));
    }
  } else {
    PetscReal res = 0.0;

    /* Check discretization error */
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Initial guess\n"));
    PetscCall(VecView(u, PETSC_VIEWER_STDOUT_WORLD));
    PetscCall(DMComputeL2Diff(dm, 0.0, user.exactFuncs, NULL, u, &error));
    if (error >= 1.0e-11) PetscCall(PetscPrintf(PETSC_COMM_WORLD, "L_2 Error: %g\n", (double)error));
    else                  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "L_2 Error: < 1.0e-11\n"));
    /* Check residual */
    PetscCall(SNESComputeFunction(snes, u, r));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Initial Residual\n"));
    PetscCall(VecChop(r, 1.0e-10));
    PetscCall(VecView(r, PETSC_VIEWER_STDOUT_WORLD));
    PetscCall(VecNorm(r, NORM_2, &res));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "L_2 Residual: %g\n", (double)res));
    /* Check Jacobian */
    {
      Vec          b;
      PetscBool    isNull;

      PetscCall(SNESComputeJacobian(snes, u, A, A));
      PetscCall(MatNullSpaceTest(nullSpace, J, &isNull));
      //PetscCheck(isNull,PETSC_COMM_WORLD, PETSC_ERR_PLIB, "The null space calculated for the system operator is invalid.");
      PetscCall(VecDuplicate(u, &b));
      PetscCall(VecSet(r, 0.0));
      PetscCall(SNESComputeFunction(snes, r, b));
      PetscCall(MatMult(A, u, r));
      PetscCall(VecAXPY(r, 1.0, b));
      PetscCall(VecDestroy(&b));
      PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Au - b = Au + F(0)\n"));
      PetscCall(VecChop(r, 1.0e-10));
      PetscCall(VecView(r, PETSC_VIEWER_STDOUT_WORLD));
      PetscCall(VecNorm(r, NORM_2, &res));
      PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Linear L_2 Residual: %g\n", (double)res));
    }
  }
  PetscCall(VecViewFromOptions(u, NULL, "-sol_vec_view"));

  /* Move particles */
  {
    DM        vdm;
    IS        vis;
    Vec       vel, locvel, pvel;
    PetscReal dt = 0.01;
    PetscInt  vf[1] = {0};
    PetscInt  dim = user.dim, numSteps = 30, tn;

    PetscCall(DMViewFromOptions(sdm, NULL, "-part_dm_view"));
    PetscCall(DMCreateSubDM(dm, 1, vf, &vis, &vdm));
    PetscCall(VecGetSubVector(u, vis, &vel));
    PetscCall(DMGetLocalVector(dm, &locvel));
    PetscCall(DMGlobalToLocalBegin(dm, vel, INSERT_VALUES, locvel));
    PetscCall(DMGlobalToLocalEnd(dm, vel, INSERT_VALUES, locvel));
    PetscCall(VecRestoreSubVector(u, vis, &vel));
    for (tn = 0; tn < numSteps; ++tn) {
      DMInterpolationInfo ictx;
      const PetscScalar  *pv;
      PetscReal          *coords;
      PetscInt            Np, p, d;

      PetscCall(DMSwarmVectorDefineField(sdm, DMSwarmPICField_coor));
      PetscCall(DMCreateGlobalVector(sdm, &pvel));
      PetscCall(DMSwarmGetLocalSize(sdm, &Np));
      PetscCall(PetscSynchronizedPrintf(PETSC_COMM_WORLD, "Timestep: %D Np: %D\n", tn, Np));
      PetscCall(PetscSynchronizedFlush(PETSC_COMM_WORLD, NULL));
      /* Interpolate velocity */
      PetscCall(DMInterpolationCreate(PETSC_COMM_SELF, &ictx));
      PetscCall(DMInterpolationSetDim(ictx, dim));
      PetscCall(DMInterpolationSetDof(ictx, dim));
      PetscCall(DMSwarmGetField(sdm, DMSwarmPICField_coor, NULL, NULL, (void **) &coords));
      PetscCall(DMInterpolationAddPoints(ictx, Np, coords));
      PetscCall(DMSwarmRestoreField(sdm, DMSwarmPICField_coor, NULL, NULL, (void **) &coords));
      PetscCall(DMInterpolationSetUp(ictx, vdm, PETSC_FALSE, PETSC_FALSE));
      PetscCall(DMInterpolationEvaluate(ictx, vdm, locvel, pvel));
      PetscCall(DMInterpolationDestroy(&ictx));
      /* Push particles */
      PetscCall(DMSwarmGetField(sdm, DMSwarmPICField_coor, NULL, NULL, (void **) &coords));
      PetscCall(VecViewFromOptions(pvel, NULL, "-vel_view"));
      PetscCall(VecGetArrayRead(pvel, &pv));
      for (p = 0; p < Np; ++p) {
        for (d = 0; d < dim; ++d) coords[p*dim+d] += dt*pv[p*dim+d];
      }
      PetscCall(DMSwarmRestoreField(sdm, DMSwarmPICField_coor, NULL, NULL, (void **) &coords));
      /* Migrate particles */
      PetscCall(DMSwarmMigrate(sdm, PETSC_TRUE));
      PetscCall(DMViewFromOptions(sdm, NULL, "-part_dm_view"));
      PetscCall(VecDestroy(&pvel));
    }
    PetscCall(DMRestoreLocalVector(dm, &locvel));
    PetscCall(ISDestroy(&vis));
    PetscCall(DMDestroy(&vdm));
  }

  PetscCall(MatNullSpaceDestroy(&nullSpace));
  if (A != J) PetscCall(MatDestroy(&A));
  PetscCall(MatDestroy(&J));
  PetscCall(VecDestroy(&u));
  PetscCall(VecDestroy(&r));
  PetscCall(SNESDestroy(&snes));
  PetscCall(DMDestroy(&sdm));
  PetscCall(DMDestroy(&dm));
  PetscCall(PetscFree(user.exactFuncs));
  PetscCall(PetscFinalize());
  return 0;
}
