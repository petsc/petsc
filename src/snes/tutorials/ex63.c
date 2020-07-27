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

PetscInt spatialDim = 0;

typedef enum {NEUMANN, DIRICHLET} BCType;
typedef enum {RUN_FULL, RUN_TEST} RunType;

typedef struct {
  PetscInt      debug;             /* The debugging level */
  RunType       runType;           /* Whether to run tests, or solve the full problem */
  PetscLogEvent createMeshEvent;
  PetscBool     showInitial, showSolution, showError;
  /* Domain and mesh definition */
  PetscInt      dim;               /* The topological mesh dimension */
  PetscBool     interpolate;       /* Generate intermediate mesh elements */
  PetscBool     simplex;           /* Use simplices or tensor product cells */
  PetscReal     refinementLimit;   /* The largest allowable cell volume */
  PetscBool     testPartition;     /* Use a fixed partitioning for testing */
  /* Problem definition */
  BCType        bcType;
  PetscErrorCode (**exactFuncs)(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nf, PetscScalar *u, void *ctx);
} AppCtx;

PetscErrorCode zero_scalar(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nf, PetscScalar *u, void *ctx)
{
  u[0] = 0.0;
  return 0;
}
PetscErrorCode zero_vector(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nf, PetscScalar *u, void *ctx)
{
  PetscInt d;
  for (d = 0; d < spatialDim; ++d) u[d] = 0.0;
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
  options->debug           = 0;
  options->runType         = RUN_FULL;
  options->dim             = 2;
  options->interpolate     = PETSC_TRUE;
  options->simplex         = PETSC_TRUE;
  options->refinementLimit = 0.0;
  options->testPartition   = PETSC_FALSE;
  options->bcType          = DIRICHLET;
  options->showInitial     = PETSC_FALSE;
  options->showSolution    = PETSC_TRUE;
  options->showError       = PETSC_FALSE;

  ierr = PetscOptionsBegin(comm, "", "Stokes Problem Options", "DMPLEX");CHKERRQ(ierr);
  ierr = PetscOptionsInt("-debug", "The debugging level", "ex62.c", options->debug, &options->debug, NULL);CHKERRQ(ierr);
  run  = options->runType;
  ierr = PetscOptionsEList("-run_type", "The run type", "ex62.c", runTypes, 2, runTypes[options->runType], &run, NULL);CHKERRQ(ierr);

  options->runType = (RunType) run;

  ierr = PetscOptionsInt("-dim", "The topological mesh dimension", "ex62.c", options->dim, &options->dim, NULL);CHKERRQ(ierr);
  spatialDim = options->dim;
  ierr = PetscOptionsBool("-interpolate", "Generate intermediate mesh elements", "ex62.c", options->interpolate, &options->interpolate, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-simplex", "Use simplices or tensor product cells", "ex62.c", options->simplex, &options->simplex, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-refinement_limit", "The largest allowable cell volume", "ex62.c", options->refinementLimit, &options->refinementLimit, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-test_partition", "Use a fixed partition for testing", "ex62.c", options->testPartition, &options->testPartition, NULL);CHKERRQ(ierr);
  bc   = options->bcType;
  ierr = PetscOptionsEList("-bc_type","Type of boundary condition","ex62.c",bcTypes,2,bcTypes[options->bcType],&bc,NULL);CHKERRQ(ierr);

  options->bcType = (BCType) bc;

  ierr = PetscOptionsBool("-show_initial", "Output the initial guess for verification", "ex62.c", options->showInitial, &options->showInitial, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-show_solution", "Output the solution for verification", "ex62.c", options->showSolution, &options->showSolution, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-show_error", "Output the error for verification", "ex62.c", options->showError, &options->showError, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();

  ierr = PetscLogEventRegister("CreateMesh", DM_CLASSID, &options->createMeshEvent);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode DMVecViewLocal(DM dm, Vec v, PetscViewer viewer)
{
  Vec            lv;
  PetscInt       p;
  PetscMPIInt    rank, size;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = MPI_Comm_rank(PetscObjectComm((PetscObject)dm), &rank);CHKERRQ(ierr);
  ierr = MPI_Comm_size(PetscObjectComm((PetscObject)dm), &size);CHKERRQ(ierr);
  ierr = DMGetLocalVector(dm, &lv);CHKERRQ(ierr);
  ierr = DMGlobalToLocalBegin(dm, v, INSERT_VALUES, lv);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(dm, v, INSERT_VALUES, lv);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD, "Local function\n");CHKERRQ(ierr);
  for (p = 0; p < size; ++p) {
    if (p == rank) {ierr = VecView(lv, PETSC_VIEWER_STDOUT_SELF);CHKERRQ(ierr);}
    ierr = PetscBarrier((PetscObject) dm);CHKERRQ(ierr);
  }
  ierr = DMRestoreLocalVector(dm, &lv);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode CreateMesh(MPI_Comm comm, AppCtx *user, DM *dm)
{
  PetscInt       dim             = user->dim;
  PetscBool      interpolate     = user->interpolate;
  PetscReal      refinementLimit = user->refinementLimit;
  const PetscInt cells[3]        = {3, 3, 3};
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = PetscLogEventBegin(user->createMeshEvent,0,0,0,0);CHKERRQ(ierr);
  ierr = DMPlexCreateBoxMesh(comm, dim, user->simplex, user->simplex ? NULL : cells, NULL, NULL, NULL, interpolate, dm);CHKERRQ(ierr);
  {
    DM refinedMesh     = NULL;
    DM distributedMesh = NULL;

    /* Refine mesh using a volume constraint */
    ierr = DMPlexSetRefinementLimit(*dm, refinementLimit);CHKERRQ(ierr);
    if (user->simplex) {ierr = DMRefine(*dm, comm, &refinedMesh);CHKERRQ(ierr);}
    if (refinedMesh) {
      ierr = DMDestroy(dm);CHKERRQ(ierr);
      *dm  = refinedMesh;
    }
    /* Setup test partitioning */
    if (user->testPartition) {
      PetscInt         triSizes_n2[2]       = {4, 4};
      PetscInt         triPoints_n2[8]      = {3, 5, 6, 7, 0, 1, 2, 4};
      PetscInt         triSizes_n3[3]       = {2, 3, 3};
      PetscInt         triPoints_n3[8]      = {3, 5, 1, 6, 7, 0, 2, 4};
      PetscInt         triSizes_n5[5]       = {1, 2, 2, 1, 2};
      PetscInt         triPoints_n5[8]      = {3, 5, 6, 4, 7, 0, 1, 2};
      PetscInt         triSizes_ref_n2[2]   = {8, 8};
      PetscInt         triPoints_ref_n2[16] = {1, 5, 6, 7, 10, 11, 14, 15, 0, 2, 3, 4, 8, 9, 12, 13};
      PetscInt         triSizes_ref_n3[3]   = {5, 6, 5};
      PetscInt         triPoints_ref_n3[16] = {1, 7, 10, 14, 15, 2, 6, 8, 11, 12, 13, 0, 3, 4, 5, 9};
      PetscInt         triSizes_ref_n5[5]   = {3, 4, 3, 3, 3};
      PetscInt         triPoints_ref_n5[16] = {1, 7, 10, 2, 11, 13, 14, 5, 6, 15, 0, 8, 9, 3, 4, 12};
      const PetscInt  *sizes = NULL;
      const PetscInt  *points = NULL;
      PetscPartitioner part;
      PetscInt         cEnd;
      PetscMPIInt      rank, size;

      ierr = MPI_Comm_rank(comm, &rank);CHKERRQ(ierr);
      ierr = MPI_Comm_size(comm, &size);CHKERRQ(ierr);
      ierr = DMPlexGetHeightStratum(*dm, 0, NULL, &cEnd);CHKERRQ(ierr);
      if (!rank) {
        if (dim == 2 && user->simplex && size == 2 && cEnd == 8) {
           sizes = triSizes_n2; points = triPoints_n2;
        } else if (dim == 2 && user->simplex && size == 3 && cEnd == 8) {
          sizes = triSizes_n3; points = triPoints_n3;
        } else if (dim == 2 && user->simplex && size == 5 && cEnd == 8) {
          sizes = triSizes_n5; points = triPoints_n5;
        } else if (dim == 2 && user->simplex && size == 2 && cEnd == 16) {
           sizes = triSizes_ref_n2; points = triPoints_ref_n2;
        } else if (dim == 2 && user->simplex && size == 3 && cEnd == 16) {
          sizes = triSizes_ref_n3; points = triPoints_ref_n3;
        } else if (dim == 2 && user->simplex && size == 5 && cEnd == 16) {
          sizes = triSizes_ref_n5; points = triPoints_ref_n5;
        } else SETERRQ(comm, PETSC_ERR_ARG_WRONG, "No stored partition matching run parameters");
      }
      ierr = DMPlexGetPartitioner(*dm, &part);CHKERRQ(ierr);
      ierr = PetscPartitionerSetType(part, PETSCPARTITIONERSHELL);CHKERRQ(ierr);
      ierr = PetscPartitionerShellSetPartition(part, size, sizes, points);CHKERRQ(ierr);
    }
    /* Distribute mesh over processes */
    ierr = DMPlexDistribute(*dm, 0, NULL, &distributedMesh);CHKERRQ(ierr);
    if (distributedMesh) {
      ierr = DMDestroy(dm);CHKERRQ(ierr);
      *dm  = distributedMesh;
    }
  }
  ierr = DMSetFromOptions(*dm);CHKERRQ(ierr);
  ierr = DMViewFromOptions(*dm, NULL, "-dm_view");CHKERRQ(ierr);
  ierr = PetscLogEventEnd(user->createMeshEvent,0,0,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode SetupProblem(DM dm, AppCtx *user)
{
  PetscDS        prob;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = DMGetDS(dm, &prob);CHKERRQ(ierr);
  ierr = PetscDSSetResidual(prob, 0, f0_u, f1_u);CHKERRQ(ierr);
  ierr = PetscDSSetResidual(prob, 1, f0_p, f1_p);CHKERRQ(ierr);
  ierr = PetscDSSetJacobian(prob, 0, 0, NULL, NULL,  NULL,  g3_uu);CHKERRQ(ierr);
  ierr = PetscDSSetJacobian(prob, 0, 1, NULL, NULL,  g2_up, NULL);CHKERRQ(ierr);
  ierr = PetscDSSetJacobian(prob, 1, 0, NULL, g1_pu, NULL,  NULL);CHKERRQ(ierr);
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
    SETERRQ1(PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE, "Invalid dimension %d", user->dim);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode SetupDiscretization(DM dm, AppCtx *user)
{
  DM              cdm   = dm;
  const PetscInt  dim   = user->dim;
  const PetscInt  id    = 1;
  PetscFE         fe[2];
  PetscDS         prob;
  MPI_Comm        comm;
  PetscErrorCode  ierr;

  PetscFunctionBeginUser;
  /* Create finite element */
  ierr = PetscObjectGetComm((PetscObject) dm, &comm);CHKERRQ(ierr);
  ierr = PetscFECreateDefault(comm, dim, dim, user->simplex, "vel_", PETSC_DEFAULT, &fe[0]);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) fe[0], "velocity");CHKERRQ(ierr);
  ierr = PetscFECreateDefault(comm, dim, 1, user->simplex, "pres_", PETSC_DEFAULT, &fe[1]);CHKERRQ(ierr);
  ierr = PetscFECopyQuadrature(fe[0], fe[1]);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) fe[1], "pressure");CHKERRQ(ierr);
  /* Set discretization and boundary conditions for each mesh */
  while (cdm) {
    ierr = DMGetDS(cdm, &prob);CHKERRQ(ierr);
    ierr = PetscDSSetDiscretization(prob, 0, (PetscObject) fe[0]);CHKERRQ(ierr);
    ierr = PetscDSSetDiscretization(prob, 1, (PetscObject) fe[1]);CHKERRQ(ierr);
    ierr = SetupProblem(cdm, user);CHKERRQ(ierr);
    ierr = DMAddBoundary(cdm, user->bcType == DIRICHLET ? PETSC_TRUE : PETSC_FALSE, "wall", user->bcType == NEUMANN ? "boundary" : "marker", 0, 0, NULL, (void (*)()) user->exactFuncs[0], NULL, 1, &id, user);CHKERRQ(ierr);
    ierr = DMGetCoarseDM(cdm, &cdm);CHKERRQ(ierr);
  }
  ierr = PetscFEDestroy(&fe[0]);CHKERRQ(ierr);
  ierr = PetscFEDestroy(&fe[1]);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode CreatePressureNullSpace(DM dm, AppCtx *user, Vec *v, MatNullSpace *nullSpace)
{
  Vec              vec;
  PetscErrorCode (*funcs[2])(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nf, PetscScalar *u, void* ctx) = {zero_vector, constant_p};
  PetscErrorCode   ierr;

  PetscFunctionBeginUser;
  ierr = DMGetGlobalVector(dm, &vec);CHKERRQ(ierr);
  ierr = DMProjectFunction(dm, 0.0, funcs, NULL, INSERT_ALL_VALUES, vec);CHKERRQ(ierr);
  ierr = VecNormalize(vec, NULL);CHKERRQ(ierr);
  if (user->debug) {
    ierr = PetscPrintf(PetscObjectComm((PetscObject)dm), "Pressure Null Space\n");CHKERRQ(ierr);
    ierr = VecView(vec, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  }
  ierr = MatNullSpaceCreate(PetscObjectComm((PetscObject)dm), PETSC_FALSE, 1, &vec, nullSpace);CHKERRQ(ierr);
  if (v) {
    ierr = DMCreateGlobalVector(dm, v);CHKERRQ(ierr);
    ierr = VecCopy(vec, *v);CHKERRQ(ierr);
  }
  ierr = DMRestoreGlobalVector(dm, &vec);CHKERRQ(ierr);
  /* New style for field null spaces */
  {
    PetscObject  pressure;
    MatNullSpace nullSpacePres;

    ierr = DMGetField(dm, 1, &pressure);CHKERRQ(ierr);
    ierr = MatNullSpaceCreate(PetscObjectComm(pressure), PETSC_TRUE, 0, NULL, &nullSpacePres);CHKERRQ(ierr);
    ierr = PetscObjectCompose(pressure, "nullspace", (PetscObject) nullSpacePres);CHKERRQ(ierr);
    ierr = MatNullSpaceDestroy(&nullSpacePres);CHKERRQ(ierr);
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
  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc, &argv, NULL,help);if (ierr) return ierr;
  ierr = ProcessOptions(PETSC_COMM_WORLD, &user);CHKERRQ(ierr);
  ierr = SNESCreate(PETSC_COMM_WORLD, &snes);CHKERRQ(ierr);
  ierr = CreateMesh(PETSC_COMM_WORLD, &user, &dm);CHKERRQ(ierr);
  ierr = SNESSetDM(snes, dm);CHKERRQ(ierr);
  ierr = DMSetApplicationContext(dm, &user);CHKERRQ(ierr);

  ierr = PetscMalloc(2 * sizeof(void (*)(const PetscReal[], PetscScalar *, void *)), &user.exactFuncs);CHKERRQ(ierr);
  ierr = SetupDiscretization(dm, &user);CHKERRQ(ierr);
  //ierr = DMPlexCreateClosureIndex(dm, NULL);CHKERRQ(ierr);

  /* Add a DMSWARM for particles */
  ierr = DMCreate(PETSC_COMM_WORLD, &sdm);CHKERRQ(ierr);
  ierr = DMSetType(sdm, DMSWARM);CHKERRQ(ierr);
  ierr = DMSetDimension(sdm, user.dim);CHKERRQ(ierr);
  ierr = DMSwarmSetCellDM(sdm, dm);CHKERRQ(ierr);

  /* Setup particle information */
  ierr = DMSwarmSetType(sdm, DMSWARM_PIC);CHKERRQ(ierr);
  ierr = DMSwarmRegisterPetscDatatypeField(sdm, "material", 1, PETSC_INT);CHKERRQ(ierr);
  ierr = DMSwarmRegisterPetscDatatypeField(sdm, "viscosity", 1, PETSC_REAL);CHKERRQ(ierr);
  ierr = DMSwarmFinalizeFieldRegister(sdm);CHKERRQ(ierr);

  /* Setup number of particles and coordinates */
  ierr = DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd);CHKERRQ(ierr);
  ierr = DMSwarmSetLocalSizes(sdm, particlesPerCell * (cEnd - cStart), 4);CHKERRQ(ierr);
  ierr = DMSwarmGetField(sdm, DMSwarmPICField_coor, &bs, NULL, (void **) &coords);CHKERRQ(ierr);
  ierr = DMSwarmGetField(sdm, "material", NULL, NULL, (void **) &materials);CHKERRQ(ierr);
  ierr = DMSwarmGetField(sdm, "viscosity", NULL, NULL, (void **) &viscosities);CHKERRQ(ierr);
  for (c = cStart; c < cEnd; ++c) {
    const PetscInt i = (c-cStart)*bs;
    PetscReal      centroid[3];

    ierr = DMPlexComputeCellGeometryFVM(dm, c, NULL, centroid, NULL);CHKERRQ(ierr);
    for (d = 0; d < user.dim; ++d) coords[i+d] = centroid[d];
    materials[c-cStart] = c % 4;
    viscosities[c-cStart] = 1.0e20 + 1e18*(cos(2*PETSC_PI*centroid[0])+1.0);
  }
  ierr = DMSwarmRestoreField(sdm, DMSwarmPICField_coor, &bs, NULL, (void **) &coords);CHKERRQ(ierr);
  ierr = DMSwarmRestoreField(sdm, "material", NULL, NULL, (void **) &materials);CHKERRQ(ierr);
  ierr = DMSwarmRestoreField(sdm, "viscosity", NULL, NULL, (void **) &viscosities);CHKERRQ(ierr);

  ierr = DMCreateGlobalVector(dm, &u);CHKERRQ(ierr);
  ierr = VecDuplicate(u, &r);CHKERRQ(ierr);

  ierr = DMSetMatType(dm,MATAIJ);CHKERRQ(ierr);
  ierr = DMCreateMatrix(dm, &J);CHKERRQ(ierr);
  A = J;
  ierr = CreatePressureNullSpace(dm, &user, NULL, &nullSpace);CHKERRQ(ierr);
  ierr = MatSetNullSpace(A, nullSpace);CHKERRQ(ierr);

  ierr = DMPlexSetSNESLocalFEM(dm,&user,&user,&user);CHKERRQ(ierr);
  ierr = SNESSetJacobian(snes, A, J, NULL, NULL);CHKERRQ(ierr);

  ierr = SNESSetFromOptions(snes);CHKERRQ(ierr);

  ierr = DMProjectFunction(dm, 0.0, user.exactFuncs, NULL, INSERT_ALL_VALUES, u);CHKERRQ(ierr);
  if (user.showInitial) {ierr = DMVecViewLocal(dm, u, PETSC_VIEWER_STDOUT_SELF);CHKERRQ(ierr);}
  if (user.runType == RUN_FULL) {
    PetscErrorCode (*initialGuess[2])(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nf, PetscScalar *u, void* ctx) = {zero_vector, zero_scalar};

    ierr = DMProjectFunction(dm, 0.0, initialGuess, NULL, INSERT_VALUES, u);CHKERRQ(ierr);
    if (user.debug) {
      ierr = PetscPrintf(PETSC_COMM_WORLD, "Initial guess\n");CHKERRQ(ierr);
      ierr = VecView(u, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
    }
    ierr = SNESSolve(snes, NULL, u);CHKERRQ(ierr);
    ierr = SNESGetIterationNumber(snes, &its);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD, "Number of SNES iterations = %D\n", its);CHKERRQ(ierr);
    ierr = DMComputeL2Diff(dm, 0.0, user.exactFuncs, NULL, u, &error);CHKERRQ(ierr);
    ierr = DMComputeL2FieldDiff(dm, 0.0, user.exactFuncs, NULL, u, ferrors);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD, "L_2 Error: %.3g [%.3g, %.3g]\n", (double)error, (double)ferrors[0], (double)ferrors[1]);CHKERRQ(ierr);
    if (user.showError) {
      Vec r;
      ierr = DMGetGlobalVector(dm, &r);CHKERRQ(ierr);
      ierr = DMProjectFunction(dm, 0.0, user.exactFuncs, NULL, INSERT_ALL_VALUES, r);CHKERRQ(ierr);
      ierr = VecAXPY(r, -1.0, u);CHKERRQ(ierr);
      ierr = PetscPrintf(PETSC_COMM_WORLD, "Solution Error\n");CHKERRQ(ierr);
      ierr = VecView(r, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
      ierr = DMRestoreGlobalVector(dm, &r);CHKERRQ(ierr);
    }
    if (user.showSolution) {
      ierr = PetscPrintf(PETSC_COMM_WORLD, "Solution\n");CHKERRQ(ierr);
      ierr = VecChop(u, 3.0e-9);CHKERRQ(ierr);
      ierr = VecView(u, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
    }
  } else {
    PetscReal res = 0.0;

    /* Check discretization error */
    ierr = PetscPrintf(PETSC_COMM_WORLD, "Initial guess\n");CHKERRQ(ierr);
    ierr = VecView(u, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
    ierr = DMComputeL2Diff(dm, 0.0, user.exactFuncs, NULL, u, &error);CHKERRQ(ierr);
    if (error >= 1.0e-11) {ierr = PetscPrintf(PETSC_COMM_WORLD, "L_2 Error: %g\n", (double)error);CHKERRQ(ierr);}
    else                  {ierr = PetscPrintf(PETSC_COMM_WORLD, "L_2 Error: < 1.0e-11\n");CHKERRQ(ierr);}
    /* Check residual */
    ierr = SNESComputeFunction(snes, u, r);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD, "Initial Residual\n");CHKERRQ(ierr);
    ierr = VecChop(r, 1.0e-10);CHKERRQ(ierr);
    ierr = VecView(r, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
    ierr = VecNorm(r, NORM_2, &res);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD, "L_2 Residual: %g\n", (double)res);CHKERRQ(ierr);
    /* Check Jacobian */
    {
      Vec          b;
      PetscBool    isNull;

      ierr = SNESComputeJacobian(snes, u, A, A);CHKERRQ(ierr);
      ierr = MatNullSpaceTest(nullSpace, J, &isNull);CHKERRQ(ierr);
      //if (!isNull) SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_PLIB, "The null space calculated for the system operator is invalid.");
      ierr = VecDuplicate(u, &b);CHKERRQ(ierr);
      ierr = VecSet(r, 0.0);CHKERRQ(ierr);
      ierr = SNESComputeFunction(snes, r, b);CHKERRQ(ierr);
      ierr = MatMult(A, u, r);CHKERRQ(ierr);
      ierr = VecAXPY(r, 1.0, b);CHKERRQ(ierr);
      ierr = VecDestroy(&b);CHKERRQ(ierr);
      ierr = PetscPrintf(PETSC_COMM_WORLD, "Au - b = Au + F(0)\n");CHKERRQ(ierr);
      ierr = VecChop(r, 1.0e-10);CHKERRQ(ierr);
      ierr = VecView(r, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
      ierr = VecNorm(r, NORM_2, &res);CHKERRQ(ierr);
      ierr = PetscPrintf(PETSC_COMM_WORLD, "Linear L_2 Residual: %g\n", (double)res);CHKERRQ(ierr);
    }
  }
  ierr = VecViewFromOptions(u, NULL, "-sol_vec_view");CHKERRQ(ierr);

  /* Move particles */
  {
    DM        vdm;
    IS        vis;
    Vec       vel, locvel, pvel;
    PetscReal dt = 0.01;
    PetscInt  vf[1] = {0};
    PetscInt  dim = user.dim, numSteps = 30, tn;

    ierr = DMViewFromOptions(sdm, NULL, "-part_dm_view");CHKERRQ(ierr);
    ierr = DMCreateSubDM(dm, 1, vf, &vis, &vdm);CHKERRQ(ierr);
    ierr = VecGetSubVector(u, vis, &vel);CHKERRQ(ierr);
    ierr = DMGetLocalVector(dm, &locvel);CHKERRQ(ierr);
    ierr = DMGlobalToLocalBegin(dm, vel, INSERT_VALUES, locvel);CHKERRQ(ierr);
    ierr = DMGlobalToLocalEnd(dm, vel, INSERT_VALUES, locvel);CHKERRQ(ierr);
    ierr = VecRestoreSubVector(u, vis, &vel);CHKERRQ(ierr);
    for (tn = 0; tn < numSteps; ++tn) {
      DMInterpolationInfo ictx;
      const PetscScalar  *pv;
      PetscReal          *coords;
      PetscInt            Np, p, d;

      ierr = DMSwarmVectorDefineField(sdm, DMSwarmPICField_coor);CHKERRQ(ierr);
      ierr = DMCreateGlobalVector(sdm, &pvel);CHKERRQ(ierr);
      ierr = DMSwarmGetLocalSize(sdm, &Np);CHKERRQ(ierr);
      ierr = PetscSynchronizedPrintf(PETSC_COMM_WORLD, "Timestep: %D Np: %D\n", tn, Np);CHKERRQ(ierr);
      ierr = PetscSynchronizedFlush(PETSC_COMM_WORLD, NULL);CHKERRQ(ierr);
      /* Interpolate velocity */
      ierr = DMInterpolationCreate(PETSC_COMM_SELF, &ictx);CHKERRQ(ierr);
      ierr = DMInterpolationSetDim(ictx, dim);CHKERRQ(ierr);
      ierr = DMInterpolationSetDof(ictx, dim);CHKERRQ(ierr);
      ierr = DMSwarmGetField(sdm, DMSwarmPICField_coor, NULL, NULL, (void **) &coords);CHKERRQ(ierr);
      ierr = DMInterpolationAddPoints(ictx, Np, coords);CHKERRQ(ierr);
      ierr = DMSwarmRestoreField(sdm, DMSwarmPICField_coor, NULL, NULL, (void **) &coords);CHKERRQ(ierr);
      ierr = DMInterpolationSetUp(ictx, vdm, PETSC_FALSE);CHKERRQ(ierr);
      ierr = DMInterpolationEvaluate(ictx, vdm, locvel, pvel);CHKERRQ(ierr);
      ierr = DMInterpolationDestroy(&ictx);CHKERRQ(ierr);
      /* Push particles */
      ierr = DMSwarmGetField(sdm, DMSwarmPICField_coor, NULL, NULL, (void **) &coords);CHKERRQ(ierr);
      ierr = VecViewFromOptions(pvel, NULL, "-vel_view");CHKERRQ(ierr);
      ierr = VecGetArrayRead(pvel, &pv);CHKERRQ(ierr);
      for (p = 0; p < Np; ++p) {
        for (d = 0; d < dim; ++d) coords[p*dim+d] += dt*pv[p*dim+d];
      }
      ierr = DMSwarmRestoreField(sdm, DMSwarmPICField_coor, NULL, NULL, (void **) &coords);CHKERRQ(ierr);
      /* Migrate particles */
      ierr = DMSwarmMigrate(sdm, PETSC_TRUE);CHKERRQ(ierr);
      ierr = DMViewFromOptions(sdm, NULL, "-part_dm_view");CHKERRQ(ierr);
      ierr = VecDestroy(&pvel);CHKERRQ(ierr);
    }
    ierr = DMRestoreLocalVector(dm, &locvel);CHKERRQ(ierr);
    ierr = ISDestroy(&vis);CHKERRQ(ierr);
    ierr = DMDestroy(&vdm);CHKERRQ(ierr);
  }

  ierr = MatNullSpaceDestroy(&nullSpace);CHKERRQ(ierr);
  if (A != J) {ierr = MatDestroy(&A);CHKERRQ(ierr);}
  ierr = MatDestroy(&J);CHKERRQ(ierr);
  ierr = VecDestroy(&u);CHKERRQ(ierr);
  ierr = VecDestroy(&r);CHKERRQ(ierr);
  ierr = SNESDestroy(&snes);CHKERRQ(ierr);
  ierr = DMDestroy(&sdm);CHKERRQ(ierr);
  ierr = DMDestroy(&dm);CHKERRQ(ierr);
  ierr = PetscFree(user.exactFuncs);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}
