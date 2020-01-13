static char help[] = "Stokes Problem in 2d and 3d with simplicial finite elements.\n\
We solve the Stokes problem in a rectangular\n\
domain, using a parallel unstructured mesh (DMPLEX) to discretize it.\n\n\n";

/*
The isoviscous Stokes problem, which we discretize using the finite
element method on an unstructured mesh. The weak form equations are

  < \nabla v, \nabla u + {\nabla u}^T > - < \nabla\cdot v, p > + < v, f > = 0
  < q, \nabla\cdot u >                                                    = 0

Viewing:

To produce nice output, use

  -dm_refine 3 -show_error -dm_view hdf5:sol1.h5 -error_vec_view hdf5:sol1.h5::append -sol_vec_view hdf5:sol1.h5::append -exact_vec_view hdf5:sol1.h5::append

You can get a LaTeX view of the mesh, with point numbering using

  -dm_view :mesh.tex:ascii_latex -dm_plex_view_scale 8.0

The data layout can be viewed using

  -dm_petscsection_view

Lots of information about the FEM assembly can be printed using

  -dm_plex_print_fem 2

Field Data:

  DMPLEX data is organized by point, and the closure operation just stacks up the
data from each sieve point in the closure. Thus, for a P_2-P_1 Stokes element, we
have

  cl{e} = {f e_0 e_1 e_2 v_0 v_1 v_2}
  x     = [u_{e_0} v_{e_0} u_{e_1} v_{e_1} u_{e_2} v_{e_2} u_{v_0} v_{v_0} p_{v_0} u_{v_1} v_{v_1} p_{v_1} u_{v_2} v_{v_2} p_{v_2}]

The problem here is that we would like to loop over each field separately for
integration. Therefore, the closure visitor in DMPlexVecGetClosure() reorders
the data so that each field is contiguous

  x'    = [u_{e_0} v_{e_0} u_{e_1} v_{e_1} u_{e_2} v_{e_2} u_{v_0} v_{v_0} u_{v_1} v_{v_1} u_{v_2} v_{v_2} p_{v_0} p_{v_1} p_{v_2}]

Likewise, DMPlexVecSetClosure() takes data partitioned by field, and correctly
puts it into the Sieve ordering.

TODO:
 - Update FETI test output
 - Reorder mesh
 - Check the q1-p0 Vanka domains are correct (I think its correct)
   - Check scaling of iterates, right now it is bad
 - Check the q2-q1 domains since convergence is bad
   - Ask Patrick about domains
 - Plot residual by fields after each smoother iterate
 - Get Diskin checks going
*/

#include <petscdmplex.h>
#include <petscsnes.h>
#include <petscds.h>

PetscInt spatialDim = 0;

typedef enum {NEUMANN, DIRICHLET, NUM_BC_TYPES} BCType;
const char *bcTypes[NUM_BC_TYPES+1]  = {"neumann", "dirichlet", "unknown"};
typedef enum {RUN_FULL, RUN_TEST, NUM_RUN_TYPES} RunType;
const char *runTypes[NUM_RUN_TYPES+1] = {"full", "test", "unknown"};
typedef enum {SOL_QUADRATIC, SOL_CUBIC, SOL_TRIG, NUM_SOL_TYPES} SolType;
const char *solTypes[NUM_SOL_TYPES+1] = {"quadratic", "cubic", "trig", "unknown"};

typedef struct {
  PetscInt      debug;             /* The debugging level */
  RunType       runType;           /* Whether to run tests, or solve the full problem */
  PetscLogEvent createMeshEvent;
  PetscBool     showInitial, showError;
  /* Domain and mesh definition */
  PetscInt      dim;               /* The topological mesh dimension */
  PetscBool     interpolate;       /* Generate intermediate mesh elements */
  PetscBool     simplex;           /* Use simplices or tensor product cells */
  PetscInt      cells[3];          /* The initial domain division */
  PetscReal     refinementLimit;   /* The largest allowable cell volume */
  PetscBool     testPartition;     /* Use a fixed partitioning for testing */
  /* Problem definition */
  BCType        bcType;
  SolType       solType;
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

/*
  In 2D we use exact solution:

    u = x^3 + y^3
    v = 2 x^3 - 3 x^2 y
    p = 3/2 x^2 + 3/2 y^2 - 1
    f_x = 6 (x + y)
    f_y = 12 x - 3 y

  so that

    -\Delta u + \nabla p + f = <-6 x - 6 y, -12 x + 6 y> + <3 x, 3 y> + <6 (x + y), 12 x - 6 y> = 0
    \nabla \cdot u           = 3 x^2 - 3 x^2 = 0
*/
PetscErrorCode cubic_u_2d(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nf, PetscScalar *u, void *ctx)
{
  u[0] =     x[0]*x[0]*x[0] +     x[1]*x[1]*x[1];
  u[1] = 2.0*x[0]*x[0]*x[0] - 3.0*x[0]*x[0]*x[1];
  return 0;
}

PetscErrorCode quadratic_p_2d(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nf, PetscScalar *p, void *ctx)
{
  *p = 3.0*x[0]*x[0]/2.0 + 3.0*x[1]*x[1]/2.0 - 1.0;
  return 0;
}

/*
  In 2D we use exact solution:

    u =  sin(n pi x) + y^2
    v = -sin(n pi y)
    p = 3/2 x^2 + 3/2 y^2 - 1
    f_x = 4 - 3x - n^2 pi^2 sin (n pi x)
    f_y =   - 3y + n^2 pi^2 sin(n pi y)

  so that

    -\Delta u + \nabla p + f = <n^2 pi^2 sin (n pi x) - 4, -n^2 pi^2 sin(n pi y)> + <3 x, 3 y> + <4 - 3x - n^2 pi^2 sin (n pi x), -3y + n^2 pi^2 sin(n pi y)> = 0
    \nabla \cdot u           = n pi cos(n pi x) - n pi cos(n pi y) = 0
*/
PetscErrorCode trig_u_2d(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nf, PetscScalar *u, void *ctx)
{
  const PetscReal n = 1.0;

  u[0] =  PetscSinReal(n*PETSC_PI*x[0]) + x[1]*x[1];
  u[1] = -PetscSinReal(n*PETSC_PI*x[1]);
  return 0;
}

void f0_quadratic_u(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                    const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                    const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                    PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f0[])
{
  PetscInt c;
  for (c = 0; c < dim; ++c) f0[c] = 3.0;
}

void f0_cubic_u(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f0[])
{
  f0[0] =  3.0*x[0] + 6.0*x[1];
  f0[1] = 12.0*x[0] - 9.0*x[1];
}

void f0_trig_u(PetscInt dim, PetscInt Nf, PetscInt NfAux,
               const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
               const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
               PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f0[])
{
  const PetscReal n = 1.0;

  f0[0] = 4.0 - 3.0*x[0] - PetscSqr(n*PETSC_PI)*PetscSinReal(n*PETSC_PI*x[0]);
  f0[1] =      -3.0*x[1] + PetscSqr(n*PETSC_PI)*PetscSinReal(n*PETSC_PI*x[1]);
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

void pressure(PetscInt dim, PetscInt Nf, PetscInt NfAux,
              const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
              const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
              PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar p[])
{
  p[0] = u[uOff[1]];
}

PetscErrorCode ProcessOptions(MPI_Comm comm, AppCtx *options)
{
  PetscInt       bc, run, sol, n;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  options->debug           = 0;
  options->runType         = RUN_FULL;
  options->dim             = 2;
  options->interpolate     = PETSC_FALSE;
  options->simplex         = PETSC_TRUE;
  options->cells[0]        = 3;
  options->cells[1]        = 3;
  options->cells[2]        = 3;
  options->refinementLimit = 0.0;
  options->testPartition   = PETSC_FALSE;
  options->bcType          = DIRICHLET;
  options->solType         = SOL_QUADRATIC;
  options->showInitial     = PETSC_FALSE;
  options->showError       = PETSC_FALSE;

  ierr = PetscOptionsBegin(comm, "", "Stokes Problem Options", "DMPLEX");CHKERRQ(ierr);
  ierr = PetscOptionsInt("-debug", "The debugging level", "ex62.c", options->debug, &options->debug, NULL);CHKERRQ(ierr);
  run  = options->runType;
  ierr = PetscOptionsEList("-run_type", "The run type", "ex62.c", runTypes, NUM_RUN_TYPES, runTypes[options->runType], &run, NULL);CHKERRQ(ierr);
  options->runType = (RunType) run;
  ierr = PetscOptionsInt("-dim", "The topological mesh dimension", "ex62.c", options->dim, &options->dim, NULL);CHKERRQ(ierr);
  spatialDim = options->dim;
  ierr = PetscOptionsBool("-interpolate", "Generate intermediate mesh elements", "ex62.c", options->interpolate, &options->interpolate, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-simplex", "Use simplices or tensor product cells", "ex62.c", options->simplex, &options->simplex, NULL);CHKERRQ(ierr);
  if (options->simplex) {
    options->cells[0] = 4 - options->dim;
    options->cells[1] = 4 - options->dim;
    options->cells[2] = 4 - options->dim;
  }
  n = 3;
  ierr = PetscOptionsIntArray("-cells", "The initial mesh division", "ex62.c", options->cells, &n, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-refinement_limit", "The largest allowable cell volume", "ex62.c", options->refinementLimit, &options->refinementLimit, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-test_partition", "Use a fixed partition for testing", "ex62.c", options->testPartition, &options->testPartition, NULL);CHKERRQ(ierr);
  bc   = options->bcType;
  ierr = PetscOptionsEList("-bc_type","Type of boundary condition","ex62.c", bcTypes, NUM_BC_TYPES, bcTypes[options->bcType], &bc, NULL);CHKERRQ(ierr);
  options->bcType = (BCType) bc;
  sol  = options->solType;
  ierr = PetscOptionsEList("-sol_type", "The solution type", "ex62.c", solTypes, NUM_SOL_TYPES, solTypes[options->solType], &sol, NULL);CHKERRQ(ierr);
  options->solType = (SolType) sol;
  ierr = PetscOptionsBool("-show_initial", "Output the initial guess for verification", "ex62.c", options->showInitial, &options->showInitial, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-show_error", "Output the error for verification", "ex62.c", options->showError, &options->showError, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();

  ierr = PetscLogEventRegister("CreateMesh", DM_CLASSID, &options->createMeshEvent);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode DMVecViewLocal(DM dm, Vec v)
{
  Vec            lv;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = DMGetLocalVector(dm, &lv);CHKERRQ(ierr);
  ierr = DMGlobalToLocalBegin(dm, v, INSERT_VALUES, lv);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(dm, v, INSERT_VALUES, lv);CHKERRQ(ierr);
  ierr = DMPrintLocalVec(dm, "Local function", 0.0, lv);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(dm, &lv);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode CreateMesh(MPI_Comm comm, AppCtx *user, DM *dm)
{
  PetscInt       dim             = user->dim;
  PetscBool      interpolate     = user->interpolate;
  PetscReal      refinementLimit = user->refinementLimit;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = PetscLogEventBegin(user->createMeshEvent,0,0,0,0);CHKERRQ(ierr);
  ierr = DMPlexCreateBoxMesh(comm, dim, user->simplex, user->cells, NULL, NULL, NULL, interpolate, dm);CHKERRQ(ierr);
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
      PetscInt         triSizes_n2[2]         = {4, 4};
      PetscInt         triPoints_n2[8]        = {3, 5, 6, 7, 0, 1, 2, 4};
      PetscInt         triSizes_n3[3]         = {2, 3, 3};
      PetscInt         triPoints_n3[8]        = {3, 5, 1, 6, 7, 0, 2, 4};
      PetscInt         triSizes_n5[5]         = {1, 2, 2, 1, 2};
      PetscInt         triPoints_n5[8]        = {3, 5, 6, 4, 7, 0, 1, 2};
      PetscInt         triSizes_ref_n2[2]     = {8, 8};
      PetscInt         triPoints_ref_n2[16]   = {1, 5, 6, 7, 10, 11, 14, 15, 0, 2, 3, 4, 8, 9, 12, 13};
      PetscInt         triSizes_ref_n3[3]     = {5, 6, 5};
      PetscInt         triPoints_ref_n3[16]   = {1, 7, 10, 14, 15, 2, 6, 8, 11, 12, 13, 0, 3, 4, 5, 9};
      PetscInt         triSizes_ref_n5[5]     = {3, 4, 3, 3, 3};
      PetscInt         triPoints_ref_n5[16]   = {1, 7, 10, 2, 11, 13, 14, 5, 6, 15, 0, 8, 9, 3, 4, 12};
      PetscInt         triSizes_ref_n5_d3[5]  = {1, 1, 1, 1, 2};
      PetscInt         triPoints_ref_n5_d3[6] = {0, 1, 2, 3, 4, 5};
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
        } else if (dim == 3 && user->simplex && size == 5 && cEnd == 6) {
          sizes = triSizes_ref_n5_d3; points = triPoints_ref_n5_d3;
        } else SETERRQ(comm, PETSC_ERR_ARG_WRONG, "No stored partition matching run parameters");
      }
      ierr = DMPlexGetPartitioner(*dm, &part);CHKERRQ(ierr);
      ierr = PetscPartitionerSetType(part, PETSCPARTITIONERSHELL);CHKERRQ(ierr);
      ierr = PetscPartitionerShellSetPartition(part, size, sizes, points);CHKERRQ(ierr);
    } else {
      PetscPartitioner part;

      ierr = DMPlexGetPartitioner(*dm, &part);CHKERRQ(ierr);
      ierr = PetscPartitionerSetFromOptions(part);CHKERRQ(ierr);
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
  const PetscInt id = 1;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = DMGetDS(dm, &prob);CHKERRQ(ierr);
  switch (user->solType) {
  case SOL_QUADRATIC:
    switch (user->dim) {
    case 2:
      ierr = PetscDSSetResidual(prob, 0, f0_quadratic_u, f1_u);CHKERRQ(ierr);
      user->exactFuncs[0] = quadratic_u_2d;
      user->exactFuncs[1] = linear_p_2d;
      break;
    case 3:
      ierr = PetscDSSetResidual(prob, 0, f0_quadratic_u, f1_u);CHKERRQ(ierr);
      user->exactFuncs[0] = quadratic_u_3d;
      user->exactFuncs[1] = linear_p_3d;
      break;
    default: SETERRQ1(PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE, "Unsupported dimension %d for quadratic solution", user->dim);
    }
    break;
  case SOL_CUBIC:
    switch (user->dim) {
    case 2:
      ierr = PetscDSSetResidual(prob, 0, f0_cubic_u, f1_u);CHKERRQ(ierr);
      user->exactFuncs[0] = cubic_u_2d;
      user->exactFuncs[1] = quadratic_p_2d;
      break;
    default: SETERRQ1(PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE, "Unsupported dimension %d for quadratic solution", user->dim);
    }
    break;
  case SOL_TRIG:
    switch (user->dim) {
    case 2:
      ierr = PetscDSSetResidual(prob, 0, f0_trig_u, f1_u);CHKERRQ(ierr);
      user->exactFuncs[0] = trig_u_2d;
      user->exactFuncs[1] = quadratic_p_2d;
      break;
    default: SETERRQ1(PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE, "Unsupported dimension %d for trigonometric solution", user->dim);
    }
    break;
  default: SETERRQ2(PetscObjectComm((PetscObject) prob), PETSC_ERR_ARG_WRONG, "Unsupported solution type: %s (%D)", solTypes[PetscMin(user->solType, NUM_SOL_TYPES)], user->solType);
  }
  ierr = PetscDSSetResidual(prob, 1, f0_p, f1_p);CHKERRQ(ierr);
  ierr = PetscDSSetJacobian(prob, 0, 0, NULL, NULL,  NULL,  g3_uu);CHKERRQ(ierr);
  ierr = PetscDSSetJacobian(prob, 0, 1, NULL, NULL,  g2_up, NULL);CHKERRQ(ierr);
  ierr = PetscDSSetJacobian(prob, 1, 0, NULL, g1_pu, NULL,  NULL);CHKERRQ(ierr);

  ierr = PetscDSAddBoundary(prob, user->bcType == DIRICHLET ? DM_BC_ESSENTIAL : DM_BC_NATURAL, "wall", user->bcType == NEUMANN ? "boundary" : "marker", 0, 0, NULL, (void (*)(void)) user->exactFuncs[0], 1, &id, user);CHKERRQ(ierr);
  ierr = PetscDSSetExactSolution(prob, 0, user->exactFuncs[0], user);CHKERRQ(ierr);
  ierr = PetscDSSetExactSolution(prob, 1, user->exactFuncs[1], user);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode SetupDiscretization(DM dm, AppCtx *user)
{
  DM              cdm   = dm;
  const PetscInt  dim   = user->dim;
  PetscFE         fe[2];
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
  ierr = DMSetField(dm, 0, NULL, (PetscObject) fe[0]);CHKERRQ(ierr);
  ierr = DMSetField(dm, 1, NULL, (PetscObject) fe[1]);CHKERRQ(ierr);
  ierr = DMCreateDS(dm);CHKERRQ(ierr);
  ierr = SetupProblem(dm, user);CHKERRQ(ierr);
  while (cdm) {
    ierr = DMCopyDisc(dm, cdm);CHKERRQ(ierr);
    ierr = DMGetCoarseDM(cdm, &cdm);CHKERRQ(ierr);
  }
  ierr = PetscFEDestroy(&fe[0]);CHKERRQ(ierr);
  ierr = PetscFEDestroy(&fe[1]);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode CreatePressureNullSpace(DM dm, PetscInt dummy, MatNullSpace *nullspace)
{
  Vec              vec;
  PetscErrorCode (*funcs[2])(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nf, PetscScalar *u, void* ctx) = {zero_vector, constant_p};
  PetscErrorCode   ierr;

  PetscFunctionBeginUser;
  ierr = DMCreateGlobalVector(dm, &vec);CHKERRQ(ierr);
  ierr = DMProjectFunction(dm, 0.0, funcs, NULL, INSERT_ALL_VALUES, vec);CHKERRQ(ierr);
  ierr = VecNormalize(vec, NULL);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) vec, "Pressure Null Space");CHKERRQ(ierr);
  ierr = VecViewFromOptions(vec, NULL, "-pressure_nullspace_view");CHKERRQ(ierr);
  ierr = MatNullSpaceCreate(PetscObjectComm((PetscObject)dm), PETSC_FALSE, 1, &vec, nullspace);CHKERRQ(ierr);
  ierr = VecDestroy(&vec);CHKERRQ(ierr);
  /* New style for field null spaces */
  {
    PetscObject  pressure;
    MatNullSpace nullspacePres;

    ierr = DMGetField(dm, 1, NULL, &pressure);CHKERRQ(ierr);
    ierr = MatNullSpaceCreate(PetscObjectComm(pressure), PETSC_TRUE, 0, NULL, &nullspacePres);CHKERRQ(ierr);
    ierr = PetscObjectCompose(pressure, "nullspace", (PetscObject) nullspacePres);CHKERRQ(ierr);
    ierr = MatNullSpaceDestroy(&nullspacePres);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/* Add a vector in the nullspace to make the continuum integral 0.

   If int(u) = a and int(n) = b, then int(u - a/b n) = a - a/b b = 0
*/
static PetscErrorCode CorrectDiscretePressure(DM dm, MatNullSpace nullspace, Vec u, AppCtx *user)
{
  PetscDS        prob;
  const Vec     *nullvecs;
  PetscScalar    pintd, intc[2], intn[2];
  MPI_Comm       comm;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = PetscObjectGetComm((PetscObject) dm, &comm);CHKERRQ(ierr);
  ierr = DMGetDS(dm, &prob);CHKERRQ(ierr);
  ierr = PetscDSSetObjective(prob, 1, pressure);CHKERRQ(ierr);
  ierr = MatNullSpaceGetVecs(nullspace, NULL, NULL, &nullvecs);CHKERRQ(ierr);
  ierr = VecDot(nullvecs[0], u, &pintd);CHKERRQ(ierr);
  if (PetscAbsScalar(pintd) > 1.0e-10) SETERRQ1(comm, PETSC_ERR_ARG_WRONG, "Discrete integral of pressure: %g\n", (double) PetscRealPart(pintd));
  ierr = DMPlexComputeIntegralFEM(dm, nullvecs[0], intn, user);CHKERRQ(ierr);
  ierr = DMPlexComputeIntegralFEM(dm, u, intc, user);CHKERRQ(ierr);
  ierr = VecAXPY(u, -intc[1]/intn[1], nullvecs[0]);CHKERRQ(ierr);
  ierr = DMPlexComputeIntegralFEM(dm, u, intc, user);CHKERRQ(ierr);
  if (PetscAbsScalar(intc[1]) > 1.0e-10) SETERRQ1(comm, PETSC_ERR_ARG_WRONG, "Continuum integral of pressure after correction: %g\n", (double) PetscRealPart(intc[1]));
  PetscFunctionReturn(0);
}

static PetscErrorCode SNESConvergenceCorrectPressure(SNES snes, PetscInt it, PetscReal xnorm, PetscReal gnorm, PetscReal f, SNESConvergedReason *reason, void *user)
{
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = SNESConvergedDefault(snes, it, xnorm, gnorm, f, reason, user);CHKERRQ(ierr);
  if (*reason > 0) {
    DM           dm;
    Mat          J;
    Vec          u;
    MatNullSpace nullspace;

    ierr = SNESGetDM(snes, &dm);CHKERRQ(ierr);
    ierr = SNESGetSolution(snes, &u);CHKERRQ(ierr);
    ierr = SNESGetJacobian(snes, &J, NULL, NULL, NULL);CHKERRQ(ierr);
    ierr = MatGetNullSpace(J, &nullspace);CHKERRQ(ierr);
    ierr = CorrectDiscretePressure(dm, nullspace, u, (AppCtx *) user);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

int main(int argc, char **argv)
{
  SNES           snes;                 /* nonlinear solver */
  DM             dm;                   /* problem definition */
  Vec            u, r;                 /* solution and residual */
  AppCtx         user;                 /* user-defined work context */
  PetscReal      error         = 0.0;  /* L_2 error in the solution */
  PetscReal      ferrors[2];
  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc, &argv, NULL,help);if (ierr) return ierr;
  ierr = ProcessOptions(PETSC_COMM_WORLD, &user);CHKERRQ(ierr);
  ierr = SNESCreate(PETSC_COMM_WORLD, &snes);CHKERRQ(ierr);
  ierr = CreateMesh(PETSC_COMM_WORLD, &user, &dm);CHKERRQ(ierr);
  ierr = SNESSetDM(snes, dm);CHKERRQ(ierr);
  ierr = DMSetApplicationContext(dm, &user);CHKERRQ(ierr);

  ierr = PetscMalloc(2 * sizeof(void (*)(const PetscReal[], PetscScalar *, void *)), &user.exactFuncs);CHKERRQ(ierr);
  ierr = SetupDiscretization(dm, &user);CHKERRQ(ierr);
  ierr = DMPlexCreateClosureIndex(dm, NULL);CHKERRQ(ierr);

  ierr = DMCreateGlobalVector(dm, &u);CHKERRQ(ierr);
  ierr = VecDuplicate(u, &r);CHKERRQ(ierr);

  ierr = DMSetNullSpaceConstructor(dm, 2, CreatePressureNullSpace);CHKERRQ(ierr);
  ierr = SNESSetConvergenceTest(snes, SNESConvergenceCorrectPressure, &user, NULL);CHKERRQ(ierr);

  ierr = DMPlexSetSNESLocalFEM(dm,&user,&user,&user);CHKERRQ(ierr);

  ierr = SNESSetFromOptions(snes);CHKERRQ(ierr);

  ierr = DMProjectFunction(dm, 0.0, user.exactFuncs, NULL, INSERT_ALL_VALUES, u);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) u, "Exact Solution");CHKERRQ(ierr);
  ierr = VecViewFromOptions(u, NULL, "-exact_vec_view");CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) u, "Solution");CHKERRQ(ierr);
  if (user.showInitial) {ierr = DMVecViewLocal(dm, u);CHKERRQ(ierr);}
  ierr = PetscObjectSetName((PetscObject) u, "Solution");CHKERRQ(ierr);
  if (user.runType == RUN_FULL) {
    PetscErrorCode (*initialGuess[2])(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nf, PetscScalar *u, void* ctx) = {zero_vector, zero_scalar};

    ierr = DMProjectFunction(dm, 0.0, initialGuess, NULL, INSERT_VALUES, u);CHKERRQ(ierr);
    if (user.debug) {
      ierr = PetscPrintf(PETSC_COMM_WORLD, "Initial guess\n");CHKERRQ(ierr);
      ierr = VecView(u, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
    }
    ierr = SNESSolve(snes, NULL, u);CHKERRQ(ierr);
    ierr = DMComputeL2Diff(dm, 0.0, user.exactFuncs, NULL, u, &error);CHKERRQ(ierr);
    ierr = DMComputeL2FieldDiff(dm, 0.0, user.exactFuncs, NULL, u, ferrors);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD, "L_2 Error: %g [%g, %g]\n", (double)error, (double)ferrors[0], (double)ferrors[1]);CHKERRQ(ierr);
    if (user.showError) {
      Vec r;

      ierr = DMGetGlobalVector(dm, &r);CHKERRQ(ierr);
      ierr = DMProjectFunction(dm, 0.0, user.exactFuncs, NULL, INSERT_ALL_VALUES, r);CHKERRQ(ierr);
      ierr = VecAXPY(r, -1.0, u);CHKERRQ(ierr);
      ierr = PetscObjectSetName((PetscObject) r, "Solution Error");CHKERRQ(ierr);
      ierr = VecViewFromOptions(r, NULL, "-error_vec_view");CHKERRQ(ierr);
      ierr = DMRestoreGlobalVector(dm, &r);CHKERRQ(ierr);
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
      Mat          J, M;
      MatNullSpace nullspace;
      Vec          b;
      PetscBool    isNull;

      ierr = SNESSetUpMatrices(snes);CHKERRQ(ierr);
      ierr = SNESGetJacobian(snes, &J, &M, NULL, NULL);CHKERRQ(ierr);
      ierr = SNESComputeJacobian(snes, u, J, M);CHKERRQ(ierr);
      ierr = MatGetNullSpace(J, &nullspace);CHKERRQ(ierr);
      ierr = MatNullSpaceTest(nullspace, J, &isNull);CHKERRQ(ierr);
      ierr = VecDuplicate(u, &b);CHKERRQ(ierr);
      ierr = VecSet(r, 0.0);CHKERRQ(ierr);
      ierr = SNESComputeFunction(snes, r, b);CHKERRQ(ierr);
      ierr = MatMult(J, u, r);CHKERRQ(ierr);
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

  ierr = VecDestroy(&u);CHKERRQ(ierr);
  ierr = VecDestroy(&r);CHKERRQ(ierr);
  ierr = SNESDestroy(&snes);CHKERRQ(ierr);
  ierr = DMDestroy(&dm);CHKERRQ(ierr);
  ierr = PetscFree(user.exactFuncs);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

  # 2D serial P1 tests 0-3
  test:
    suffix: 0
    requires: triangle
    args: -run_type test -refinement_limit 0.0    -bc_type dirichlet -interpolate 0 -vel_petscspace_degree 1 -pres_petscspace_degree 1 -show_initial -dm_plex_print_fem 1
  test:
    suffix: 1
    requires: triangle
    args: -run_type test -refinement_limit 0.0    -bc_type dirichlet -interpolate 1 -vel_petscspace_degree 1 -pres_petscspace_degree 1 -show_initial -dm_plex_print_fem 1
  test:
    suffix: 2
    requires: triangle
    args: -run_type test -refinement_limit 0.0625 -bc_type dirichlet -interpolate 0 -vel_petscspace_degree 1 -pres_petscspace_degree 1 -show_initial -dm_plex_print_fem 1
  test:
    suffix: 3
    requires: triangle
    args: -run_type test -refinement_limit 0.0625 -bc_type dirichlet -interpolate 1 -vel_petscspace_degree 1 -pres_petscspace_degree 1 -show_initial -dm_plex_print_fem 1
  # 2D serial P2 tests 4-5
  test:
    suffix: 4
    requires: triangle
    args: -run_type test -refinement_limit 0.0    -bc_type dirichlet -interpolate 1 -vel_petscspace_degree 2 -pres_petscspace_degree 1 -show_initial -dm_plex_print_fem 1
  test:
    suffix: 5
    requires: triangle
    args: -run_type test -refinement_limit 0.0625 -bc_type dirichlet -interpolate 1 -vel_petscspace_degree 2 -pres_petscspace_degree 1 -show_initial -dm_plex_print_fem 1
  # 2D serial P3 tests
  test:
    suffix: 2d_p3_0
    requires: triangle
    args: -run_type test -bc_type dirichlet -interpolate 1 -vel_petscspace_degree 3 -pres_petscspace_degree 2
  test:
    suffix: 2d_p3_1
    requires: triangle !single
    args: -run_type full -bc_type dirichlet -interpolate 1 -vel_petscspace_degree 3 -pres_petscspace_degree 2
  # Parallel tests 6-17
  test:
    suffix: 6
    requires: triangle
    nsize: 2
    args: -run_type test -refinement_limit 0.0    -test_partition -bc_type dirichlet -interpolate 0 -vel_petscspace_degree 1 -pres_petscspace_degree 1 -dm_plex_print_fem 1
  test:
    suffix: 7
    requires: triangle
    nsize: 3
    args: -run_type test -refinement_limit 0.0    -test_partition -bc_type dirichlet -interpolate 0 -vel_petscspace_degree 1 -pres_petscspace_degree 1 -dm_plex_print_fem 1
  test:
    suffix: 8
    requires: triangle
    nsize: 5
    args: -run_type test -refinement_limit 0.0    -test_partition -bc_type dirichlet -interpolate 0 -vel_petscspace_degree 1 -pres_petscspace_degree 1 -dm_plex_print_fem 1
  test:
    suffix: 9
    requires: triangle
    nsize: 2
    args: -run_type test -refinement_limit 0.0    -test_partition -bc_type dirichlet -interpolate 1 -vel_petscspace_degree 1 -pres_petscspace_degree 1 -dm_plex_print_fem 1
  test:
    suffix: 10
    requires: triangle
    nsize: 3
    args: -run_type test -refinement_limit 0.0    -test_partition -bc_type dirichlet -interpolate 1 -vel_petscspace_degree 1 -pres_petscspace_degree 1 -dm_plex_print_fem 1
  test:
    suffix: 11
    requires: triangle
    nsize: 5
    args: -run_type test -refinement_limit 0.0    -test_partition -bc_type dirichlet -interpolate 1 -vel_petscspace_degree 1 -pres_petscspace_degree 1 -dm_plex_print_fem 1
  test:
    suffix: 12
    requires: triangle
    nsize: 2
    args: -run_type test -refinement_limit 0.0625 -test_partition -bc_type dirichlet -interpolate 0 -vel_petscspace_degree 1 -pres_petscspace_degree 1 -dm_plex_print_fem 1
  test:
    suffix: 13
    requires: triangle
    nsize: 3
    args: -run_type test -refinement_limit 0.0625 -test_partition -bc_type dirichlet -interpolate 0 -vel_petscspace_degree 1 -pres_petscspace_degree 1 -dm_plex_print_fem 1
  test:
    suffix: 14
    requires: triangle
    nsize: 5
    args: -run_type test -refinement_limit 0.0625 -test_partition -bc_type dirichlet -interpolate 0 -vel_petscspace_degree 1 -pres_petscspace_degree 1 -dm_plex_print_fem 1
  test:
    suffix: 15
    requires: triangle
    nsize: 2
    args: -run_type test -refinement_limit 0.0625 -test_partition -bc_type dirichlet -interpolate 1 -vel_petscspace_degree 1 -pres_petscspace_degree 1 -dm_plex_print_fem 1
  test:
    suffix: 16
    requires: triangle
    nsize: 3
    args: -run_type test -refinement_limit 0.0625 -test_partition -bc_type dirichlet -interpolate 1 -vel_petscspace_degree 1 -pres_petscspace_degree 1 -dm_plex_print_fem 1
  test:
    suffix: 17
    requires: triangle
    nsize: 5
    args: -run_type test -refinement_limit 0.0625 -test_partition -bc_type dirichlet -interpolate 1 -vel_petscspace_degree 1 -pres_petscspace_degree 1 -dm_plex_print_fem 1
  # 3D serial P1 tests 43-46
  test:
    suffix: 43
    requires: ctetgen
    args: -run_type test -dim 3 -refinement_limit 0.0    -bc_type dirichlet -interpolate 0 -vel_petscspace_degree 1 -pres_petscspace_degree 1 -show_initial -dm_plex_print_fem 1
  test:
    suffix: 44
    requires: ctetgen
    args: -run_type test -dim 3 -refinement_limit 0.0    -bc_type dirichlet -interpolate 1 -vel_petscspace_degree 1 -pres_petscspace_degree 1 -show_initial -dm_plex_print_fem 1
  test:
    suffix: 45
    requires: ctetgen
    args: -run_type test -dim 3 -refinement_limit 0.0125 -bc_type dirichlet -interpolate 0 -vel_petscspace_degree 1 -pres_petscspace_degree 1 -show_initial -dm_plex_print_fem 1
  test:
    suffix: 46
    requires: ctetgen
    args: -run_type test -dim 3 -refinement_limit 0.0125 -bc_type dirichlet -interpolate 1 -vel_petscspace_degree 1 -pres_petscspace_degree 1 -show_initial -dm_plex_print_fem 1
  # Full solutions 18-29
  test:
    suffix: 18
    requires: triangle !single
    filter:  sed -e "s/total number of linear solver iterations=11/total number of linear solver iterations=12/g"
    args: -run_type full -refinement_limit 0.0625 -bc_type dirichlet -interpolate 0 -vel_petscspace_degree 1 -pres_petscspace_degree 1 -pc_type jacobi -ksp_rtol 1.0e-9 -snes_error_if_not_converged -ksp_error_if_not_converged -snes_view
  test:
    suffix: 19
    requires: triangle !single
    nsize: 2
    filter:  sed -e "s/total number of linear solver iterations=11/total number of linear solver iterations=12/g"
    args: -run_type full -petscpartitioner_type simple -refinement_limit 0.0625 -bc_type dirichlet -interpolate 0 -vel_petscspace_degree 1 -pres_petscspace_degree 1 -pc_type jacobi -ksp_rtol 1.0e-9 -snes_error_if_not_converged -ksp_error_if_not_converged -snes_view
  test:
    suffix: 20
    requires: triangle !single
    filter:  sed -e "s/total number of linear solver iterations=11/total number of linear solver iterations=12/g"
    nsize: 3
    args: -run_type full -petscpartitioner_type simple -refinement_limit 0.0625 -bc_type dirichlet -interpolate 0 -vel_petscspace_degree 1 -pres_petscspace_degree 1 -pc_type jacobi -ksp_rtol 1.0e-9 -snes_error_if_not_converged -ksp_error_if_not_converged -snes_view
  test:
    suffix: 20_parmetis
    requires: parmetis triangle !single
    filter:  sed -e "s/total number of linear solver iterations=11/total number of linear solver iterations=12/g"
    nsize: 3
    args: -run_type full -petscpartitioner_type parmetis -refinement_limit 0.0625 -bc_type dirichlet -interpolate 0 -vel_petscspace_degree 1 -pres_petscspace_degree 1 -pc_type jacobi -ksp_rtol 1.0e-9 -snes_error_if_not_converged -ksp_error_if_not_converged -snes_view
  test:
    suffix: 21
    requires: triangle !single
    filter:  sed -e "s/total number of linear solver iterations=11/total number of linear solver iterations=12/g"
    nsize: 5
    args: -run_type full -petscpartitioner_type simple -refinement_limit 0.0625 -bc_type dirichlet -interpolate 0 -vel_petscspace_degree 1 -pres_petscspace_degree 1 -pc_type jacobi -ksp_rtol 1.0e-9 -snes_error_if_not_converged -ksp_error_if_not_converged -snes_view
  test:
    suffix: 22
    requires: triangle !single
    filter:  sed -e "s/total number of linear solver iterations=11/total number of linear solver iterations=12/g"
    args: -run_type full -refinement_limit 0.0625 -bc_type dirichlet -interpolate 1 -vel_petscspace_degree 1 -pres_petscspace_degree 1 -pc_type jacobi -ksp_rtol 1.0e-9 -snes_error_if_not_converged -ksp_error_if_not_converged -snes_view
  test:
    suffix: 23
    requires: triangle !single
    nsize: 2
    filter:  sed -e "s/total number of linear solver iterations=11/total number of linear solver iterations=12/g"
    args: -run_type full -petscpartitioner_type simple -refinement_limit 0.0625 -bc_type dirichlet -interpolate 1 -vel_petscspace_degree 1 -pres_petscspace_degree 1 -pc_type jacobi -ksp_rtol 1.0e-9 -snes_error_if_not_converged -ksp_error_if_not_converged -snes_view
  test:
    suffix: 24
    requires: triangle !single
    nsize: 3
    filter:  sed -e "s/total number of linear solver iterations=11/total number of linear solver iterations=12/g"
    args: -run_type full -petscpartitioner_type simple -refinement_limit 0.0625 -bc_type dirichlet -interpolate 1 -vel_petscspace_degree 1 -pres_petscspace_degree 1 -pc_type jacobi -ksp_rtol 1.0e-9 -snes_error_if_not_converged -ksp_error_if_not_converged -snes_view
  test:
    suffix: 25
    requires: triangle !single
    filter:  sed -e "s/total number of linear solver iterations=11/total number of linear solver iterations=12/g"
    nsize: 5
    args: -run_type full -petscpartitioner_type simple -refinement_limit 0.0625 -bc_type dirichlet -interpolate 1 -vel_petscspace_degree 1 -pres_petscspace_degree 1 -pc_type jacobi -ksp_rtol 1.0e-9 -snes_error_if_not_converged -ksp_error_if_not_converged -snes_view
  test:
    suffix: 26
    requires: triangle !single
    args: -run_type full -refinement_limit 0.0625 -bc_type dirichlet -interpolate 1 -vel_petscspace_degree 2 -pres_petscspace_degree 1 -pc_type jacobi -ksp_rtol 1.0e-9 -snes_error_if_not_converged -ksp_error_if_not_converged -snes_view
  test:
    suffix: 27
    requires: triangle !single
    nsize: 2
    args: -run_type full -petscpartitioner_type simple -refinement_limit 0.0625 -bc_type dirichlet -interpolate 1 -vel_petscspace_degree 2 -pres_petscspace_degree 1 -pc_type jacobi -ksp_rtol 1.0e-9 -snes_error_if_not_converged -ksp_error_if_not_converged -snes_view
  test:
    suffix: 28
    requires: triangle !single
    nsize: 3
    args: -run_type full -petscpartitioner_type simple -refinement_limit 0.0625 -bc_type dirichlet -interpolate 1 -vel_petscspace_degree 2 -pres_petscspace_degree 1 -pc_type jacobi -ksp_rtol 1.0e-9 -snes_error_if_not_converged -ksp_error_if_not_converged -snes_view
  test:
    suffix: 29
    requires: triangle !single
    nsize: 5
    args: -run_type full -petscpartitioner_type simple -refinement_limit 0.0625 -bc_type dirichlet -interpolate 1 -vel_petscspace_degree 2 -pres_petscspace_degree 1 -pc_type jacobi -ksp_rtol 1.0e-9 -snes_error_if_not_converged -ksp_error_if_not_converged -snes_view
  # Full solutions with quads
  #   FULL Schur with LU/Jacobi
  test:
    suffix: quad_q2q1_full
    requires: !single
    args: -run_type full -simplex 0 -refinement_limit 0.00625 -bc_type dirichlet -interpolate 1 -vel_petscspace_degree 2 -pres_petscspace_degree 1 -ksp_type fgmres -ksp_gmres_restart 10 -ksp_rtol 1.0e-9 -pc_type fieldsplit -pc_fieldsplit_type schur -pc_fieldsplit_schur_factorization_type full -fieldsplit_pressure_ksp_rtol 1e-10 -fieldsplit_velocity_ksp_type gmres -fieldsplit_velocity_pc_type lu -fieldsplit_pressure_pc_type jacobi -snes_error_if_not_converged -ksp_error_if_not_converged -snes_view
  test:
    suffix: quad_q2p1_full
    requires: !single
    args: -run_type full -simplex 0 -refinement_limit 0.00625 -bc_type dirichlet -interpolate 1 -vel_petscspace_degree 2 -pres_petscspace_degree 1 -pres_petscspace_poly_tensor 0 -pres_petscdualspace_lagrange_continuity 0 -ksp_type fgmres -ksp_gmres_restart 10 -ksp_rtol 1.0e-9 -pc_type fieldsplit -pc_fieldsplit_type schur -pc_fieldsplit_schur_factorization_type full -fieldsplit_pressure_ksp_rtol 1e-10 -fieldsplit_velocity_ksp_type gmres -fieldsplit_velocity_pc_type lu -fieldsplit_pressure_pc_type jacobi -snes_error_if_not_converged -ksp_error_if_not_converged -snes_view
  # Stokes preconditioners 30-36
  #   Jacobi
  test:
    suffix: 30
    requires: triangle !single
    filter:  sed -e "s/total number of linear solver iterations=756/total number of linear solver iterations=757/g" -e "s/total number of linear solver iterations=758/total number of linear solver iterations=757/g"
    args: -run_type full -refinement_limit 0.00625 -bc_type dirichlet -interpolate 1 -vel_petscspace_degree 2 -pres_petscspace_degree 1 -ksp_gmres_restart 100 -pc_type jacobi -ksp_rtol 1.0e-9 -snes_error_if_not_converged -ksp_error_if_not_converged -snes_view
  #  Block diagonal \begin{pmatrix} A & 0 \\ 0 & I \end{pmatrix}
  test:
    suffix: 31
    requires: triangle !single
    args: -run_type full -refinement_limit 0.00625 -bc_type dirichlet -interpolate 1 -vel_petscspace_degree 2 -pres_petscspace_degree 1 -ksp_type fgmres -ksp_gmres_restart 100 -ksp_rtol 1.0e-4 -pc_type fieldsplit -pc_fieldsplit_type additive -fieldsplit_velocity_pc_type lu -fieldsplit_pressure_pc_type jacobi -snes_error_if_not_converged -ksp_error_if_not_converged -snes_view
  #  Block triangular \begin{pmatrix} A & B \\ 0 & I \end{pmatrix}
  test:
    suffix: 32
    requires: triangle !single
    args: -run_type full -refinement_limit 0.00625 -bc_type dirichlet -interpolate 1 -vel_petscspace_degree 2 -pres_petscspace_degree 1 -ksp_type fgmres -ksp_gmres_restart 100 -ksp_rtol 1.0e-9 -pc_type fieldsplit -pc_fieldsplit_type multiplicative -fieldsplit_velocity_pc_type lu -fieldsplit_pressure_pc_type jacobi -snes_error_if_not_converged -ksp_error_if_not_converged -snes_view
  #  Diagonal Schur complement \begin{pmatrix} A & 0 \\ 0 & S \end{pmatrix}
  test:
    suffix: 33
    requires: triangle !single
    args: -run_type full -refinement_limit 0.00625 -bc_type dirichlet -interpolate 1 -vel_petscspace_degree 2 -pres_petscspace_degree 1 -ksp_type fgmres -ksp_gmres_restart 100 -ksp_rtol 1.0e-9 -pc_type fieldsplit -pc_fieldsplit_type schur -pc_fieldsplit_schur_factorization_type diag -fieldsplit_pressure_ksp_rtol 1e-10 -fieldsplit_velocity_ksp_type gmres -fieldsplit_velocity_pc_type lu -fieldsplit_pressure_pc_type jacobi -snes_error_if_not_converged -ksp_error_if_not_converged -snes_view
  #  Upper triangular Schur complement \begin{pmatrix} A & B \\ 0 & S \end{pmatrix}
  test:
    suffix: 34
    requires: triangle !single
    args: -run_type full -refinement_limit 0.00625 -bc_type dirichlet -interpolate 1 -vel_petscspace_degree 2 -pres_petscspace_degree 1 -ksp_type fgmres -ksp_gmres_restart 100 -ksp_rtol 1.0e-9 -pc_type fieldsplit -pc_fieldsplit_type schur -pc_fieldsplit_schur_factorization_type upper -fieldsplit_pressure_ksp_rtol 1e-10 -fieldsplit_velocity_ksp_type gmres -fieldsplit_velocity_pc_type lu -fieldsplit_pressure_pc_type jacobi -snes_error_if_not_converged -ksp_error_if_not_converged -snes_view
  #  Lower triangular Schur complement \begin{pmatrix} A & B \\ 0 & S \end{pmatrix}
  test:
    suffix: 35
    requires: triangle !single
    args: -run_type full -refinement_limit 0.00625 -bc_type dirichlet -interpolate 1 -vel_petscspace_degree 2 -pres_petscspace_degree 1 -ksp_type fgmres -ksp_gmres_restart 100 -ksp_rtol 1.0e-9 -pc_type fieldsplit -pc_fieldsplit_type schur -pc_fieldsplit_schur_factorization_type lower -fieldsplit_pressure_ksp_rtol 1e-10 -fieldsplit_velocity_ksp_type gmres -fieldsplit_velocity_pc_type lu -fieldsplit_pressure_pc_type jacobi -snes_error_if_not_converged -ksp_error_if_not_converged -snes_view
  #  Full Schur complement \begin{pmatrix} I & 0 \\ B^T A^{-1} & I \end{pmatrix} \begin{pmatrix} A & 0 \\ 0 & S \end{pmatrix} \begin{pmatrix} I & A^{-1} B \\ 0 & I \end{pmatrix}
  test:
    suffix: 36
    requires: triangle !single
    args: -run_type full -refinement_limit 0.00625 -bc_type dirichlet -interpolate 1 -vel_petscspace_degree 2 -pres_petscspace_degree 1 -ksp_type fgmres -ksp_gmres_restart 100 -ksp_rtol 1.0e-9 -pc_type fieldsplit -pc_fieldsplit_type schur -pc_fieldsplit_schur_factorization_type full -fieldsplit_pressure_ksp_rtol 1e-10 -fieldsplit_velocity_ksp_type gmres -fieldsplit_velocity_pc_type lu -fieldsplit_pressure_pc_type jacobi -snes_error_if_not_converged -ksp_error_if_not_converged -snes_view
  #  SIMPLE \begin{pmatrix} I & 0 \\ B^T A^{-1} & I \end{pmatrix} \begin{pmatrix} A & 0 \\ 0 & B^T diag(A)^{-1} B \end{pmatrix} \begin{pmatrix} I & diag(A)^{-1} B \\ 0 & I \end{pmatrix}
  test:
    suffix: pc_simple
    requires: triangle !single
    args: -run_type full -refinement_limit 0.00625 -bc_type dirichlet -interpolate 1 -vel_petscspace_degree 2 -pres_petscspace_degree 1 -ksp_type fgmres -ksp_gmres_restart 100 -ksp_rtol 1.0e-9 -pc_type fieldsplit -pc_fieldsplit_type schur -pc_fieldsplit_schur_factorization_type full -fieldsplit_pressure_ksp_rtol 1e-10 -fieldsplit_velocity_ksp_type gmres -fieldsplit_velocity_pc_type lu -fieldsplit_pressure_pc_type jacobi -fieldsplit_pressure_inner_ksp_type preonly -fieldsplit_pressure_inner_pc_type jacobi -fieldsplit_pressure_upper_ksp_type preonly -fieldsplit_pressure_upper_pc_type jacobi -snes_error_if_not_converged -ksp_error_if_not_converged -snes_view
  #  SIMPLEC \begin{pmatrix} I & 0 \\ B^T A^{-1} & I \end{pmatrix} \begin{pmatrix} A & 0 \\ 0 & B^T rowsum(A)^{-1} B \end{pmatrix} \begin{pmatrix} I & rowsum(A)^{-1} B \\ 0 & I \end{pmatrix}
  test:
    suffix: pc_simplec
    requires: triangle
    args: -run_type full -dm_refine 3 -bc_type dirichlet -interpolate 1 -vel_petscspace_degree 2 -pres_petscspace_degree 1 -ksp_type fgmres -ksp_max_it 5 -ksp_gmres_restart 100 -ksp_rtol 1.0e-9 -pc_type fieldsplit -pc_fieldsplit_type schur -pc_fieldsplit_schur_factorization_type full -fieldsplit_pressure_ksp_rtol 1e-10 -fieldsplit_pressure_ksp_max_it 10 -fieldsplit_velocity_ksp_type gmres -fieldsplit_velocity_pc_type lu -fieldsplit_pressure_pc_type jacobi -fieldsplit_pressure_inner_ksp_type preonly -fieldsplit_pressure_inner_pc_type jacobi -fieldsplit_pressure_inner_pc_jacobi_type rowsum -fieldsplit_pressure_upper_ksp_type preonly -fieldsplit_pressure_upper_pc_type jacobi -fieldsplit_pressure_upper_pc_jacobi_type rowsum -snes_converged_reason -ksp_converged_reason -snes_view
  # FETI-DP solvers
  test:
    suffix: fetidp_2d_tri
    requires: triangle mumps
    filter: grep -v "variant HERMITIAN"
    nsize: 5
    args: -run_type full -dm_refine 2 -bc_type dirichlet -interpolate 1 -vel_petscspace_degree 2 -pres_petscspace_degree 1 -snes_view -snes_error_if_not_converged -dm_mat_type is -ksp_type fetidp -ksp_rtol 1.0e-8 -ksp_fetidp_saddlepoint -fetidp_ksp_type cg -fetidp_fieldsplit_p_ksp_max_it 1 -fetidp_fieldsplit_p_ksp_type richardson -fetidp_fieldsplit_p_ksp_richardson_scale 200 -fetidp_fieldsplit_p_pc_type none -ksp_fetidp_saddlepoint_flip 1 -fetidp_bddc_pc_bddc_dirichlet_pc_factor_mat_solver_type mumps -fetidp_bddc_pc_bddc_neumann_pc_factor_mat_solver_type mumps -petscpartitioner_type simple -fetidp_fieldsplit_lag_ksp_type preonly
  test:
    suffix: fetidp_3d_tet
    requires: ctetgen suitesparse mumps
    filter: grep -v "variant HERMITIAN" | sed -e "s/linear solver iterations=10[0-9]/linear solver iterations=100/g" | sed -e "s/linear solver iterations=9[0-9]/linear solver iterations=100/g"
    nsize: 5
    args: -run_type full -dm_refine 2 -bc_type dirichlet -interpolate 1 -vel_petscspace_degree 2 -pres_petscspace_degree 1 -snes_view -snes_error_if_not_converged -dm_mat_type is -ksp_type fetidp -ksp_rtol 1.0e-8 -ksp_fetidp_saddlepoint -fetidp_ksp_type cg -fetidp_fieldsplit_p_ksp_max_it 1 -fetidp_fieldsplit_p_ksp_type richardson -fetidp_fieldsplit_p_ksp_richardson_scale 1000 -fetidp_fieldsplit_p_pc_type none -ksp_fetidp_saddlepoint_flip 1 -fetidp_bddc_pc_bddc_use_deluxe_scaling -fetidp_bddc_pc_bddc_benign_trick -fetidp_bddc_pc_bddc_deluxe_singlemat -dim 3 -fetidp_pc_discrete_harmonic -fetidp_harmonic_pc_factor_mat_solver_type cholmod -fetidp_harmonic_pc_type cholesky -fetidp_bddelta_pc_factor_mat_solver_type umfpack -fetidp_fieldsplit_lag_ksp_type preonly -test_partition

  test:
    suffix: fetidp_2d_quad
    requires: mumps double
    filter: grep -v "variant HERMITIAN"
    nsize: 5
    args: -run_type full -dm_refine 2 -bc_type dirichlet -interpolate 1 -vel_petscspace_degree 2 -pres_petscspace_degree 1 -snes_view -snes_error_if_not_converged -dm_mat_type is -ksp_type fetidp -ksp_rtol 1.0e-8 -ksp_fetidp_saddlepoint -fetidp_ksp_type cg -fetidp_fieldsplit_p_ksp_max_it 1 -fetidp_fieldsplit_p_ksp_type richardson -fetidp_fieldsplit_p_ksp_richardson_scale 200 -fetidp_fieldsplit_p_pc_type none -ksp_fetidp_saddlepoint_flip 1 -fetidp_bddc_pc_bddc_dirichlet_pc_factor_mat_solver_type mumps -fetidp_bddc_pc_bddc_neumann_pc_factor_mat_solver_type mumps -simplex 0 -petscpartitioner_type simple -fetidp_fieldsplit_lag_ksp_type preonly
  test:
    suffix: fetidp_3d_hex
    requires: suitesparse
    filter: grep -v "variant HERMITIAN" | sed -e "s/linear solver iterations=7[0-9]/linear solver iterations=71/g"
    nsize: 5
    args: -run_type full -dm_refine 1 -bc_type dirichlet -interpolate 1 -vel_petscspace_degree 2 -pres_petscspace_degree 1 -snes_view -snes_error_if_not_converged -dm_mat_type is -ksp_type fetidp -ksp_rtol 1.0e-8 -ksp_fetidp_saddlepoint -fetidp_ksp_type cg -fetidp_fieldsplit_p_ksp_max_it 1 -fetidp_fieldsplit_p_ksp_type richardson -fetidp_fieldsplit_p_ksp_richardson_scale 2000 -fetidp_fieldsplit_p_pc_type none -ksp_fetidp_saddlepoint_flip 1 -dim 3 -simplex 0 -fetidp_pc_discrete_harmonic -fetidp_harmonic_pc_factor_mat_solver_type cholmod -fetidp_harmonic_pc_type cholesky -petscpartitioner_type simple -fetidp_fieldsplit_lag_ksp_type preonly -fetidp_bddc_pc_bddc_dirichlet_pc_factor_mat_solver_type umfpack -fetidp_bddc_pc_bddc_neumann_pc_factor_mat_solver_type umfpack
  # Convergence
  test:
    suffix: 2d_quad_q1_p0_conv
    requires: !single
    args: -run_type full -bc_type dirichlet -simplex 0 -interpolate 1 -dm_refine 0 -vel_petscspace_degree 1 -pres_petscspace_degree 0 \
      -snes_convergence_estimate -convest_num_refine 3 -snes_error_if_not_converged \
      -ksp_type fgmres -ksp_gmres_restart 10 -ksp_rtol 1.0e-9 -ksp_error_if_not_converged \
      -pc_type fieldsplit -pc_fieldsplit_type schur -pc_fieldsplit_schur_factorization_type full \
        -fieldsplit_velocity_pc_type lu \
        -fieldsplit_pressure_ksp_rtol 1e-10 -fieldsplit_pressure_pc_type jacobi
  test:
    suffix: 2d_tri_p2_p1_conv
    requires: triangle !single
    args: -run_type full -sol_type cubic -bc_type dirichlet -interpolate 1 -dm_refine 0 \
      -vel_petscspace_degree 2 -pres_petscspace_degree 1 \
      -snes_convergence_estimate -convest_num_refine 3 -snes_error_if_not_converged \
      -ksp_type fgmres -ksp_gmres_restart 10 -ksp_rtol 1.0e-9 -ksp_error_if_not_converged \
      -pc_type fieldsplit -pc_fieldsplit_type schur -pc_fieldsplit_schur_factorization_type full \
        -fieldsplit_velocity_pc_type lu \
        -fieldsplit_pressure_ksp_rtol 1e-10 -fieldsplit_pressure_pc_type jacobi
  test:
    suffix: 2d_quad_q2_q1_conv
    requires: !single
    args: -run_type full -sol_type cubic -bc_type dirichlet -simplex 0 -interpolate 1 -dm_refine 0 \
      -vel_petscspace_degree 2 -pres_petscspace_degree 1 \
      -snes_convergence_estimate -convest_num_refine 3 -snes_error_if_not_converged \
      -ksp_type fgmres -ksp_gmres_restart 10 -ksp_rtol 1.0e-9 -ksp_error_if_not_converged \
      -pc_type fieldsplit -pc_fieldsplit_type schur -pc_fieldsplit_schur_factorization_type full \
        -fieldsplit_velocity_pc_type lu \
        -fieldsplit_pressure_ksp_rtol 1e-10 -fieldsplit_pressure_pc_type jacobi
  test:
    suffix: 2d_quad_q2_p1_conv
    requires: !single
    args: -run_type full -sol_type cubic -bc_type dirichlet -simplex 0 -interpolate 1 -dm_refine 0 \
      -vel_petscspace_degree 2 -pres_petscspace_degree 1 -pres_petscspace_poly_tensor 0 -pres_petscdualspace_lagrange_continuity 0 \
      -snes_convergence_estimate -convest_num_refine 3 -snes_error_if_not_converged \
      -ksp_type fgmres -ksp_gmres_restart 10 -ksp_rtol 1.0e-9 -ksp_error_if_not_converged \
      -pc_type fieldsplit -pc_fieldsplit_type schur -pc_fieldsplit_schur_factorization_type full \
        -fieldsplit_velocity_pc_type lu \
        -fieldsplit_pressure_ksp_rtol 1e-10 -fieldsplit_pressure_pc_type jacobi
  # GMG solver
  test:
    suffix: 2d_tri_p2_p1_gmg_vcycle
    requires: triangle
    args: -run_type full -sol_type cubic -bc_type dirichlet -interpolate 1 -cells 2,2 -dm_refine_hierarchy 1 \
      -vel_petscspace_degree 2 -pres_petscspace_degree 1 \
      -snes_convergence_estimate -convest_num_refine 1 -snes_error_if_not_converged \
      -ksp_type fgmres -ksp_gmres_restart 10 -ksp_rtol 1.0e-9 -ksp_error_if_not_converged \
      -pc_type fieldsplit -pc_fieldsplit_type schur -pc_fieldsplit_schur_factorization_type full \
        -fieldsplit_velocity_pc_type mg \
        -fieldsplit_pressure_ksp_rtol 1e-10 -fieldsplit_pressure_pc_type jacobi
  # Vanka solver
  test:
    suffix: 2d_quad_q1_p0_vanka_add
    requires: double !complex
    filter: sed -e "s/linear solver iterations=[0-9][0-9]*""/linear solver iterations=49/g" -e "s/Linear solve converged due to CONVERGED_RTOL iterations [0-9][0-9]*""/Linear solve converged due to CONVERGED_RTOL iterations 49/g"
    args: -run_type full -bc_type dirichlet -simplex 0 -dm_refine 1 -interpolate 1 -vel_petscspace_degree 1 -pres_petscspace_degree 0 -petscds_jac_pre 0 \
      -snes_rtol 1.0e-4 -snes_error_if_not_converged -snes_view -snes_monitor -snes_converged_reason \
      -ksp_type gmres -ksp_rtol 1.0e-5 -ksp_error_if_not_converged -ksp_converged_reason \
      -pc_type patch -pc_patch_partition_of_unity 0 -pc_patch_construct_codim 0 -pc_patch_construct_type vanka \
        -sub_ksp_type preonly -sub_pc_type lu
  test:
    suffix: 2d_quad_q1_p0_vanka_add_unity
    requires: double !complex
    filter: sed -e "s/linear solver iterations=[0-9][0-9]*""/linear solver iterations=45/g" -e "s/Linear solve converged due to CONVERGED_RTOL iterations [0-9][0-9]*""/Linear solve converged due to CONVERGED_RTOL iterations 45/g"
    args: -run_type full -bc_type dirichlet -simplex 0 -dm_refine 1 -interpolate 1 -vel_petscspace_degree 1 -pres_petscspace_degree 0 -petscds_jac_pre 0 \
      -snes_rtol 1.0e-4 -snes_error_if_not_converged -snes_view -snes_monitor -snes_converged_reason \
      -ksp_type gmres -ksp_rtol 1.0e-5 -ksp_error_if_not_converged -ksp_converged_reason \
      -pc_type patch -pc_patch_partition_of_unity 1 -pc_patch_construct_codim 0 -pc_patch_construct_type vanka \
        -sub_ksp_type preonly -sub_pc_type lu
  test:
    suffix: 2d_quad_q2_q1_vanka_add
    requires: double !complex
    filter: sed -e "s/linear solver iterations=[0-9][0-9][0-9]*""/linear solver iterations=489/g"
    args: -run_type full -bc_type dirichlet -simplex 0 -dm_refine 0 -interpolate 1 -vel_petscspace_degree 2 -pres_petscspace_degree 1 -petscds_jac_pre 0 \
      -snes_rtol 1.0e-4 -snes_error_if_not_converged -snes_view -snes_monitor -snes_converged_reason \
      -ksp_type gmres -ksp_rtol 1.0e-5 -ksp_error_if_not_converged \
      -pc_type patch -pc_patch_partition_of_unity 0 -pc_patch_construct_dim 0 -pc_patch_construct_type vanka \
        -sub_ksp_type preonly -sub_pc_type lu
  test:
    suffix: 2d_quad_q2_q1_vanka_add_unity
    requires: double !complex
    filter: sed -e "s/linear solver iterations=[0-9][0-9][0-9]*""/linear solver iterations=795/g"
    args: -run_type full -bc_type dirichlet -simplex 0 -dm_refine 0 -interpolate 1 -vel_petscspace_degree 2 -pres_petscspace_degree 1 -petscds_jac_pre 0 \
      -snes_rtol 1.0e-4 -snes_error_if_not_converged -snes_view -snes_monitor -snes_converged_reason \
      -ksp_type gmres -ksp_rtol 1.0e-5 -ksp_error_if_not_converged \
      -pc_type patch -pc_patch_partition_of_unity 1 -pc_patch_construct_dim 0 -pc_patch_construct_type vanka \
        -sub_ksp_type preonly -sub_pc_type lu
  # Vanka smoother
  test:
    suffix: 2d_quad_q1_p0_gmg_vanka_add
    requires: double !complex long_runtime
    args: -run_type full -bc_type dirichlet -simplex 0 -dm_refine_hierarchy 3 -interpolate 1 -vel_petscspace_degree 1 -pres_petscspace_degree 0 -petscds_jac_pre 0 \
      -snes_rtol 1.0e-4 -snes_error_if_not_converged -snes_view -snes_monitor -snes_converged_reason \
      -ksp_type gmres -ksp_rtol 1.0e-5 -ksp_error_if_not_converged -ksp_monitor_true_residual \
      -pc_type mg -pc_mg_levels 3 \
        -mg_levels_ksp_type gmres -mg_levels_ksp_max_it 30 -mg_levels_ksp_monitor_true_residual_no \
        -mg_levels_pc_type patch -mg_levels_pc_patch_partition_of_unity 0 -mg_levels_pc_patch_construct_codim 0 -mg_levels_pc_patch_construct_type vanka \
          -mg_levels_sub_ksp_type preonly -mg_levels_sub_pc_type lu \
        -mg_coarse_pc_type svd

  test:
    requires: !single
    suffix: bddc_quad
    nsize: 2
    args: -run_type full -dm_refine 1 -bc_type dirichlet -interpolate 1 -vel_petscspace_degree 2 -pres_petscspace_degree 1 -snes_view -snes_error_if_not_converged -dm_mat_type is -ksp_type gmres -ksp_rtol 1.e-8 -pc_type bddc -pc_bddc_corner_selection -pc_bddc_dirichlet_pc_type svd -pc_bddc_neumann_pc_type svd -pc_bddc_coarse_redundant_pc_type svd -simplex 0 -petscpartitioner_type simple -ksp_monitor_short -pc_bddc_symmetric 0

TEST*/
