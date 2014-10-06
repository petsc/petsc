static char help[] = "Hybrid Finite Element-Finite Volume Example.\n";
/*F
  Here we are advecting a passive tracer in a harmonic velocity field, defined by
a forcing function $f$:
\begin{align}
  -\Delta \mathbf{u} + f &= 0 \\
  \frac{\partial\phi}{\partial t} - \nabla\cdot \phi \mathbf{u} &= 0
\end{align}
F*/
#include <petscdmplex.h>
#include <petscds.h>
#include <petscts.h>

#include <petsc-private/dmpleximpl.h> /* For DotD */

PetscInt spatialDim = 0;

typedef enum {ZERO, CONSTANT, GAUSSIAN} PorosityDistribution;

typedef struct {
  /* Domain and mesh definition */
  PetscInt       dim;               /* The topological mesh dimension */
  DMBoundaryType xbd, ybd;          /* The boundary type for the x- and y-boundary */
  /* Problem definition */
  PetscBool      useFV;             /* Use a finite volume scheme for advection */
  void         (*exactFuncs[2])(const PetscReal x[], PetscScalar *u, void *ctx);
  void         (*initialGuess[2])(const PetscReal x[], PetscScalar *u, void *ctx);
  PorosityDistribution porosityDist;
} AppCtx;

#undef __FUNCT__
#define __FUNCT__ "ProcessOptions"
static PetscErrorCode ProcessOptions(MPI_Comm comm, AppCtx *options)
{
  const char    *porosityDist[2]  = {"constant", "Gaussian"};
  PetscInt       bd, pd;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  options->dim       = 2;
  options->xbd       = DM_BOUNDARY_PERIODIC;
  options->ybd       = DM_BOUNDARY_PERIODIC;
  options->useFV     = PETSC_FALSE;
  options->porosityDist = ZERO;

  ierr = PetscOptionsBegin(comm, "", "Magma Dynamics Options", "DMPLEX");CHKERRQ(ierr);
  ierr = PetscOptionsInt("-dim", "The topological mesh dimension", "ex18.c", options->dim, &options->dim, NULL);CHKERRQ(ierr);
  spatialDim = options->dim;
  bd   = options->xbd;
  ierr = PetscOptionsEList("-x_bd_type", "The x-boundary type", "ex18.c", DMBoundaryTypes, 5, DMBoundaryTypes[options->xbd], &bd, NULL);CHKERRQ(ierr);
  options->xbd = (DMBoundaryType) bd;
  bd   = options->ybd;
  ierr = PetscOptionsEList("-y_bd_type", "The y-boundary type", "ex18.c", DMBoundaryTypes, 5, DMBoundaryTypes[options->ybd], &bd, NULL);CHKERRQ(ierr);
  options->ybd = (DMBoundaryType) bd;
  ierr = PetscOptionsBool("-use_fv", "Use the finite volume method for advection", "ex18.c", options->useFV, &options->useFV, NULL);CHKERRQ(ierr);
  pd   = options->porosityDist;
  ierr = PetscOptionsEList("-porosity_dist","Initial porosity distribution type","ex18.c",porosityDist,2,porosityDist[options->porosityDist],&pd,NULL);CHKERRQ(ierr);
  options->porosityDist = (PorosityDistribution) pd;
  ierr = PetscOptionsEnd();

  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "CreateMesh"
static PetscErrorCode CreateMesh(MPI_Comm comm, AppCtx *user, DM *dm)
{
  DM              distributedMesh = NULL;
  PetscBool       periodic        = user->xbd == DM_BOUNDARY_PERIODIC || user->xbd == DM_BOUNDARY_TWIST || user->ybd == DM_BOUNDARY_PERIODIC || user->ybd == DM_BOUNDARY_TWIST ? PETSC_TRUE : PETSC_FALSE;
  const PetscInt  cells[3]        = {3, 3, 3};
  const PetscReal L[3]            = {1.0, 1.0, 1.0};
  PetscReal       maxCell[3];
  DMLabel         label;
  PetscInt        d;
  PetscErrorCode  ierr;

  PetscFunctionBeginUser;
  ierr = DMPlexCreateHexBoxMesh(comm, user->dim, cells, user->xbd, user->ybd, DM_BOUNDARY_NONE, dm);CHKERRQ(ierr);
  if (periodic) {for (d = 0; d < 3; ++d) maxCell[d] = 1.1*(L[d]/cells[d]); ierr = DMSetPeriodicity(*dm, maxCell, L);CHKERRQ(ierr);}
  ierr = PetscObjectSetName((PetscObject) *dm, "Mesh");CHKERRQ(ierr);
  ierr = DMPlexGetLabel(*dm, "marker", &label);CHKERRQ(ierr);
  if (label) {ierr = DMPlexLabelComplete(*dm, label);CHKERRQ(ierr);}
  ierr = DMPlexDistribute(*dm, NULL, 0, NULL, &distributedMesh);CHKERRQ(ierr);
  if (distributedMesh) {
    ierr = DMDestroy(dm);CHKERRQ(ierr);
    *dm  = distributedMesh;
  }
  ierr = DMSetFromOptions(*dm);CHKERRQ(ierr);
  ierr = DMPlexLocalizeCoordinates(*dm);CHKERRQ(ierr);
  ierr = DMViewFromOptions(*dm, NULL, "-dm_view");CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static void f0_lap_u(const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], const PetscReal x[], PetscScalar f0[])
{
  const PetscInt Nc = spatialDim;
  PetscInt       comp;
  for (comp = 0; comp < Nc; ++comp) f0[comp] = 4.0;
}

static void f1_lap_u(const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], const PetscReal x[], PetscScalar f1[])
{
  const PetscInt Nc  = spatialDim;
  const PetscInt dim = spatialDim;
  PetscInt       comp, d;

  for (comp = 0; comp < Nc; ++comp) {
    for (d = 0; d < dim; ++d) {
      f1[comp*dim+d] = u_x[comp*dim+d];
    }
  }
}

static void f0_lap_periodic_u(const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], const PetscReal x[], PetscScalar f0[])
{
  f0[0] = -sin(2.0*PETSC_PI*x[0]);
  f0[1] = 2.0*PETSC_PI*x[1]*cos(2.0*PETSC_PI*x[0]);
}

static void f0_advection(const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                         const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], const PetscReal x[], PetscScalar f0[])
{
  PetscInt d;
  f0[0] = u_t[spatialDim+1];
  for (d = 0; d < spatialDim; ++d) f0[0] -= u[spatialDim+1]*u_x[d*spatialDim+d] + u_x[(spatialDim+1)*spatialDim+d]*u[d];
}

static void f1_advection(const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                         const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], const PetscReal x[], PetscScalar f1[])
{
  PetscInt d;
  for (d = 0; d < spatialDim; ++d) f1[0] = 0.0;
}

static void riemann_advection(const PetscReal *qp, const PetscReal *n, const PetscScalar *uL, const PetscScalar *uR, PetscScalar *flux, void *ctx)
{
  PetscReal wn = DMPlex_DotD_Internal(spatialDim, uL, n);

  flux[0] = (wn > 0 ? uL[spatialDim] : uR[spatialDim]) * wn;
}

/*
  In 2D we use the exact solution:

    u   = x^2 + y^2
    v   = 2 x^2 - 2xy
    phi = h(x + y + (u + v) t)
    f_x = f_y = 4

  so that

    -\Delta u + f = <-4, -4> + <4, 4> = 0
    {\partial\phi}{\partial t} - \nabla\cdot \phi u = 0
    u h_t(x + y + (u + v) t) - u . grad phi - phi div u
  = u h' + v h'              - u h_x - v h_y
  = 0

We will conserve phi since

    \nabla \cdot u = 2x - 2x = 0
*/
void quadratic_u_2d(const PetscReal x[], PetscScalar *u, void *ctx)
{
  u[0] = x[0]*x[0] + x[1]*x[1];
  u[1] = 2.0*x[0]*x[0] - 2.0*x[0]*x[1];
}

/*
  In 2D we use the exact, periodic solution:

    u   =  sin(2 pi x)/4 pi^2
    v   = -y cos(2 pi x)/2 pi
    phi = h(x + y + (u + v) t)
    f_x = -sin(2 pi x)
    f_y = 2 pi y cos(2 pi x)

  so that

    -\Delta u + f = <sin(2pi x),  -2pi y cos(2pi x)> + <-sin(2pi x), 2pi y cos(2pi x)> = 0

We will conserve phi since

    \nabla \cdot u = cos(2pi x)/2pi - cos(2pi x)/2pi = 0
*/
void periodic_u_2d(const PetscReal x[], PetscScalar *u, void *ctx)
{
  u[0] = sin(2.0*PETSC_PI*x[0])/PetscSqr(2.0*PETSC_PI);
  u[1] = -x[1]*cos(2.0*PETSC_PI*x[0])/(2.0*PETSC_PI);
}

void initialVelocity(const PetscReal x[], PetscScalar *u, void *ctx)
{
  PetscInt d;
  for (d = 0; d < spatialDim; ++d) u[d] = 0.0;
}

void zero_phi(const PetscReal x[], PetscScalar *u, void *ctx)
{
  u[0] = 0.0;
}

void constant_phi(const PetscReal x[], PetscScalar *u, void *ctx)
{
  u[0] = 1.0;
}

void gaussian_phi_2d(const PetscReal x[], PetscScalar *u, void *ctx)
{
  const PetscReal r2 = x[0]*x[0] + x[1]*x[1];
  u[0] = PetscExpReal(-r2)/(2.0*PETSC_PI);
}

#undef __FUNCT__
#define __FUNCT__ "SetupProblem"
static PetscErrorCode SetupProblem(DM dm, AppCtx *user)
{
  PetscDS        prob;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = DMGetDS(dm, &prob);CHKERRQ(ierr);
  switch (user->xbd) {
  case DM_BOUNDARY_PERIODIC:
    ierr = PetscDSSetResidual(prob, 0, f0_lap_periodic_u, f1_lap_u);CHKERRQ(ierr);
    break;
  default:
    ierr = PetscDSSetResidual(prob, 0, f0_lap_u, f1_lap_u);CHKERRQ(ierr);
    break;
  }
  ierr = PetscDSSetResidual(prob, 1, f0_advection, f1_advection);CHKERRQ(ierr);
  ierr = PetscDSSetRiemannSolver(prob, 1, riemann_advection);CHKERRQ(ierr);
  switch (user->dim) {
  case 2:
    user->initialGuess[0] = quadratic_u_2d/*initialVelocity*/;
    switch(user->porosityDist) {
    case ZERO:     user->initialGuess[1] = zero_phi;break;
    case CONSTANT: user->initialGuess[1] = constant_phi;break;
    case GAUSSIAN: user->initialGuess[1] = gaussian_phi_2d;break;
    }
    switch (user->xbd) {
    case DM_BOUNDARY_PERIODIC:
      user->exactFuncs[0] = periodic_u_2d;
      user->exactFuncs[1] = user->initialGuess[1];
      break;
    default:
      user->exactFuncs[0] = quadratic_u_2d;
      user->exactFuncs[1] = user->initialGuess[1];
      break;
    }
    break;
  default:
    SETERRQ1(PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE, "Invalid dimension %d", user->dim);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SetupDiscretization"
PetscErrorCode SetupDiscretization(DM dm, AppCtx *user)
{
  DM              cdm   = dm;
  const PetscInt  dim   = user->dim;
  const PetscInt  id    = 1;
  PetscQuadrature q;
  PetscFE         fe[2];
  PetscFV         fv;
  PetscDS         prob;
  PetscInt        order;
  PetscErrorCode  ierr;

  PetscFunctionBeginUser;
  /* Create finite element */
  ierr = PetscFECreateDefault(dm, dim, dim, PETSC_FALSE, "velocity_", -1, &fe[0]);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) fe[0], "velocity");CHKERRQ(ierr);
  ierr = PetscFEGetQuadrature(fe[0], &q);CHKERRQ(ierr);
  ierr = PetscQuadratureGetOrder(q, &order);CHKERRQ(ierr);
  ierr = PetscFECreateDefault(dm, dim, 1, PETSC_FALSE, "porosity_", order, &fe[1]);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) fe[1], "porosity");CHKERRQ(ierr);

  ierr = PetscFVCreate(PetscObjectComm((PetscObject) dm), &fv);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) fv, "porosity");CHKERRQ(ierr);
  ierr = PetscFVSetFromOptions(fv);CHKERRQ(ierr);
  ierr = PetscFVSetNumComponents(fv, 1);CHKERRQ(ierr);
  ierr = PetscFVSetSpatialDimension(fv, dim);CHKERRQ(ierr);
  ierr = PetscFVSetQuadrature(fv, q);CHKERRQ(ierr);

  /* Set discretization and boundary conditions for each mesh */
  while (cdm) {
    DMLabel label;

    ierr = DMGetDS(cdm, &prob);CHKERRQ(ierr);
    ierr = PetscDSSetDiscretization(prob, 0, (PetscObject) fe[0]);CHKERRQ(ierr);
    if (user->useFV) {ierr = PetscDSSetDiscretization(prob, 1, (PetscObject) fv);CHKERRQ(ierr);}
    else             {ierr = PetscDSSetDiscretization(prob, 1, (PetscObject) fe[1]);CHKERRQ(ierr);}

    ierr = SetupProblem(cdm, user);CHKERRQ(ierr);
    ierr = DMPlexGetLabel(cdm, "marker", &label);CHKERRQ(ierr);
    if (label) {ierr = DMPlexAddBoundary(cdm, 1, "wall", "marker", 0, user->exactFuncs[0], 1, &id, user);CHKERRQ(ierr);}
    ierr = DMPlexGetCoarseDM(cdm, &cdm);CHKERRQ(ierr);

    /* Coordinates were never localized for coarse meshes */
    if (cdm) {ierr = DMPlexLocalizeCoordinates(cdm);CHKERRQ(ierr);}
  }
  ierr = PetscFEDestroy(&fe[0]);CHKERRQ(ierr);
  ierr = PetscFEDestroy(&fe[1]);CHKERRQ(ierr);
  ierr = PetscFVDestroy(&fv);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc, char **argv)
{
  MPI_Comm       comm;
  TS             ts;
  DM             dm;
  Vec            u;
  AppCtx         user;
  void          *ctxs[2] = {NULL, NULL};
  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc, &argv, (char*) 0, help);CHKERRQ(ierr);
  comm = PETSC_COMM_WORLD;
  ierr = ProcessOptions(comm, &user);CHKERRQ(ierr);
  ierr = TSCreate(comm, &ts);CHKERRQ(ierr);
  ierr = TSSetType(ts, TSBEULER);CHKERRQ(ierr);
  ierr = CreateMesh(comm, &user, &dm);CHKERRQ(ierr);
  ierr = TSSetDM(ts, dm);CHKERRQ(ierr);
  ierr = SetupDiscretization(dm, &user);CHKERRQ(ierr);

  ierr = DMCreateGlobalVector(dm, &u);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) u, "solution");CHKERRQ(ierr);
  ierr = DMTSSetIFunctionLocal(dm, DMPlexTSComputeIFunctionFEM, &user);CHKERRQ(ierr);
  ierr = TSSetDuration(ts, 1, 2.0);CHKERRQ(ierr);
  ierr = TSSetInitialTimeStep(ts, 0.0, 0.01);CHKERRQ(ierr);
  ierr = TSSetFromOptions(ts);CHKERRQ(ierr);

  ierr = DMPlexProjectFunction(dm, user.exactFuncs, ctxs, INSERT_ALL_VALUES, u);CHKERRQ(ierr);
  ierr = DMPlexProjectFunction(dm, user.initialGuess, ctxs, INSERT_VALUES, u);CHKERRQ(ierr);
  ierr = VecViewFromOptions(u, "init_", "-vec_view");CHKERRQ(ierr);
  ierr = DMTSCheckFromOptions(ts, u, user.exactFuncs);CHKERRQ(ierr);
  ierr = TSSolve(ts, u);CHKERRQ(ierr);
  ierr = VecViewFromOptions(u, "sol_", "-vec_view");CHKERRQ(ierr);

  ierr = VecDestroy(&u);CHKERRQ(ierr);
  ierr = DMDestroy(&dm);CHKERRQ(ierr);
  ierr = TSDestroy(&ts);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return(0);
}
