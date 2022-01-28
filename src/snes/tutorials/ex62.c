static char help[] = "Stokes Problem discretized with finite elements,\n\
using a parallel unstructured mesh (DMPLEX) to represent the domain.\n\n\n";

/*
For the isoviscous Stokes problem, which we discretize using the finite
element method on an unstructured mesh, the weak form equations are

  < \nabla v, \nabla u + {\nabla u}^T > - < \nabla\cdot v, p > - < v, f > = 0
  < q, -\nabla\cdot u >                                                   = 0

Viewing:

To produce nice output, use

  -dm_refine 3 -dm_view hdf5:sol1.h5 -error_vec_view hdf5:sol1.h5::append -snes_view_solution hdf5:sol1.h5::append -exact_vec_view hdf5:sol1.h5::append

You can get a LaTeX view of the mesh, with point numbering using

  -dm_view :mesh.tex:ascii_latex -dm_plex_view_scale 8.0

The data layout can be viewed using

  -dm_petscsection_view

Lots of information about the FEM assembly can be printed using

  -dm_plex_print_fem 3
*/

#include <petscdmplex.h>
#include <petscsnes.h>
#include <petscds.h>
#include <petscbag.h>

// TODO: Plot residual by fields after each smoother iterate

typedef enum {SOL_QUADRATIC, SOL_TRIG, SOL_UNKNOWN} SolType;
const char *SolTypes[] = {"quadratic", "trig", "unknown", "SolType", "SOL_", 0};

typedef struct {
  PetscScalar mu; /* dynamic shear viscosity */
} Parameter;

typedef struct {
  PetscBag bag; /* Problem parameters */
  SolType  sol; /* MMS solution */
} AppCtx;

static void f1_u(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                 const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                 const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                 PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f1[])
{
  const PetscReal mu = PetscRealPart(constants[0]);
  const PetscInt  Nc = uOff[1]-uOff[0];
  PetscInt        c, d;

  for (c = 0; c < Nc; ++c) {
    for (d = 0; d < dim; ++d) {
      f1[c*dim+d] = mu * (u_x[c*dim+d] + u_x[d*dim+c]);
    }
    f1[c*dim+c] -= u[uOff[1]];
  }
}

static void f0_p(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                 const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                 const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                 PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f0[])
{
  PetscInt d;
  for (d = 0, f0[0] = 0.0; d < dim; ++d) f0[0] -= u_x[d*dim+d];
}

static void g1_pu(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                  const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                  const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                  PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g1[])
{
  PetscInt d;
  for (d = 0; d < dim; ++d) g1[d*dim+d] = -1.0; /* < q, -\nabla\cdot u > */
}

static void g2_up(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                  const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                  const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                  PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g2[])
{
  PetscInt d;
  for (d = 0; d < dim; ++d) g2[d*dim+d] = -1.0; /* -< \nabla\cdot v, p > */
}

static void g3_uu(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                  const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                  const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                  PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g3[])
{
  const PetscReal mu = PetscRealPart(constants[0]);
  const PetscInt  Nc = uOff[1]-uOff[0];
  PetscInt        c, d;

  for (c = 0; c < Nc; ++c) {
    for (d = 0; d < dim; ++d) {
      g3[((c*Nc+c)*dim+d)*dim+d] += mu; /* < \nabla v, \nabla u > */
      g3[((c*Nc+d)*dim+d)*dim+c] += mu; /* < \nabla v, {\nabla u}^T > */
    }
  }
}

static void g0_pp(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                  const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                  const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                  PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g0[])
{
  const PetscReal mu = PetscRealPart(constants[0]);

  g0[0] = 1.0/mu;
}

/* Quadratic MMS Solution
   2D:

     u = x^2 + y^2
     v = 2 x^2 - 2xy
     p = x + y - 1
     f = <1 - 4 mu, 1 - 4 mu>

   so that

     e(u) = (grad u + grad u^T) = / 4x  4x \
                                  \ 4x -4x /
     div mu e(u) - \nabla p + f = mu <4, 4> - <1, 1> + <1 - 4 mu, 1 - 4 mu> = 0
     \nabla \cdot u             = 2x - 2x = 0

   3D:

     u = 2 x^2 + y^2 + z^2
     v = 2 x^2 - 2xy
     w = 2 x^2 - 2xz
     p = x + y + z - 3/2
     f = <1 - 8 mu, 1 - 4 mu, 1 - 4 mu>

   so that

     e(u) = (grad u + grad u^T) = / 8x  4x  4x \
                                  | 4x -4x  0  |
                                  \ 4x  0  -4x /
     div mu e(u) - \nabla p + f = mu <8, 4, 4> - <1, 1, 1> + <1 - 8 mu, 1 - 4 mu, 1 - 4 mu> = 0
     \nabla \cdot u             = 4x - 2x - 2x = 0
*/
static PetscErrorCode quadratic_u(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nc, PetscScalar *u, void *ctx)
{
  PetscInt c;

  u[0] = (dim-1)*PetscSqr(x[0]);
  for (c = 1; c < Nc; ++c) {
    u[0] += PetscSqr(x[c]);
    u[c]  = 2.0*PetscSqr(x[0]) - 2.0*x[0]*x[c];
  }
  return 0;
}

static PetscErrorCode quadratic_p(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nc, PetscScalar *u, void *ctx)
{
  PetscInt d;

  u[0] = -0.5*dim;
  for (d = 0; d < dim; ++d) u[0] += x[d];
  return 0;
}

static void f0_quadratic_u(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                           const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                           const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                           PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f0[])
{
  const PetscReal mu = PetscRealPart(constants[0]);
  PetscInt        d;

  f0[0] = (dim-1)*4.0*mu - 1.0;
  for (d = 1; d < dim; ++d) f0[d] = 4.0*mu - 1.0;
}

/* Trigonometric MMS Solution
   2D:

     u = sin(pi x) + sin(pi y)
     v = -pi cos(pi x) y
     p = sin(2 pi x) + sin(2 pi y)
     f = <2pi cos(2 pi x) + mu pi^2 sin(pi x) + mu pi^2 sin(pi y), 2pi cos(2 pi y) - mu pi^3 cos(pi x) y>

   so that

     e(u) = (grad u + grad u^T) = /        2pi cos(pi x)             pi cos(pi y) + pi^2 sin(pi x) y \
                                  \ pi cos(pi y) + pi^2 sin(pi x) y          -2pi cos(pi x)          /
     div mu e(u) - \nabla p + f = mu <-pi^2 sin(pi x) - pi^2 sin(pi y), pi^3 cos(pi x) y> - <2pi cos(2 pi x), 2pi cos(2 pi y)> + <f_x, f_y> = 0
     \nabla \cdot u             = pi cos(pi x) - pi cos(pi x) = 0

   3D:

     u = 2 sin(pi x) + sin(pi y) + sin(pi z)
     v = -pi cos(pi x) y
     w = -pi cos(pi x) z
     p = sin(2 pi x) + sin(2 pi y) + sin(2 pi z)
     f = <2pi cos(2 pi x) + mu 2pi^2 sin(pi x) + mu pi^2 sin(pi y) + mu pi^2 sin(pi z), 2pi cos(2 pi y) - mu pi^3 cos(pi x) y, 2pi cos(2 pi z) - mu pi^3 cos(pi x) z>

   so that

     e(u) = (grad u + grad u^T) = /        4pi cos(pi x)             pi cos(pi y) + pi^2 sin(pi x) y  pi cos(pi z) + pi^2 sin(pi x) z \
                                  | pi cos(pi y) + pi^2 sin(pi x) y          -2pi cos(pi x)                        0                  |
                                  \ pi cos(pi z) + pi^2 sin(pi x) z               0                         -2pi cos(pi x)            /
     div mu e(u) - \nabla p + f = mu <-2pi^2 sin(pi x) - pi^2 sin(pi y) - pi^2 sin(pi z), pi^3 cos(pi x) y, pi^3 cos(pi x) z> - <2pi cos(2 pi x), 2pi cos(2 pi y), 2pi cos(2 pi z)> + <f_x, f_y, f_z> = 0
     \nabla \cdot u             = 2 pi cos(pi x) - pi cos(pi x) - pi cos(pi x) = 0
*/
static PetscErrorCode trig_u(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nc, PetscScalar *u, void *ctx)
{
  PetscInt c;

  u[0] = (dim-1)*PetscSinReal(PETSC_PI*x[0]);
  for (c = 1; c < Nc; ++c) {
    u[0] += PetscSinReal(PETSC_PI*x[c]);
    u[c]  = -PETSC_PI*PetscCosReal(PETSC_PI*x[0]) * x[c];
  }
  return 0;
}

static PetscErrorCode trig_p(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nc, PetscScalar *u, void *ctx)
{
  PetscInt d;

  for (d = 0, u[0] = 0.0; d < dim; ++d) u[0] += PetscSinReal(2.0*PETSC_PI*x[d]);
  return 0;
}

static void f0_trig_u(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                      const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                      const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                      PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f0[])
{
  const PetscReal mu = PetscRealPart(constants[0]);
  PetscInt        d;

  f0[0] = -2.0*PETSC_PI*PetscCosReal(2.0*PETSC_PI*x[0]) - (dim-1)*mu*PetscSqr(PETSC_PI)*PetscSinReal(PETSC_PI*x[0]);
  for (d = 1; d < dim; ++d) {
    f0[0] -= mu*PetscSqr(PETSC_PI)*PetscSinReal(PETSC_PI*x[d]);
    f0[d]  = -2.0*PETSC_PI*PetscCosReal(2.0*PETSC_PI*x[d]) + mu*PetscPowRealInt(PETSC_PI, 3)*PetscCosReal(PETSC_PI*x[0])*x[d];
  }
}

static PetscErrorCode ProcessOptions(MPI_Comm comm, AppCtx *options)
{
  PetscInt       sol;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  options->sol = SOL_QUADRATIC;

  ierr = PetscOptionsBegin(comm, "", "Stokes Problem Options", "DMPLEX");CHKERRQ(ierr);
  sol  = options->sol;
  ierr = PetscOptionsEList("-sol", "The MMS solution", "ex62.c", SolTypes, (sizeof(SolTypes)/sizeof(SolTypes[0]))-3, SolTypes[options->sol], &sol, NULL);CHKERRQ(ierr);
  options->sol = (SolType) sol;
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode CreateMesh(MPI_Comm comm, AppCtx *user, DM *dm)
{
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = DMCreate(comm, dm);CHKERRQ(ierr);
  ierr = DMSetType(*dm, DMPLEX);CHKERRQ(ierr);
  ierr = DMSetFromOptions(*dm);CHKERRQ(ierr);
  ierr = DMViewFromOptions(*dm, NULL, "-dm_view");CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode SetupParameters(MPI_Comm comm, AppCtx *ctx)
{
  Parameter     *p;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  /* setup PETSc parameter bag */
  ierr = PetscBagCreate(PETSC_COMM_SELF, sizeof(Parameter), &ctx->bag);CHKERRQ(ierr);
  ierr = PetscBagGetData(ctx->bag, (void **) &p);CHKERRQ(ierr);
  ierr = PetscBagSetName(ctx->bag, "par", "Stokes Parameters");CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(ctx->bag, &p->mu, 1.0, "mu", "Dynamic Shear Viscosity, Pa s");CHKERRQ(ierr);
  ierr = PetscBagSetFromOptions(ctx->bag);CHKERRQ(ierr);
  {
    PetscViewer       viewer;
    PetscViewerFormat format;
    PetscBool         flg;

    ierr = PetscOptionsGetViewer(comm, NULL, NULL, "-param_view", &viewer, &format, &flg);CHKERRQ(ierr);
    if (flg) {
      ierr = PetscViewerPushFormat(viewer, format);CHKERRQ(ierr);
      ierr = PetscBagView(ctx->bag, viewer);CHKERRQ(ierr);
      ierr = PetscViewerFlush(viewer);CHKERRQ(ierr);
      ierr = PetscViewerPopFormat(viewer);CHKERRQ(ierr);
      ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode SetupEqn(DM dm, AppCtx *user)
{
  PetscErrorCode (*exactFuncs[2])(PetscInt, PetscReal, const PetscReal[], PetscInt, PetscScalar *, void *);
  PetscDS          ds;
  DMLabel          label;
  const PetscInt   id = 1;
  PetscErrorCode   ierr;

  PetscFunctionBeginUser;
  ierr = DMGetDS(dm, &ds);CHKERRQ(ierr);
  switch (user->sol) {
    case SOL_QUADRATIC:
      ierr = PetscDSSetResidual(ds, 0, f0_quadratic_u, f1_u);CHKERRQ(ierr);
      exactFuncs[0] = quadratic_u;
      exactFuncs[1] = quadratic_p;
      break;
    case SOL_TRIG:
      ierr = PetscDSSetResidual(ds, 0, f0_trig_u, f1_u);CHKERRQ(ierr);
      exactFuncs[0] = trig_u;
      exactFuncs[1] = trig_p;
      break;
    default: SETERRQ(PetscObjectComm((PetscObject) dm), PETSC_ERR_ARG_WRONG, "Unsupported solution type: %s (%D)", SolTypes[PetscMin(user->sol, SOL_UNKNOWN)], user->sol);
  }
  ierr = PetscDSSetResidual(ds, 1, f0_p, NULL);CHKERRQ(ierr);
  ierr = PetscDSSetJacobian(ds, 0, 0, NULL, NULL,  NULL,  g3_uu);CHKERRQ(ierr);
  ierr = PetscDSSetJacobian(ds, 0, 1, NULL, NULL,  g2_up, NULL);CHKERRQ(ierr);
  ierr = PetscDSSetJacobian(ds, 1, 0, NULL, g1_pu, NULL,  NULL);CHKERRQ(ierr);
  ierr = PetscDSSetJacobianPreconditioner(ds, 0, 0, NULL, NULL, NULL, g3_uu);CHKERRQ(ierr);
  ierr = PetscDSSetJacobianPreconditioner(ds, 1, 1, g0_pp, NULL, NULL, NULL);CHKERRQ(ierr);

  ierr = PetscDSSetExactSolution(ds, 0, exactFuncs[0], user);CHKERRQ(ierr);
  ierr = PetscDSSetExactSolution(ds, 1, exactFuncs[1], user);CHKERRQ(ierr);

  ierr = DMGetLabel(dm, "marker", &label);CHKERRQ(ierr);
  ierr = DMAddBoundary(dm, DM_BC_ESSENTIAL, "wall", label, 1, &id, 0, 0, NULL, (void (*)(void)) exactFuncs[0], NULL, user, NULL);CHKERRQ(ierr);

  /* Make constant values available to pointwise functions */
  {
    Parameter  *param;
    PetscScalar constants[1];

    ierr = PetscBagGetData(user->bag, (void **) &param);CHKERRQ(ierr);
    constants[0] = param->mu; /* dynamic shear viscosity, Pa s */
    ierr = PetscDSSetConstants(ds, 1, constants);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode zero(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nc, PetscScalar *u, void *ctx)
{
  PetscInt c;
  for (c = 0; c < Nc; ++c) u[c] = 0.0;
  return 0;
}
static PetscErrorCode one(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nc, PetscScalar *u, void *ctx)
{
  PetscInt c;
  for (c = 0; c < Nc; ++c) u[c] = 1.0;
  return 0;
}

static PetscErrorCode CreatePressureNullSpace(DM dm, PetscInt origField, PetscInt field, MatNullSpace *nullspace)
{
  Vec              vec;
  PetscErrorCode (*funcs[2])(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nf, PetscScalar *u, void* ctx) = {zero, one};
  PetscErrorCode   ierr;

  PetscFunctionBeginUser;
  if (origField != 1) SETERRQ(PetscObjectComm((PetscObject) dm), PETSC_ERR_ARG_WRONG, "Field %D should be 1 for pressure", origField);
  funcs[field] = one;
  {
    PetscDS ds;
    ierr = DMGetDS(dm, &ds);CHKERRQ(ierr);
    ierr = PetscObjectViewFromOptions((PetscObject) ds, NULL, "-ds_view");CHKERRQ(ierr);
  }
  ierr = DMCreateGlobalVector(dm, &vec);CHKERRQ(ierr);
  ierr = DMProjectFunction(dm, 0.0, funcs, NULL, INSERT_ALL_VALUES, vec);CHKERRQ(ierr);
  ierr = VecNormalize(vec, NULL);CHKERRQ(ierr);
  ierr = MatNullSpaceCreate(PetscObjectComm((PetscObject)dm), PETSC_FALSE, 1, &vec, nullspace);CHKERRQ(ierr);
  ierr = VecDestroy(&vec);CHKERRQ(ierr);
  /* New style for field null spaces */
  {
    PetscObject  pressure;
    MatNullSpace nullspacePres;

    ierr = DMGetField(dm, field, NULL, &pressure);CHKERRQ(ierr);
    ierr = MatNullSpaceCreate(PetscObjectComm(pressure), PETSC_TRUE, 0, NULL, &nullspacePres);CHKERRQ(ierr);
    ierr = PetscObjectCompose(pressure, "nullspace", (PetscObject) nullspacePres);CHKERRQ(ierr);
    ierr = MatNullSpaceDestroy(&nullspacePres);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode SetupProblem(DM dm, PetscErrorCode (*setupEqn)(DM, AppCtx *), AppCtx *user)
{
  DM              cdm = dm;
  PetscQuadrature q   = NULL;
  PetscBool       simplex;
  PetscInt        dim, Nf = 2, f, Nc[2];
  const char     *name[2]   = {"velocity", "pressure"};
  const char     *prefix[2] = {"vel_",     "pres_"};
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);
  ierr = DMPlexIsSimplex(dm, &simplex);CHKERRQ(ierr);
  Nc[0] = dim;
  Nc[1] = 1;
  for (f = 0; f < Nf; ++f) {
    PetscFE fe;

    ierr = PetscFECreateDefault(PETSC_COMM_SELF, dim, Nc[f], simplex, prefix[f], -1, &fe);CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject) fe, name[f]);CHKERRQ(ierr);
    if (!q) {ierr = PetscFEGetQuadrature(fe, &q);CHKERRQ(ierr);}
    ierr = PetscFESetQuadrature(fe, q);CHKERRQ(ierr);
    ierr = DMSetField(dm, f, NULL, (PetscObject) fe);CHKERRQ(ierr);
    ierr = PetscFEDestroy(&fe);CHKERRQ(ierr);
  }
  ierr = DMCreateDS(dm);CHKERRQ(ierr);
  ierr = (*setupEqn)(dm, user);CHKERRQ(ierr);
  while (cdm) {
    ierr = DMCopyDisc(dm, cdm);CHKERRQ(ierr);
    ierr = DMSetNullSpaceConstructor(cdm, 1, CreatePressureNullSpace);CHKERRQ(ierr);
    ierr = DMGetCoarseDM(cdm, &cdm);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

int main(int argc, char **argv)
{
  SNES           snes;
  DM             dm;
  Vec            u;
  AppCtx         user;
  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc, &argv, NULL, help);if (ierr) return ierr;
  ierr = ProcessOptions(PETSC_COMM_WORLD, &user);CHKERRQ(ierr);
  ierr = CreateMesh(PETSC_COMM_WORLD, &user, &dm);CHKERRQ(ierr);
  ierr = SNESCreate(PetscObjectComm((PetscObject) dm), &snes);CHKERRQ(ierr);
  ierr = SNESSetDM(snes, dm);CHKERRQ(ierr);
  ierr = DMSetApplicationContext(dm, &user);CHKERRQ(ierr);

  ierr = SetupParameters(PETSC_COMM_WORLD, &user);CHKERRQ(ierr);
  ierr = SetupProblem(dm, SetupEqn, &user);CHKERRQ(ierr);
  ierr = DMPlexCreateClosureIndex(dm, NULL);CHKERRQ(ierr);

  ierr = DMCreateGlobalVector(dm, &u);CHKERRQ(ierr);
  ierr = DMPlexSetSNESLocalFEM(dm, &user, &user, &user);CHKERRQ(ierr);
  ierr = SNESSetFromOptions(snes);CHKERRQ(ierr);
  ierr = DMSNESCheckFromOptions(snes, u);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) u, "Solution");CHKERRQ(ierr);
  {
    Mat          J;
    MatNullSpace sp;

    ierr = SNESSetUp(snes);CHKERRQ(ierr);
    ierr = CreatePressureNullSpace(dm, 1, 1, &sp);CHKERRQ(ierr);
    ierr = SNESGetJacobian(snes, &J, NULL, NULL, NULL);CHKERRQ(ierr);
    ierr = MatSetNullSpace(J, sp);CHKERRQ(ierr);
    ierr = MatNullSpaceDestroy(&sp);CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject) J, "Jacobian");CHKERRQ(ierr);
    ierr = MatViewFromOptions(J, NULL, "-J_view");CHKERRQ(ierr);
  }
  ierr = SNESSolve(snes, NULL, u);CHKERRQ(ierr);

  ierr = VecDestroy(&u);CHKERRQ(ierr);
  ierr = SNESDestroy(&snes);CHKERRQ(ierr);
  ierr = DMDestroy(&dm);CHKERRQ(ierr);
  ierr = PetscBagDestroy(&user.bag);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}
/*TEST

  test:
    suffix: 2d_p2_p1_check
    requires: triangle
    args: -sol quadratic -vel_petscspace_degree 2 -pres_petscspace_degree 1 -dmsnes_check 0.0001

  test:
    suffix: 2d_p2_p1_check_parallel
    nsize: {{2 3 5}}
    requires: triangle
    args: -sol quadratic -dm_refine 2 -dm_distribute -petscpartitioner_type simple -vel_petscspace_degree 2 -pres_petscspace_degree 1 -dmsnes_check 0.0001

  test:
    suffix: 3d_p2_p1_check
    requires: ctetgen
    args: -sol quadratic -dm_plex_dim 3 -dm_plex_box_faces 2,2,2 -vel_petscspace_degree 2 -pres_petscspace_degree 1 -dmsnes_check 0.0001

  test:
    suffix: 3d_p2_p1_check_parallel
    nsize: {{2 3 5}}
    requires: ctetgen
    args: -sol quadratic -dm_refine 2 -dm_plex_dim 3 -dm_plex_box_faces 2,2,2 -dm_distribute -petscpartitioner_type simple -vel_petscspace_degree 2 -pres_petscspace_degree 1 -dmsnes_check 0.0001

  test:
    suffix: 2d_p2_p1_conv
    requires: triangle
    # Using -dm_refine 3 gives L_2 convergence rate: [3.0, 2.1]
    args: -sol trig -vel_petscspace_degree 2 -pres_petscspace_degree 1 -snes_convergence_estimate -convest_num_refine 2 -ksp_error_if_not_converged \
      -ksp_atol 1e-10 -ksp_error_if_not_converged -pc_use_amat \
      -pc_type fieldsplit -pc_fieldsplit_type schur -pc_fieldsplit_schur_fact_type full -pc_fieldsplit_schur_precondition a11 -pc_fieldsplit_off_diag_use_amat \
        -fieldsplit_velocity_pc_type lu -fieldsplit_pressure_ksp_rtol 1e-10 -fieldsplit_pressure_pc_type lu

  test:
    suffix: 2d_p2_p1_conv_gamg
    requires: triangle
    args: -sol trig -vel_petscspace_degree 2 -pres_petscspace_degree 1 -snes_convergence_estimate -convest_num_refine 2  \
      -pc_type fieldsplit -pc_fieldsplit_type schur -pc_fieldsplit_schur_fact_type full -pc_fieldsplit_schur_precondition full \
        -fieldsplit_velocity_pc_type lu -fieldsplit_pressure_ksp_rtol 1e-10 -fieldsplit_pressure_pc_type gamg -fieldsplit_pressure_mg_coarse_pc_type svd

  test:
    suffix: 3d_p2_p1_conv
    requires: ctetgen !single
    # Using -dm_refine 2 -convest_num_refine 2 gives L_2 convergence rate: [2.8, 2.8]
    args: -sol trig -dm_plex_dim 3 -dm_refine 1 -vel_petscspace_degree 2 -pres_petscspace_degree 1 -snes_convergence_estimate -convest_num_refine 1 \
      -ksp_atol 1e-10 -ksp_error_if_not_converged -pc_use_amat \
      -pc_type fieldsplit -pc_fieldsplit_type schur -pc_fieldsplit_schur_fact_type full -pc_fieldsplit_schur_precondition a11 -pc_fieldsplit_off_diag_use_amat \
        -fieldsplit_velocity_pc_type lu -fieldsplit_pressure_ksp_rtol 1e-10 -fieldsplit_pressure_pc_type lu

  test:
    suffix: 2d_q2_q1_check
    args: -sol quadratic -dm_plex_simplex 0 -vel_petscspace_degree 2 -pres_petscspace_degree 1 -dmsnes_check 0.0001

  test:
    suffix: 3d_q2_q1_check
    args: -sol quadratic -dm_plex_simplex 0 -dm_plex_dim 3 -dm_plex_box_faces 2,2,2 -vel_petscspace_degree 2 -pres_petscspace_degree 1 -dmsnes_check 0.0001

  test:
    suffix: 2d_q2_q1_conv
    # Using -dm_refine 3 -convest_num_refine 1 gives L_2 convergence rate: [3.0, 2.1]
    args: -sol trig -dm_plex_simplex 0 -vel_petscspace_degree 2 -pres_petscspace_degree 1 -snes_convergence_estimate -convest_num_refine 1 -ksp_error_if_not_converged \
      -ksp_atol 1e-10 -ksp_error_if_not_converged -pc_use_amat \
      -pc_type fieldsplit -pc_fieldsplit_type schur -pc_fieldsplit_schur_fact_type full -pc_fieldsplit_schur_precondition a11 -pc_fieldsplit_off_diag_use_amat \
        -fieldsplit_velocity_pc_type lu -fieldsplit_pressure_ksp_rtol 1e-10 -fieldsplit_pressure_pc_type lu

  test:
    suffix: 3d_q2_q1_conv
    requires: !single
    # Using -dm_refine 2 -convest_num_refine 2 gives L_2 convergence rate: [2.8, 2.4]
    args: -sol trig -dm_plex_simplex 0 -dm_plex_dim 3 -vel_petscspace_degree 2 -pres_petscspace_degree 1 -snes_convergence_estimate -convest_num_refine 1 \
      -ksp_atol 1e-10 -ksp_error_if_not_converged -pc_use_amat \
      -pc_type fieldsplit -pc_fieldsplit_type schur -pc_fieldsplit_schur_fact_type full -pc_fieldsplit_schur_precondition a11 -pc_fieldsplit_off_diag_use_amat \
        -fieldsplit_velocity_pc_type lu -fieldsplit_pressure_ksp_rtol 1e-10 -fieldsplit_pressure_pc_type lu

  test:
    suffix: 2d_p3_p2_check
    requires: triangle
    args: -sol quadratic -vel_petscspace_degree 3 -pres_petscspace_degree 2 -dmsnes_check 0.0001

  test:
    suffix: 3d_p3_p2_check
    requires: ctetgen !single
    args: -sol quadratic -dm_plex_dim 3 -dm_plex_box_faces 2,2,2 -vel_petscspace_degree 3 -pres_petscspace_degree 2 -dmsnes_check 0.0001

  test:
    suffix: 2d_p3_p2_conv
    requires: triangle
    # Using -dm_refine 2 gives L_2 convergence rate: [3.8, 3.0]
    args: -sol trig -vel_petscspace_degree 3 -pres_petscspace_degree 2 -snes_convergence_estimate -convest_num_refine 2 -ksp_error_if_not_converged \
      -ksp_atol 1e-10 -ksp_error_if_not_converged -pc_use_amat \
      -pc_type fieldsplit -pc_fieldsplit_type schur -pc_fieldsplit_schur_fact_type full -pc_fieldsplit_schur_precondition a11 -pc_fieldsplit_off_diag_use_amat \
        -fieldsplit_velocity_pc_type lu -fieldsplit_pressure_ksp_rtol 1e-10 -fieldsplit_pressure_pc_type lu

  test:
    suffix: 3d_p3_p2_conv
    requires: ctetgen long_runtime
    # Using -dm_refine 1 -convest_num_refine 2 gives L_2 convergence rate: [3.6, 3.9]
    args: -sol trig -dm_plex_dim 3 -dm_refine 1 -vel_petscspace_degree 3 -pres_petscspace_degree 2 -snes_convergence_estimate -convest_num_refine 2 \
      -ksp_atol 1e-10 -ksp_error_if_not_converged -pc_use_amat \
      -pc_type fieldsplit -pc_fieldsplit_type schur -pc_fieldsplit_schur_fact_type full -pc_fieldsplit_schur_precondition a11 -pc_fieldsplit_off_diag_use_amat \
        -fieldsplit_velocity_pc_type lu -fieldsplit_pressure_ksp_rtol 1e-10 -fieldsplit_pressure_pc_type lu

  test:
    suffix: 2d_q1_p0_conv
    requires: !single
    # Using -dm_refine 3 gives L_2 convergence rate: [1.9, 1.0]
    args: -sol quadratic -dm_plex_simplex 0 -vel_petscspace_degree 1 -pres_petscspace_degree 0 -snes_convergence_estimate -convest_num_refine 2 \
      -ksp_atol 1e-10 -petscds_jac_pre 0 \
      -pc_type fieldsplit -pc_fieldsplit_type schur -pc_fieldsplit_schur_fact_type full -pc_fieldsplit_schur_precondition full \
        -fieldsplit_velocity_pc_type lu -fieldsplit_pressure_ksp_rtol 1e-10 -fieldsplit_pressure_pc_type gamg -fieldsplit_pressure_mg_levels_pc_type jacobi -fieldsplit_pressure_mg_coarse_pc_type svd

  test:
    suffix: 3d_q1_p0_conv
    requires: !single
    # Using -dm_refine 2 -convest_num_refine 2 gives L_2 convergence rate: [1.7, 1.0]
    args: -sol quadratic -dm_plex_simplex 0 -dm_plex_dim 3 -dm_refine 1 -vel_petscspace_degree 1 -pres_petscspace_degree 0 -snes_convergence_estimate -convest_num_refine 1 \
      -ksp_atol 1e-10 -petscds_jac_pre 0 \
      -pc_type fieldsplit -pc_fieldsplit_type schur -pc_fieldsplit_schur_fact_type full -pc_fieldsplit_schur_precondition full \
        -fieldsplit_velocity_pc_type lu -fieldsplit_pressure_ksp_rtol 1e-10 -fieldsplit_pressure_pc_type gamg -fieldsplit_pressure_mg_levels_pc_type jacobi -fieldsplit_pressure_mg_coarse_pc_type svd

  # Stokes preconditioners
  #   Block diagonal \begin{pmatrix} A & 0 \\ 0 & I \end{pmatrix}
  test:
    suffix: 2d_p2_p1_block_diagonal
    requires: triangle
    args: -sol quadratic -dm_refine 2 -vel_petscspace_degree 2 -pres_petscspace_degree 1 -petscds_jac_pre 0 \
      -snes_error_if_not_converged \
      -ksp_type fgmres -ksp_gmres_restart 100 -ksp_rtol 1.0e-4 -ksp_error_if_not_converged \
      -pc_type fieldsplit -pc_fieldsplit_type additive -fieldsplit_velocity_pc_type lu -fieldsplit_pressure_pc_type jacobi
  #   Block triangular \begin{pmatrix} A & B \\ 0 & I \end{pmatrix}
  test:
    suffix: 2d_p2_p1_block_triangular
    requires: triangle
    args: -sol quadratic -dm_refine 2 -vel_petscspace_degree 2 -pres_petscspace_degree 1 -petscds_jac_pre 0 \
      -snes_error_if_not_converged \
      -ksp_type fgmres -ksp_gmres_restart 100 -ksp_rtol 1.0e-9 -ksp_error_if_not_converged \
      -pc_type fieldsplit -pc_fieldsplit_type multiplicative -fieldsplit_velocity_pc_type lu -fieldsplit_pressure_pc_type jacobi
  #   Diagonal Schur complement \begin{pmatrix} A & 0 \\ 0 & S \end{pmatrix}
  test:
    suffix: 2d_p2_p1_schur_diagonal
    requires: triangle
    args: -sol quadratic -dm_refine 2 -vel_petscspace_degree 2 -pres_petscspace_degree 1 \
      -snes_error_if_not_converged \
      -ksp_type fgmres -ksp_gmres_restart 100 -ksp_rtol 1.0e-9 -ksp_error_if_not_converged -pc_use_amat \
      -pc_type fieldsplit -pc_fieldsplit_type schur -pc_fieldsplit_schur_factorization_type diag -pc_fieldsplit_off_diag_use_amat \
        -fieldsplit_velocity_pc_type lu -fieldsplit_pressure_ksp_rtol 1e-10 -fieldsplit_pressure_pc_type jacobi
  #   Upper triangular Schur complement \begin{pmatrix} A & B \\ 0 & S \end{pmatrix}
  test:
    suffix: 2d_p2_p1_schur_upper
    requires: triangle
    args: -sol quadratic -dm_refine 2 -vel_petscspace_degree 2 -pres_petscspace_degree 1 -dmsnes_check 0.0001 \
      -ksp_type fgmres -ksp_gmres_restart 100 -ksp_rtol 1.0e-9 -ksp_error_if_not_converged -pc_use_amat \
      -pc_type fieldsplit -pc_fieldsplit_type schur -pc_fieldsplit_schur_factorization_type upper -pc_fieldsplit_off_diag_use_amat \
        -fieldsplit_velocity_pc_type lu -fieldsplit_pressure_ksp_rtol 1e-10 -fieldsplit_pressure_pc_type jacobi
  #   Lower triangular Schur complement \begin{pmatrix} A & B \\ 0 & S \end{pmatrix}
  test:
    suffix: 2d_p2_p1_schur_lower
    requires: triangle
    args: -sol quadratic -dm_refine 2 -vel_petscspace_degree 2 -pres_petscspace_degree 1 \
      -snes_error_if_not_converged \
      -ksp_type fgmres -ksp_gmres_restart 100 -ksp_rtol 1.0e-9 -ksp_error_if_not_converged -pc_use_amat \
      -pc_type fieldsplit -pc_fieldsplit_type schur -pc_fieldsplit_schur_factorization_type lower -pc_fieldsplit_off_diag_use_amat \
        -fieldsplit_velocity_pc_type lu -fieldsplit_pressure_ksp_rtol 1e-10 -fieldsplit_pressure_pc_type jacobi
  #   Full Schur complement \begin{pmatrix} I & 0 \\ B^T A^{-1} & I \end{pmatrix} \begin{pmatrix} A & 0 \\ 0 & S \end{pmatrix} \begin{pmatrix} I & A^{-1} B \\ 0 & I \end{pmatrix}
  test:
    suffix: 2d_p2_p1_schur_full
    requires: triangle
    args: -sol quadratic -dm_refine 2 -vel_petscspace_degree 2 -pres_petscspace_degree 1 \
      -snes_error_if_not_converged \
      -ksp_type fgmres -ksp_gmres_restart 100 -ksp_rtol 1.0e-9 -ksp_error_if_not_converged -pc_use_amat \
      -pc_type fieldsplit -pc_fieldsplit_type schur -pc_fieldsplit_schur_factorization_type full -pc_fieldsplit_off_diag_use_amat \
        -fieldsplit_velocity_pc_type lu -fieldsplit_pressure_ksp_rtol 1e-10 -fieldsplit_pressure_pc_type jacobi
  #   Full Schur + Velocity GMG
  test:
    suffix: 2d_p2_p1_gmg_vcycle
    requires: triangle
    args: -sol quadratic -dm_refine_hierarchy 2 -vel_petscspace_degree 2 -pres_petscspace_degree 1 \
      -ksp_type fgmres -ksp_atol 1e-9 -snes_error_if_not_converged -pc_use_amat \
      -pc_type fieldsplit -pc_fieldsplit_type schur -pc_fieldsplit_schur_fact_type full -pc_fieldsplit_off_diag_use_amat \
        -fieldsplit_velocity_pc_type mg -fieldsplit_pressure_ksp_rtol 1e-10 -fieldsplit_pressure_pc_type gamg -fieldsplit_pressure_mg_coarse_pc_type svd
  #   SIMPLE \begin{pmatrix} I & 0 \\ B^T A^{-1} & I \end{pmatrix} \begin{pmatrix} A & 0 \\ 0 & B^T diag(A)^{-1} B \end{pmatrix} \begin{pmatrix} I & diag(A)^{-1} B \\ 0 & I \end{pmatrix}
  test:
    suffix: 2d_p2_p1_simple
    requires: triangle
    args: -sol quadratic -dm_refine 2 -vel_petscspace_degree 2 -pres_petscspace_degree 1 -petscds_jac_pre 0 \
      -snes_error_if_not_converged \
      -ksp_type fgmres -ksp_gmres_restart 100 -ksp_rtol 1.0e-9 -ksp_error_if_not_converged \
      -pc_type fieldsplit -pc_fieldsplit_type schur -pc_fieldsplit_schur_factorization_type full \
        -fieldsplit_velocity_pc_type lu -fieldsplit_pressure_ksp_rtol 1e-10 -fieldsplit_pressure_pc_type jacobi \
        -fieldsplit_pressure_inner_ksp_type preonly -fieldsplit_pressure_inner_pc_type jacobi -fieldsplit_pressure_upper_ksp_type preonly -fieldsplit_pressure_upper_pc_type jacobi
  #   FETI-DP solvers (these solvers are quite inefficient, they are here to exercise the code)
  test:
    suffix: 2d_p2_p1_fetidp
    requires: triangle mumps
    nsize: 5
    args: -sol quadratic -dm_refine 2 -dm_mat_type is -dm_distribute -petscpartitioner_type simple -vel_petscspace_degree 2 -pres_petscspace_degree 1 -petscds_jac_pre 0 \
      -snes_error_if_not_converged \
      -ksp_type fetidp -ksp_rtol 1.0e-8 \
      -ksp_fetidp_saddlepoint -fetidp_ksp_type cg \
        -fetidp_fieldsplit_p_ksp_max_it 1 -fetidp_fieldsplit_p_ksp_type richardson -fetidp_fieldsplit_p_ksp_richardson_scale 200 -fetidp_fieldsplit_p_pc_type none \
        -fetidp_bddc_pc_bddc_dirichlet_pc_factor_mat_solver_type mumps -fetidp_bddc_pc_bddc_neumann_pc_factor_mat_solver_type mumps -fetidp_fieldsplit_lag_ksp_type preonly
  test:
    suffix: 2d_q2_q1_fetidp
    requires: mumps
    nsize: 5
    args: -sol quadratic -dm_plex_simplex 0 -dm_refine 2 -dm_mat_type is -dm_distribute -petscpartitioner_type simple -vel_petscspace_degree 2 -pres_petscspace_degree 1 -petscds_jac_pre 0 \
      -ksp_type fetidp -ksp_rtol 1.0e-8 -ksp_error_if_not_converged \
      -ksp_fetidp_saddlepoint -fetidp_ksp_type cg \
        -fetidp_fieldsplit_p_ksp_max_it 1 -fetidp_fieldsplit_p_ksp_type richardson -fetidp_fieldsplit_p_ksp_richardson_scale 200 -fetidp_fieldsplit_p_pc_type none \
        -fetidp_bddc_pc_bddc_dirichlet_pc_factor_mat_solver_type mumps -fetidp_bddc_pc_bddc_neumann_pc_factor_mat_solver_type mumps -fetidp_fieldsplit_lag_ksp_type preonly
  test:
    suffix: 3d_p2_p1_fetidp
    requires: ctetgen mumps suitesparse
    nsize: 5
    args: -sol quadratic -dm_plex_dim 3 -dm_plex_box_faces 2,2,2 -dm_refine 1 -dm_mat_type is -dm_distribute -petscpartitioner_type simple -vel_petscspace_degree 2 -pres_petscspace_degree 1 -petscds_jac_pre 0 \
      -snes_error_if_not_converged \
      -ksp_type fetidp -ksp_rtol 1.0e-9  \
      -ksp_fetidp_saddlepoint -fetidp_ksp_type cg \
        -fetidp_fieldsplit_p_ksp_max_it 1 -fetidp_fieldsplit_p_ksp_type richardson -fetidp_fieldsplit_p_ksp_richardson_scale 1000 -fetidp_fieldsplit_p_pc_type none \
        -fetidp_bddc_pc_bddc_use_deluxe_scaling -fetidp_bddc_pc_bddc_benign_trick -fetidp_bddc_pc_bddc_deluxe_singlemat \
        -fetidp_pc_discrete_harmonic -fetidp_harmonic_pc_factor_mat_solver_type petsc -fetidp_harmonic_pc_type cholesky \
        -fetidp_bddelta_pc_factor_mat_solver_type umfpack -fetidp_fieldsplit_lag_ksp_type preonly -fetidp_bddc_sub_schurs_mat_solver_type mumps -fetidp_bddc_sub_schurs_mat_mumps_icntl_14 100000 \
        -fetidp_bddelta_pc_factor_mat_ordering_type external \
        -fetidp_bddc_pc_bddc_dirichlet_pc_factor_mat_solver_type umfpack -fetidp_bddc_pc_bddc_neumann_pc_factor_mat_solver_type umfpack \
        -fetidp_bddc_pc_bddc_dirichlet_pc_factor_mat_ordering_type external -fetidp_bddc_pc_bddc_neumann_pc_factor_mat_ordering_type external
  test:
    suffix: 3d_q2_q1_fetidp
    requires: suitesparse
    nsize: 5
    args: -sol quadratic -dm_plex_simplex 0 -dm_plex_dim 3 -dm_plex_box_faces 2,2,2 -dm_refine 1 -dm_mat_type is -dm_distribute -petscpartitioner_type simple -vel_petscspace_degree 2 -pres_petscspace_degree 1 -petscds_jac_pre 0 \
      -snes_error_if_not_converged \
      -ksp_type fetidp -ksp_rtol 1.0e-8 \
      -ksp_fetidp_saddlepoint -fetidp_ksp_type cg \
        -fetidp_fieldsplit_p_ksp_max_it 1 -fetidp_fieldsplit_p_ksp_type richardson -fetidp_fieldsplit_p_ksp_richardson_scale 2000 -fetidp_fieldsplit_p_pc_type none \
        -fetidp_pc_discrete_harmonic -fetidp_harmonic_pc_factor_mat_solver_type petsc -fetidp_harmonic_pc_type cholesky \
        -fetidp_bddc_pc_bddc_symmetric -fetidp_fieldsplit_lag_ksp_type preonly \
        -fetidp_bddc_pc_bddc_dirichlet_pc_factor_mat_solver_type umfpack -fetidp_bddc_pc_bddc_neumann_pc_factor_mat_solver_type umfpack \
        -fetidp_bddc_pc_bddc_dirichlet_pc_factor_mat_ordering_type external -fetidp_bddc_pc_bddc_neumann_pc_factor_mat_ordering_type external
  #   BDDC solvers (these solvers are quite inefficient, they are here to exercise the code)
  test:
    suffix: 2d_p2_p1_bddc
    nsize: 2
    requires: triangle !single
    args: -sol quadratic -dm_plex_box_faces 2,2,2 -dm_refine 1 -dm_mat_type is -dm_distribute -petscpartitioner_type simple -vel_petscspace_degree 2 -pres_petscspace_degree 1 -petscds_jac_pre 0 \
      -snes_error_if_not_converged \
      -ksp_type gmres -ksp_gmres_restart 100 -ksp_rtol 1.0e-8 -ksp_error_if_not_converged \
        -pc_type bddc -pc_bddc_corner_selection -pc_bddc_dirichlet_pc_type svd -pc_bddc_neumann_pc_type svd -pc_bddc_coarse_redundant_pc_type svd
  #   Vanka
  test:
    suffix: 2d_q1_p0_vanka
    requires: double !complex
    args: -sol quadratic -dm_plex_simplex 0 -dm_refine 2 -vel_petscspace_degree 1 -pres_petscspace_degree 0 -petscds_jac_pre 0 \
      -snes_rtol 1.0e-4 \
      -ksp_type fgmres -ksp_atol 1e-5 -ksp_error_if_not_converged \
      -pc_type patch -pc_patch_partition_of_unity 0 -pc_patch_construct_codim 0 -pc_patch_construct_type vanka \
        -sub_ksp_type preonly -sub_pc_type lu
  test:
    suffix: 2d_q1_p0_vanka_denseinv
    requires: double !complex
    args: -sol quadratic -dm_plex_simplex 0 -dm_refine 2 -vel_petscspace_degree 1 -pres_petscspace_degree 0 -petscds_jac_pre 0 \
      -snes_rtol 1.0e-4 \
      -ksp_type fgmres -ksp_atol 1e-5 -ksp_error_if_not_converged \
      -pc_type patch -pc_patch_partition_of_unity 0 -pc_patch_construct_codim 0 -pc_patch_construct_type vanka \
        -pc_patch_dense_inverse -pc_patch_sub_mat_type seqdense
  #   Vanka smoother
  test:
    suffix: 2d_q1_p0_gmg_vanka
    requires: double !complex
    args: -sol quadratic -dm_plex_simplex 0 -dm_refine_hierarchy 2 -vel_petscspace_degree 1 -pres_petscspace_degree 0 -petscds_jac_pre 0 \
      -snes_rtol 1.0e-4 \
      -ksp_type fgmres -ksp_atol 1e-5 -ksp_error_if_not_converged \
      -pc_type mg \
        -mg_levels_ksp_type gmres -mg_levels_ksp_max_it 30 \
        -mg_levels_pc_type patch -mg_levels_pc_patch_partition_of_unity 0 -mg_levels_pc_patch_construct_codim 0 -mg_levels_pc_patch_construct_type vanka \
          -mg_levels_sub_ksp_type preonly -mg_levels_sub_pc_type lu \
        -mg_coarse_pc_type svd

TEST*/
