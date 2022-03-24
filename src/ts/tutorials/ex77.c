static char help[] = "Time-dependent reactive low Mach Flow in 2d and 3d channels with finite elements.\n\
We solve the reactive low Mach flow problem in a rectangular domain\n\
using a parallel unstructured mesh (DMPLEX) to discretize the flow\n\
and particles (DWSWARM) to discretize the chemical species.\n\n\n";

/*F
This low Mach flow is time-dependent isoviscous Navier-Stokes flow. We discretize using the
finite element method on an unstructured mesh. The weak form equations are

\begin{align*}
    < q, \nabla\cdot u > = 0
    <v, du/dt> + <v, u \cdot \nabla u> + < \nabla v, \nu (\nabla u + {\nabla u}^T) > - < \nabla\cdot v, p >  - < v, f  >  = 0
    < w, u \cdot \nabla T > + < \nabla w, \alpha \nabla T > - < w, Q > = 0
\end{align*}

where $\nu$ is the kinematic viscosity and $\alpha$ is thermal diffusivity.

For visualization, use

  -dm_view hdf5:$PWD/sol.h5 -sol_vec_view hdf5:$PWD/sol.h5::append -exact_vec_view hdf5:$PWD/sol.h5::append

The particles can be visualized using

  -part_dm_view draw -part_dm_view_swarm_radius 0.03

F*/

#include <petscdmplex.h>
#include <petscdmswarm.h>
#include <petscts.h>
#include <petscds.h>
#include <petscbag.h>

typedef enum {SOL_TRIG_TRIG, NUM_SOL_TYPES} SolType;
const char *solTypes[NUM_SOL_TYPES+1] = {"trig_trig",  "unknown"};

typedef enum {PART_LAYOUT_CELL, PART_LAYOUT_BOX, NUM_PART_LAYOUT_TYPES} PartLayoutType;
const char *partLayoutTypes[NUM_PART_LAYOUT_TYPES+1] = {"cell", "box",  "unknown"};

typedef struct {
  PetscReal nu;    /* Kinematic viscosity */
  PetscReal alpha; /* Thermal diffusivity */
  PetscReal T_in;  /* Inlet temperature*/
  PetscReal omega; /* Rotation speed in MMS benchmark */
} Parameter;

typedef struct {
  /* Problem definition */
  PetscBag       bag;          /* Holds problem parameters */
  SolType        solType;      /* MMS solution type */
  PartLayoutType partLayout;   /* Type of particle distribution */
  PetscInt       Npc;          /* The initial number of particles per cell */
  PetscReal      partLower[3]; /* Lower left corner of particle box */
  PetscReal      partUpper[3]; /* Upper right corner of particle box */
  PetscInt       Npb;          /* The initial number of particles per box dimension */
} AppCtx;

typedef struct {
  PetscReal ti; /* The time for ui, at the beginning of the advection solve */
  PetscReal tf; /* The time for uf, at the end of the advection solve */
  Vec       ui; /* The PDE solution field at ti */
  Vec       uf; /* The PDE solution field at tf */
  Vec       x0; /* The initial particle positions at t = 0 */
  PetscErrorCode (*exact)(PetscInt, PetscReal, const PetscReal[], PetscInt, PetscScalar *, void *);
  AppCtx   *ctx; /* Context for exact solution */
} AdvCtx;

static PetscErrorCode zero(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nc, PetscScalar *u, void *ctx)
{
  PetscInt d;
  for (d = 0; d < Nc; ++d) u[d] = 0.0;
  return 0;
}

static PetscErrorCode constant(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nc, PetscScalar *u, void *ctx)
{
  PetscInt d;
  for (d = 0; d < Nc; ++d) u[d] = 1.0;
  return 0;
}

/*
  CASE: trigonometric-trigonometric
  In 2D we use exact solution:

    x = r0 cos(w t + theta0)  r0     = sqrt(x0^2 + y0^2)
    y = r0 sin(w t + theta0)  theta0 = arctan(y0/x0)
    u = -w r0 sin(theta0) = -w y
    v =  w r0 cos(theta0) =  w x
    p = x + y - 1
    T = t + x + y
    f = <1, 1>
    Q = 1 + w (x - y)/r

  so that

    \nabla \cdot u = 0 + 0 = 0

  f = du/dt + u \cdot \nabla u - \nu \Delta u + \nabla p
    = <0, 0> + u_i d_i u_j - \nu 0 + <1, 1>
    = <1, 1> + w^2 <-y, x> . <<0, 1>, <-1, 0>>
    = <1, 1> + w^2 <-x, -y>
    = <1, 1> - w^2 <x, y>

  Q = dT/dt + u \cdot \nabla T - \alpha \Delta T
    = 1 + <u, v> . <1, 1> - \alpha 0
    = 1 + u + v
*/
static PetscErrorCode trig_trig_x(PetscInt dim, PetscReal time, const PetscReal X[], PetscInt Nf, PetscScalar *x, void *ctx)
{
  const PetscReal x0     = X[0];
  const PetscReal y0     = X[1];
  const PetscReal R0     = PetscSqrtReal(x0*x0 + y0*y0);
  const PetscReal theta0 = PetscAtan2Real(y0, x0);
  Parameter      *p      = (Parameter *) ctx;

  x[0] = R0*PetscCosReal(p->omega*time + theta0);
  x[1] = R0*PetscSinReal(p->omega*time + theta0);
  return 0;
}
static PetscErrorCode trig_trig_u(PetscInt dim, PetscReal time, const PetscReal X[], PetscInt Nf, PetscScalar *u, void *ctx)
{
  Parameter *p = (Parameter *) ctx;

  u[0] = -p->omega*X[1];
  u[1] =  p->omega*X[0];
  return 0;
}
static PetscErrorCode trig_trig_u_t(PetscInt dim, PetscReal time, const PetscReal X[], PetscInt Nf, PetscScalar *u, void *ctx)
{
  u[0] = 0.0;
  u[1] = 0.0;
  return 0;
}

static PetscErrorCode trig_trig_p(PetscInt dim, PetscReal time, const PetscReal X[], PetscInt Nf, PetscScalar *p, void *ctx)
{
  p[0] = X[0] + X[1] - 1.0;
  return 0;
}

static PetscErrorCode trig_trig_T(PetscInt dim, PetscReal time, const PetscReal X[], PetscInt Nf, PetscScalar *T, void *ctx)
{
  T[0] = time + X[0] + X[1];
  return 0;
}
static PetscErrorCode trig_trig_T_t(PetscInt dim, PetscReal time, const PetscReal X[], PetscInt Nf, PetscScalar *T, void *ctx)
{
  T[0] = 1.0;
  return 0;
}

static void f0_trig_trig_v(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                           const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                           const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                           PetscReal t, const PetscReal X[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f0[])
{
  const PetscReal omega = PetscRealPart(constants[3]);
  PetscInt        Nc    = dim;
  PetscInt        c, d;

  for (d = 0; d < dim; ++d) f0[d] = u_t[uOff[0]+d];

  for (c = 0; c < Nc; ++c) {
    for (d = 0; d < dim; ++d) f0[c] += u[d]*u_x[c*dim+d];
  }
  f0[0] -= 1.0 - omega*omega*X[0];
  f0[1] -= 1.0 - omega*omega*X[1];
}

static void f0_trig_trig_w(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                           const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                           const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                           PetscReal t, const PetscReal X[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f0[])
{
  const PetscReal omega = PetscRealPart(constants[3]);
  PetscInt        d;

  for (d = 0, f0[0] = 0; d < dim; ++d) f0[0] += u[uOff[0]+d]*u_x[uOff_x[2]+d];
  f0[0] += u_t[uOff[2]] - (1.0 + omega*(X[0] - X[1]));
}

static void f0_q(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                 const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                 const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                 PetscReal t, const PetscReal X[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f0[])
{
  PetscInt d;
  for (d = 0, f0[0] = 0.0; d < dim; ++d) f0[0] += u_x[d*dim+d];
}

/*f1_v = \nu[grad(u) + grad(u)^T] - pI */
static void f1_v(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                 const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                 const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                 PetscReal t, const PetscReal X[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f1[])
{
  const PetscReal nu = PetscRealPart(constants[0]);
  const PetscInt    Nc = dim;
  PetscInt        c, d;

  for (c = 0; c < Nc; ++c) {
    for (d = 0; d < dim; ++d) {
      f1[c*dim+d] = nu*(u_x[c*dim+d] + u_x[d*dim+c]);
    }
    f1[c*dim+c] -= u[uOff[1]];
  }
}

static void f1_w(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                 const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                 const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                 PetscReal t, const PetscReal X[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f1[])
{
  const PetscReal alpha = PetscRealPart(constants[1]);
  PetscInt d;
  for (d = 0; d < dim; ++d) f1[d] = alpha*u_x[uOff_x[2]+d];
}

/*Jacobians*/
static void g1_qu(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                 const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                 const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                 PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g1[])
{
  PetscInt d;
  for (d = 0; d < dim; ++d) g1[d*dim+d] = 1.0;
}

static void g0_vu(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                  const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                  const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                  PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g0[])
{
  PetscInt c, d;
  const PetscInt  Nc = dim;

  for (d = 0; d < dim; ++d) g0[d*dim+d] = u_tShift;

  for (c = 0; c < Nc; ++c) {
    for (d = 0; d < dim; ++d) {
      g0[c*Nc+d] += u_x[c*Nc+d];
    }
  }
}

static void g1_vu(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                  const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                  const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                  PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g1[])
{
  PetscInt NcI = dim;
  PetscInt NcJ = dim;
  PetscInt c, d, e;

  for (c = 0; c < NcI; ++c) {
    for (d = 0; d < NcJ; ++d) {
      for (e = 0; e < dim; ++e) {
        if (c == d) {
          g1[(c*NcJ+d)*dim+e] += u[e];
        }
      }
    }
  }
}

static void g2_vp(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                  const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                  const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                  PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g2[])
{
  PetscInt d;
  for (d = 0; d < dim; ++d) g2[d*dim+d] = -1.0;
}

static void g3_vu(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                  const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                  const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                  PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g3[])
{
   const PetscReal nu = PetscRealPart(constants[0]);
   const PetscInt  Nc = dim;
   PetscInt        c, d;

  for (c = 0; c < Nc; ++c) {
    for (d = 0; d < dim; ++d) {
      g3[((c*Nc+c)*dim+d)*dim+d] += nu; // gradU
      g3[((c*Nc+d)*dim+d)*dim+c] += nu; // gradU transpose
    }
  }
}

static void g0_wT(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                  const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                  const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                  PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g0[])
{
  PetscInt d;
  for (d = 0; d < dim; ++d) g0[d] = u_tShift;
}

static void g0_wu(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                  const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                  const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                  PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g0[])
{
  PetscInt d;
  for (d = 0; d < dim; ++d) g0[d] = u_x[uOff_x[2]+d];
}

static void g1_wT(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                  const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                  const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                  PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g1[])
{
  PetscInt d;
  for (d = 0; d < dim; ++d) g1[d] = u[uOff[0]+d];
}

static void g3_wT(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                  const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                  const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                  PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g3[])
{
  const PetscReal alpha = PetscRealPart(constants[1]);
  PetscInt               d;

  for (d = 0; d < dim; ++d) g3[d*dim+d] = alpha;
}

static PetscErrorCode ProcessOptions(MPI_Comm comm, AppCtx *options)
{
  PetscInt       sol, pl, n;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  options->solType    = SOL_TRIG_TRIG;
  options->partLayout = PART_LAYOUT_CELL;
  options->Npc        = 1;
  options->Npb        = 1;

  options->partLower[0] = options->partLower[1] = options->partLower[2] = 0.;
  options->partUpper[0] = options->partUpper[1] = options->partUpper[2] = 1.;
  ierr = PetscOptionsBegin(comm, "", "Low Mach flow Problem Options", "DMPLEX");CHKERRQ(ierr);
  sol  = options->solType;
  CHKERRQ(PetscOptionsEList("-sol_type", "The solution type", "ex77.c", solTypes, NUM_SOL_TYPES, solTypes[options->solType], &sol, NULL));
  options->solType = (SolType) sol;
  pl   = options->partLayout;
  CHKERRQ(PetscOptionsEList("-part_layout_type", "The particle layout type", "ex77.c", partLayoutTypes, NUM_PART_LAYOUT_TYPES, partLayoutTypes[options->partLayout], &pl, NULL));
  options->partLayout = (PartLayoutType) pl;
  CHKERRQ(PetscOptionsInt("-Npc", "The initial number of particles per cell", "ex77.c", options->Npc, &options->Npc, NULL));
  n    = 3;
  CHKERRQ(PetscOptionsRealArray("-part_lower", "The lower left corner of the particle box", "ex77.c", options->partLower, &n, NULL));
  n    = 3;
  CHKERRQ(PetscOptionsRealArray("-part_upper", "The upper right corner of the particle box", "ex77.c", options->partUpper, &n, NULL));
  CHKERRQ(PetscOptionsInt("-Npb", "The initial number of particles per box dimension", "ex77.c", options->Npb, &options->Npb, NULL));
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode SetupParameters(AppCtx *user)
{
  PetscBag       bag;
  Parameter     *p;

  PetscFunctionBeginUser;
  /* setup PETSc parameter bag */
  CHKERRQ(PetscBagGetData(user->bag, (void **) &p));
  CHKERRQ(PetscBagSetName(user->bag, "par", "Low Mach flow parameters"));
  bag  = user->bag;
  CHKERRQ(PetscBagRegisterReal(bag, &p->nu,    1.0, "nu",    "Kinematic viscosity"));
  CHKERRQ(PetscBagRegisterReal(bag, &p->alpha, 1.0, "alpha", "Thermal diffusivity"));
  CHKERRQ(PetscBagRegisterReal(bag, &p->T_in,  1.0, "T_in",  "Inlet temperature"));
  CHKERRQ(PetscBagRegisterReal(bag, &p->omega, 1.0, "omega", "Rotation speed in MMS benchmark"));
  PetscFunctionReturn(0);
}

static PetscErrorCode CreateMesh(MPI_Comm comm, AppCtx *user, DM *dm)
{
  PetscFunctionBeginUser;
  CHKERRQ(DMCreate(comm, dm));
  CHKERRQ(DMSetType(*dm, DMPLEX));
  CHKERRQ(DMSetFromOptions(*dm));
  CHKERRQ(DMViewFromOptions(*dm, NULL, "-dm_view"));
  PetscFunctionReturn(0);
}

static PetscErrorCode SetupProblem(DM dm, AppCtx *user)
{
  PetscErrorCode (*exactFuncs[3])(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nf, PetscScalar *u, void *ctx);
  PetscErrorCode (*exactFuncs_t[3])(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nf, PetscScalar *u, void *ctx);
  PetscDS          prob;
  DMLabel          label;
  Parameter       *ctx;
  PetscInt         id;

  PetscFunctionBeginUser;
  CHKERRQ(DMGetLabel(dm, "marker", &label));
  CHKERRQ(DMGetDS(dm, &prob));
  switch(user->solType) {
  case SOL_TRIG_TRIG:
    CHKERRQ(PetscDSSetResidual(prob, 0, f0_trig_trig_v, f1_v));
    CHKERRQ(PetscDSSetResidual(prob, 2, f0_trig_trig_w, f1_w));

    exactFuncs[0]   = trig_trig_u;
    exactFuncs[1]   = trig_trig_p;
    exactFuncs[2]   = trig_trig_T;
    exactFuncs_t[0] = trig_trig_u_t;
    exactFuncs_t[1] = NULL;
    exactFuncs_t[2] = trig_trig_T_t;
    break;
   default: SETERRQ(PetscObjectComm((PetscObject) prob), PETSC_ERR_ARG_WRONG, "Unsupported solution type: %s (%D)", solTypes[PetscMin(user->solType, NUM_SOL_TYPES)], user->solType);
  }

  CHKERRQ(PetscDSSetResidual(prob, 1, f0_q, NULL));

  CHKERRQ(PetscDSSetJacobian(prob, 0, 0, g0_vu, g1_vu,  NULL,  g3_vu));
  CHKERRQ(PetscDSSetJacobian(prob, 0, 1, NULL, NULL,  g2_vp, NULL));
  CHKERRQ(PetscDSSetJacobian(prob, 1, 0, NULL, g1_qu, NULL,  NULL));
  CHKERRQ(PetscDSSetJacobian(prob, 2, 0, g0_wu, NULL, NULL,  NULL));
  CHKERRQ(PetscDSSetJacobian(prob, 2, 2, g0_wT, g1_wT, NULL,  g3_wT));
  /* Setup constants */
  {
    Parameter  *param;
    PetscScalar constants[4];

    CHKERRQ(PetscBagGetData(user->bag, (void **) &param));

    constants[0] = param->nu;
    constants[1] = param->alpha;
    constants[2] = param->T_in;
    constants[3] = param->omega;
    CHKERRQ(PetscDSSetConstants(prob, 4, constants));
  }
  /* Setup Boundary Conditions */
  CHKERRQ(PetscBagGetData(user->bag, (void **) &ctx));
  id   = 3;
  CHKERRQ(PetscDSAddBoundary(prob, DM_BC_ESSENTIAL, "top wall velocity",    label, 1, &id, 0, 0, NULL, (void (*)(void)) exactFuncs[0], (void (*)(void)) exactFuncs_t[0], ctx, NULL));
  id   = 1;
  CHKERRQ(PetscDSAddBoundary(prob, DM_BC_ESSENTIAL, "bottom wall velocity", label, 1, &id, 0, 0, NULL, (void (*)(void)) exactFuncs[0], (void (*)(void)) exactFuncs_t[0], ctx, NULL));
  id   = 2;
  CHKERRQ(PetscDSAddBoundary(prob, DM_BC_ESSENTIAL, "right wall velocity",  label, 1, &id, 0, 0, NULL, (void (*)(void)) exactFuncs[0], (void (*)(void)) exactFuncs_t[0], ctx, NULL));
  id   = 4;
  CHKERRQ(PetscDSAddBoundary(prob, DM_BC_ESSENTIAL, "left wall velocity",   label, 1, &id, 0, 0, NULL, (void (*)(void)) exactFuncs[0], (void (*)(void)) exactFuncs_t[0], ctx, NULL));
  id   = 3;
  CHKERRQ(PetscDSAddBoundary(prob, DM_BC_ESSENTIAL, "top wall temp",    label, 1, &id, 2, 0, NULL, (void (*)(void)) exactFuncs[2], (void (*)(void)) exactFuncs_t[2], ctx, NULL));
  id   = 1;
  CHKERRQ(PetscDSAddBoundary(prob, DM_BC_ESSENTIAL, "bottom wall temp", label, 1, &id, 2, 0, NULL, (void (*)(void)) exactFuncs[2], (void (*)(void)) exactFuncs_t[2], ctx, NULL));
  id   = 2;
  CHKERRQ(PetscDSAddBoundary(prob, DM_BC_ESSENTIAL, "right wall temp",  label, 1, &id, 2, 0, NULL, (void (*)(void)) exactFuncs[2], (void (*)(void)) exactFuncs_t[2], ctx, NULL));
  id   = 4;
  CHKERRQ(PetscDSAddBoundary(prob, DM_BC_ESSENTIAL, "left wall temp",   label, 1, &id, 2, 0, NULL, (void (*)(void)) exactFuncs[2], (void (*)(void)) exactFuncs_t[2], ctx, NULL));

  /*setup exact solution.*/
  CHKERRQ(PetscDSSetExactSolution(prob, 0, exactFuncs[0], ctx));
  CHKERRQ(PetscDSSetExactSolution(prob, 1, exactFuncs[1], ctx));
  CHKERRQ(PetscDSSetExactSolution(prob, 2, exactFuncs[2], ctx));
  CHKERRQ(PetscDSSetExactSolutionTimeDerivative(prob, 0, exactFuncs_t[0], ctx));
  CHKERRQ(PetscDSSetExactSolutionTimeDerivative(prob, 1, exactFuncs_t[1], ctx));
  CHKERRQ(PetscDSSetExactSolutionTimeDerivative(prob, 2, exactFuncs_t[2], ctx));
  PetscFunctionReturn(0);
}

/* x_t = v

   Note that here we use the velocity field at t_{n+1} to advect the particles from
   t_n to t_{n+1}. If we use both of these fields, we could use Crank-Nicholson or
   the method of characteristics.
*/
static PetscErrorCode FreeStreaming(TS ts, PetscReal t, Vec X, Vec F, void *ctx)
{
  AdvCtx             *adv = (AdvCtx *) ctx;
  Vec                 u   = adv->ui;
  DM                  sdm, dm, vdm;
  Vec                 vel, locvel, pvel;
  IS                  vis;
  DMInterpolationInfo ictx;
  const PetscScalar  *coords, *v;
  PetscScalar        *f;
  PetscInt            vf[1] = {0};
  PetscInt            dim, Np;

  PetscFunctionBeginUser;
  CHKERRQ(TSGetDM(ts, &sdm));
  CHKERRQ(DMSwarmGetCellDM(sdm, &dm));
  CHKERRQ(DMGetGlobalVector(sdm, &pvel));
  CHKERRQ(DMSwarmGetLocalSize(sdm, &Np));
  CHKERRQ(DMGetDimension(dm, &dim));
  /* Get local velocity */
  CHKERRQ(DMCreateSubDM(dm, 1, vf, &vis, &vdm));
  CHKERRQ(VecGetSubVector(u, vis, &vel));
  CHKERRQ(DMGetLocalVector(vdm, &locvel));
  CHKERRQ(DMPlexInsertBoundaryValues(vdm, PETSC_TRUE, locvel, adv->ti, NULL, NULL, NULL));
  CHKERRQ(DMGlobalToLocalBegin(vdm, vel, INSERT_VALUES, locvel));
  CHKERRQ(DMGlobalToLocalEnd(vdm, vel, INSERT_VALUES, locvel));
  CHKERRQ(VecRestoreSubVector(u, vis, &vel));
  CHKERRQ(ISDestroy(&vis));
  /* Interpolate velocity */
  CHKERRQ(DMInterpolationCreate(PETSC_COMM_SELF, &ictx));
  CHKERRQ(DMInterpolationSetDim(ictx, dim));
  CHKERRQ(DMInterpolationSetDof(ictx, dim));
  CHKERRQ(VecGetArrayRead(X, &coords));
  CHKERRQ(DMInterpolationAddPoints(ictx, Np, (PetscReal *) coords));
  CHKERRQ(VecRestoreArrayRead(X, &coords));
  /* Particles that lie outside the domain should be dropped,
     whereas particles that move to another partition should trigger a migration */
  CHKERRQ(DMInterpolationSetUp(ictx, vdm, PETSC_FALSE, PETSC_TRUE));
  CHKERRQ(VecSet(pvel, 0.));
  CHKERRQ(DMInterpolationEvaluate(ictx, vdm, locvel, pvel));
  CHKERRQ(DMInterpolationDestroy(&ictx));
  CHKERRQ(DMRestoreLocalVector(vdm, &locvel));
  CHKERRQ(DMDestroy(&vdm));

  CHKERRQ(VecGetArray(F, &f));
  CHKERRQ(VecGetArrayRead(pvel, &v));
  CHKERRQ(PetscArraycpy(f, v, Np*dim));
  CHKERRQ(VecRestoreArrayRead(pvel, &v));
  CHKERRQ(VecRestoreArray(F, &f));
  CHKERRQ(DMRestoreGlobalVector(sdm, &pvel));
  PetscFunctionReturn(0);
}

static PetscErrorCode SetInitialParticleConditions(TS ts, Vec u)
{
  AppCtx        *user;
  void          *ctx;
  DM             dm;
  PetscScalar   *coords;
  PetscReal      x[3], dx[3];
  PetscInt       n[3];
  PetscInt       dim, d, i, j, k;

  PetscFunctionBegin;
  CHKERRQ(TSGetApplicationContext(ts, &ctx));
  user = ((AdvCtx *) ctx)->ctx;
  CHKERRQ(TSGetDM(ts, &dm));
  CHKERRQ(DMGetDimension(dm, &dim));
  switch (user->partLayout) {
    case PART_LAYOUT_CELL:
      CHKERRQ(DMSwarmSetPointCoordinatesRandom(dm, user->Npc));
      break;
    case PART_LAYOUT_BOX:
      for (d = 0; d < dim; ++d) {
        n[d]  = user->Npb;
        dx[d] = (user->partUpper[d] - user->partLower[d])/PetscMax(1, n[d] - 1);
      }
      CHKERRQ(VecGetArray(u, &coords));
      switch (dim) {
        case 2:
          x[0] = user->partLower[0];
          for (i = 0; i < n[0]; ++i, x[0] += dx[0]) {
            x[1] = user->partLower[1];
            for (j = 0; j < n[1]; ++j, x[1] += dx[1]) {
              const PetscInt p = j*n[0] + i;
              for (d = 0; d < dim; ++d) coords[p*dim + d] = x[d];
            }
          }
          break;
        case 3:
          x[0] = user->partLower[0];
          for (i = 0; i < n[0]; ++i, x[0] += dx[0]) {
            x[1] = user->partLower[1];
            for (j = 0; j < n[1]; ++j, x[1] += dx[1]) {
              x[2] = user->partLower[2];
              for (k = 0; k < n[2]; ++k, x[2] += dx[2]) {
                const PetscInt p = (k*n[1] + j)*n[0] + i;
                for (d = 0; d < dim; ++d) coords[p*dim + d] = x[d];
              }
            }
          }
          break;
        default: SETERRQ(PetscObjectComm((PetscObject) ts), PETSC_ERR_SUP, "Do not support particle layout in dimension %D", dim);
      }
      CHKERRQ(VecRestoreArray(u, &coords));
      break;
    default: SETERRQ(PetscObjectComm((PetscObject) ts), PETSC_ERR_ARG_WRONG, "Invalid particle layout type %s", partLayoutTypes[PetscMin(user->partLayout, NUM_PART_LAYOUT_TYPES)]);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode SetupDiscretization(DM dm, DM sdm, AppCtx *user)
{
  DM              cdm = dm;
  PetscFE         fe[3];
  Parameter      *param;
  PetscInt       *cellid, n[3];
  PetscReal       x[3], dx[3];
  PetscScalar    *coords;
  DMPolytopeType  ct;
  PetscInt        dim, d, cStart, cEnd, c, Np, p, i, j, k;
  PetscBool       simplex;
  MPI_Comm        comm;

  PetscFunctionBeginUser;
  CHKERRQ(DMGetDimension(dm, &dim));
  CHKERRQ(DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd));
  CHKERRQ(DMPlexGetCellType(dm, cStart, &ct));
  simplex = DMPolytopeTypeGetNumVertices(ct) == DMPolytopeTypeGetDim(ct)+1 ? PETSC_TRUE : PETSC_FALSE;
  /* Create finite element */
  CHKERRQ(PetscObjectGetComm((PetscObject) dm, &comm));
  CHKERRQ(PetscFECreateDefault(comm, dim, dim, simplex, "vel_", PETSC_DEFAULT, &fe[0]));
  CHKERRQ(PetscObjectSetName((PetscObject) fe[0], "velocity"));

  CHKERRQ(PetscFECreateDefault(comm, dim, 1, simplex, "pres_", PETSC_DEFAULT, &fe[1]));
  CHKERRQ(PetscFECopyQuadrature(fe[0], fe[1]));
  CHKERRQ(PetscObjectSetName((PetscObject) fe[1], "pressure"));

  CHKERRQ(PetscFECreateDefault(comm, dim, 1, simplex, "temp_", PETSC_DEFAULT, &fe[2]));
  CHKERRQ(PetscFECopyQuadrature(fe[0], fe[2]));
  CHKERRQ(PetscObjectSetName((PetscObject) fe[2], "temperature"));

  /* Set discretization and boundary conditions for each mesh */
  CHKERRQ(DMSetField(dm, 0, NULL, (PetscObject) fe[0]));
  CHKERRQ(DMSetField(dm, 1, NULL, (PetscObject) fe[1]));
  CHKERRQ(DMSetField(dm, 2, NULL, (PetscObject) fe[2]));
  CHKERRQ(DMCreateDS(dm));
  CHKERRQ(SetupProblem(dm, user));
  CHKERRQ(PetscBagGetData(user->bag, (void **) &param));
  while (cdm) {
    CHKERRQ(DMCopyDisc(dm, cdm));
    CHKERRQ(DMGetCoarseDM(cdm, &cdm));
  }
  CHKERRQ(PetscFEDestroy(&fe[0]));
  CHKERRQ(PetscFEDestroy(&fe[1]));
  CHKERRQ(PetscFEDestroy(&fe[2]));

  {
    PetscObject  pressure;
    MatNullSpace nullspacePres;

    CHKERRQ(DMGetField(dm, 1, NULL, &pressure));
    CHKERRQ(MatNullSpaceCreate(PetscObjectComm(pressure), PETSC_TRUE, 0, NULL, &nullspacePres));
    CHKERRQ(PetscObjectCompose(pressure, "nullspace", (PetscObject) nullspacePres));
    CHKERRQ(MatNullSpaceDestroy(&nullspacePres));
  }

  /* Setup particle information */
  CHKERRQ(DMSwarmSetType(sdm, DMSWARM_PIC));
  CHKERRQ(DMSwarmRegisterPetscDatatypeField(sdm, "mass", 1, PETSC_REAL));
  CHKERRQ(DMSwarmFinalizeFieldRegister(sdm));
  switch (user->partLayout) {
    case PART_LAYOUT_CELL:
      CHKERRQ(DMSwarmSetLocalSizes(sdm, (cEnd - cStart) * user->Npc, 0));
      CHKERRQ(DMSetFromOptions(sdm));
      CHKERRQ(DMSwarmGetField(sdm, DMSwarmPICField_cellid, NULL, NULL, (void **) &cellid));
      for (c = cStart; c < cEnd; ++c) {
        for (p = 0; p < user->Npc; ++p) {
          const PetscInt n = c*user->Npc + p;

          cellid[n] = c;
        }
      }
      CHKERRQ(DMSwarmRestoreField(sdm, DMSwarmPICField_cellid, NULL, NULL, (void **) &cellid));
      CHKERRQ(DMSwarmSetPointCoordinatesRandom(sdm, user->Npc));
      break;
    case PART_LAYOUT_BOX:
      Np = 1;
      for (d = 0; d < dim; ++d) {
        n[d]  = user->Npb;
        dx[d] = (user->partUpper[d] - user->partLower[d])/PetscMax(1, n[d] - 1);
        Np   *= n[d];
      }
      CHKERRQ(DMSwarmSetLocalSizes(sdm, Np, 0));
      CHKERRQ(DMSetFromOptions(sdm));
      CHKERRQ(DMSwarmGetField(sdm, DMSwarmPICField_coor, NULL, NULL, (void **) &coords));
      switch (dim) {
        case 2:
          x[0] = user->partLower[0];
          for (i = 0; i < n[0]; ++i, x[0] += dx[0]) {
            x[1] = user->partLower[1];
            for (j = 0; j < n[1]; ++j, x[1] += dx[1]) {
              const PetscInt p = j*n[0] + i;
              for (d = 0; d < dim; ++d) coords[p*dim + d] = x[d];
            }
          }
          break;
        case 3:
          x[0] = user->partLower[0];
          for (i = 0; i < n[0]; ++i, x[0] += dx[0]) {
            x[1] = user->partLower[1];
            for (j = 0; j < n[1]; ++j, x[1] += dx[1]) {
              x[2] = user->partLower[2];
              for (k = 0; k < n[2]; ++k, x[2] += dx[2]) {
                const PetscInt p = (k*n[1] + j)*n[0] + i;
                for (d = 0; d < dim; ++d) coords[p*dim + d] = x[d];
              }
            }
          }
          break;
        default: SETERRQ(comm, PETSC_ERR_SUP, "Do not support particle layout in dimension %D", dim);
      }
      CHKERRQ(DMSwarmRestoreField(sdm, DMSwarmPICField_coor, NULL, NULL, (void **) &coords));
      CHKERRQ(DMSwarmGetField(sdm, DMSwarmPICField_cellid, NULL, NULL, (void **) &cellid));
      for (p = 0; p < Np; ++p) cellid[p] = 0;
      CHKERRQ(DMSwarmRestoreField(sdm, DMSwarmPICField_cellid, NULL, NULL, (void **) &cellid));
      CHKERRQ(DMSwarmMigrate(sdm, PETSC_TRUE));
      break;
    default: SETERRQ(comm, PETSC_ERR_ARG_WRONG, "Invalid particle layout type %s", partLayoutTypes[PetscMin(user->partLayout, NUM_PART_LAYOUT_TYPES)]);
  }
  CHKERRQ(PetscObjectSetName((PetscObject) sdm, "Particles"));
  CHKERRQ(DMViewFromOptions(sdm, NULL, "-dm_view"));
  PetscFunctionReturn(0);
}

static PetscErrorCode CreatePressureNullSpace(DM dm, PetscInt ofield, PetscInt nfield, MatNullSpace *nullSpace)
{
  Vec              vec;
  PetscErrorCode (*funcs[3])(PetscInt, PetscReal, const PetscReal[], PetscInt, PetscScalar *, void *) = {zero, zero, zero};

  PetscFunctionBeginUser;
  PetscCheck(ofield == 1,PetscObjectComm((PetscObject) dm), PETSC_ERR_ARG_WRONG, "Nullspace must be for pressure field at index 1, not %D", ofield);
  funcs[nfield] = constant;
  CHKERRQ(DMCreateGlobalVector(dm, &vec));
  CHKERRQ(DMProjectFunction(dm, 0.0, funcs, NULL, INSERT_ALL_VALUES, vec));
  CHKERRQ(VecNormalize(vec, NULL));
  CHKERRQ(PetscObjectSetName((PetscObject) vec, "Pressure Null Space"));
  CHKERRQ(VecViewFromOptions(vec, NULL, "-pressure_nullspace_view"));
  CHKERRQ(MatNullSpaceCreate(PetscObjectComm((PetscObject) dm), PETSC_FALSE, 1, &vec, nullSpace));
  CHKERRQ(VecDestroy(&vec));
  PetscFunctionReturn(0);
}

static PetscErrorCode RemoveDiscretePressureNullspace_Private(TS ts, Vec u)
{
  DM             dm;
  MatNullSpace   nullsp;

  PetscFunctionBegin;
  CHKERRQ(TSGetDM(ts, &dm));
  CHKERRQ(CreatePressureNullSpace(dm, 1, 1, &nullsp));
  CHKERRQ(MatNullSpaceRemove(nullsp, u));
  CHKERRQ(MatNullSpaceDestroy(&nullsp));
  PetscFunctionReturn(0);
}

/* Make the discrete pressure discretely divergence free */
static PetscErrorCode RemoveDiscretePressureNullspace(TS ts)
{
  Vec            u;

  PetscFunctionBegin;
  CHKERRQ(TSGetSolution(ts, &u));
  CHKERRQ(RemoveDiscretePressureNullspace_Private(ts, u));
  PetscFunctionReturn(0);
}

static PetscErrorCode SetInitialConditions(TS ts, Vec u)
{
  DM             dm;
  PetscReal      t;

  PetscFunctionBegin;
  CHKERRQ(TSGetDM(ts, &dm));
  CHKERRQ(TSGetTime(ts, &t));
  CHKERRQ(DMComputeExactSolution(dm, t, u, NULL));
  CHKERRQ(RemoveDiscretePressureNullspace_Private(ts, u));
  PetscFunctionReturn(0);
}

static PetscErrorCode MonitorError(TS ts, PetscInt step, PetscReal crtime, Vec u, void *ctx)
{
  PetscErrorCode (*exactFuncs[3])(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nf, PetscScalar *u, void *ctx);
  void            *ctxs[3];
  DM               dm;
  PetscDS          ds;
  Vec              v;
  PetscReal        ferrors[3];
  PetscInt         tl, l, f;

  PetscFunctionBeginUser;
  CHKERRQ(TSGetDM(ts, &dm));
  CHKERRQ(DMGetDS(dm, &ds));

  for (f = 0; f < 3; ++f) CHKERRQ(PetscDSGetExactSolution(ds, f, &exactFuncs[f], &ctxs[f]));
  CHKERRQ(DMComputeL2FieldDiff(dm, crtime, exactFuncs, ctxs, u, ferrors));
  CHKERRQ(PetscObjectGetTabLevel((PetscObject) ts, &tl));
  for (l = 0; l < tl; ++l) CHKERRQ(PetscPrintf(PETSC_COMM_WORLD, "\t"));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD, "Timestep: %04d time = %-8.4g \t L_2 Error: [%2.3g, %2.3g, %2.3g]\n", (int) step, (double) crtime, (double) ferrors[0], (double) ferrors[1], (double) ferrors[2]));

  CHKERRQ(DMGetGlobalVector(dm, &u));
  CHKERRQ(PetscObjectSetName((PetscObject) u, "Numerical Solution"));
  CHKERRQ(VecViewFromOptions(u, NULL, "-sol_vec_view"));
  CHKERRQ(DMRestoreGlobalVector(dm, &u));

  CHKERRQ(DMGetGlobalVector(dm, &v));
  CHKERRQ(DMProjectFunction(dm, 0.0, exactFuncs, ctxs, INSERT_ALL_VALUES, v));
  CHKERRQ(PetscObjectSetName((PetscObject) v, "Exact Solution"));
  CHKERRQ(VecViewFromOptions(v, NULL, "-exact_vec_view"));
  CHKERRQ(DMRestoreGlobalVector(dm, &v));

  PetscFunctionReturn(0);
}

/* Note that adv->x0 will not be correct after migration */
static PetscErrorCode ComputeParticleError(TS ts, Vec u, Vec e)
{
  AdvCtx            *adv;
  DM                 sdm;
  Parameter         *param;
  const PetscScalar *xp0, *xp;
  PetscScalar       *ep;
  PetscReal          time;
  PetscInt           dim, Np, p;
  MPI_Comm           comm;

  PetscFunctionBeginUser;
  CHKERRQ(TSGetTime(ts, &time));
  CHKERRQ(TSGetApplicationContext(ts, &adv));
  CHKERRQ(PetscBagGetData(adv->ctx->bag, (void **) &param));
  CHKERRQ(PetscObjectGetComm((PetscObject) ts, &comm));
  CHKERRQ(TSGetDM(ts, &sdm));
  CHKERRQ(DMGetDimension(sdm, &dim));
  CHKERRQ(DMSwarmGetLocalSize(sdm, &Np));
  CHKERRQ(VecGetArrayRead(adv->x0, &xp0));
  CHKERRQ(VecGetArrayRead(u, &xp));
  CHKERRQ(VecGetArrayWrite(e, &ep));
  for (p = 0; p < Np; ++p) {
    PetscScalar x[3];
    PetscReal   x0[3];
    PetscInt    d;

    for (d = 0; d < dim; ++d) x0[d] = PetscRealPart(xp0[p*dim+d]);
    CHKERRQ(adv->exact(dim, time, x0, 1, x, param));
    for (d = 0; d < dim; ++d) ep[p*dim+d] += x[d] - xp[p*dim+d];
  }
  CHKERRQ(VecRestoreArrayRead(adv->x0, &xp0));
  CHKERRQ(VecRestoreArrayRead(u, &xp));
  CHKERRQ(VecRestoreArrayWrite(e, &ep));
  PetscFunctionReturn(0);
}

static PetscErrorCode MonitorParticleError(TS ts, PetscInt step, PetscReal time, Vec u, void *ctx)
{
  AdvCtx            *adv = (AdvCtx *) ctx;
  DM                 sdm;
  Parameter         *param;
  const PetscScalar *xp0, *xp;
  PetscReal          error = 0.0;
  PetscInt           dim, tl, l, Np, p;
  MPI_Comm           comm;

  PetscFunctionBeginUser;
  CHKERRQ(PetscBagGetData(adv->ctx->bag, (void **) &param));
  CHKERRQ(PetscObjectGetComm((PetscObject) ts, &comm));
  CHKERRQ(TSGetDM(ts, &sdm));
  CHKERRQ(DMGetDimension(sdm, &dim));
  CHKERRQ(DMSwarmGetLocalSize(sdm, &Np));
  CHKERRQ(VecGetArrayRead(adv->x0, &xp0));
  CHKERRQ(VecGetArrayRead(u, &xp));
  for (p = 0; p < Np; ++p) {
    PetscScalar x[3];
    PetscReal   x0[3];
    PetscReal   perror = 0.0;
    PetscInt    d;

    for (d = 0; d < dim; ++d) x0[d] = PetscRealPart(xp0[p*dim+d]);
    CHKERRQ(adv->exact(dim, time, x0, 1, x, param));
    for (d = 0; d < dim; ++d) perror += PetscSqr(PetscRealPart(x[d] - xp[p*dim+d]));
    error += perror;
  }
  CHKERRQ(VecRestoreArrayRead(adv->x0, &xp0));
  CHKERRQ(VecRestoreArrayRead(u, &xp));
  CHKERRQ(PetscObjectGetTabLevel((PetscObject) ts, &tl));
  for (l = 0; l < tl; ++l) CHKERRQ(PetscPrintf(PETSC_COMM_WORLD, "\t"));
  CHKERRQ(PetscPrintf(comm, "Timestep: %04d time = %-8.4g \t L_2 Particle Error: [%2.3g]\n", (int) step, (double) time, (double) error));
  PetscFunctionReturn(0);
}

static PetscErrorCode AdvectParticles(TS ts)
{
  TS             sts;
  DM             sdm;
  Vec            coordinates;
  AdvCtx        *adv;
  PetscReal      time;
  PetscBool      lreset, reset;
  PetscInt       dim, n, N, newn, newN;

  PetscFunctionBeginUser;
  CHKERRQ(PetscObjectQuery((PetscObject) ts, "_SwarmTS",  (PetscObject *) &sts));
  CHKERRQ(TSGetDM(sts, &sdm));
  CHKERRQ(TSGetRHSFunction(sts, NULL, NULL, (void **) &adv));
  CHKERRQ(DMGetDimension(sdm, &dim));
  CHKERRQ(DMSwarmGetSize(sdm, &N));
  CHKERRQ(DMSwarmGetLocalSize(sdm, &n));
  CHKERRQ(DMSwarmCreateGlobalVectorFromField(sdm, DMSwarmPICField_coor, &coordinates));
  CHKERRQ(TSGetTime(ts, &time));
  CHKERRQ(TSSetMaxTime(sts, time));
  adv->tf = time;
  CHKERRQ(TSSolve(sts, coordinates));
  CHKERRQ(DMSwarmDestroyGlobalVectorFromField(sdm, DMSwarmPICField_coor, &coordinates));
  CHKERRQ(VecCopy(adv->uf, adv->ui));
  adv->ti = adv->tf;

  CHKERRQ(DMSwarmMigrate(sdm, PETSC_TRUE));
  CHKERRQ(DMSwarmGetSize(sdm, &newN));
  CHKERRQ(DMSwarmGetLocalSize(sdm, &newn));
  lreset = (n != newn || N != newN) ? PETSC_TRUE : PETSC_FALSE;
  CHKERRMPI(MPI_Allreduce(&lreset, &reset, 1, MPIU_BOOL, MPI_LOR, PetscObjectComm((PetscObject) sts)));
  if (reset) {
    CHKERRQ(TSReset(sts));
    CHKERRQ(DMSwarmVectorDefineField(sdm, DMSwarmPICField_coor));
  }
  CHKERRQ(DMViewFromOptions(sdm, NULL, "-dm_view"));
  PetscFunctionReturn(0);
}

int main(int argc, char **argv)
{
  DM              dm, sdm;
  TS              ts, sts;
  Vec             u, xtmp;
  AppCtx          user;
  AdvCtx          adv;
  PetscReal       t;
  PetscInt        dim;

  CHKERRQ(PetscInitialize(&argc, &argv, NULL,help));
  CHKERRQ(ProcessOptions(PETSC_COMM_WORLD, &user));
  CHKERRQ(PetscBagCreate(PETSC_COMM_WORLD, sizeof(Parameter), &user.bag));
  CHKERRQ(SetupParameters(&user));
  CHKERRQ(TSCreate(PETSC_COMM_WORLD, &ts));
  CHKERRQ(CreateMesh(PETSC_COMM_WORLD, &user, &dm));
  CHKERRQ(TSSetDM(ts, dm));
  CHKERRQ(DMSetApplicationContext(dm, &user));
  /* Discretize chemical species */
  CHKERRQ(DMCreate(PETSC_COMM_WORLD, &sdm));
  CHKERRQ(PetscObjectSetOptionsPrefix((PetscObject) sdm, "part_"));
  CHKERRQ(DMSetType(sdm, DMSWARM));
  CHKERRQ(DMGetDimension(dm, &dim));
  CHKERRQ(DMSetDimension(sdm, dim));
  CHKERRQ(DMSwarmSetCellDM(sdm, dm));
  /* Setup problem */
  CHKERRQ(SetupDiscretization(dm, sdm, &user));
  CHKERRQ(DMPlexCreateClosureIndex(dm, NULL));

  CHKERRQ(DMCreateGlobalVector(dm, &u));
  CHKERRQ(DMSetNullSpaceConstructor(dm, 1, CreatePressureNullSpace));

  CHKERRQ(DMTSSetBoundaryLocal(dm, DMPlexTSComputeBoundary, &user));
  CHKERRQ(DMTSSetIFunctionLocal(dm, DMPlexTSComputeIFunctionFEM, &user));
  CHKERRQ(DMTSSetIJacobianLocal(dm, DMPlexTSComputeIJacobianFEM, &user));
  CHKERRQ(TSSetExactFinalTime(ts, TS_EXACTFINALTIME_MATCHSTEP));
  CHKERRQ(TSSetPreStep(ts, RemoveDiscretePressureNullspace));
  CHKERRQ(TSMonitorSet(ts, MonitorError, &user, NULL));
  CHKERRQ(TSSetFromOptions(ts));

  CHKERRQ(TSSetComputeInitialCondition(ts, SetInitialConditions)); /* Must come after SetFromOptions() */
  CHKERRQ(SetInitialConditions(ts, u));
  CHKERRQ(TSGetTime(ts, &t));
  CHKERRQ(DMSetOutputSequenceNumber(dm, 0, t));
  CHKERRQ(DMTSCheckFromOptions(ts, u));

  /* Setup particle position integrator */
  CHKERRQ(TSCreate(PETSC_COMM_WORLD, &sts));
  CHKERRQ(PetscObjectSetOptionsPrefix((PetscObject) sts, "part_"));
  CHKERRQ(PetscObjectIncrementTabLevel((PetscObject) sts, (PetscObject) ts, 1));
  CHKERRQ(TSSetDM(sts, sdm));
  CHKERRQ(TSSetProblemType(sts, TS_NONLINEAR));
  CHKERRQ(TSSetExactFinalTime(sts, TS_EXACTFINALTIME_MATCHSTEP));
  CHKERRQ(TSMonitorSet(sts, MonitorParticleError, &adv, NULL));
  CHKERRQ(TSSetFromOptions(sts));
  CHKERRQ(TSSetApplicationContext(sts, &adv));
  CHKERRQ(TSSetComputeExactError(sts, ComputeParticleError));
  CHKERRQ(TSSetComputeInitialCondition(sts, SetInitialParticleConditions));
  adv.ti = t;
  adv.uf = u;
  CHKERRQ(VecDuplicate(adv.uf, &adv.ui));
  CHKERRQ(VecCopy(u, adv.ui));
  CHKERRQ(TSSetRHSFunction(sts, NULL, FreeStreaming, &adv));
  CHKERRQ(TSSetPostStep(ts, AdvectParticles));
  CHKERRQ(PetscObjectCompose((PetscObject) ts, "_SwarmTS", (PetscObject) sts));
  CHKERRQ(DMSwarmVectorDefineField(sdm, DMSwarmPICField_coor));
  CHKERRQ(DMCreateGlobalVector(sdm, &adv.x0));
  CHKERRQ(DMSwarmCreateGlobalVectorFromField(sdm, DMSwarmPICField_coor, &xtmp));
  CHKERRQ(VecCopy(xtmp, adv.x0));
  CHKERRQ(DMSwarmDestroyGlobalVectorFromField(sdm, DMSwarmPICField_coor, &xtmp));
  switch(user.solType) {
    case SOL_TRIG_TRIG: adv.exact = trig_trig_x;break;
    default: SETERRQ(PetscObjectComm((PetscObject) sdm), PETSC_ERR_ARG_WRONG, "Unsupported solution type: %s (%D)", solTypes[PetscMin(user.solType, NUM_SOL_TYPES)], user.solType);
  }
  adv.ctx = &user;

  CHKERRQ(TSSolve(ts, u));
  CHKERRQ(DMTSCheckFromOptions(ts, u));
  CHKERRQ(PetscObjectSetName((PetscObject) u, "Numerical Solution"));

  CHKERRQ(VecDestroy(&u));
  CHKERRQ(VecDestroy(&adv.x0));
  CHKERRQ(VecDestroy(&adv.ui));
  CHKERRQ(DMDestroy(&dm));
  CHKERRQ(DMDestroy(&sdm));
  CHKERRQ(TSDestroy(&ts));
  CHKERRQ(TSDestroy(&sts));
  CHKERRQ(PetscBagDestroy(&user.bag));
  CHKERRQ(PetscFinalize());
  return 0;
}

/*TEST

  # Swarm does not work with complex
  test:
    suffix: 2d_tri_p2_p1_p1_tconvp
    requires: triangle !single !complex
    args: -dm_plex_separate_marker -sol_type trig_trig -dm_refine 2 \
      -vel_petscspace_degree 2 -pres_petscspace_degree 1 -temp_petscspace_degree 1 \
      -dmts_check .001 -ts_max_steps 4 -ts_dt 0.1 -ts_monitor_cancel \
      -ksp_type fgmres -ksp_gmres_restart 10 -ksp_rtol 1.0e-9 -ksp_error_if_not_converged \
      -pc_type fieldsplit -pc_fieldsplit_0_fields 0,2 -pc_fieldsplit_1_fields 1 -pc_fieldsplit_type schur -pc_fieldsplit_schur_factorization_type full \
        -fieldsplit_0_pc_type lu \
        -fieldsplit_pressure_ksp_rtol 1e-10 -fieldsplit_pressure_pc_type jacobi \
      -omega 0.5 -part_layout_type box -part_lower 0.25,0.25 -part_upper 0.75,0.75 -Npb 5 \
      -part_ts_max_steps 2 -part_ts_dt 0.05 -part_ts_convergence_estimate -convest_num_refine 1 -part_ts_monitor_cancel
  test:
    suffix: 2d_tri_p2_p1_p1_exit
    requires: triangle !single !complex
    args: -dm_plex_separate_marker -sol_type trig_trig -dm_refine 1 \
      -vel_petscspace_degree 2 -pres_petscspace_degree 1 -temp_petscspace_degree 1 \
      -dmts_check .001 -ts_max_steps 10 -ts_dt 0.1 \
      -ksp_type fgmres -ksp_gmres_restart 10 -ksp_rtol 1.0e-9 -ksp_error_if_not_converged \
      -pc_type fieldsplit -pc_fieldsplit_0_fields 0,2 -pc_fieldsplit_1_fields 1 -pc_fieldsplit_type schur -pc_fieldsplit_schur_factorization_type full \
        -fieldsplit_0_pc_type lu \
        -fieldsplit_pressure_ksp_rtol 1e-10 -fieldsplit_pressure_pc_type jacobi \
      -omega 0.5 -part_layout_type box -part_lower 0.25,0.25 -part_upper 0.75,0.75 -Npb 5 \
      -part_ts_max_steps 20 -part_ts_dt 0.05

TEST*/
