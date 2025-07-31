static char help[] = "Poisson Problem in 2d and 3d with simplicial finite elements in both primal and mixed form.\n\
We solve the Poisson problem in a rectangular\n\
domain, using a parallel unstructured mesh (DMPLEX) to discretize it.\n\
This example solves mixed form equation to get the flux field to calculate flux norm. We then use that for adaptive mesh refinement. \n\n\n";

/*
The primal (or original) Poisson problem, in the strong form, is given by,

\begin{align}
  - \nabla \cdot ( \nabla u ) = f
\end{align}
where $u$ is potential.

The weak form of this equation is

\begin{align}
  < \nabla v, \nabla u > - < v, \nabla u \cdot \hat{n} >_\Gamma - < v, f > = 0
\end{align}

The mixed Poisson problem, in the strong form, is given by,

\begin{align}
  q - \nabla u &= 0 \\
  - \nabla \cdot q &= f
\end{align}
where $u$ is the potential and $q$ is the flux.

The weak form of this equation is

\begin{align}
  < t, q > + < \nabla \cdot t, u > - < t \cdot \hat{n}, u >_\Gamma &= 0 \\
  <v, \nabla \cdot q> - < v, f > &= 0
\end{align}

We solve both primal and mixed problem and calculate the error in the flux norm, namely || e || = || q^m_h - \nabla u^p_h ||. Here superscript 'm' represents field from mixed form and 'p' represents field from the primal form.

The following boundary conditions are prescribed.

Primal problem:
\begin{align}
  u = u_0                    on \Gamma_D
  \nabla u \cdot \hat{n} = g on \Gamma_N
\end{align}

Mixed problem:
\begin{align}
  u = u_0             on \Gamma_D
  q \cdot \hat{n} = g on \Gamma_N
\end{align}
        __________\Gamma_D_____________
        |                              |
        |                              |
        |                              |
\Gamma_N                               \Gamma_N
        |                              |
        |                              |
        |                              |
        |_________\Gamma_D_____________|

To visualize the automated adaptation

  -dm_adapt_pre_view draw -dm_adapt_view draw -draw_pause -1 -geometry 0,0,1024,1024

and to compare with a naice gradient estimator use

  -adaptor_type gradient

To see a sequence of adaptations

  -snes_adapt_sequence 8 -adaptor_monitor_error draw::draw_lg
  -dm_adapt_pre_view draw -dm_adapt_iter_view draw -dm_adapt_view draw -draw_pause 1 -geometry 0,0,1024,1024

To get a better view of the by-hand process, use

  -dm_view hdf5:${PWD}/mesh.h5
  -primal_sol_vec_view hdf5:${PWD}/mesh.h5::append
  -flux_error_vec_view hdf5:${PWD}/mesh.h5::append
  -exact_error_vec_view hdf5:${PWD}/mesh.h5::append
  -refine_vec_view hdf5:${PWD}/mesh.h5::append
  -adapt_dm_view draw -draw_pause -1

This is also possible with the automated path

  -dm_view hdf5:${PWD}/mesh.h5
  -adapt_primal_sol_vec_view hdf5:${PWD}/mesh.h5::append
  -adapt_error_vec_view hdf5:${PWD}/mesh.h5::append
  -adapt_vec_view hdf5:${PWD}/mesh.h5::append
*/

#include <petsc/private/petscfeimpl.h>
#include <petscdmplex.h>
#include <petscsnes.h>
#include <petscdmadaptor.h>
#include <petscds.h>
#include <petscviewerhdf5.h>
#include <petscbag.h>

PETSC_EXTERN PetscErrorCode SetupMixed(DMAdaptor, DM);

typedef enum {
  SOL_QUADRATIC,
  SOL_TRIG,
  SOL_SENSOR,
  SOL_UNKNOWN,
  NUM_SOL_TYPE
} SolType;
const char *SolTypeNames[NUM_SOL_TYPE + 4] = {"quadratic", "trig", "sensor", "unknown", "SolType", "SOL_", NULL};

typedef struct {
  PetscBag  param;
  SolType   solType;
  PetscBool byHand;
  PetscInt  numAdapt;
} AppCtx;

typedef struct {
  PetscReal k;
} Parameter;

/* Exact solution: u = x^2 + y^2 */
static PetscErrorCode quadratic_u(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nc, PetscScalar *u, void *ctx)
{
  PetscInt d;

  u[0] = 0.0;
  for (d = 0; d < dim; ++d) u[0] += x[d] * x[d];
  return PETSC_SUCCESS;
}
/* Exact solution: q = (2x, 2y) */
static PetscErrorCode quadratic_q(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nc, PetscScalar *u, void *ctx)
{
  PetscInt c;
  for (c = 0; c < Nc; ++c) u[c] = 2.0 * x[c];
  return PETSC_SUCCESS;
}

/* Exact solution: u = sin( n \pi x ) * sin( n \pi y ) */
static PetscErrorCode trig_u(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nc, PetscScalar *u, void *ctx)
{
  const PetscReal n = 5.5;

  u[0] = 1.0;
  for (PetscInt d = 0; d < dim; ++d) u[0] *= PetscSinReal(n * PETSC_PI * x[d]);
  return PETSC_SUCCESS;
}
static PetscErrorCode trig_q(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nc, PetscScalar *u, void *ctx)
{
  const PetscReal n = 5.5;

  for (PetscInt c = 0; c < Nc; c++) u[c] = n * PETSC_PI * PetscCosReal(n * PETSC_PI * x[c]) * PetscSinReal(n * PETSC_PI * x[Nc - c - 1]);
  return PETSC_SUCCESS;
}

/*
Classic hyperbolic sensor function for testing multi-scale anisotropic mesh adaptation:

  f:[-1, 1]x[-1, 1] \to R,
    f(x, y) = sin(50xy)/100 if |xy| > 2\pi/50 else sin(50xy)

(mapped to have domain [0,1] x [0,1] in this case).
*/
static PetscErrorCode sensor_u(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nf, PetscScalar u[], void *ctx)
{
  const PetscReal xref = 2. * x[0] - 1.;
  const PetscReal yref = 2. * x[1] - 1.;
  const PetscReal xy   = xref * yref;

  u[0] = PetscSinReal(50. * xy);
  if (PetscAbsReal(xy) > 2. * PETSC_PI / 50.) u[0] *= 0.01;

  return PETSC_SUCCESS;
}

/* Flux is (cos(50xy) * 50y/100, cos(50xy) * 50x/100) if |xy| > 2\pi/50 else (cos(50xy) * 50y, cos(50xy) * 50x) */
static PetscErrorCode sensor_q(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nf, PetscScalar u[], void *ctx)
{
  const PetscReal xref = 2. * x[0] - 1.;
  const PetscReal yref = 2. * x[1] - 1.;
  const PetscReal xy   = xref * yref;

  u[0] = 50. * yref * PetscCosReal(50. * xy) * 2.0;
  u[1] = 50. * xref * PetscCosReal(50. * xy) * 2.0;
  if (PetscAbsReal(xy) > 2. * PETSC_PI / 50.) {
    u[0] *= 0.01;
    u[1] *= 0.01;
  }
  return PETSC_SUCCESS;
}

/* We set up residuals and Jacobians for the primal problem. */
static void f0_quadratic_primal(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f0[])
{
  f0[0] = 4.0;
}

static void f0_trig_primal(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f0[])
{
  const PetscReal n = 5.5;

  f0[0] = -2.0 * PetscSqr(n * PETSC_PI) * PetscSinReal(n * PETSC_PI * x[0]) * PetscSinReal(n * PETSC_PI * x[1]);
}

static void f0_sensor_primal(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f0[])
{
  const PetscReal xref = 2. * x[0] - 1.;
  const PetscReal yref = 2. * x[1] - 1.;
  const PetscReal xy   = xref * yref;

  f0[0] = -2500.0 * PetscSinReal(50. * xy) * (xref * xref + yref * yref) * 4.0;
  if (PetscAbsReal(xy) > 2. * PETSC_PI / 50.) f0[0] *= 0.01;
}

static void f1_primal(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f1[])
{
  const PetscReal k = PetscRealPart(constants[0]);

  for (PetscInt d = 0; d < dim; ++d) f1[d] = k * u_x[uOff_x[0] + d];
}

static void f0_quadratic_bd_primal(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, const PetscReal x[], const PetscReal n[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f0[])
{
  const PetscReal k = PetscRealPart(constants[0]);
  PetscScalar     flux;

  PetscCallAbort(PETSC_COMM_SELF, quadratic_q(dim, t, x, dim, &flux, NULL));
  for (PetscInt d = 0; d < dim; ++d) f0[d] = -k * flux * n[d];
}

static void f0_trig_bd_primal(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, const PetscReal x[], const PetscReal n[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f0[])
{
  const PetscReal k = PetscRealPart(constants[0]);
  PetscScalar     flux;

  PetscCallAbort(PETSC_COMM_SELF, trig_q(dim, t, x, dim, &flux, NULL));
  for (PetscInt d = 0; d < dim; ++d) f0[d] = -k * flux * n[d];
}

static void f0_sensor_bd_primal(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, const PetscReal x[], const PetscReal n[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f0[])
{
  const PetscReal k = PetscRealPart(constants[0]);
  PetscScalar     flux[2];

  PetscCallAbort(PETSC_COMM_SELF, sensor_q(dim, t, x, dim, flux, NULL));
  for (PetscInt d = 0; d < dim; ++d) f0[d] = -k * flux[d] * n[d];
}

static void g3_primal_uu(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g3[])
{
  const PetscReal k = PetscRealPart(constants[0]);

  for (PetscInt d = 0; d < dim; ++d) g3[d * dim + d] = k;
}

/* Now we set up the residuals and Jacobians mixed problem. */
static void f0_mixed_quadratic_u(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f0[])
{
  f0[0] = 4.0;
  for (PetscInt d = 0; d < dim; ++d) f0[0] += -u_x[uOff_x[0] + d * dim + d];
}
static void f0_mixed_trig_u(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f0[])
{
  PetscReal n = 5.5;

  f0[0] = -2.0 * PetscSqr(n * PETSC_PI) * PetscSinReal(n * PETSC_PI * x[0]) * PetscSinReal(n * PETSC_PI * x[1]);
  for (PetscInt d = 0; d < dim; ++d) f0[0] += -u_x[uOff_x[0] + d * dim + d];
}
static void f0_mixed_sensor_u(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f0[])
{
  const PetscReal xref = 2. * x[0] - 1.;
  const PetscReal yref = 2. * x[1] - 1.;
  const PetscReal xy   = xref * yref;

  f0[0] = -2500.0 * PetscSinReal(50. * xy) * (xref * xref + yref * yref) * 4.0;
  if (PetscAbsReal(xy) > 2. * PETSC_PI / 50.) f0[0] *= 0.01;
  for (PetscInt d = 0; d < dim; ++d) f0[0] += -u_x[uOff_x[0] + d * dim + d];
}

static void f0_mixed_q(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f0[])
{
  for (PetscInt d = 0; d < dim; d++) f0[d] = u[uOff[0] + d];
}

static void f1_mixed_q(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f1[])
{
  const PetscReal k = PetscRealPart(constants[0]);

  for (PetscInt d = 0; d < dim; d++) f1[d * dim + d] = k * u[uOff[1]];
}

static void f0_quadratic_bd_mixed_q(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, const PetscReal x[], const PetscReal n[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f0[])
{
  const PetscReal k = PetscRealPart(constants[0]);
  PetscScalar     potential;

  PetscCallAbort(PETSC_COMM_SELF, quadratic_u(dim, t, x, dim, &potential, NULL));
  for (PetscInt d = 0; d < dim; ++d) f0[d] = -k * potential * n[d];
}

static void f0_trig_bd_mixed_q(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, const PetscReal x[], const PetscReal n[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f0[])
{
  const PetscReal k = PetscRealPart(constants[0]);
  PetscScalar     potential;

  PetscCallAbort(PETSC_COMM_SELF, trig_u(dim, t, x, dim, &potential, NULL));
  for (PetscInt d = 0; d < dim; ++d) f0[d * dim + d] = -k * potential * n[d];
}

static void f0_sensor_bd_mixed_q(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, const PetscReal x[], const PetscReal n[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f0[])
{
  const PetscReal k = PetscRealPart(constants[0]);
  PetscScalar     potential;

  PetscCallAbort(PETSC_COMM_SELF, sensor_u(dim, t, x, dim, &potential, NULL));
  for (PetscInt d = 0; d < dim; ++d) f0[d * dim + d] = -k * potential * n[d];
}

/* <v, \nabla\cdot q> */
static void g1_mixed_uq(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g1[])
{
  for (PetscInt d = 0; d < dim; d++) g1[d * dim + d] = -1.0;
}

/* < t, q> */
static void g0_mixed_qq(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g0[])
{
  for (PetscInt d = 0; d < dim; d++) g0[d * dim + d] = 1.0;
}

/* <\nabla\cdot t, u> = <\nabla t, Iu> */
static void g2_mixed_qu(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g2[])
{
  const PetscReal k = PetscRealPart(constants[0]);

  for (PetscInt d = 0; d < dim; d++) g2[d * dim + d] = k;
}

static PetscErrorCode ProcessOptions(MPI_Comm comm, AppCtx *user)
{
  PetscFunctionBeginUser;
  PetscOptionsBegin(comm, "", "Flux norm error in primal poisson problem Options", "DMPLEX");
  user->byHand   = PETSC_TRUE;
  user->numAdapt = 1;
  user->solType  = SOL_QUADRATIC;

  PetscCall(PetscOptionsGetBool(NULL, NULL, "-by_hand", &user->byHand, NULL));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-num_adapt", &user->numAdapt, NULL));
  PetscCall(PetscOptionsEnum("-sol_type", "Type of exact solution", "ex27.c", SolTypeNames, (PetscEnum)user->solType, (PetscEnum *)&user->solType, NULL));
  PetscOptionsEnd();
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode SetupParameters(PetscBag bag, AppCtx *user)
{
  Parameter *param;

  PetscFunctionBeginUser;
  PetscCall(PetscBagGetData(bag, (void **)&param));
  PetscCall(PetscBagSetName(bag, "par", "Poisson parameters"));
  PetscCall(PetscBagRegisterReal(bag, &param->k, 1.0, "k", "Thermal conductivity"));
  PetscCall(PetscBagSetFromOptions(bag));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode CreateMesh(MPI_Comm comm, AppCtx *user, DM *dm)
{
  PetscFunctionBeginUser;
  PetscCall(DMCreate(comm, dm));
  PetscCall(DMSetType(*dm, DMPLEX));
  PetscCall(DMSetFromOptions(*dm));
  PetscCall(DMSetApplicationContext(*dm, &user));
  PetscCall(DMViewFromOptions(*dm, NULL, "-dm_view"));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode SetupPrimalProblem(DM dm, AppCtx *user)
{
  PetscDS       ds;
  DMLabel       label;
  PetscInt      id, bd;
  Parameter    *param;
  PetscWeakForm wf;

  PetscFunctionBeginUser;
  PetscCall(DMGetDS(dm, &ds));
  PetscCall(DMGetLabel(dm, "marker", &label));

  PetscCall(PetscDSSetJacobian(ds, 0, 0, NULL, NULL, NULL, g3_primal_uu));

  switch (user->solType) {
  case SOL_QUADRATIC:
    PetscCall(PetscDSSetResidual(ds, 0, f0_quadratic_primal, f1_primal));

    id = 1;
    PetscCall(DMAddBoundary(dm, DM_BC_ESSENTIAL, "bottom wall primal potential", label, 1, &id, 0, 0, NULL, (PetscVoidFn *)quadratic_u, NULL, user, NULL));

    id = 2;
    PetscCall(DMAddBoundary(dm, DM_BC_NATURAL, "right wall flux", label, 1, &id, 0, 0, NULL, NULL, NULL, user, &bd));
    PetscCall(PetscDSGetBoundary(ds, bd, &wf, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL));
    PetscCall(PetscWeakFormSetIndexBdResidual(wf, label, id, 0, 0, 0, f0_quadratic_bd_primal, 0, NULL));

    id = 3;
    PetscCall(DMAddBoundary(dm, DM_BC_ESSENTIAL, "top wall primal potential", label, 1, &id, 0, 0, NULL, (PetscVoidFn *)quadratic_u, NULL, user, NULL));

    id = 4;
    PetscCall(DMAddBoundary(dm, DM_BC_NATURAL, "left wall flux", label, 1, &id, 0, 0, NULL, NULL, NULL, user, &bd));
    PetscCall(PetscDSGetBoundary(ds, bd, &wf, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL));
    PetscCall(PetscWeakFormSetIndexBdResidual(wf, label, id, 0, 0, 0, f0_quadratic_bd_primal, 0, NULL));

    PetscCall(PetscDSSetExactSolution(ds, 0, quadratic_u, user));
    break;
  case SOL_TRIG:
    PetscCall(PetscDSSetResidual(ds, 0, f0_trig_primal, f1_primal));

    id = 1;
    PetscCall(DMAddBoundary(dm, DM_BC_ESSENTIAL, "bottom wall primal potential", label, 1, &id, 0, 0, NULL, (PetscVoidFn *)trig_u, NULL, user, NULL));

    id = 2;
    PetscCall(DMAddBoundary(dm, DM_BC_NATURAL, "right wall flux", label, 1, &id, 0, 0, NULL, NULL, NULL, user, &bd));
    PetscCall(PetscDSGetBoundary(ds, bd, &wf, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL));
    PetscCall(PetscWeakFormSetIndexBdResidual(wf, label, id, 0, 0, 0, f0_trig_bd_primal, 0, NULL));

    id = 3;
    PetscCall(DMAddBoundary(dm, DM_BC_ESSENTIAL, "top wall primal potential", label, 1, &id, 0, 0, NULL, (PetscVoidFn *)trig_u, NULL, user, NULL));

    id = 4;
    PetscCall(DMAddBoundary(dm, DM_BC_NATURAL, "left wall flux", label, 1, &id, 0, 0, NULL, NULL, NULL, user, &bd));
    PetscCall(PetscDSGetBoundary(ds, bd, &wf, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL));
    PetscCall(PetscWeakFormSetIndexBdResidual(wf, label, id, 0, 0, 0, f0_trig_bd_primal, 0, NULL));

    PetscCall(PetscDSSetExactSolution(ds, 0, trig_u, user));
    break;
  case SOL_SENSOR:
    PetscCall(PetscDSSetResidual(ds, 0, f0_sensor_primal, f1_primal));

    id = 1;
    PetscCall(DMAddBoundary(dm, DM_BC_ESSENTIAL, "bottom wall primal potential", label, 1, &id, 0, 0, NULL, (PetscVoidFn *)sensor_u, NULL, user, NULL));

    id = 2;
    PetscCall(DMAddBoundary(dm, DM_BC_NATURAL, "right wall flux", label, 1, &id, 0, 0, NULL, NULL, NULL, user, &bd));
    PetscCall(PetscDSGetBoundary(ds, bd, &wf, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL));
    PetscCall(PetscWeakFormSetIndexBdResidual(wf, label, id, 0, 0, 0, f0_sensor_bd_primal, 0, NULL));

    id = 3;
    PetscCall(DMAddBoundary(dm, DM_BC_ESSENTIAL, "top wall primal potential", label, 1, &id, 0, 0, NULL, (PetscVoidFn *)sensor_u, NULL, user, NULL));

    id = 4;
    PetscCall(DMAddBoundary(dm, DM_BC_NATURAL, "left wall flux", label, 1, &id, 0, 0, NULL, NULL, NULL, user, &bd));
    PetscCall(PetscDSGetBoundary(ds, bd, &wf, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL));
    PetscCall(PetscWeakFormSetIndexBdResidual(wf, label, id, 0, 0, 0, f0_sensor_bd_primal, 0, NULL));

    PetscCall(PetscDSSetExactSolution(ds, 0, sensor_u, user));
    break;
  default:
    SETERRQ(PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_WRONG, "Invalid exact solution type %s", SolTypeNames[PetscMin(user->solType, SOL_UNKNOWN)]);
  }

  /* Setup constants */
  {
    PetscCall(PetscBagGetData(user->param, (void **)&param));
    PetscScalar constants[1];

    constants[0] = param->k;

    PetscCall(PetscDSSetConstants(ds, 1, constants));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode SetupPrimalDiscretization(DM dm, AppCtx *user)
{
  DM             cdm = dm;
  PetscFE        fe[1];
  DMPolytopeType ct;
  PetscInt       dim, cStart;
  MPI_Comm       comm;

  PetscFunctionBeginUser;
  PetscCall(PetscObjectGetComm((PetscObject)dm, &comm));
  PetscCall(DMGetDimension(dm, &dim));

  /* Create finite element */
  PetscCall(DMPlexGetHeightStratum(dm, 0, &cStart, NULL));
  PetscCall(DMPlexGetCellType(dm, cStart, &ct));
  PetscCall(PetscFECreateByCell(comm, dim, 1, ct, "primal_potential_", PETSC_DEFAULT, &fe[0]));
  PetscCall(PetscObjectSetName((PetscObject)fe[0], "primal potential"));

  /* Set discretization and boundary conditions for each mesh */
  PetscCall(DMSetField(dm, 0, NULL, (PetscObject)fe[0]));
  PetscCall(DMCreateDS(dm));
  while (cdm) {
    PetscCall(DMCopyDisc(dm, cdm));
    PetscCall(DMGetCoarseDM(cdm, &cdm));
  }

  PetscCall(PetscFEDestroy(&fe[0]));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode SetupMixedProblem(DM dm, AppCtx *user)
{
  PetscDS       ds;
  DMLabel       label;
  PetscInt      id, bd;
  Parameter    *param;
  PetscWeakForm wf;

  PetscFunctionBeginUser;
  PetscCall(DMGetDS(dm, &ds));
  PetscCall(DMGetLabel(dm, "marker", &label));

  /* Residual terms */
  PetscCall(PetscDSSetResidual(ds, 0, f0_mixed_q, f1_mixed_q));

  PetscCall(PetscDSSetJacobian(ds, 0, 0, g0_mixed_qq, NULL, NULL, NULL));
  PetscCall(PetscDSSetJacobian(ds, 0, 1, NULL, NULL, g2_mixed_qu, NULL));

  PetscCall(PetscDSSetJacobian(ds, 1, 0, NULL, g1_mixed_uq, NULL, NULL));

  switch (user->solType) {
  case SOL_QUADRATIC:
    PetscCall(PetscDSSetResidual(ds, 1, f0_mixed_quadratic_u, NULL));

    id = 1;
    PetscCall(DMAddBoundary(dm, DM_BC_NATURAL, "Dirichlet Bd Integral bottom wall", label, 1, &id, 0, 0, NULL, NULL, NULL, user, &bd));
    PetscCall(PetscDSGetBoundary(ds, bd, &wf, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL));
    PetscCall(PetscWeakFormSetIndexBdResidual(wf, label, id, 0, 0, 0, f0_quadratic_bd_mixed_q, 0, NULL));

    id = 2;
    PetscCall(DMAddBoundary(dm, DM_BC_ESSENTIAL, "right wall flux", label, 1, &id, 0, 0, NULL, (PetscVoidFn *)quadratic_q, NULL, user, NULL));

    id = 4;
    PetscCall(DMAddBoundary(dm, DM_BC_ESSENTIAL, "left wall flux", label, 1, &id, 0, 0, NULL, (PetscVoidFn *)quadratic_q, NULL, user, NULL));

    id = 3;
    PetscCall(DMAddBoundary(dm, DM_BC_NATURAL, "Dirichlet Bd Integral top wall", label, 1, &id, 0, 0, NULL, NULL, NULL, user, &bd));
    PetscCall(PetscDSGetBoundary(ds, bd, &wf, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL));
    PetscCall(PetscWeakFormSetIndexBdResidual(wf, label, id, 0, 0, 0, f0_quadratic_bd_mixed_q, 0, NULL));

    PetscCall(PetscDSSetExactSolution(ds, 0, quadratic_q, user));
    PetscCall(PetscDSSetExactSolution(ds, 1, quadratic_u, user));
    break;
  case SOL_TRIG:
    PetscCall(PetscDSSetResidual(ds, 1, f0_mixed_trig_u, NULL));

    id = 1;
    PetscCall(DMAddBoundary(dm, DM_BC_NATURAL, "Dirichlet Bd Integral bottom wall", label, 1, &id, 0, 0, NULL, NULL, NULL, user, &bd));
    PetscCall(PetscDSGetBoundary(ds, bd, &wf, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL));
    PetscCall(PetscWeakFormSetIndexBdResidual(wf, label, id, 0, 0, 0, f0_trig_bd_mixed_q, 0, NULL));

    id = 2;
    PetscCall(DMAddBoundary(dm, DM_BC_ESSENTIAL, "right wall flux", label, 1, &id, 1, 0, NULL, (PetscVoidFn *)trig_q, NULL, user, NULL));

    id = 4;
    PetscCall(DMAddBoundary(dm, DM_BC_ESSENTIAL, "left wall flux", label, 1, &id, 1, 0, NULL, (PetscVoidFn *)trig_q, NULL, user, NULL));

    id = 3;
    PetscCall(DMAddBoundary(dm, DM_BC_NATURAL, "Dirichlet Bd Integral top wall", label, 1, &id, 0, 0, NULL, NULL, NULL, user, &bd));
    PetscCall(PetscDSGetBoundary(ds, bd, &wf, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL));
    PetscCall(PetscWeakFormSetIndexBdResidual(wf, label, id, 0, 0, 0, f0_trig_bd_mixed_q, 0, NULL));

    PetscCall(PetscDSSetExactSolution(ds, 0, trig_q, user));
    PetscCall(PetscDSSetExactSolution(ds, 1, trig_u, user));
    break;
  case SOL_SENSOR:
    PetscCall(PetscDSSetResidual(ds, 1, f0_mixed_sensor_u, NULL));

    id = 1;
    PetscCall(DMAddBoundary(dm, DM_BC_NATURAL, "Dirichlet Bd Integral bottom wall", label, 1, &id, 0, 0, NULL, NULL, NULL, user, &bd));
    PetscCall(PetscDSGetBoundary(ds, bd, &wf, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL));
    PetscCall(PetscWeakFormSetIndexBdResidual(wf, label, id, 0, 0, 0, f0_sensor_bd_mixed_q, 0, NULL));

    id = 2;
    PetscCall(DMAddBoundary(dm, DM_BC_ESSENTIAL, "right wall flux", label, 1, &id, 1, 0, NULL, (PetscVoidFn *)sensor_q, NULL, user, NULL));

    id = 4;
    PetscCall(DMAddBoundary(dm, DM_BC_ESSENTIAL, "left wall flux", label, 1, &id, 1, 0, NULL, (PetscVoidFn *)sensor_q, NULL, user, NULL));

    id = 3;
    PetscCall(DMAddBoundary(dm, DM_BC_NATURAL, "Dirichlet Bd Integral top wall", label, 1, &id, 0, 0, NULL, NULL, NULL, user, &bd));
    PetscCall(PetscDSGetBoundary(ds, bd, &wf, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL));
    PetscCall(PetscWeakFormSetIndexBdResidual(wf, label, id, 0, 0, 0, f0_sensor_bd_mixed_q, 0, NULL));

    PetscCall(PetscDSSetExactSolution(ds, 0, sensor_q, user));
    PetscCall(PetscDSSetExactSolution(ds, 1, sensor_u, user));
    break;
  default:
    SETERRQ(PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_WRONG, "Invalid exact solution type %s", SolTypeNames[PetscMin(user->solType, SOL_UNKNOWN)]);
  }

  /* Setup constants */
  {
    PetscCall(PetscBagGetData(user->param, (void **)&param));
    PetscScalar constants[1];

    constants[0] = param->k;

    PetscCall(PetscDSSetConstants(ds, 1, constants));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode SetupMixedDiscretization(DM dm, AppCtx *user)
{
  DM             cdm = dm;
  PetscFE        fe[2];
  DMPolytopeType ct;
  PetscInt       dim, cStart;
  MPI_Comm       comm;

  PetscFunctionBeginUser;
  PetscCall(PetscObjectGetComm((PetscObject)dm, &comm));
  PetscCall(DMGetDimension(dm, &dim));

  /* Create finite element */
  PetscCall(DMPlexGetHeightStratum(dm, 0, &cStart, NULL));
  PetscCall(DMPlexGetCellType(dm, cStart, &ct));
  PetscCall(PetscFECreateByCell(comm, dim, dim, ct, "mixed_flux_", PETSC_DEFAULT, &fe[0]));
  PetscCall(PetscObjectSetName((PetscObject)fe[0], "mixed flux"));
  /* NOTE:  Set the same quadrature order as the primal problem here or use the command line option. */

  PetscCall(PetscFECreateByCell(comm, dim, 1, ct, "mixed_potential_", PETSC_DEFAULT, &fe[1]));
  PetscCall(PetscFECopyQuadrature(fe[0], fe[1]));
  PetscCall(PetscObjectSetName((PetscObject)fe[1], "mixed potential"));

  /* Set discretization and boundary conditions for each mesh */
  PetscCall(DMSetField(dm, 0, NULL, (PetscObject)fe[0]));
  PetscCall(DMSetField(dm, 1, NULL, (PetscObject)fe[1]));
  PetscCall(DMCreateDS(dm));
  while (cdm) {
    PetscCall(DMCopyDisc(dm, cdm));
    PetscCall(DMGetCoarseDM(cdm, &cdm));
  }
  PetscCall(PetscFEDestroy(&fe[0]));
  PetscCall(PetscFEDestroy(&fe[1]));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode SetupMixed(DMAdaptor adaptor, DM mdm)
{
  AppCtx *ctx;

  PetscFunctionBeginUser;
  PetscCall(DMGetApplicationContext(mdm, (void **)&ctx));
  PetscCall(SetupMixedDiscretization(mdm, ctx));
  PetscCall(SetupMixedProblem(mdm, ctx));
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv)
{
  DM        dm, mdm;               /* problem specification */
  SNES      snes, msnes;           /* nonlinear solvers */
  Vec       u, mu;                 /* solution vectors */
  Vec       fluxError, exactError; /* Element wise error vector */
  PetscReal fluxNorm, exactNorm;   /* Flux error norm */
  AppCtx    user;                  /* user-defined work context */

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  PetscCall(PetscBagCreate(PETSC_COMM_WORLD, sizeof(Parameter), &user.param));
  PetscCall(SetupParameters(user.param, &user));
  PetscCall(ProcessOptions(PETSC_COMM_WORLD, &user));

  // Set up and solve primal problem
  PetscCall(CreateMesh(PETSC_COMM_WORLD, &user, &dm));
  PetscCall(DMSetApplicationContext(dm, &user));
  PetscCall(SNESCreate(PETSC_COMM_WORLD, &snes));
  PetscCall(SNESSetDM(snes, dm));

  PetscCall(SetupPrimalDiscretization(dm, &user));
  PetscCall(SetupPrimalProblem(dm, &user));
  PetscCall(DMCreateGlobalVector(dm, &u));
  PetscCall(VecSet(u, 0.0));
  PetscCall(DMPlexSetSNESLocalFEM(dm, PETSC_FALSE, &user));
  PetscCall(SNESSetFromOptions(snes));
  PetscCall(DMSNESCheckFromOptions(snes, u));

  for (PetscInt a = 0; a < user.numAdapt; ++a) {
    if (a > 0) {
      char prefix[16];

      PetscCall(PetscSNPrintf(prefix, 16, "a%d_", (int)a));
      PetscCall(SNESSetOptionsPrefix(snes, prefix));
    }
    PetscCall(SNESSolve(snes, NULL, u));

    // Needed if you allow SNES to refine
    PetscCall(SNESGetSolution(snes, &u));
    PetscCall(VecGetDM(u, &dm));
  }

  PetscCall(PetscObjectSetName((PetscObject)u, "Primal Solution "));
  PetscCall(VecViewFromOptions(u, NULL, "-primal_sol_vec_view"));

  if (user.byHand) {
    // Set up and solve mixed problem
    PetscCall(DMClone(dm, &mdm));
    PetscCall(SNESCreate(PETSC_COMM_WORLD, &msnes));
    PetscCall(SNESSetOptionsPrefix(msnes, "mixed_"));
    PetscCall(SNESSetDM(msnes, mdm));

    PetscCall(SetupMixedDiscretization(mdm, &user));
    PetscCall(SetupMixedProblem(mdm, &user));
    PetscCall(DMCreateGlobalVector(mdm, &mu));
    PetscCall(VecSet(mu, 0.0));
    PetscCall(DMPlexSetSNESLocalFEM(mdm, PETSC_FALSE, &user));
    PetscCall(SNESSetFromOptions(msnes));

    PetscCall(DMSNESCheckFromOptions(msnes, mu));
    PetscCall(SNESSolve(msnes, NULL, mu));
    PetscCall(PetscObjectSetName((PetscObject)mu, "Mixed Solution "));
    PetscCall(VecViewFromOptions(mu, NULL, "-mixed_sol_vec_view"));

    // Create the error space of piecewise constants
    DM             dmErr;
    PetscFE        feErr;
    DMPolytopeType ct;
    PetscInt       dim, cStart;

    PetscCall(DMClone(dm, &dmErr));
    PetscCall(DMGetDimension(dmErr, &dim));
    PetscCall(DMPlexGetHeightStratum(dmErr, 0, &cStart, NULL));
    PetscCall(DMPlexGetCellType(dmErr, cStart, &ct));
    PetscCall(PetscFECreateLagrangeByCell(PETSC_COMM_SELF, dim, 1, ct, 0, PETSC_DEFAULT, &feErr));
    PetscCall(PetscObjectSetName((PetscObject)feErr, "Error"));
    PetscCall(DMSetField(dmErr, 0, NULL, (PetscObject)feErr));
    PetscCall(PetscFEDestroy(&feErr));
    PetscCall(DMCreateDS(dmErr));
    PetscCall(DMViewFromOptions(dmErr, NULL, "-dmerr_view"));

    // Compute the flux norm
    PetscCall(DMGetGlobalVector(dmErr, &fluxError));
    PetscCall(PetscObjectSetName((PetscObject)fluxError, "Flux Error"));
    PetscCall(DMGetGlobalVector(dmErr, &exactError));
    PetscCall(PetscObjectSetName((PetscObject)exactError, "Analytical Error"));
    PetscCall(DMPlexComputeL2FluxDiffVec(u, 0, mu, 0, fluxError));
    {
      PetscDS             ds;
      PetscSimplePointFn *func[2] = {NULL, NULL};
      void               *ctx[2]  = {NULL, NULL};

      PetscCall(DMGetDS(mdm, &ds));
      PetscCall(PetscDSGetExactSolution(ds, 0, &func[0], &ctx[0]));
      PetscCall(DMPlexComputeL2DiffVec(mdm, 0.0, func, ctx, mu, exactError));
    }
    PetscCall(VecNorm(fluxError, NORM_2, &fluxNorm));
    PetscCall(VecNorm(exactError, NORM_2, &exactNorm));
    PetscCall(PetscPrintf(PETSC_COMM_SELF, "Flux error norm = %g\t Exact flux error norm = %g\n", (double)fluxNorm, (double)exactNorm));
    PetscCall(VecViewFromOptions(fluxError, NULL, "-flux_error_vec_view"));
    PetscCall(VecViewFromOptions(exactError, NULL, "-exact_error_vec_view"));

    // Adaptive refinement based on calculated error
    DM        rdm;
    VecTagger refineTag;
    DMLabel   adaptLabel;
    IS        refineIS;
    Vec       ref;

    PetscCall(DMLabelCreate(PETSC_COMM_WORLD, "adapt", &adaptLabel));
    PetscCall(DMLabelSetDefaultValue(adaptLabel, DM_ADAPT_COARSEN));
    PetscCall(VecTaggerCreate(PETSC_COMM_WORLD, &refineTag));
    PetscCall(VecTaggerSetFromOptions(refineTag));
    PetscCall(VecTaggerSetUp(refineTag));
    PetscCall(PetscObjectViewFromOptions((PetscObject)refineTag, NULL, "-tag_view"));

    PetscCall(VecTaggerComputeIS(refineTag, fluxError, &refineIS, NULL));
    PetscCall(VecTaggerDestroy(&refineTag));
    PetscCall(DMLabelSetStratumIS(adaptLabel, DM_ADAPT_REFINE, refineIS));
    PetscCall(ISViewFromOptions(refineIS, NULL, "-refine_is_view"));
    PetscCall(ISDestroy(&refineIS));

    PetscCall(DMPlexCreateLabelField(dm, adaptLabel, &ref));
    PetscCall(VecViewFromOptions(ref, NULL, "-refine_vec_view"));
    PetscCall(VecDestroy(&ref));

    // Mark adaptation phase with prefix ref_
    PetscCall(PetscObjectSetOptionsPrefix((PetscObject)dm, "adapt_"));
    PetscCall(DMAdaptLabel(dm, adaptLabel, &rdm));
    PetscCall(PetscObjectSetOptionsPrefix((PetscObject)dm, NULL));
    PetscCall(PetscObjectSetName((PetscObject)rdm, "Adaptively Refined DM"));
    PetscCall(DMViewFromOptions(rdm, NULL, "-adapt_dm_view"));
    PetscCall(DMDestroy(&rdm));
    PetscCall(DMLabelDestroy(&adaptLabel));

    // Destroy the error structures
    PetscCall(DMRestoreGlobalVector(dmErr, &fluxError));
    PetscCall(DMRestoreGlobalVector(dmErr, &exactError));
    PetscCall(DMDestroy(&dmErr));

    // Destroy the mixed structures
    PetscCall(VecDestroy(&mu));
    PetscCall(DMDestroy(&mdm));
    PetscCall(SNESDestroy(&msnes));
  }

  // Destroy the primal structures
  PetscCall(VecDestroy(&u));
  PetscCall(DMDestroy(&dm));
  PetscCall(SNESDestroy(&snes));
  PetscCall(PetscBagDestroy(&user.param));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

  # Tests using the explicit code above
  testset:
    suffix: 2d_p2_rt0p0_byhand
    requires: triangle
    args: -dm_plex_box_faces 1,1 -dm_plex_box_lower 0,0 -dm_plex_box_upper 1,1 -dm_refine 3 \
          -primal_potential_petscspace_degree 2 \
          -mixed_potential_petscdualspace_lagrange_continuity 0 \
          -mixed_flux_petscspace_type ptrimmed \
          -mixed_flux_petscspace_components 2 \
          -mixed_flux_petscspace_ptrimmed_form_degree -1 \
          -mixed_flux_petscdualspace_order 1 \
          -mixed_flux_petscdualspace_form_degree -1 \
          -mixed_flux_petscdualspace_lagrange_trimmed true \
          -mixed_flux_petscfe_default_quadrature_order 2 \
          -vec_tagger_type cdf -vec_tagger_box 0.9,1.0 \
            -tag_view \
            -adapt_dm_adaptor cellrefiner -adapt_dm_plex_transform_type refine_sbr \
          -dmsnes_check 0.001 -mixed_dmsnes_check 0.001 -pc_type jacobi -mixed_pc_type jacobi
    test:
      suffix: quadratic
      args: -sol_type quadratic
    test:
      suffix: trig
      args: -sol_type trig
    test:
      suffix: sensor
      args: -sol_type sensor

  # Tests using the embedded adaptor in SNES
  testset:
    suffix: 2d_p2_rt0p0
    requires: triangle defined(PETSC_HAVE_EXECUTABLE_EXPORT)
    args: -dm_plex_box_faces 1,1 -dm_plex_box_lower 0,0 -dm_plex_box_upper 1,1 -dm_refine 3 \
          -primal_potential_petscspace_degree 2 \
          -mixed_potential_petscdualspace_lagrange_continuity 0 \
          -mixed_flux_petscspace_type ptrimmed \
          -mixed_flux_petscspace_components 2 \
          -mixed_flux_petscspace_ptrimmed_form_degree -1 \
          -mixed_flux_petscdualspace_order 1 \
          -mixed_flux_petscdualspace_form_degree -1 \
          -mixed_flux_petscdualspace_lagrange_trimmed true \
          -mixed_flux_petscfe_default_quadrature_order 2 \
          -by_hand 0 \
          -refine_vec_tagger_type cdf -refine_vec_tagger_box 0.9,1.0 \
            -snes_adapt_view \
            -adapt_dm_adaptor cellrefiner -adapt_dm_plex_transform_type refine_sbr \
            -adaptor_criterion label -adaptor_type flux -adaptor_mixed_setup_function SetupMixed \
          -snes_adapt_sequence 1 -pc_type jacobi -mixed_pc_type jacobi
    test:
      suffix: quadratic
      args: -sol_type quadratic -adaptor_monitor_error
    test:
      suffix: trig
      args: -sol_type trig -adaptor_monitor_error
    test:
      suffix: sensor
      args: -sol_type sensor

  # Tests using multiple adaptor loops
  testset:
    suffix: 2d_p2_rt0p0_a2
    requires: triangle defined(PETSC_HAVE_EXECUTABLE_EXPORT)
    args: -dm_plex_box_faces 1,1 -dm_plex_box_lower 0,0 -dm_plex_box_upper 1,1 -dm_refine 3 \
          -primal_potential_petscspace_degree 2 \
          -mixed_potential_petscdualspace_lagrange_continuity 0 \
          -mixed_flux_petscspace_type ptrimmed \
          -mixed_flux_petscspace_components 2 \
          -mixed_flux_petscspace_ptrimmed_form_degree -1 \
          -mixed_flux_petscdualspace_order 1 \
          -mixed_flux_petscdualspace_form_degree -1 \
          -mixed_flux_petscdualspace_lagrange_trimmed true \
          -mixed_flux_petscfe_default_quadrature_order 2 \
          -by_hand 0 \
          -num_adapt 2 \
          -refine_vec_tagger_type cdf -refine_vec_tagger_box 0.9,1.0 \
            -snes_adapt_view \
            -adapt_dm_adaptor cellrefiner -adapt_dm_plex_transform_type refine_sbr \
            -adaptor_criterion label -adaptor_type gradient -adaptor_mixed_setup_function SetupMixed \
          -snes_adapt_sequence 2 -pc_type jacobi \
          -a1_refine_vec_tagger_type cdf -a1_refine_vec_tagger_box 0.9,1.0 \
            -a1_snes_adapt_view \
            -a1_adaptor_criterion label -a1_adaptor_type flux -a1_adaptor_mixed_setup_function SetupMixed \
          -a1_snes_adapt_sequence 1 -a1_pc_type jacobi -a1_mixed_pc_type jacobi
    test:
      suffix: sensor
      args: -sol_type sensor

TEST*/
