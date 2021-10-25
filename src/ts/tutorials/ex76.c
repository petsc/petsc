static char help[] = "Time-dependent Low Mach Flow in 2d and 3d channels with finite elements.\n\
We solve the Low Mach flow problem for both conducting and non-conducting fluids,\n\
using a parallel unstructured mesh (DMPLEX) to discretize it.\n\n\n";

/*F
The non-conducting Low Mach flow is time-dependent isoviscous Navier-Stokes flow. We discretize using the
finite element method on an unstructured mesh. The weak form equations are

\begin{align*}
    < q, \nabla\cdot u > = 0
    <v, du/dt> + <v, u \cdot \nabla u> + < \nabla v, \nu (\nabla u + {\nabla u}^T) > - < \nabla\cdot v, p >  - < v, f  >  = 0
    < w, u \cdot \nabla T > + < \nabla w, \alpha \nabla T > - < w, Q > = 0
\end{align*}

where $\nu$ is the kinematic viscosity and $\alpha$ is thermal diffusivity.

The conducting form is given in the ABLATE documentation [1,2] and derived in Principe and Codina [2].

For visualization, use

  -dm_view hdf5:$PWD/sol.h5 -sol_vec_view hdf5:$PWD/sol.h5::append -exact_vec_view hdf5:$PWD/sol.h5::append

[1] https://ubchrest.github.io/ablate/content/formulations/lowMachFlow/
[2] https://github.com/UBCHREST/ablate/blob/main/ablateCore/flow/lowMachFlow.c
[3] J. Principe and R. Codina, "Mathematical models for thermally coupled low speed flows", Adv. in Theo. and App. Mech., 2(1), pp.93--112, 2009.
F*/

#include <petscdmplex.h>
#include <petscsnes.h>
#include <petscts.h>
#include <petscds.h>
#include <petscbag.h>

typedef enum {MOD_INCOMPRESSIBLE, MOD_CONDUCTING, NUM_MOD_TYPES} ModType;
const char *modTypes[NUM_MOD_TYPES+1] = {"incompressible", "conducting", "unknown"};

typedef enum {SOL_QUADRATIC, SOL_CUBIC, SOL_CUBIC_TRIG, SOL_TAYLOR_GREEN, SOL_PIPE, SOL_PIPE_WIGGLY, NUM_SOL_TYPES} SolType;
const char *solTypes[NUM_SOL_TYPES+1] = {"quadratic", "cubic", "cubic_trig", "taylor_green", "pipe", "pipe_wiggly", "unknown"};

/* Fields */
const PetscInt VEL      = 0;
const PetscInt PRES     = 1;
const PetscInt TEMP     = 2;
/* Sources */
const PetscInt MOMENTUM = 0;
const PetscInt MASS     = 1;
const PetscInt ENERGY   = 2;
/* Constants */
const PetscInt STROUHAL = 0;
const PetscInt FROUDE   = 1;
const PetscInt REYNOLDS = 2;
const PetscInt PECLET   = 3;
const PetscInt P_TH     = 4;
const PetscInt MU       = 5;
const PetscInt NU       = 6;
const PetscInt C_P      = 7;
const PetscInt K        = 8;
const PetscInt ALPHA    = 9;
const PetscInt T_IN     = 10;
const PetscInt G_DIR    = 11;
const PetscInt EPSILON  = 12;

typedef struct {
  PetscReal Strouhal; /* Strouhal number */
  PetscReal Froude;   /* Froude number */
  PetscReal Reynolds; /* Reynolds number */
  PetscReal Peclet;   /* Peclet number */
  PetscReal p_th;     /* Thermodynamic pressure */
  PetscReal mu;       /* Dynamic viscosity */
  PetscReal nu;       /* Kinematic viscosity */
  PetscReal c_p;      /* Specific heat at constant pressure */
  PetscReal k;        /* Thermal conductivity */
  PetscReal alpha;    /* Thermal diffusivity */
  PetscReal T_in;     /* Inlet temperature */
  PetscReal g_dir;    /* Gravity direction */
  PetscReal epsilon;  /* Strength of perturbation */
} Parameter;

typedef struct {
  /* Problem definition */
  PetscBag  bag;          /* Holds problem parameters */
  ModType   modType;      /* Model type */
  SolType   solType;      /* MMS solution type */
  PetscBool hasNullSpace; /* Problem has the constant null space for pressure */
  /* Flow diagnostics */
  DM        dmCell;       /* A DM with piecewise constant discretization */
} AppCtx;

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
  CASE: quadratic
  In 2D we use exact solution:

    u = t + x^2 + y^2
    v = t + 2x^2 - 2xy
    p = x + y - 1
    T = t + x + y + 1
    f = <t (2x + 2y) + 2x^3 + 4x^2y - 2xy^2 -4\nu + 2, t (2x - 2y) + 4xy^2 + 2x^2y - 2y^3 -4\nu + 2>
    Q = 1 + 2t + 3x^2 - 2xy + y^2

  so that

    \nabla \cdot u = 2x - 2x = 0

  f = du/dt + u \cdot \nabla u - \nu \Delta u + \nabla p
    = <1, 1> + <t + x^2 + y^2, t + 2x^2 - 2xy> . <<2x, 4x - 2y>, <2y, -2x>> - \nu <4, 4> + <1, 1>
    = <t (2x + 2y) + 2x^3 + 4x^2y - 2xy^2, t (2x - 2y) + 2x^2y + 4xy^2 - 2y^3> + <-4 \nu + 2, -4\nu + 2>
    = <t (2x + 2y) + 2x^3 + 4x^2y - 2xy^2 - 4\nu + 2, t (2x - 2y) + 4xy^2 + 2x^2y - 2y^3 - 4\nu + 2>

  Q = dT/dt + u \cdot \nabla T - \alpha \Delta T
    = 1 + <t + x^2 + y^2, t + 2x^2 - 2xy> . <1, 1> - \alpha 0
    = 1 + 2t + 3x^2 - 2xy + y^2
*/

static PetscErrorCode quadratic_u(PetscInt Dim, PetscReal time, const PetscReal X[], PetscInt Nf, PetscScalar *u, void *ctx)
{
  u[0] = time + X[0]*X[0] + X[1]*X[1];
  u[1] = time + 2.0*X[0]*X[0] - 2.0*X[0]*X[1];
  return 0;
}
static PetscErrorCode quadratic_u_t(PetscInt Dim, PetscReal time, const PetscReal X[], PetscInt Nf, PetscScalar *u, void *ctx)
{
  u[0] = 1.0;
  u[1] = 1.0;
  return 0;
}

static PetscErrorCode quadratic_p(PetscInt Dim, PetscReal time, const PetscReal X[], PetscInt Nf, PetscScalar *p, void *ctx)
{
  p[0] = X[0] + X[1] - 1.0;
  return 0;
}

static PetscErrorCode quadratic_T(PetscInt Dim, PetscReal time, const PetscReal X[], PetscInt Nf, PetscScalar *T, void *ctx)
{
  T[0] = time + X[0] + X[1] + 1.0;
  return 0;
}
static PetscErrorCode quadratic_T_t(PetscInt Dim, PetscReal time, const PetscReal X[], PetscInt Nf, PetscScalar *T, void *ctx)
{
  T[0] = 1.0;
  return 0;
}

static void f0_quadratic_v(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                           const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                           const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                           PetscReal t, const PetscReal X[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f0[])
{
  const PetscReal nu = PetscRealPart(constants[NU]);

  f0[0] -= t*(2*X[0] + 2*X[1]) + 2*X[0]*X[0]*X[0] + 4*X[0]*X[0]*X[1] - 2*X[0]*X[1]*X[1] - 4.0*nu + 2;
  f0[1] -= t*(2*X[0] - 2*X[1]) + 4*X[0]*X[1]*X[1] + 2*X[0]*X[0]*X[1] - 2*X[1]*X[1]*X[1] - 4.0*nu + 2;
}

static void f0_quadratic_w(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                           const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                           const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                           PetscReal t, const PetscReal X[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f0[])
{
  f0[0] -= 2*t + 1 + 3*X[0]*X[0] - 2*X[0]*X[1] + X[1]*X[1];
}

/*
  CASE: quadratic
  In 2D we use exact solution:

    u = t + x^2 + y^2
    v = t + 2x^2 - 2xy
    p = x + y - 1
    T = t + x + y + 1
  rho = p^{th} / T

  so that

    \nabla \cdot u = 2x - 2x = 0
    grad u = <<2 x, 4x - 2y>, <2 y, -2x>>
    epsilon(u) = 1/2 (grad u + grad u^T) = <<2x, 2x>, <2x, -2x>>
    epsilon'(u) = epsilon(u) - 1/3 (div u) I = epsilon(u)
    div epsilon'(u) = <2, 2>

  f = rho S du/dt + rho u \cdot \nabla u - 2\mu/Re div \epsilon'(u) + \nabla p + rho / F^2 \hat y
    = rho S <1, 1> + rho <t + x^2 + y^2, t + 2x^2 - 2xy> . <<2x, 4x - 2y>, <2y, -2x>> - 2\mu/Re <2, 2> + <1, 1> + rho/F^2 <0, 1>
    = rho S <1, 1> + rho <t (2x + 2y) + 2x^3 + 4x^2y - 2xy^2, t (2x - 2y) + 2x^2y + 4xy^2 - 2y^3> - mu/Re <4, 4> + <1, 1> + rho/F^2 <0, 1>

  g = S rho_t + div (rho u)
    = -S pth T_t/T^2 + rho div (u) + u . grad rho
    = -S pth 1/T^2 - pth u . grad T / T^2
    = -pth / T^2 (S + 2t + 3 x^2 - 2xy + y^2)

  Q = rho c_p S dT/dt + rho c_p u . grad T - 1/Pe div k grad T
    = c_p S pth / T + c_p pth (2t + 3 x^2 - 2xy + y^2) / T - k/Pe 0
    = c_p pth / T (S + 2t + 3 x^2 - 2xy + y^2)
*/
static void f0_conduct_quadratic_v(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                                   const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                                   const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                                   PetscReal t, const PetscReal X[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f0[])
{
  const PetscReal S    = PetscRealPart(constants[STROUHAL]);
  const PetscReal F    = PetscRealPart(constants[FROUDE]);
  const PetscReal Re   = PetscRealPart(constants[REYNOLDS]);
  const PetscReal mu   = PetscRealPart(constants[MU]);
  const PetscReal p_th = PetscRealPart(constants[P_TH]);
  const PetscReal rho  = p_th / (t + X[0] + X[1] + 1.);
  const PetscInt  gd   = (PetscInt) PetscRealPart(constants[G_DIR]);

  f0[0]  -= rho * S + rho * (2.*t*(X[0] + X[1]) + 2.*X[0]*X[0]*X[0] + 4.*X[0]*X[0]*X[1] - 2.*X[0]*X[1]*X[1]) - 4.*mu/Re + 1.;
  f0[1]  -= rho * S + rho * (2.*t*(X[0] - X[1]) + 2.*X[0]*X[0]*X[1] + 4.*X[0]*X[1]*X[1] - 2.*X[1]*X[1]*X[1]) - 4.*mu/Re + 1.;
  f0[gd] -= rho/PetscSqr(F);
}

static void f0_conduct_quadratic_q(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                                   const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                                   const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                                   PetscReal t, const PetscReal X[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f0[])
{
  const PetscReal S    = PetscRealPart(constants[STROUHAL]);
  const PetscReal p_th = PetscRealPart(constants[P_TH]);

  f0[0] += p_th * (S + 2.*t + 3.*X[0]*X[0] - 2.*X[0]*X[1] + X[1]*X[1]) / PetscSqr(t + X[0] + X[1] + 1.);
}

static void f0_conduct_quadratic_w(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                                   const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                                   const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                                   PetscReal t, const PetscReal X[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f0[])
{
  const PetscReal S    = PetscRealPart(constants[STROUHAL]);
  const PetscReal c_p  = PetscRealPart(constants[C_P]);
  const PetscReal p_th = PetscRealPart(constants[P_TH]);

  f0[0] -= c_p*p_th * (S + 2.*t + 3.*X[0]*X[0] - 2.*X[0]*X[1] + X[1]*X[1]) / (t + X[0] + X[1] + 1.);
}

/*
  CASE: cubic
  In 2D we use exact solution:

    u = t + x^3 + y^3
    v = t + 2x^3 - 3x^2y
    p = 3/2 x^2 + 3/2 y^2 - 1
    T = t + 1/2 x^2 + 1/2 y^2
    f = < t(3x^2 + 3y^2) + 3x^5 + 6x^3y^2 - 6x^2y^3 - \nu(6x + 6y) + 3x + 1,
          t(3x^2 - 6xy) + 6x^2y^3 + 3x^4y - 6xy^4 - \nu(12x - 6y) + 3y + 1>
    Q = x^4 + xy^3 + 2x^3y - 3x^2y^2 + xt + yt - 2\alpha + 1

  so that

    \nabla \cdot u = 3x^2 - 3x^2 = 0

  du/dt + u \cdot \nabla u - \nu \Delta u + \nabla p - f
  = <1,1> + <t(3x^2 + 3y^2) + 3x^5 + 6x^3y^2 - 6x^2y^3, t(3x^2 - 6xy) + 6x^2y^3 + 3x^4y - 6xy^4> - \nu<6x + 6y, 12x - 6y> + <3x, 3y> - <t(3x^2 + 3y^2) + 3x^5 + 6x^3y^2 - 6x^2y^3 - \nu(6x + 6y) + 3x + 1, t(3x^2 - 6xy) + 6x^2y^3 + 3x^4y - 6xy^4 - \nu(12x - 6y) + 3y + 1>  = 0

  dT/dt + u \cdot \nabla T - \alpha \Delta T - Q = 1 + (x^3 + y^3) x + (2x^3 - 3x^2y) y - 2*\alpha - (x^4 + xy^3 + 2x^3y - 3x^2y^2 - 2*\alpha +1)   = 0
*/
static PetscErrorCode cubic_u(PetscInt Dim, PetscReal time, const PetscReal X[], PetscInt Nf, PetscScalar *u, void *ctx)
{
  u[0] = time + X[0]*X[0]*X[0] + X[1]*X[1]*X[1];
  u[1] = time + 2.0*X[0]*X[0]*X[0] - 3.0*X[0]*X[0]*X[1];
  return 0;
}
static PetscErrorCode cubic_u_t(PetscInt Dim, PetscReal time, const PetscReal X[], PetscInt Nf, PetscScalar *u, void *ctx)
{
  u[0] = 1.0;
  u[1] = 1.0;
  return 0;
}

static PetscErrorCode cubic_p(PetscInt Dim, PetscReal time, const PetscReal X[], PetscInt Nf, PetscScalar *p, void *ctx)
{
  p[0] = 3.0*X[0]*X[0]/2.0 + 3.0*X[1]*X[1]/2.0 - 1.0;
  return 0;
}

static PetscErrorCode cubic_T(PetscInt Dim, PetscReal time, const PetscReal X[], PetscInt Nf, PetscScalar *T, void *ctx)
{
  T[0] = time + X[0]*X[0]/2.0 + X[1]*X[1]/2.0;
  return 0;
}
static PetscErrorCode cubic_T_t(PetscInt Dim, PetscReal time, const PetscReal X[], PetscInt Nf, PetscScalar *T, void *ctx)
{
  T[0] = 1.0;
  return 0;
}

static void f0_cubic_v(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                       const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                       const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                       PetscReal t, const PetscReal X[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f0[])
{
  const PetscReal nu = PetscRealPart(constants[NU]);

  f0[0] -= (t*(3*X[0]*X[0] + 3*X[1]*X[1]) + 3*X[0]*X[0]*X[0]*X[0]*X[0] + 6*X[0]*X[0]*X[0]*X[1]*X[1] - 6*X[0]*X[0]*X[1]*X[1]*X[1] - ( 6*X[0] + 6*X[1])*nu + 3*X[0] + 1);
  f0[1] -= (t*(3*X[0]*X[0] - 6*X[0]*X[1]) + 3*X[0]*X[0]*X[0]*X[0]*X[1] + 6*X[0]*X[0]*X[1]*X[1]*X[1] - 6*X[0]*X[1]*X[1]*X[1]*X[1] - (12*X[0] - 6*X[1])*nu + 3*X[1] + 1);
}

static void f0_cubic_w(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                       const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                       const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                       PetscReal t, const PetscReal X[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f0[])
{
  const PetscReal alpha = PetscRealPart(constants[ALPHA]);

  f0[0] -= X[0]*X[0]*X[0]*X[0] + 2.0*X[0]*X[0]*X[0]*X[1] - 3.0*X[0]*X[0]*X[1]*X[1] + X[0]*X[1]*X[1]*X[1] + X[0]*t + X[1]*t - 2.0*alpha + 1;
}

/*
  CASE: cubic-trigonometric
  In 2D we use exact solution:

    u = beta cos t + x^3 + y^3
    v = beta sin t + 2x^3 - 3x^2y
    p = 3/2 x^2 + 3/2 y^2 - 1
    T = 20 cos t + 1/2 x^2 + 1/2 y^2
    f = < beta cos t 3x^2         + beta sin t (3y^2 - 1) + 3x^5 + 6x^3y^2 - 6x^2y^3 - \nu(6x + 6y)  + 3x,
          beta cos t (6x^2 - 6xy) - beta sin t (3x^2)     + 3x^4y + 6x^2y^3 - 6xy^4  - \nu(12x - 6y) + 3y>
    Q = beta cos t x + beta sin t (y - 1) + x^4 + 2x^3y - 3x^2y^2 + xy^3 - 2\alpha

  so that

    \nabla \cdot u = 3x^2 - 3x^2 = 0

  f = du/dt + u \cdot \nabla u - \nu \Delta u + \nabla p
    = <-sin t, cos t> + <cos t + x^3 + y^3, sin t + 2x^3 - 3x^2y> <<3x^2, 6x^2 - 6xy>, <3y^2, -3x^2>> - \nu <6x + 6y, 12x - 6y> + <3x, 3y>
    = <-sin t, cos t> + <cos t 3x^2 + 3x^5 + 3x^2y^3 + sin t 3y^2 + 6x^3y^2 - 9x^2y^3, cos t (6x^2 - 6xy) + 6x^5 - 6x^4y + 6x^2y^3 - 6xy^4 + sin t (-3x^2) - 6x^5 + 9x^4y> - \nu <6x + 6y, 12x - 6y> + <3x, 3y>
    = <cos t (3x^2)       + sin t (3y^2 - 1) + 3x^5 + 6x^3y^2 - 6x^2y^3 - \nu (6x + 6y)  + 3x,
       cos t (6x^2 - 6xy) - sin t (3x^2)     + 3x^4y + 6x^2y^3 - 6xy^4  - \nu (12x - 6y) + 3y>

  Q = dT/dt + u \cdot \nabla T - \alpha \Delta T
    = -sin t + <cos t + x^3 + y^3, sin t + 2x^3 - 3x^2y> . <x, y> - 2 \alpha
    = -sin t + cos t (x) + x^4 + xy^3 + sin t (y) + 2x^3y - 3x^2y^2 - 2 \alpha
    = cos t x + sin t (y - 1) + (x^4 + 2x^3y - 3x^2y^2 + xy^3 - 2 \alpha)
*/
static PetscErrorCode cubic_trig_u(PetscInt Dim, PetscReal time, const PetscReal X[], PetscInt Nf, PetscScalar *u, void *ctx)
{
  u[0] = 100.*PetscCosReal(time) + X[0]*X[0]*X[0] + X[1]*X[1]*X[1];
  u[1] = 100.*PetscSinReal(time) + 2.0*X[0]*X[0]*X[0] - 3.0*X[0]*X[0]*X[1];
  return 0;
}
static PetscErrorCode cubic_trig_u_t(PetscInt Dim, PetscReal time, const PetscReal X[], PetscInt Nf, PetscScalar *u, void *ctx)
{
  u[0] = -100.*PetscSinReal(time);
  u[1] =  100.*PetscCosReal(time);
  return 0;
}

static PetscErrorCode cubic_trig_p(PetscInt Dim, PetscReal time, const PetscReal X[], PetscInt Nf, PetscScalar *p, void *ctx)
{
  p[0] = 3.0*X[0]*X[0]/2.0 + 3.0*X[1]*X[1]/2.0 - 1.0;
  return 0;
}

static PetscErrorCode cubic_trig_T(PetscInt Dim, PetscReal time, const PetscReal X[], PetscInt Nf, PetscScalar *T, void *ctx)
{
  T[0] = 100.*PetscCosReal(time) + X[0]*X[0]/2.0 + X[1]*X[1]/2.0;
  return 0;
}
static PetscErrorCode cubic_trig_T_t(PetscInt Dim, PetscReal time, const PetscReal X[], PetscInt Nf, PetscScalar *T, void *ctx)
{
  T[0] = -100.*PetscSinReal(time);
  return 0;
}

static void f0_cubic_trig_v(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                            const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                            const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                            PetscReal t, const PetscReal X[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f0[])
{
  const PetscReal nu = PetscRealPart(constants[NU]);

  f0[0] -= 100.*PetscCosReal(t)*(3*X[0]*X[0])               + 100.*PetscSinReal(t)*(3*X[1]*X[1] - 1.) + 3*X[0]*X[0]*X[0]*X[0]*X[0] + 6*X[0]*X[0]*X[0]*X[1]*X[1] - 6*X[0]*X[0]*X[1]*X[1]*X[1] - ( 6*X[0] + 6*X[1])*nu + 3*X[0];
  f0[1] -= 100.*PetscCosReal(t)*(6*X[0]*X[0] - 6*X[0]*X[1]) - 100.*PetscSinReal(t)*(3*X[0]*X[0])      + 3*X[0]*X[0]*X[0]*X[0]*X[1] + 6*X[0]*X[0]*X[1]*X[1]*X[1] - 6*X[0]*X[1]*X[1]*X[1]*X[1] - (12*X[0] - 6*X[1])*nu + 3*X[1];
}

static void f0_cubic_trig_w(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                            const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                            const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                            PetscReal t, const PetscReal X[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f0[])
{
  const PetscReal alpha = PetscRealPart(constants[ALPHA]);

  f0[0] -= 100.*PetscCosReal(t)*X[0] + 100.*PetscSinReal(t)*(X[1] - 1.) + X[0]*X[0]*X[0]*X[0] + 2.0*X[0]*X[0]*X[0]*X[1] - 3.0*X[0]*X[0]*X[1]*X[1] + X[0]*X[1]*X[1]*X[1] - 2.0*alpha;
}

/*
  CASE: Taylor-Green vortex
  In 2D we use exact solution:

    u = 1 - cos(\pi(x - t)) sin(\pi(y - t)) exp(-2 \pi^2 \nu t)
    v = 1 + sin(\pi(x - t)) cos(\pi(y - t)) exp(-2 \pi^2 \nu t)
    p = -1/4 [cos(2 \pi(x - t)) + cos(2 \pi(y - t))] exp(-4 \pi^2 \nu t)
    T = t + x + y
    f = <\nu \pi^2 exp(-2\nu \pi^2 t) cos(\pi(x-t)) sin(\pi(y-t)), -\nu \pi^2 exp(-2\nu \pi^2 t) sin(\pi(x-t)) cos(\pi(y-t))  >
    Q = 3 + sin(\pi(x-y)) exp(-2\nu \pi^2 t)

  so that

  \nabla \cdot u = \pi sin(\pi(x - t)) sin(\pi(y - t)) exp(-2 \pi^2 \nu t) - \pi sin(\pi(x - t)) sin(\pi(y - t)) exp(-2 \pi^2 \nu t) = 0

  f = du/dt + u \cdot \nabla u - \nu \Delta u + \nabla p
    = <-\pi (sin(\pi(x - t)) sin(\pi(y - t)) - cos(\pi(x - t)) cos(\pi(y - t)) - 2\pi cos(\pi(x - t)) sin(\pi(y - t))) exp(-2 \pi^2 \nu t),
        \pi (sin(\pi(x - t)) sin(\pi(y - t)) - cos(\pi(x - t)) cos(\pi(y - t)) - 2\pi sin(\pi(x - t)) cos(\pi(y - t))) exp(-2 \pi^2 \nu t)>
    + < \pi (1 - cos(\pi(x - t)) sin(\pi(y - t)) exp(-2 \pi^2 \nu t)) sin(\pi(x - t)) sin(\pi(y - t)) exp(-2 \pi^2 \nu t),
        \pi (1 - cos(\pi(x - t)) sin(\pi(y - t)) exp(-2 \pi^2 \nu t)) cos(\pi(x - t)) cos(\pi(y - t)) exp(-2 \pi^2 \nu t)>
    + <-\pi (1 + sin(\pi(x - t)) cos(\pi(y - t)) exp(-2 \pi^2 \nu t)) cos(\pi(x - t)) cos(\pi(y - t)) exp(-2 \pi^2 \nu t),
       -\pi (1 + sin(\pi(x - t)) cos(\pi(y - t)) exp(-2 \pi^2 \nu t)) sin(\pi(x - t)) sin(\pi(y - t)) exp(-2 \pi^2 \nu t)>
    + <-2\pi^2 cos(\pi(x - t)) sin(\pi(y - t)) exp(-2 \pi^2 \nu t),
        2\pi^2 sin(\pi(x - t)) cos(\pi(y - t)) exp(-2 \pi^2 \nu t)>
    + < \pi/2 sin(2\pi(x - t)) exp(-4 \pi^2 \nu t),
        \pi/2 sin(2\pi(y - t)) exp(-4 \pi^2 \nu t)>
    = <-\pi (sin(\pi(x - t)) sin(\pi(y - t)) - cos(\pi(x - t)) cos(\pi(y - t)) - 2\pi cos(\pi(x - t)) sin(\pi(y - t))) exp(-2 \pi^2 \nu t),
        \pi (sin(\pi(x - t)) sin(\pi(y - t)) - cos(\pi(x - t)) cos(\pi(y - t)) - 2\pi sin(\pi(x - t)) cos(\pi(y - t))) exp(-2 \pi^2 \nu t)>
    + < \pi sin(\pi(x - t)) sin(\pi(y - t)) exp(-2 \pi^2 \nu t),
        \pi cos(\pi(x - t)) cos(\pi(y - t)) exp(-2 \pi^2 \nu t)>
    + <-\pi cos(\pi(x - t)) cos(\pi(y - t)) exp(-2 \pi^2 \nu t),
       -\pi sin(\pi(x - t)) sin(\pi(y - t)) exp(-2 \pi^2 \nu t)>
    + <-\pi/2 sin(2\pi(x - t)) exp(-4 \pi^2 \nu t),
       -\pi/2 sin(2\pi(y - t)) exp(-4 \pi^2 \nu t)>
    + <-2\pi^2 cos(\pi(x - t)) sin(\pi(y - t)) exp(-2 \pi^2 \nu t),
        2\pi^2 sin(\pi(x - t)) cos(\pi(y - t)) exp(-2 \pi^2 \nu t)>
    + < \pi/2 sin(2\pi(x - t)) exp(-4 \pi^2 \nu t),
        \pi/2 sin(2\pi(y - t)) exp(-4 \pi^2 \nu t)>
    = <-\pi (sin(\pi(x - t)) sin(\pi(y - t)) - cos(\pi(x - t)) cos(\pi(y - t))) exp(-2 \pi^2 \nu t),
        \pi (sin(\pi(x - t)) sin(\pi(y - t)) - cos(\pi(x - t)) cos(\pi(y - t))) exp(-2 \pi^2 \nu t)>
    + < \pi sin(\pi(x - t)) sin(\pi(y - t)) exp(-2 \pi^2 \nu t),
        \pi cos(\pi(x - t)) cos(\pi(y - t)) exp(-2 \pi^2 \nu t)>
    + <-\pi cos(\pi(x - t)) cos(\pi(y - t)) exp(-2 \pi^2 \nu t),
       -\pi sin(\pi(x - t)) sin(\pi(y - t)) exp(-2 \pi^2 \nu t)>
    = < \pi cos(\pi(x - t)) cos(\pi(y - t)),
        \pi sin(\pi(x - t)) sin(\pi(y - t))>
    + <-\pi cos(\pi(x - t)) cos(\pi(y - t)),
       -\pi sin(\pi(x - t)) sin(\pi(y - t))> = 0
  Q = dT/dt + u \cdot \nabla T - \alpha \Delta T
    = 1 + u \cdot <1, 1> - 0
    = 1 + u + v
*/

static PetscErrorCode taylor_green_u(PetscInt Dim, PetscReal time, const PetscReal X[], PetscInt Nf, PetscScalar *u, void *ctx)
{
  u[0] = 1 - PetscCosReal(PETSC_PI*(X[0]-time))*PetscSinReal(PETSC_PI*(X[1]-time))*PetscExpReal(-2*PETSC_PI*PETSC_PI*time);
  u[1] = 1 + PetscSinReal(PETSC_PI*(X[0]-time))*PetscCosReal(PETSC_PI*(X[1]-time))*PetscExpReal(-2*PETSC_PI*PETSC_PI*time);
  return 0;
}
static PetscErrorCode taylor_green_u_t(PetscInt Dim, PetscReal time, const PetscReal X[], PetscInt Nf, PetscScalar *u, void *ctx)
{
  u[0] = -PETSC_PI*(PetscSinReal(PETSC_PI*(X[0]-time))*PetscSinReal(PETSC_PI*(X[1]-time))
                  - PetscCosReal(PETSC_PI*(X[0]-time))*PetscCosReal(PETSC_PI*(X[1]-time))
                  - 2*PETSC_PI*PetscCosReal(PETSC_PI*(X[0]-time))*PetscSinReal(PETSC_PI*(X[1]-time)))*PetscExpReal(-2*PETSC_PI*PETSC_PI*time);
  u[1] =  PETSC_PI*(PetscSinReal(PETSC_PI*(X[0]-time))*PetscSinReal(PETSC_PI*(X[1]-time))
                  - PetscCosReal(PETSC_PI*(X[0]-time))*PetscCosReal(PETSC_PI*(X[1]-time))
                  - 2*PETSC_PI*PetscSinReal(PETSC_PI*(X[0]-time))*PetscCosReal(PETSC_PI*(X[1]-time)))*PetscExpReal(-2*PETSC_PI*PETSC_PI*time);
  return 0;
}

static PetscErrorCode taylor_green_p(PetscInt Dim, PetscReal time, const PetscReal X[], PetscInt Nf, PetscScalar *p, void *ctx)
{
  p[0] = -0.25*(PetscCosReal(2*PETSC_PI*(X[0]-time)) + PetscCosReal(2*PETSC_PI*(X[1]-time)))*PetscExpReal(-4*PETSC_PI*PETSC_PI*time);
  return 0;
}

static PetscErrorCode taylor_green_p_t(PetscInt Dim, PetscReal time, const PetscReal X[], PetscInt Nf, PetscScalar *p, void *ctx)
{
  p[0] = PETSC_PI*(0.5*(PetscSinReal(2*PETSC_PI*(X[0]-time)) + PetscSinReal(2*PETSC_PI*(X[1]-time)))
                 + PETSC_PI*(PetscCosReal(2*PETSC_PI*(X[0]-time)) + PetscCosReal(2*PETSC_PI*(X[1]-time))))*PetscExpReal(-4*PETSC_PI*PETSC_PI*time);
  return 0;
}

static PetscErrorCode taylor_green_T(PetscInt Dim, PetscReal time, const PetscReal X[], PetscInt Nf, PetscScalar *T, void *ctx)
{
  T[0] = time + X[0] + X[1];
  return 0;
}
static PetscErrorCode taylor_green_T_t(PetscInt Dim, PetscReal time, const PetscReal X[], PetscInt Nf, PetscScalar *T, void *ctx)
{
  T[0] = 1.0;
  return 0;
}

static void f0_taylor_green_w(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                            const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                            const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                            PetscReal t, const PetscReal X[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f0[])
{
  PetscScalar vel[2];

  taylor_green_u(dim, t, X, Nf, vel, NULL);
  f0[0] -= 1.0 + vel[0] + vel[1];
}

/*
  CASE: Pipe flow
  Poiseuille flow, with the incoming fluid having a parabolic temperature profile and the side walls being held at T_in

    u = \Delta Re/(2 mu) y (1 - y)
    v = 0
    p = -\Delta x
    T = y (1 - y) + T_in
  rho = p^{th} / T

  so that

    \nabla \cdot u = 0 - 0 = 0
    grad u = \Delta Re/(2 mu) <<0, 0>, <1 - 2y, 0>>
    epsilon(u) = 1/2 (grad u + grad u^T) = \Delta Re/(4 mu) <<0, 1 - 2y>, <<1 - 2y, 0>>
    epsilon'(u) = epsilon(u) - 1/3 (div u) I = epsilon(u)
    div epsilon'(u) = -\Delta Re/(2 mu) <1, 0>

  f = rho S du/dt + rho u \cdot \nabla u - 2\mu/Re div \epsilon'(u) + \nabla p + rho / F^2 \hat y
    = 0 + 0 - div (2\mu/Re \epsilon'(u) - pI) + rho / F^2 \hat y
    = -\Delta div <<x, (1 - 2y)/2>, <<(1 - 2y)/2, x>> + rho / F^2 \hat y
    = \Delta <1, 0> - \Delta <1, 0> + rho/F^2 <0, 1>
    = rho/F^2 <0, 1>

  g = S rho_t + div (rho u)
    = 0 + rho div (u) + u . grad rho
    = 0 + 0 - pth u . grad T / T^2
    = 0

  Q = rho c_p S dT/dt + rho c_p u . grad T - 1/Pe div k grad T
    = 0 + c_p pth / T 0 + 2 k/Pe
    = 2 k/Pe

  The boundary conditions on the top and bottom are zero velocity and T_in temperature. The boundary term is

    (2\mu/Re \epsilon'(u) - p I) . n = \Delta <<x, (1 - 2y)/2>, <<(1 - 2y)/2, x>> . n

  so that

    x = 0: \Delta <<0, (1 - 2y)/2>, <<(1 - 2y)/2, 0>> . <-1, 0> = <0, (2y - 1)/2>
    x = 1: \Delta <<1, (1 - 2y)/2>, <<(1 - 2y)/2, 1>> . <1, 0> = <1, (1 - 2y)/2>
*/
static PetscErrorCode pipe_u(PetscInt dim, PetscReal time, const PetscReal X[], PetscInt Nf, PetscScalar *u, void *ctx)
{
  Parameter *param = (Parameter *) ctx;

  u[0] = (0.5*param->Reynolds / param->mu) * X[1]*(1.0 - X[1]);
  u[1] = 0.0;
  return 0;
}
static PetscErrorCode pipe_u_t(PetscInt dim, PetscReal time, const PetscReal X[], PetscInt Nf, PetscScalar *u, void *ctx)
{
  u[0] = 0.0;
  u[1] = 0.0;
  return 0;
}

static PetscErrorCode pipe_p(PetscInt dim, PetscReal time, const PetscReal X[], PetscInt Nf, PetscScalar *p, void *ctx)
{
  p[0] = -X[0];
  return 0;
}
static PetscErrorCode pipe_p_t(PetscInt dim, PetscReal time, const PetscReal X[], PetscInt Nf, PetscScalar *p, void *ctx)
{
  p[0] = 0.0;
  return 0;
}

static PetscErrorCode pipe_T(PetscInt dim, PetscReal time, const PetscReal X[], PetscInt Nf, PetscScalar *T, void *ctx)
{
  Parameter *param = (Parameter *) ctx;

  T[0] = X[1]*(1.0 - X[1]) + param->T_in;
  return 0;
}
static PetscErrorCode pipe_T_t(PetscInt dim, PetscReal time, const PetscReal X[], PetscInt Nf, PetscScalar *T, void *ctx)
{
  T[0] = 0.0;
  return 0;
}

static void f0_conduct_pipe_v(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                              const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                              const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                              PetscReal t, const PetscReal X[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f0[])
{
  const PetscReal F    = PetscRealPart(constants[FROUDE]);
  const PetscReal p_th = PetscRealPart(constants[P_TH]);
  const PetscReal T_in = PetscRealPart(constants[T_IN]);
  const PetscReal rho  = p_th / (X[1]*(1. - X[1]) + T_in);
  const PetscInt  gd   = (PetscInt) PetscRealPart(constants[G_DIR]);

  f0[gd] -= rho/PetscSqr(F);
}

static void f0_conduct_bd_pipe_v(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                                 const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                                 const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                                 PetscReal t, const PetscReal X[], const PetscReal n[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f0[])
{
  PetscReal sigma[4] = {X[0], 0.5*(1. - 2.*X[1]), 0.5*(1. - 2.*X[1]), X[0]};
  PetscInt  d, e;

  for (d = 0; d < dim; ++d) {
    for (e = 0; e < dim; ++e) {
      f0[d] -= sigma[d*dim + e] * n[e];
    }
  }
}

static void f0_conduct_pipe_q(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                              const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                              const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                              PetscReal t, const PetscReal X[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f0[])
{
  f0[0] += 0.0;
}

static void f0_conduct_pipe_w(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                              const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                              const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                              PetscReal t, const PetscReal X[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f0[])
{
  const PetscReal k  = PetscRealPart(constants[K]);
  const PetscReal Pe = PetscRealPart(constants[PECLET]);

  f0[0] -= 2*k/Pe;
}

/*
  CASE: Wiggly pipe flow
  Perturbed Poiseuille flow, with the incoming fluid having a perturbed parabolic temperature profile and the side walls being held at T_in

    u = \Delta Re/(2 mu) [y (1 - y) + a sin(pi y)]
    v = 0
    p = -\Delta x
    T = y (1 - y) + a sin(pi y) + T_in
  rho = p^{th} / T

  so that

    \nabla \cdot u = 0 - 0 = 0
    grad u = \Delta Re/(2 mu) <<0, 0>, <1 - 2y + a pi cos(pi y), 0>>
    epsilon(u) = 1/2 (grad u + grad u^T) = \Delta Re/(4 mu) <<0, 1 - 2y + a pi cos(pi y)>, <<1 - 2y + a pi cos(pi y), 0>>
    epsilon'(u) = epsilon(u) - 1/3 (div u) I = epsilon(u)
    div epsilon'(u) = -\Delta Re/(2 mu) <1 + a pi^2/2 sin(pi y), 0>

  f = rho S du/dt + rho u \cdot \nabla u - 2\mu/Re div \epsilon'(u) + \nabla p + rho / F^2 \hat y
    = 0 + 0 - div (2\mu/Re \epsilon'(u) - pI) + rho / F^2 \hat y
    = -\Delta div <<x, (1 - 2y)/2 + a pi/2 cos(pi y)>, <<(1 - 2y)/2 + a pi/2 cos(pi y), x>> + rho / F^2 \hat y
    = -\Delta <1 - 1 - a pi^2/2 sin(pi y), 0> + rho/F^2 <0, 1>
    = a \Delta pi^2/2 sin(pi y) <1, 0> + rho/F^2 <0, 1>

  g = S rho_t + div (rho u)
    = 0 + rho div (u) + u . grad rho
    = 0 + 0 - pth u . grad T / T^2
    = 0

  Q = rho c_p S dT/dt + rho c_p u . grad T - 1/Pe div k grad T
    = 0 + c_p pth / T 0 - k/Pe div <0, 1 - 2y + a pi cos(pi y)>
    = - k/Pe (-2 - a pi^2 sin(pi y))
    = 2 k/Pe (1 + a pi^2/2 sin(pi y))

  The boundary conditions on the top and bottom are zero velocity and T_in temperature. The boundary term is

    (2\mu/Re \epsilon'(u) - p I) . n = \Delta <<x, (1 - 2y)/2 + a pi/2 cos(pi y)>, <<(1 - 2y)/2 + a pi/2 cos(pi y), x>> . n

  so that

    x = 0: \Delta <<0, (1 - 2y)/2>, <<(1 - 2y)/2, 0>> . <-1, 0> = <0, (2y - 1)/2 - a pi/2 cos(pi y)>
    x = 1: \Delta <<1, (1 - 2y)/2>, <<(1 - 2y)/2, 1>> . < 1, 0> = <1, (1 - 2y)/2 + a pi/2 cos(pi y)>
*/
static PetscErrorCode pipe_wiggly_u(PetscInt dim, PetscReal time, const PetscReal X[], PetscInt Nf, PetscScalar *u, void *ctx)
{
  Parameter *param = (Parameter *) ctx;

  u[0] = (0.5*param->Reynolds / param->mu) * (X[1]*(1.0 - X[1]) + param->epsilon*PetscSinReal(PETSC_PI*X[1]));
  u[1] = 0.0;
  return 0;
}
static PetscErrorCode pipe_wiggly_u_t(PetscInt dim, PetscReal time, const PetscReal X[], PetscInt Nf, PetscScalar *u, void *ctx)
{
  u[0] = 0.0;
  u[1] = 0.0;
  return 0;
}

static PetscErrorCode pipe_wiggly_p(PetscInt dim, PetscReal time, const PetscReal X[], PetscInt Nf, PetscScalar *p, void *ctx)
{
  p[0] = -X[0];
  return 0;
}
static PetscErrorCode pipe_wiggly_p_t(PetscInt dim, PetscReal time, const PetscReal X[], PetscInt Nf, PetscScalar *p, void *ctx)
{
  p[0] = 0.0;
  return 0;
}

static PetscErrorCode pipe_wiggly_T(PetscInt dim, PetscReal time, const PetscReal X[], PetscInt Nf, PetscScalar *T, void *ctx)
{
  Parameter *param = (Parameter *) ctx;

  T[0] = X[1]*(1.0 - X[1]) + param->epsilon*PetscSinReal(PETSC_PI*X[1]) + param->T_in;
  return 0;
}
static PetscErrorCode pipe_wiggly_T_t(PetscInt dim, PetscReal time, const PetscReal X[], PetscInt Nf, PetscScalar *T, void *ctx)
{
  T[0] = 0.0;
  return 0;
}

static void f0_conduct_pipe_wiggly_v(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                                     const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                                     const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                                     PetscReal t, const PetscReal X[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f0[])
{
  const PetscReal F    = PetscRealPart(constants[FROUDE]);
  const PetscReal p_th = PetscRealPart(constants[P_TH]);
  const PetscReal T_in = PetscRealPart(constants[T_IN]);
  const PetscReal eps  = PetscRealPart(constants[EPSILON]);
  const PetscReal rho  = p_th / (X[1]*(1. - X[1]) + T_in);
  const PetscInt  gd   = (PetscInt) PetscRealPart(constants[G_DIR]);

  f0[0]  -= eps*0.5*PetscSqr(PETSC_PI)*PetscSinReal(PETSC_PI*X[1]);
  f0[gd] -= rho/PetscSqr(F);
}

static void f0_conduct_bd_pipe_wiggly_v(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                                        const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                                        const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                                        PetscReal t, const PetscReal X[], const PetscReal n[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f0[])
{
  const PetscReal eps = PetscRealPart(constants[EPSILON]);
  PetscReal sigma[4] = {X[0], 0.5*(1. - 2.*X[1]) + eps*0.5*PETSC_PI*PetscCosReal(PETSC_PI*X[1]), 0.5*(1. - 2.*X[1]) + eps*0.5*PETSC_PI*PetscCosReal(PETSC_PI*X[1]), X[0]};
  PetscInt  d, e;

  for (d = 0; d < dim; ++d) {
    for (e = 0; e < dim; ++e) {
      f0[d] -= sigma[d*dim + e] * n[e];
    }
  }
}

static void f0_conduct_pipe_wiggly_q(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                                     const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                                     const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                                     PetscReal t, const PetscReal X[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f0[])
{
  f0[0] += 0.0;
}

static void f0_conduct_pipe_wiggly_w(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                                     const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                                     const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                                     PetscReal t, const PetscReal X[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f0[])
{
  const PetscReal k  = PetscRealPart(constants[K]);
  const PetscReal Pe = PetscRealPart(constants[PECLET]);
  const PetscReal eps = PetscRealPart(constants[EPSILON]);

  f0[0] -= 2*k/Pe*(1.0 + eps*0.5*PetscSqr(PETSC_PI)*PetscSinReal(PETSC_PI*X[1]));
}

/*      Physics Kernels      */

static void f0_q(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                 const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                 const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                 PetscReal t, const PetscReal X[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f0[])
{
  PetscInt d;
  for (d = 0, f0[0] = 0.0; d < dim; ++d) f0[0] += u_x[d*dim+d];
}

/* -\frac{Sp^{th}}{T^2} \frac{\partial T}{\partial t} + \frac{p^{th}}{T} \nabla \cdot \vb{u} - \frac{p^{th}}{T^2} \vb{u} \cdot \nabla T */
static void f0_conduct_q(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                         const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                         const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                         PetscReal t, const PetscReal X[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f0[])
{
  const PetscReal S    = PetscRealPart(constants[STROUHAL]);
  const PetscReal p_th = PetscRealPart(constants[P_TH]);
  PetscInt        d;

  // -\frac{S p^{th}}{T^2} \frac{\partial T}{\partial t}
  f0[0] += -u_t[uOff[TEMP]] * S * p_th / PetscSqr(u[uOff[TEMP]]);

  // \frac{p^{th}}{T} \nabla \cdot \vb{u}
  for (d = 0; d < dim; ++d) {
    f0[0] += p_th / u[uOff[TEMP]] * u_x[uOff_x[VEL] + d*dim + d];
  }

  // - \frac{p^{th}}{T^2} \vb{u} \cdot \nabla T
  for (d = 0; d < dim; ++d) {
    f0[0] -= p_th / (u[uOff[TEMP]] * u[uOff[TEMP]]) * u[uOff[VEL] + d] * u_x[uOff_x[TEMP] + d];
  }

  // Add in any fixed source term
  if (NfAux > 0) {
    f0[0] += a[aOff[MASS]];
  }
}

/* \vb{u}_t + \vb{u} \cdot \nabla\vb{u} */
static void f0_v(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                 const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                 const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                 PetscReal t, const PetscReal X[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f0[])
{
  const PetscInt Nc = dim;
  PetscInt       c, d;

  for (c = 0; c < Nc; ++c) {
    /* \vb{u}_t */
    f0[c] += u_t[uOff[VEL] + c];
    /* \vb{u} \cdot \nabla\vb{u} */
    for (d = 0; d < dim; ++d) f0[c] += u[uOff[VEL] + d]*u_x[uOff_x[VEL] + c*dim + d];
  }
}

/* \rho S \frac{\partial \vb{u}}{\partial t} + \rho \vb{u} \cdot \nabla \vb{u} + \rho \frac{\hat{\vb{z}}}{F^2} */
static void f0_conduct_v(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                         const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                         const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                         PetscReal t, const PetscReal X[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f0[])
{
  const PetscReal S    = PetscRealPart(constants[STROUHAL]);
  const PetscReal F    = PetscRealPart(constants[FROUDE]);
  const PetscReal p_th = PetscRealPart(constants[P_TH]);
  const PetscReal rho  = p_th / PetscRealPart(u[uOff[TEMP]]);
  const PetscInt  gdir = (PetscInt) PetscRealPart(constants[G_DIR]);
  PetscInt        Nc   = dim;
  PetscInt        c, d;

  // \rho S \frac{\partial \vb{u}}{\partial t}
  for (d = 0; d < dim; ++d) {
    f0[d] = rho * S * u_t[uOff[VEL] + d];
  }

  // \rho \vb{u} \cdot \nabla \vb{u}
  for (c = 0; c < Nc; ++c) {
    for (d = 0; d < dim; ++d) {
      f0[c] += rho * u[uOff[VEL] + d] * u_x[uOff_x[VEL] + c*dim + d];
    }
  }

  // rho \hat{z}/F^2
  f0[gdir] += rho / (F*F);

  // Add in any fixed source term
  if (NfAux > 0) {
    for (d = 0; d < dim; ++d) {
      f0[d] += a[aOff[MOMENTUM] + d];
    }
  }
}

/*f1_v = \nu[grad(u) + grad(u)^T] - pI */
static void f1_v(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                 const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                 const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                 PetscReal t, const PetscReal X[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f1[])
{
  const PetscReal nu = PetscRealPart(constants[NU]);
  const PetscInt  Nc = dim;
  PetscInt        c, d;

  for (c = 0; c < Nc; ++c) {
    for (d = 0; d < dim; ++d) {
      f1[c*dim+d] = nu*(u_x[c*dim+d] + u_x[d*dim+c]);
    }
    f1[c*dim+c] -= u[uOff[1]];
  }
}

/* 2 \mu/Re (1/2 (\nabla \vb{u} + \nabla \vb{u}^T) - 1/3 (\nabla \cdot \vb{u}) I) - p I */
static void f1_conduct_v(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                         const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                         const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                         PetscReal t, const PetscReal X[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f1[])
{
  const PetscReal Re    = PetscRealPart(constants[REYNOLDS]);
  const PetscReal mu    = PetscRealPart(constants[MU]);
  const PetscReal coef  = mu / Re;
  PetscReal       u_div = 0.0;
  const PetscInt  Nc    = dim;
  PetscInt        c, d;

  for (c = 0; c < Nc; ++c) {
    u_div += PetscRealPart(u_x[uOff_x[VEL] + c*dim + c]);
  }

  for (c = 0; c < Nc; ++c) {
    // 2 \mu/Re 1/2 (\nabla \vb{u} + \nabla \vb{u}^T
    for (d = 0; d < dim; ++d) {
      f1[c*dim + d] += coef * (u_x[uOff_x[VEL] + c*dim + d] + u_x[uOff_x[VEL] + d*dim + c]);
    }
    // -2/3 \mu/Re (\nabla \cdot \vb{u}) I
    f1[c * dim + c] -= 2.0 * coef / 3.0 * u_div;
  }

  // -p I
  for (c = 0; c < Nc; ++c) {
    f1[c*dim + c] -= u[uOff[PRES]];
  }
}

/* T_t + \vb{u} \cdot \nabla T */
static void f0_w(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                 const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                 const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                 PetscReal t, const PetscReal X[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f0[])
{
  PetscInt d;

  /* T_t */
  f0[0] += u_t[uOff[TEMP]];
  /* \vb{u} \cdot \nabla T */
  for (d = 0; d < dim; ++d) f0[0] += u[uOff[VEL] + d] * u_x[uOff_x[TEMP] + d];
}

/* \frac{C_p S p^{th}}{T} \frac{\partial T}{\partial t} + \frac{C_p p^{th}}{T} \vb{u} \cdot \nabla T */
static void f0_conduct_w(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                         const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                         const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                         PetscReal t, const PetscReal X[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f0[])
{
  const PetscReal S    = PetscRealPart(constants[STROUHAL]);
  const PetscReal c_p  = PetscRealPart(constants[C_P]);
  const PetscReal p_th = PetscRealPart(constants[P_TH]);
  PetscInt        d;

  // \frac{C_p S p^{th}}{T} \frac{\partial T}{\partial t}
  f0[0] = c_p * S * p_th / u[uOff[TEMP]] * u_t[uOff[TEMP]];

  // \frac{C_p p^{th}}{T} \vb{u} \cdot \nabla T
  for (d = 0; d < dim; ++d) {
    f0[0] += c_p * p_th / u[uOff[TEMP]] * u[uOff[VEL] + d] * u_x[uOff_x[TEMP] + d];
  }

  // Add in any fixed source term
  if (NfAux > 0) {
    f0[0] += a[aOff[ENERGY]];
  }
}

static void f1_w(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                 const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                 const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                 PetscReal t, const PetscReal X[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f1[])
{
  const PetscReal alpha = PetscRealPart(constants[ALPHA]);
  PetscInt        d;

  for (d = 0; d < dim; ++d) f1[d] = alpha*u_x[uOff_x[2]+d];
}

/* \frac{k}{Pe} \nabla T */
static void f1_conduct_w(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                         const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                         const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                         PetscReal t, const PetscReal X[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f1[])
{
  const PetscReal Pe = PetscRealPart(constants[PECLET]);
  const PetscReal k  = PetscRealPart(constants[K]);
  PetscInt        d;

  // \frac{k}{Pe} \nabla T
  for (d = 0; d < dim; ++d) {
    f1[d] = k / Pe * u_x[uOff_x[TEMP] + d];
  }
}

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
      g0[c*Nc+d] += u_x[ c*Nc+d];
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

static void g0_conduct_qu(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                          const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                          const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                          PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g0[])
{
  const PetscReal p_th = PetscRealPart(constants[P_TH]);
  PetscInt        d;

  // - \phi_i \frac{p^{th}}{T^2} \frac{\partial T}{\partial x_c} \psi_{j, u_c}
  for (d = 0; d < dim; ++d) {
    g0[d] = -p_th / PetscSqr(u[uOff[TEMP]]) * u_x[uOff_x[TEMP] + d];
  }
}

static void g1_conduct_qu(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                          const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                          const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                          PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g1[])
{
  const PetscReal p_th = PetscRealPart(constants[P_TH]);
  PetscInt        d;

  // \phi_i \frac{p^{th}}{T} \frac{\partial \psi_{u_c,j}}{\partial x_c}
  for (d = 0; d < dim; ++d) {
    g1[d * dim + d] = p_th / u[uOff[TEMP]];
  }
}

static void g0_conduct_qT(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                          const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                          const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                          PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g0[])
{
  const PetscReal S    = PetscRealPart(constants[STROUHAL]);
  const PetscReal p_th = PetscRealPart(constants[P_TH]);
  PetscInt        d;

  // - \phi_i \frac{S p^{th}}{T^2} \psi_j
  g0[0] -= S * p_th / PetscSqr(u[uOff[TEMP]]) * u_tShift;
  // \phi_i 2 \frac{S p^{th}}{T^3} T_t \psi_j
  g0[0] += 2.0 * S * p_th / PetscPowScalarInt(u[uOff[TEMP]], 3) * u_t[uOff[TEMP]];
  // \phi_i \frac{p^{th}}{T^2} \left( - \nabla \cdot \vb{u} \psi_j + \frac{2}{T} \vb{u} \cdot \nabla T \psi_j \right)
  for (d = 0; d < dim; ++d) {
    g0[0] += p_th / PetscSqr(u[uOff[TEMP]]) * (-u_x[uOff_x[VEL] + d * dim + d] + 2.0 / u[uOff[TEMP]] * u[uOff[VEL] + d] * u_x[uOff_x[TEMP] + d]);
  }
}

static void g1_conduct_qT(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                          const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                          const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                          PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g1[])
{
  const PetscReal p_th = PetscRealPart(constants[P_TH]);
  PetscInt        d;

  // - \phi_i \frac{p^{th}}{T^2} \vb{u} \cdot \nabla \psi_j
  for (d = 0; d < dim; ++d) {
    g1[d] = -p_th / PetscSqr(u[uOff[TEMP]]) * u[uOff[VEL] + d];
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
   const PetscReal nu = PetscRealPart(constants[NU]);
   const PetscInt  Nc = dim;
   PetscInt        c, d;

  for (c = 0; c < Nc; ++c) {
    for (d = 0; d < dim; ++d) {
      g3[((c*Nc+c)*dim+d)*dim+d] += nu;
      g3[((c*Nc+d)*dim+d)*dim+c] += nu;
    }
  }
}

static void g0_conduct_vT(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                          const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                          const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                          PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g0[])
{
  const PetscReal S    = PetscRealPart(constants[STROUHAL]);
  const PetscReal F    = PetscRealPart(constants[FROUDE]);
  const PetscReal p_th = PetscRealPart(constants[P_TH]);
  const PetscInt  gdir = (PetscInt) PetscRealPart(constants[G_DIR]);
  const PetscInt  Nc = dim;
  PetscInt        c, d;

  // - \vb{\phi}_i \cdot \vb{u}_t \frac{p^{th} S}{T^2} \psi_j
  for (d = 0; d < dim; ++d) {
    g0[d] -= p_th * S / PetscSqr(u[uOff[TEMP]]) * u_t[uOff[VEL] + d];
  }

  // - \vb{\phi}_i \cdot \vb{u} \cdot \nabla \vb{u} \frac{p^{th}}{T^2} \psi_j
  for (c = 0; c < Nc; ++c) {
    for (d = 0; d < dim; ++d) {
      g0[c] -= p_th / PetscSqr(u[uOff[TEMP]]) * u[uOff[VEL] + d] * u_x[uOff_x[VEL] + c * dim + d];
    }
  }

  // - \vb{\phi}_i \cdot \vu{z} \frac{p^{th}}{T^2 F^2} \psi_j
  g0[gdir] -= p_th / PetscSqr(u[uOff[TEMP]] * F);
}

static void g0_conduct_vu(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                          const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                          const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                          PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g0[])
{
  const PetscReal S    = PetscRealPart(constants[STROUHAL]);
  const PetscReal p_th = PetscRealPart(constants[P_TH]);
  const PetscInt  Nc   = dim;
  PetscInt        c, d;

  // \vb{\phi}_i \cdot S \rho \psi_j
  for (d = 0; d < dim; ++d) {
    g0[d * dim + d] = S * p_th / u[uOff[TEMP]] * u_tShift;
  }

  // \phi^c_i \cdot \rho \frac{\partial u^c}{\partial x^d} \psi^d_j
  for (c = 0; c < Nc; ++c) {
    for (d = 0; d < dim; ++d) {
      g0[c * Nc + d] += p_th / u[uOff[TEMP]] * u_x[uOff_x[VEL] + c * Nc + d];
    }
  }
}

static void g1_conduct_vu(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                          const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                          const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                          PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g1[])
{
  const PetscReal p_th = PetscRealPart(constants[P_TH]);
  const PetscInt  NcI = dim;
  const PetscInt  NcJ = dim;
  PetscInt        c, d, e;

  // \phi^c_i \rho u^e \frac{\partial \psi^d_j}{\partial x^e}
  for (c = 0; c < NcI; ++c) {
    for (d = 0; d < NcJ; ++d) {
      for (e = 0; e < dim; ++e) {
        if (c == d) {
          g1[(c * NcJ + d) * dim + e] += p_th / u[uOff[TEMP]] * u[uOff[VEL] + e];
        }
      }
    }
  }
}

static void g3_conduct_vu(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                          const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                          const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                          PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g3[])
{
  const PetscReal Re = PetscRealPart(constants[REYNOLDS]);
  const PetscReal mu = PetscRealPart(constants[MU]);
  const PetscInt  Nc = dim;
  PetscInt        c, d;

  for (c = 0; c < Nc; ++c) {
    for (d = 0; d < dim; ++d) {
      // \frac{\partial \phi^c_i}{\partial x^d} \mu/Re \frac{\partial \psi^c_i}{\partial x^d}
      g3[((c * Nc + c) * dim + d) * dim + d] += mu / Re;  // gradU
      // \frac{\partial \phi^c_i}{\partial x^d} \mu/Re \frac{\partial \psi^d_i}{\partial x^c}
      g3[((c * Nc + d) * dim + d) * dim + c] += mu / Re;  // gradU transpose
      // \frac{\partial \phi^c_i}{\partial x^d} -2/3 \mu/Re \frac{\partial \psi^d_i}{\partial x^c}
      g3[((c * Nc + d) * dim + c) * dim + d] -= 2.0 / 3.0 * mu / Re;
    }
  }
}

static void g2_conduct_vp(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                          const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                          const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                          PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g2[])
{
  PetscInt d;
  for (d = 0; d < dim; ++d) {
    g2[d * dim + d] = -1.0;
  }
}

static void g0_wT(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                  const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                  const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                  PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g0[])
{
  g0[0] = u_tShift;
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
  const PetscReal alpha = PetscRealPart(constants[ALPHA]);
  PetscInt        d;

  for (d = 0; d < dim; ++d) g3[d*dim+d] = alpha;
}

static void g0_conduct_wu(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                          const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                          const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                          PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g0[])
{
  const PetscReal p_th = PetscRealPart(constants[P_TH]);
  const PetscReal c_p  = PetscRealPart(constants[C_P]);
  PetscInt        d;

  // \phi_i \frac{C_p p^{th}}{T} \nabla T \cdot \psi_j
  for (d = 0; d < dim; ++d) {
    g0[d] = c_p * p_th / u[uOff[TEMP]] * u_x[uOff_x[TEMP] + d];
  }
}

static void g0_conduct_wT(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                          const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                          const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                          PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g0[])
{
  const PetscReal S    = PetscRealPart(constants[STROUHAL]);
  const PetscReal p_th = PetscRealPart(constants[P_TH]);
  const PetscReal c_p  = PetscRealPart(constants[C_P]);
  PetscInt        d;

  // \psi_i C_p S p^{th}\T \psi_{j}
  g0[0] += c_p * S * p_th / u[uOff[TEMP]] * u_tShift;
  // - \phi_i C_p S p^{th}/T^2 T_t \psi_j
  g0[0] -= c_p * S * p_th / PetscSqr(u[uOff[TEMP]]) * u_t[uOff[TEMP]];
  // - \phi_i C_p p^{th}/T^2 \vb{u} \cdot \nabla T \psi_j
  for (d = 0; d < dim; ++d) {
    g0[0] -= c_p * p_th / PetscSqr(u[uOff[TEMP]]) * u[uOff[VEL] + d] * u_x[uOff_x[TEMP] + d];
  }
}

static void g1_conduct_wT(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                          const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                          const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                          PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g1[])
{
  const PetscReal p_th = PetscRealPart(constants[P_TH]);
  const PetscReal c_p  = PetscRealPart(constants[C_P]);
  PetscInt        d;

  // \phi_i C_p p^{th}/T \vb{u} \cdot \nabla \psi_j
  for (d = 0; d < dim; ++d) {
    g1[d] += c_p * p_th / u[uOff[TEMP]] * u[uOff[VEL] + d];
  }
}

static void g3_conduct_wT(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                          const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                          const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                          PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g3[])
{
  const PetscReal Pe = PetscRealPart(constants[PECLET]);
  const PetscReal k  = PetscRealPart(constants[K]);
  PetscInt        d;

  // \nabla \phi_i \frac{k}{Pe} \nabla \phi_j
  for (d = 0; d < dim; ++d) {
    g3[d * dim + d] = k / Pe;
  }
}

static PetscErrorCode ProcessOptions(MPI_Comm comm, AppCtx *options)
{
  PetscInt       mod, sol;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  options->modType      = MOD_INCOMPRESSIBLE;
  options->solType      = SOL_QUADRATIC;
  options->hasNullSpace = PETSC_TRUE;
  options->dmCell       = NULL;

  ierr = PetscOptionsBegin(comm, "", "Low Mach flow Problem Options", "DMPLEX");CHKERRQ(ierr);
  mod = options->modType;
  ierr = PetscOptionsEList("-mod_type", "The model type", "ex76.c", modTypes, NUM_MOD_TYPES, modTypes[options->modType], &mod, NULL);CHKERRQ(ierr);
  options->modType = (ModType) mod;
  sol = options->solType;
  ierr = PetscOptionsEList("-sol_type", "The solution type", "ex76.c", solTypes, NUM_SOL_TYPES, solTypes[options->solType], &sol, NULL);CHKERRQ(ierr);
  options->solType = (SolType) sol;
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode SetupParameters(DM dm, AppCtx *user)
{
  PetscBag       bag;
  Parameter     *p;
  PetscReal      dir;
  PetscInt       dim;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);
  dir  = (PetscReal) (dim-1);
  /* setup PETSc parameter bag */
  ierr = PetscBagGetData(user->bag, (void **) &p);CHKERRQ(ierr);
  ierr = PetscBagSetName(user->bag, "par", "Low Mach flow parameters");CHKERRQ(ierr);
  bag  = user->bag;
  ierr = PetscBagRegisterReal(bag, &p->Strouhal, 1.0, "S",     "Strouhal number");CHKERRQ(ierr);
  ierr = PetscBagRegisterReal(bag, &p->Froude,   1.0, "Fr",    "Froude number");CHKERRQ(ierr);
  ierr = PetscBagRegisterReal(bag, &p->Reynolds, 1.0, "Re",    "Reynolds number");CHKERRQ(ierr);
  ierr = PetscBagRegisterReal(bag, &p->Peclet,   1.0, "Pe",    "Peclet number");CHKERRQ(ierr);
  ierr = PetscBagRegisterReal(bag, &p->p_th,     1.0, "p_th",  "Thermodynamic pressure");CHKERRQ(ierr);
  ierr = PetscBagRegisterReal(bag, &p->mu,       1.0, "mu",    "Dynamic viscosity");CHKERRQ(ierr);
  ierr = PetscBagRegisterReal(bag, &p->nu,       1.0, "nu",    "Kinematic viscosity");CHKERRQ(ierr);
  ierr = PetscBagRegisterReal(bag, &p->c_p,      1.0, "c_p",   "Specific heat at constant pressure");CHKERRQ(ierr);
  ierr = PetscBagRegisterReal(bag, &p->k,        1.0, "k",     "Thermal conductivity");CHKERRQ(ierr);
  ierr = PetscBagRegisterReal(bag, &p->alpha,    1.0, "alpha", "Thermal diffusivity");CHKERRQ(ierr);
  ierr = PetscBagRegisterReal(bag, &p->T_in,     1.0, "T_in",  "Inlet temperature");CHKERRQ(ierr);
  ierr = PetscBagRegisterReal(bag, &p->g_dir,    dir, "g_dir", "Gravity direction");CHKERRQ(ierr);
  ierr = PetscBagRegisterReal(bag, &p->epsilon,  1.0, "epsilon", "Perturbation strength");CHKERRQ(ierr);
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

static PetscErrorCode UniformBoundaryConditions(DM dm, DMLabel label, PetscSimplePointFunc exactFuncs[], PetscSimplePointFunc exactFuncs_t[], AppCtx *user)
{
  PetscDS        ds;
  PetscInt       id;
  void          *ctx;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = DMGetDS(dm, &ds);CHKERRQ(ierr);
  ierr = PetscBagGetData(user->bag, &ctx);CHKERRQ(ierr);
  id   = 3;
  ierr = PetscDSAddBoundary(ds, DM_BC_ESSENTIAL, "top wall velocity",    label, 1, &id, VEL, 0, NULL, (void (*)(void)) exactFuncs[VEL], (void (*)(void)) exactFuncs_t[VEL], ctx, NULL);CHKERRQ(ierr);
  id   = 1;
  ierr = PetscDSAddBoundary(ds, DM_BC_ESSENTIAL, "bottom wall velocity", label, 1, &id, VEL, 0, NULL, (void (*)(void)) exactFuncs[VEL], (void (*)(void)) exactFuncs_t[VEL], ctx, NULL);CHKERRQ(ierr);
  id   = 2;
  ierr = PetscDSAddBoundary(ds, DM_BC_ESSENTIAL, "right wall velocity",  label, 1, &id, VEL, 0, NULL, (void (*)(void)) exactFuncs[VEL], (void (*)(void)) exactFuncs_t[VEL], ctx, NULL);CHKERRQ(ierr);
  id   = 4;
  ierr = PetscDSAddBoundary(ds, DM_BC_ESSENTIAL, "left wall velocity",   label, 1, &id, VEL, 0, NULL, (void (*)(void)) exactFuncs[VEL], (void (*)(void)) exactFuncs_t[VEL], ctx, NULL);CHKERRQ(ierr);
  id   = 3;
  ierr = PetscDSAddBoundary(ds, DM_BC_ESSENTIAL, "top wall temp",    label, 1, &id, TEMP, 0, NULL, (void (*)(void)) exactFuncs[TEMP], (void (*)(void)) exactFuncs_t[TEMP], ctx, NULL);CHKERRQ(ierr);
  id   = 1;
  ierr = PetscDSAddBoundary(ds, DM_BC_ESSENTIAL, "bottom wall temp", label, 1, &id, TEMP, 0, NULL, (void (*)(void)) exactFuncs[TEMP], (void (*)(void)) exactFuncs_t[TEMP], ctx, NULL);CHKERRQ(ierr);
  id   = 2;
  ierr = PetscDSAddBoundary(ds, DM_BC_ESSENTIAL, "right wall temp",  label, 1, &id, TEMP, 0, NULL, (void (*)(void)) exactFuncs[TEMP], (void (*)(void)) exactFuncs_t[TEMP], ctx, NULL);CHKERRQ(ierr);
  id   = 4;
  ierr = PetscDSAddBoundary(ds, DM_BC_ESSENTIAL, "left wall temp",   label, 1, &id, TEMP, 0, NULL, (void (*)(void)) exactFuncs[TEMP], (void (*)(void)) exactFuncs_t[TEMP], ctx, NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode SetupProblem(DM dm, AppCtx *user)
{
  PetscSimplePointFunc exactFuncs[3];
  PetscSimplePointFunc exactFuncs_t[3];
  PetscDS              ds;
  PetscWeakForm        wf;
  DMLabel              label;
  Parameter           *ctx;
  PetscInt             id, bd;
  PetscErrorCode       ierr;

  PetscFunctionBeginUser;
  ierr = DMGetLabel(dm, "marker", &label);CHKERRQ(ierr);
  ierr = DMGetDS(dm, &ds);CHKERRQ(ierr);
  ierr = PetscDSGetWeakForm(ds, &wf);CHKERRQ(ierr);

  switch(user->modType) {
    case MOD_INCOMPRESSIBLE:
      ierr = PetscDSSetResidual(ds, VEL,  f0_v, f1_v);CHKERRQ(ierr);
      ierr = PetscDSSetResidual(ds, PRES, f0_q, NULL);CHKERRQ(ierr);
      ierr = PetscDSSetResidual(ds, TEMP, f0_w, f1_w);CHKERRQ(ierr);

      ierr = PetscDSSetJacobian(ds, VEL,  VEL,  g0_vu, g1_vu, NULL,  g3_vu);CHKERRQ(ierr);
      ierr = PetscDSSetJacobian(ds, VEL,  PRES, NULL,  NULL,  g2_vp, NULL);CHKERRQ(ierr);
      ierr = PetscDSSetJacobian(ds, PRES, VEL,  NULL,  g1_qu, NULL,  NULL);CHKERRQ(ierr);
      ierr = PetscDSSetJacobian(ds, TEMP, VEL,  g0_wu, NULL,  NULL,  NULL);CHKERRQ(ierr);
      ierr = PetscDSSetJacobian(ds, TEMP, TEMP, g0_wT, g1_wT, NULL,  g3_wT);CHKERRQ(ierr);

      switch(user->solType) {
      case SOL_QUADRATIC:
        ierr = PetscWeakFormSetIndexResidual(wf, NULL, 0, VEL,  0, 1, f0_quadratic_v, 0, NULL);CHKERRQ(ierr);
        ierr = PetscWeakFormSetIndexResidual(wf, NULL, 0, TEMP, 0, 1, f0_quadratic_w, 0, NULL);CHKERRQ(ierr);

        exactFuncs[VEL]    = quadratic_u;
        exactFuncs[PRES]   = quadratic_p;
        exactFuncs[TEMP]   = quadratic_T;
        exactFuncs_t[VEL]  = quadratic_u_t;
        exactFuncs_t[PRES] = NULL;
        exactFuncs_t[TEMP] = quadratic_T_t;

        ierr = UniformBoundaryConditions(dm, label, exactFuncs, exactFuncs_t, user);CHKERRQ(ierr);
        break;
      case SOL_CUBIC:
        ierr = PetscWeakFormSetIndexResidual(wf, NULL, 0, VEL,  0, 1, f0_cubic_v, 0, NULL);CHKERRQ(ierr);
        ierr = PetscWeakFormSetIndexResidual(wf, NULL, 0, TEMP, 0, 1, f0_cubic_w, 0, NULL);CHKERRQ(ierr);

        exactFuncs[VEL]    = cubic_u;
        exactFuncs[PRES]   = cubic_p;
        exactFuncs[TEMP]   = cubic_T;
        exactFuncs_t[VEL]  = cubic_u_t;
        exactFuncs_t[PRES] = NULL;
        exactFuncs_t[TEMP] = cubic_T_t;

        ierr = UniformBoundaryConditions(dm, label, exactFuncs, exactFuncs_t, user);CHKERRQ(ierr);
        break;
      case SOL_CUBIC_TRIG:
        ierr = PetscWeakFormSetIndexResidual(wf, NULL, 0, VEL,  0, 1, f0_cubic_trig_v, 0, NULL);CHKERRQ(ierr);
        ierr = PetscWeakFormSetIndexResidual(wf, NULL, 0, TEMP, 0, 1, f0_cubic_trig_w, 0, NULL);CHKERRQ(ierr);

        exactFuncs[VEL]    = cubic_trig_u;
        exactFuncs[PRES]   = cubic_trig_p;
        exactFuncs[TEMP]   = cubic_trig_T;
        exactFuncs_t[VEL]  = cubic_trig_u_t;
        exactFuncs_t[PRES] = NULL;
        exactFuncs_t[TEMP] = cubic_trig_T_t;

        ierr = UniformBoundaryConditions(dm, label, exactFuncs, exactFuncs_t, user);CHKERRQ(ierr);
        break;
      case SOL_TAYLOR_GREEN:
        ierr = PetscWeakFormSetIndexResidual(wf, NULL, 0, TEMP, 0, 1, f0_taylor_green_w, 0, NULL);CHKERRQ(ierr);

        exactFuncs[VEL]    = taylor_green_u;
        exactFuncs[PRES]   = taylor_green_p;
        exactFuncs[TEMP]   = taylor_green_T;
        exactFuncs_t[VEL]  = taylor_green_u_t;
        exactFuncs_t[PRES] = taylor_green_p_t;
        exactFuncs_t[TEMP] = taylor_green_T_t;

        ierr = UniformBoundaryConditions(dm, label, exactFuncs, exactFuncs_t, user);CHKERRQ(ierr);
        break;
       default: SETERRQ2(PetscObjectComm((PetscObject) ds), PETSC_ERR_ARG_WRONG, "Unsupported solution type: %s (%D)", solTypes[PetscMin(user->solType, NUM_SOL_TYPES)], user->solType);
      }
      break;
    case MOD_CONDUCTING:
      ierr = PetscDSSetResidual(ds, VEL,  f0_conduct_v, f1_conduct_v);CHKERRQ(ierr);
      ierr = PetscDSSetResidual(ds, PRES, f0_conduct_q, NULL);CHKERRQ(ierr);
      ierr = PetscDSSetResidual(ds, TEMP, f0_conduct_w, f1_conduct_w);CHKERRQ(ierr);

      ierr = PetscDSSetJacobian(ds, VEL,  VEL,  g0_conduct_vu, g1_conduct_vu, NULL,          g3_conduct_vu);CHKERRQ(ierr);
      ierr = PetscDSSetJacobian(ds, VEL,  PRES, NULL,          NULL,          g2_conduct_vp, NULL);CHKERRQ(ierr);
      ierr = PetscDSSetJacobian(ds, VEL,  TEMP, g0_conduct_vT, NULL,          NULL,          NULL);CHKERRQ(ierr);
      ierr = PetscDSSetJacobian(ds, PRES, VEL,  g0_conduct_qu, g1_conduct_qu, NULL,          NULL);CHKERRQ(ierr);
      ierr = PetscDSSetJacobian(ds, PRES, TEMP, g0_conduct_qT, g1_conduct_qT, NULL,          NULL);CHKERRQ(ierr);
      ierr = PetscDSSetJacobian(ds, TEMP, VEL,  g0_conduct_wu, NULL,          NULL,          NULL);CHKERRQ(ierr);
      ierr = PetscDSSetJacobian(ds, TEMP, TEMP, g0_conduct_wT, g1_conduct_wT, NULL,          g3_conduct_wT);CHKERRQ(ierr);

      switch(user->solType) {
      case SOL_QUADRATIC:
        ierr = PetscWeakFormSetIndexResidual(wf, NULL, 0, VEL,  0, 1, f0_conduct_quadratic_v, 0, NULL);CHKERRQ(ierr);
        ierr = PetscWeakFormSetIndexResidual(wf, NULL, 0, PRES, 0, 1, f0_conduct_quadratic_q, 0, NULL);CHKERRQ(ierr);
        ierr = PetscWeakFormSetIndexResidual(wf, NULL, 0, TEMP, 0, 1, f0_conduct_quadratic_w, 0, NULL);CHKERRQ(ierr);

        exactFuncs[VEL]    = quadratic_u;
        exactFuncs[PRES]   = quadratic_p;
        exactFuncs[TEMP]   = quadratic_T;
        exactFuncs_t[VEL]  = quadratic_u_t;
        exactFuncs_t[PRES] = NULL;
        exactFuncs_t[TEMP] = quadratic_T_t;

        ierr = UniformBoundaryConditions(dm, label, exactFuncs, exactFuncs_t, user);CHKERRQ(ierr);
        break;
      case SOL_PIPE:
        user->hasNullSpace = PETSC_FALSE;
        ierr = PetscWeakFormSetIndexResidual(wf, NULL, 0, VEL,  0, 1, f0_conduct_pipe_v, 0, NULL);CHKERRQ(ierr);
        ierr = PetscWeakFormSetIndexResidual(wf, NULL, 0, PRES, 0, 1, f0_conduct_pipe_q, 0, NULL);CHKERRQ(ierr);
        ierr = PetscWeakFormSetIndexResidual(wf, NULL, 0, TEMP, 0, 1, f0_conduct_pipe_w, 0, NULL);CHKERRQ(ierr);

        exactFuncs[VEL]    = pipe_u;
        exactFuncs[PRES]   = pipe_p;
        exactFuncs[TEMP]   = pipe_T;
        exactFuncs_t[VEL]  = pipe_u_t;
        exactFuncs_t[PRES] = pipe_p_t;
        exactFuncs_t[TEMP] = pipe_T_t;

        ierr = PetscBagGetData(user->bag, (void **) &ctx);CHKERRQ(ierr);
        id   = 2;
        ierr = DMAddBoundary(dm, DM_BC_NATURAL, "right wall", label, 1, &id, 0, 0, NULL, NULL, NULL, ctx, &bd);CHKERRQ(ierr);
        ierr = PetscDSGetBoundary(ds, bd, &wf, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL);CHKERRQ(ierr);
        ierr = PetscWeakFormSetIndexBdResidual(wf, label, id, VEL, 0, 0, f0_conduct_bd_pipe_v, 0, NULL);CHKERRQ(ierr);
        id   = 4;
        ierr = DMAddBoundary(dm, DM_BC_NATURAL, "left wall", label, 1, &id, 0, 0, NULL, NULL, NULL, ctx, &bd);CHKERRQ(ierr);
        ierr = PetscDSGetBoundary(ds, bd, &wf, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL);CHKERRQ(ierr);
        ierr = PetscWeakFormSetIndexBdResidual(wf, label, id, VEL, 0, 0, f0_conduct_bd_pipe_v, 0, NULL);CHKERRQ(ierr);
        id   = 4;
        ierr = PetscDSAddBoundary(ds, DM_BC_ESSENTIAL, "left wall temperature",   label, 1, &id, TEMP, 0, NULL, (void (*)(void)) exactFuncs[TEMP], (void (*)(void)) exactFuncs_t[TEMP], ctx, NULL);CHKERRQ(ierr);
        id   = 3;
        ierr = PetscDSAddBoundary(ds, DM_BC_ESSENTIAL, "top wall velocity",       label, 1, &id, VEL,  0, NULL, (void (*)(void)) exactFuncs[VEL], (void (*)(void)) exactFuncs_t[VEL], ctx, NULL);CHKERRQ(ierr);
        ierr = PetscDSAddBoundary(ds, DM_BC_ESSENTIAL, "top wall temperature",    label, 1, &id, TEMP, 0, NULL, (void (*)(void)) exactFuncs[TEMP], (void (*)(void)) exactFuncs_t[TEMP], ctx, NULL);CHKERRQ(ierr);
        id   = 1;
        ierr = PetscDSAddBoundary(ds, DM_BC_ESSENTIAL, "bottom wall velocity",    label, 1, &id, VEL,  0, NULL, (void (*)(void)) exactFuncs[VEL], (void (*)(void)) exactFuncs_t[VEL], ctx, NULL);CHKERRQ(ierr);
        ierr = PetscDSAddBoundary(ds, DM_BC_ESSENTIAL, "bottom wall temperature", label, 1, &id, TEMP, 0, NULL, (void (*)(void)) exactFuncs[TEMP], (void (*)(void)) exactFuncs_t[TEMP], ctx, NULL);CHKERRQ(ierr);
        break;
      case SOL_PIPE_WIGGLY:
        user->hasNullSpace = PETSC_FALSE;
        ierr = PetscWeakFormSetIndexResidual(wf, NULL, 0, VEL,  0, 1, f0_conduct_pipe_wiggly_v, 0, NULL);CHKERRQ(ierr);
        ierr = PetscWeakFormSetIndexResidual(wf, NULL, 0, PRES, 0, 1, f0_conduct_pipe_wiggly_q, 0, NULL);CHKERRQ(ierr);
        ierr = PetscWeakFormSetIndexResidual(wf, NULL, 0, TEMP, 0, 1, f0_conduct_pipe_wiggly_w, 0, NULL);CHKERRQ(ierr);

        exactFuncs[VEL]    = pipe_wiggly_u;
        exactFuncs[PRES]   = pipe_wiggly_p;
        exactFuncs[TEMP]   = pipe_wiggly_T;
        exactFuncs_t[VEL]  = pipe_wiggly_u_t;
        exactFuncs_t[PRES] = pipe_wiggly_p_t;
        exactFuncs_t[TEMP] = pipe_wiggly_T_t;

        ierr = PetscBagGetData(user->bag, (void **) &ctx);CHKERRQ(ierr);
        id   = 2;
        ierr = DMAddBoundary(dm, DM_BC_NATURAL, "right wall", label, 1, &id, 0, 0, NULL, NULL, NULL, ctx, &bd);CHKERRQ(ierr);
        ierr = PetscDSGetBoundary(ds, bd, &wf, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL);CHKERRQ(ierr);
        ierr = PetscWeakFormSetIndexBdResidual(wf, label, id, VEL, 0, 0, f0_conduct_bd_pipe_wiggly_v, 0, NULL);CHKERRQ(ierr);
        id   = 4;
        ierr = DMAddBoundary(dm, DM_BC_NATURAL, "left wall", label, 1, &id, 0, 0, NULL, NULL, NULL, ctx, &bd);CHKERRQ(ierr);
        ierr = PetscDSGetBoundary(ds, bd, &wf, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL);CHKERRQ(ierr);
        ierr = PetscWeakFormSetIndexBdResidual(wf, label, id, VEL, 0, 0, f0_conduct_bd_pipe_wiggly_v, 0, NULL);CHKERRQ(ierr);
        id   = 4;
        ierr = PetscDSAddBoundary(ds, DM_BC_ESSENTIAL, "left wall temperature",   label, 1, &id, TEMP, 0, NULL, (void (*)(void)) exactFuncs[TEMP], (void (*)(void)) exactFuncs_t[TEMP], ctx, NULL);CHKERRQ(ierr);
        id   = 3;
        ierr = PetscDSAddBoundary(ds, DM_BC_ESSENTIAL, "top wall velocity",       label, 1, &id, VEL,  0, NULL, (void (*)(void)) exactFuncs[VEL], (void (*)(void)) exactFuncs_t[VEL], ctx, NULL);CHKERRQ(ierr);
        ierr = PetscDSAddBoundary(ds, DM_BC_ESSENTIAL, "top wall temperature",    label, 1, &id, TEMP, 0, NULL, (void (*)(void)) exactFuncs[TEMP], (void (*)(void)) exactFuncs_t[TEMP], ctx, NULL);CHKERRQ(ierr);
        id   = 1;
        ierr = PetscDSAddBoundary(ds, DM_BC_ESSENTIAL, "bottom wall velocity",    label, 1, &id, VEL,  0, NULL, (void (*)(void)) exactFuncs[VEL], (void (*)(void)) exactFuncs_t[VEL], ctx, NULL);CHKERRQ(ierr);
        ierr = PetscDSAddBoundary(ds, DM_BC_ESSENTIAL, "bottom wall temperature", label, 1, &id, TEMP, 0, NULL, (void (*)(void)) exactFuncs[TEMP], (void (*)(void)) exactFuncs_t[TEMP], ctx, NULL);CHKERRQ(ierr);
        break;
       default: SETERRQ2(PetscObjectComm((PetscObject) ds), PETSC_ERR_ARG_WRONG, "Unsupported solution type: %s (%D)", solTypes[PetscMin(user->solType, NUM_SOL_TYPES)], user->solType);
      }
      break;
    default: SETERRQ2(PetscObjectComm((PetscObject) ds), PETSC_ERR_ARG_WRONG, "Unsupported model type: %s (%D)", solTypes[PetscMin(user->modType, NUM_MOD_TYPES)], user->modType);
  }
  /* Setup constants */
  {
    Parameter  *param;
    PetscScalar constants[13];

    ierr = PetscBagGetData(user->bag, (void **) &param);CHKERRQ(ierr);

    constants[STROUHAL] = param->Strouhal;
    constants[FROUDE]   = param->Froude;
    constants[REYNOLDS] = param->Reynolds;
    constants[PECLET]   = param->Peclet;
    constants[P_TH]     = param->p_th;
    constants[MU]       = param->mu;
    constants[NU]       = param->nu;
    constants[C_P]      = param->c_p;
    constants[K]        = param->k;
    constants[ALPHA]    = param->alpha;
    constants[T_IN]     = param->T_in;
    constants[G_DIR]    = param->g_dir;
    constants[EPSILON]  = param->epsilon;
    ierr = PetscDSSetConstants(ds, 13, constants);CHKERRQ(ierr);
  }

  ierr = PetscBagGetData(user->bag, (void **) &ctx);CHKERRQ(ierr);
  ierr = PetscDSSetExactSolution(ds, VEL,  exactFuncs[VEL],  ctx);CHKERRQ(ierr);
  ierr = PetscDSSetExactSolution(ds, PRES, exactFuncs[PRES], ctx);CHKERRQ(ierr);
  ierr = PetscDSSetExactSolution(ds, TEMP, exactFuncs[TEMP], ctx);CHKERRQ(ierr);
  ierr = PetscDSSetExactSolutionTimeDerivative(ds, VEL,  exactFuncs_t[VEL],  ctx);CHKERRQ(ierr);
  ierr = PetscDSSetExactSolutionTimeDerivative(ds, PRES, exactFuncs_t[PRES], ctx);CHKERRQ(ierr);
  ierr = PetscDSSetExactSolutionTimeDerivative(ds, TEMP, exactFuncs_t[TEMP], ctx);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode CreateCellDM(DM dm, AppCtx *user)
{
  PetscFE        fe, fediv;
  DMPolytopeType ct;
  PetscInt       dim, cStart;
  PetscBool      simplex;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm, 0, &cStart, NULL);CHKERRQ(ierr);
  ierr = DMPlexGetCellType(dm, cStart, &ct);CHKERRQ(ierr);
  simplex = DMPolytopeTypeGetNumVertices(ct) == DMPolytopeTypeGetDim(ct)+1 ? PETSC_TRUE : PETSC_FALSE;

  ierr = DMGetField(dm, VEL,  NULL, (PetscObject *) &fe);CHKERRQ(ierr);
  ierr = PetscFECreateDefault(PETSC_COMM_SELF, dim, 1, simplex, "div_", PETSC_DEFAULT, &fediv);CHKERRQ(ierr);
  ierr = PetscFECopyQuadrature(fe, fediv);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) fediv, "divergence");CHKERRQ(ierr);

  ierr = DMDestroy(&user->dmCell);CHKERRQ(ierr);
  ierr = DMClone(dm, &user->dmCell);CHKERRQ(ierr);
  ierr = DMSetField(user->dmCell, 0, NULL, (PetscObject) fediv);CHKERRQ(ierr);
  ierr = DMCreateDS(user->dmCell);CHKERRQ(ierr);
  ierr = PetscFEDestroy(&fediv);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode GetCellDM(DM dm, AppCtx *user, DM *dmCell)
{
  PetscInt       cStart, cEnd, cellStart = -1, cellEnd = -1;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = DMPlexGetSimplexOrBoxCells(dm, 0, &cStart, &cEnd);CHKERRQ(ierr);
  if (user->dmCell) {ierr = DMPlexGetSimplexOrBoxCells(user->dmCell, 0, &cellStart, &cellEnd);CHKERRQ(ierr);}
  if (cStart != cellStart || cEnd != cellEnd) {ierr = CreateCellDM(dm, user);CHKERRQ(ierr);}
  *dmCell = user->dmCell;
  PetscFunctionReturn(0);
}

static PetscErrorCode SetupDiscretization(DM dm, AppCtx *user)
{
  DM              cdm = dm;
  PetscFE         fe[3];
  Parameter      *param;
  DMPolytopeType  ct;
  PetscInt        dim, cStart;
  PetscBool       simplex;
  PetscErrorCode  ierr;

  PetscFunctionBeginUser;
  ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm, 0, &cStart, NULL);CHKERRQ(ierr);
  ierr = DMPlexGetCellType(dm, cStart, &ct);CHKERRQ(ierr);
  simplex = DMPolytopeTypeGetNumVertices(ct) == DMPolytopeTypeGetDim(ct)+1 ? PETSC_TRUE : PETSC_FALSE;
  /* Create finite element */
  ierr = PetscFECreateDefault(PETSC_COMM_SELF, dim, dim, simplex, "vel_", PETSC_DEFAULT, &fe[0]);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) fe[0], "velocity");CHKERRQ(ierr);

  ierr = PetscFECreateDefault(PETSC_COMM_SELF, dim, 1, simplex, "pres_", PETSC_DEFAULT, &fe[1]);CHKERRQ(ierr);
  ierr = PetscFECopyQuadrature(fe[0], fe[1]);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) fe[1], "pressure");CHKERRQ(ierr);

  ierr = PetscFECreateDefault(PETSC_COMM_SELF, dim, 1, simplex, "temp_", PETSC_DEFAULT, &fe[2]);CHKERRQ(ierr);
  ierr = PetscFECopyQuadrature(fe[0], fe[2]);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) fe[2], "temperature");CHKERRQ(ierr);

  /* Set discretization and boundary conditions for each mesh */
  ierr = DMSetField(dm, VEL,  NULL, (PetscObject) fe[VEL]);CHKERRQ(ierr);
  ierr = DMSetField(dm, PRES, NULL, (PetscObject) fe[PRES]);CHKERRQ(ierr);
  ierr = DMSetField(dm, TEMP, NULL, (PetscObject) fe[TEMP]);CHKERRQ(ierr);
  ierr = DMCreateDS(dm);CHKERRQ(ierr);
  ierr = SetupProblem(dm, user);CHKERRQ(ierr);
  ierr = PetscBagGetData(user->bag, (void **) &param);CHKERRQ(ierr);
  while (cdm) {
    ierr = DMCopyDisc(dm, cdm);CHKERRQ(ierr);
    ierr = DMGetCoarseDM(cdm, &cdm);CHKERRQ(ierr);
  }
  ierr = PetscFEDestroy(&fe[VEL]);CHKERRQ(ierr);
  ierr = PetscFEDestroy(&fe[PRES]);CHKERRQ(ierr);
  ierr = PetscFEDestroy(&fe[TEMP]);CHKERRQ(ierr);

  if (user->hasNullSpace) {
    PetscObject  pressure;
    MatNullSpace nullspacePres;

    ierr = DMGetField(dm, PRES, NULL, &pressure);CHKERRQ(ierr);
    ierr = MatNullSpaceCreate(PetscObjectComm(pressure), PETSC_TRUE, 0, NULL, &nullspacePres);CHKERRQ(ierr);
    ierr = PetscObjectCompose(pressure, "nullspace", (PetscObject) nullspacePres);CHKERRQ(ierr);
    ierr = MatNullSpaceDestroy(&nullspacePres);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode CreatePressureNullSpace(DM dm, PetscInt ofield, PetscInt nfield, MatNullSpace *nullSpace)
{
  Vec              vec;
  PetscErrorCode (*funcs[3])(PetscInt, PetscReal, const PetscReal[], PetscInt, PetscScalar *, void *) = {zero, zero, zero};
  PetscErrorCode   ierr;

  PetscFunctionBeginUser;
  if (ofield != PRES) SETERRQ2(PetscObjectComm((PetscObject) dm), PETSC_ERR_ARG_WRONG, "Nullspace must be for pressure field at index %D, not %D", PRES, ofield);
  funcs[nfield] = constant;
  ierr = DMCreateGlobalVector(dm, &vec);CHKERRQ(ierr);
  ierr = DMProjectFunction(dm, 0.0, funcs, NULL, INSERT_ALL_VALUES, vec);CHKERRQ(ierr);
  ierr = VecNormalize(vec, NULL);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) vec, "Pressure Null Space");CHKERRQ(ierr);
  ierr = VecViewFromOptions(vec, NULL, "-pressure_nullspace_view");CHKERRQ(ierr);
  ierr = MatNullSpaceCreate(PetscObjectComm((PetscObject) dm), PETSC_FALSE, 1, &vec, nullSpace);CHKERRQ(ierr);
  ierr = VecDestroy(&vec);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode RemoveDiscretePressureNullspace_Private(TS ts, Vec u)
{
  DM             dm;
  AppCtx        *user;
  MatNullSpace   nullsp;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = TSGetDM(ts, &dm);CHKERRQ(ierr);
  ierr = DMGetApplicationContext(dm, &user);CHKERRQ(ierr);
  if (!user->hasNullSpace) PetscFunctionReturn(0);
  ierr = CreatePressureNullSpace(dm, 1, 1, &nullsp);CHKERRQ(ierr);
  ierr = MatNullSpaceRemove(nullsp, u);CHKERRQ(ierr);
  ierr = MatNullSpaceDestroy(&nullsp);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* Make the discrete pressure discretely divergence free */
static PetscErrorCode RemoveDiscretePressureNullspace(TS ts)
{
  Vec            u;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = TSGetSolution(ts, &u);CHKERRQ(ierr);
  ierr = RemoveDiscretePressureNullspace_Private(ts, u);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static void divergence(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                       const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                       const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                       PetscReal t, const PetscReal X[], PetscInt numConstants, const PetscScalar constants[], PetscScalar divu[])
{
  PetscInt d;

  divu[0] = 0.;
  for (d = 0; d < dim; ++d) divu[0] += u_x[d*dim+d];
}

static PetscErrorCode SetInitialConditions(TS ts, Vec u)
{
  AppCtx        *user;
  DM             dm;
  PetscReal      t;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = TSGetDM(ts, &dm);CHKERRQ(ierr);
  ierr = TSGetTime(ts, &t);CHKERRQ(ierr);
  ierr = DMComputeExactSolution(dm, t, u, NULL);CHKERRQ(ierr);
  ierr = DMGetApplicationContext(dm, &user);CHKERRQ(ierr);
  ierr = RemoveDiscretePressureNullspace_Private(ts, u);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MonitorError(TS ts, PetscInt step, PetscReal crtime, Vec u, void *ctx)
{
  PetscErrorCode (*exactFuncs[3])(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nf, PetscScalar *u, void *ctx);
  void            *ctxs[3];
  PetscPointFunc   diagnostics[1] = {divergence};
  DM               dm, dmCell = NULL;
  PetscDS          ds;
  Vec              v, divu;
  PetscReal        ferrors[3], massFlux;
  PetscInt         f;
  PetscErrorCode   ierr;

  PetscFunctionBeginUser;
  ierr = TSGetDM(ts, &dm);CHKERRQ(ierr);
  ierr = DMGetDS(dm, &ds);CHKERRQ(ierr);

  for (f = 0; f < 3; ++f) {ierr = PetscDSGetExactSolution(ds, f, &exactFuncs[f], &ctxs[f]);CHKERRQ(ierr);}
  ierr = DMComputeL2FieldDiff(dm, crtime, exactFuncs, ctxs, u, ferrors);CHKERRQ(ierr);
  ierr = GetCellDM(dm, (AppCtx *) ctx, &dmCell);CHKERRQ(ierr);
  ierr = DMGetGlobalVector(dmCell, &divu);CHKERRQ(ierr);
  ierr = DMProjectField(dmCell, crtime, u, diagnostics, INSERT_VALUES, divu);CHKERRQ(ierr);
  ierr = VecViewFromOptions(divu, NULL, "-divu_vec_view");CHKERRQ(ierr);
  ierr = VecNorm(divu, NORM_2, &massFlux);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD, "Timestep: %04d time = %-8.4g \t L_2 Error: [%2.3g, %2.3g, %2.3g] ||div u||: %2.3g\n", (int) step, (double) crtime, (double) ferrors[0], (double) ferrors[1], (double) ferrors[2], (double) massFlux);CHKERRQ(ierr);

  ierr = VecViewFromOptions(u, NULL, "-sol_vec_view");CHKERRQ(ierr);

  ierr = DMGetGlobalVector(dm, &v);CHKERRQ(ierr);
  ierr = DMProjectFunction(dm, crtime, exactFuncs, ctxs, INSERT_ALL_VALUES, v);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) v, "Exact Solution");CHKERRQ(ierr);
  ierr = VecViewFromOptions(v, NULL, "-exact_vec_view");CHKERRQ(ierr);
  ierr = DMRestoreGlobalVector(dm, &v);CHKERRQ(ierr);

  ierr = VecViewFromOptions(divu, NULL, "-div_vec_view");CHKERRQ(ierr);
  ierr = DMRestoreGlobalVector(dmCell, &divu);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

int main(int argc, char **argv)
{
  DM              dm;   /* problem definition */
  TS              ts;   /* timestepper */
  Vec             u;    /* solution */
  AppCtx          user; /* user-defined work context */
  PetscReal       t;
  PetscErrorCode  ierr;

  ierr = PetscInitialize(&argc, &argv, NULL,help);if (ierr) return ierr;
  ierr = ProcessOptions(PETSC_COMM_WORLD, &user);CHKERRQ(ierr);
  ierr = PetscBagCreate(PETSC_COMM_WORLD, sizeof(Parameter), &user.bag);CHKERRQ(ierr);
  ierr = CreateMesh(PETSC_COMM_WORLD, &user, &dm);CHKERRQ(ierr);
  ierr = SetupParameters(dm, &user);CHKERRQ(ierr);
  ierr = TSCreate(PETSC_COMM_WORLD, &ts);CHKERRQ(ierr);
  ierr = TSSetDM(ts, dm);CHKERRQ(ierr);
  ierr = DMSetApplicationContext(dm, &user);CHKERRQ(ierr);
  /* Setup problem */
  ierr = SetupDiscretization(dm, &user);CHKERRQ(ierr);
  ierr = DMPlexCreateClosureIndex(dm, NULL);CHKERRQ(ierr);

  ierr = DMCreateGlobalVector(dm, &u);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) u, "Numerical Solution");CHKERRQ(ierr);
  if (user.hasNullSpace) {ierr = DMSetNullSpaceConstructor(dm, 1, CreatePressureNullSpace);CHKERRQ(ierr);}

  ierr = DMTSSetBoundaryLocal(dm, DMPlexTSComputeBoundary, &user);CHKERRQ(ierr);
  ierr = DMTSSetIFunctionLocal(dm, DMPlexTSComputeIFunctionFEM, &user);CHKERRQ(ierr);
  ierr = DMTSSetIJacobianLocal(dm, DMPlexTSComputeIJacobianFEM, &user);CHKERRQ(ierr);
  ierr = TSSetExactFinalTime(ts, TS_EXACTFINALTIME_MATCHSTEP);CHKERRQ(ierr);
  ierr = TSSetPreStep(ts, RemoveDiscretePressureNullspace);CHKERRQ(ierr);
  ierr = TSSetFromOptions(ts);CHKERRQ(ierr);

  ierr = TSSetComputeInitialCondition(ts, SetInitialConditions);CHKERRQ(ierr); /* Must come after SetFromOptions() */
  ierr = SetInitialConditions(ts, u);CHKERRQ(ierr);
  ierr = TSGetTime(ts, &t);CHKERRQ(ierr);
  ierr = DMSetOutputSequenceNumber(dm, 0, t);CHKERRQ(ierr);
  ierr = DMTSCheckFromOptions(ts, u);CHKERRQ(ierr);
  ierr = TSMonitorSet(ts, MonitorError, &user, NULL);CHKERRQ(ierr);CHKERRQ(ierr);

  ierr = TSSolve(ts, u);CHKERRQ(ierr);
  ierr = DMTSCheckFromOptions(ts, u);CHKERRQ(ierr);

  ierr = VecDestroy(&u);CHKERRQ(ierr);
  ierr = DMDestroy(&user.dmCell);CHKERRQ(ierr);
  ierr = DMDestroy(&dm);CHKERRQ(ierr);
  ierr = TSDestroy(&ts);CHKERRQ(ierr);
  ierr = PetscBagDestroy(&user.bag);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

  testset:
    requires: triangle !single
    args: -dm_plex_separate_marker \
          -div_petscdualspace_lagrange_use_moments -div_petscdualspace_lagrange_moment_order 3 \
          -snes_error_if_not_converged -snes_convergence_test correct_pressure \
          -ksp_type fgmres -ksp_gmres_restart 10 -ksp_rtol 1.0e-9 -ksp_error_if_not_converged \
          -pc_type fieldsplit -pc_fieldsplit_0_fields 0,2 -pc_fieldsplit_1_fields 1 \
          -pc_fieldsplit_type schur -pc_fieldsplit_schur_factorization_type full \
            -fieldsplit_0_pc_type lu \
            -fieldsplit_pressure_ksp_rtol 1e-10 -fieldsplit_pressure_pc_type jacobi

    test:
      suffix: 2d_tri_p2_p1_p1
      args: -sol_type quadratic \
            -vel_petscspace_degree 2 -pres_petscspace_degree 1 -temp_petscspace_degree 1 \
            -dmts_check .001 -ts_max_steps 4 -ts_dt 0.1

    test:
      # Using -dm_refine 5 -convest_num_refine 2 gives L_2 convergence rate: [0.89, 0.011, 1.0]
      suffix: 2d_tri_p2_p1_p1_tconv
      args: -sol_type cubic_trig \
            -vel_petscspace_degree 2 -pres_petscspace_degree 1 -temp_petscspace_degree 1 \
            -ts_max_steps 4 -ts_dt 0.1 -ts_convergence_estimate -convest_num_refine 1

    test:
      # Using -dm_refine 3 -convest_num_refine 3 gives L_2 convergence rate: [3.0, 2.5, 1.9]
      suffix: 2d_tri_p2_p1_p1_sconv
      args: -sol_type cubic \
            -vel_petscspace_degree 2 -pres_petscspace_degree 1 -temp_petscspace_degree 1 \
            -ts_max_steps 1 -ts_dt 1e-4 -ts_convergence_estimate -ts_convergence_temporal 0 -convest_num_refine 1

    test:
      suffix: 2d_tri_p3_p2_p2
      args: -sol_type cubic \
            -vel_petscspace_degree 3 -pres_petscspace_degree 2 -temp_petscspace_degree 2 \
            -dmts_check .001 -ts_max_steps 4 -ts_dt 0.1

    test:
      # Using -dm_refine 3 -convest_num_refine 3 gives L_2 convergence rate: [3.0, 2.1, 3.1]
      suffix: 2d_tri_p2_p1_p1_tg_sconv
      args: -sol_type taylor_green \
            -vel_petscspace_degree 2 -pres_petscspace_degree 1 -temp_petscspace_degree 1 \
            -ts_max_steps 1 -ts_dt 1e-8 -ts_convergence_estimate -ts_convergence_temporal 0 -convest_num_refine 1

    test:
      # Using -dm_refine 3 -convest_num_refine 2 gives L_2 convergence rate: [1.2, 1.5, 1.2]
      suffix: 2d_tri_p2_p1_p1_tg_tconv
      args: -sol_type taylor_green \
            -vel_petscspace_degree 2 -pres_petscspace_degree 1 -temp_petscspace_degree 1 \
            -ts_max_steps 4 -ts_dt 0.1 -ts_convergence_estimate -convest_num_refine 1

  testset:
    requires: triangle !single
    args: -dm_plex_separate_marker -mod_type conducting \
          -div_petscdualspace_lagrange_use_moments -div_petscdualspace_lagrange_moment_order 3 \
          -snes_error_if_not_converged -snes_max_linear_solve_fail 5 \
          -ksp_type fgmres -ksp_max_it 2 -ksp_gmres_restart 10 -ksp_rtol 1.0e-9 \
          -pc_type fieldsplit -pc_fieldsplit_0_fields 0,2 -pc_fieldsplit_1_fields 1 \
          -pc_fieldsplit_type schur -pc_fieldsplit_schur_factorization_type full \
            -fieldsplit_0_pc_type lu \
            -fieldsplit_pressure_ksp_rtol 1e-10 -fieldsplit_pressure_pc_type jacobi

    test:
      # At this resolution, the rhs is inconsistent on some Newton steps
      suffix: 2d_tri_p2_p1_p1_conduct
      args: -sol_type quadratic \
            -vel_petscspace_degree 2 -pres_petscspace_degree 1 -temp_petscspace_degree 1 \
            -dmts_check .001 -ts_max_steps 4 -ts_dt 0.1 \
            -pc_fieldsplit_schur_precondition full \
              -fieldsplit_pressure_ksp_max_it 2 -fieldsplit_pressure_pc_type svd

    test:
      suffix: 2d_tri_p2_p1_p2_conduct_pipe
      args: -sol_type pipe \
            -vel_petscspace_degree 2 -pres_petscspace_degree 1 -temp_petscspace_degree 2 \
            -dmts_check .001 -ts_max_steps 4 -ts_dt 0.1

    test:
      suffix: 2d_tri_p2_p1_p2_conduct_pipe_wiggly_sconv
      args: -sol_type pipe_wiggly -Fr 1e10 \
            -vel_petscspace_degree 2 -pres_petscspace_degree 1 -temp_petscspace_degree 2 \
            -ts_convergence_estimate -ts_convergence_temporal 0 -convest_num_refine 1 \
            -ts_max_steps 1 -ts_dt 1e10 \
            -ksp_atol 1e-12 -ksp_max_it 300 \
              -fieldsplit_pressure_ksp_atol 1e-14

TEST*/
