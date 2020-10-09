static char help[] = "Time-dependent Low Mach Flow in 2d and 3d channels with finite elements.\n\
We solve the Low Mach flow problem in a rectangular\n\
domain, using a parallel unstructured mesh (DMPLEX) to discretize it.\n\n\n";

/*F
This Low Mach flow is time-dependent isoviscous Navier-Stokes flow. We discretize using the
finite element method on an unstructured mesh. The weak form equations are

\begin{align*}
    < q, \nabla\cdot u > = 0
    <v, du/dt> + <v, u \cdot \nabla u> + < \nabla v, \nu (\nabla u + {\nabla u}^T) > - < \nabla\cdot v, p >  - < v, f  >  = 0
    < w, u \cdot \nabla T > + < \nabla w, \alpha \nabla T > - < w, Q > = 0
\end{align*}

where $\nu$ is the kinematic viscosity and $\alpha$ is thermal diffusivity.

For visualization, use

  -dm_view hdf5:$PWD/sol.h5 -sol_vec_view hdf5:$PWD/sol.h5::append -exact_vec_view hdf5:$PWD/sol.h5::append
F*/

#include <petscdmplex.h>
#include <petscsnes.h>
#include <petscts.h>
#include <petscds.h>
#include <petscbag.h>

typedef enum {SOL_QUADRATIC, SOL_CUBIC, SOL_CUBIC_TRIG, NUM_SOL_TYPES} SolType;
const char *solTypes[NUM_SOL_TYPES+1] = {"quadratic", "cubic", "cubic_trig",  "unknown"};

typedef struct {
  PetscReal nu;    /* Kinematic viscosity */
  PetscReal alpha; /* Thermal diffusivity */
  PetscReal T_in;  /* Inlet temperature*/
} Parameter;

typedef struct {
  /* Problem definition */
  PetscBag bag;     /* Holds problem parameters */
  SolType  solType; /* MMS solution type */
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
    T = t + x + y
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
  T[0] = time + X[0] + X[1];
  return 0;
}
static PetscErrorCode quadratic_T_t(PetscInt Dim, PetscReal time, const PetscReal X[], PetscInt Nf, PetscScalar *T, void *ctx)
{
  T[0] = 1.0;
  return 0;
}

/* f0_v = du/dt - f */
static void f0_quadratic_v(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                           const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                           const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                           PetscReal t, const PetscReal X[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f0[])
{
  const PetscReal nu = PetscRealPart(constants[0]);
  PetscInt        Nc = dim;
  PetscInt        c, d;

  for (d = 0; d<dim; ++d) f0[d] = u_t[uOff[0]+d];

  for (c = 0; c < Nc; ++c) {
    for (d = 0; d < dim; ++d) f0[c] += u[d]*u_x[c*dim+d];
  }
  f0[0] -= (t*(2*X[0] + 2*X[1]) + 2*X[0]*X[0]*X[0] + 4*X[0]*X[0]*X[1] - 2*X[0]*X[1]*X[1] - 4.0*nu + 2);
  f0[1] -= (t*(2*X[0] - 2*X[1]) + 4*X[0]*X[1]*X[1] + 2*X[0]*X[0]*X[1] - 2*X[1]*X[1]*X[1] - 4.0*nu + 2);
}

/* f0_w = dT/dt + u.grad(T) - Q */
static void f0_quadratic_w(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                           const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                           const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                           PetscReal t, const PetscReal X[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f0[])
{
  PetscInt d;
  f0[0] = 0;
  for (d = 0; d < dim; ++d) f0[0] += u[uOff[0]+d]*u_x[uOff_x[2]+d];
  f0[0] += u_t[uOff[2]] - (2*t + 1 + 3*X[0]*X[0] - 2*X[0]*X[1] + X[1]*X[1]);
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
  PetscInt                   c, d;
  PetscInt                   Nc = dim;
  const PetscReal            nu = PetscRealPart(constants[0]);

  for (d=0; d<dim; ++d) f0[d] = u_t[uOff[0]+d];

  for (c=0; c<Nc; ++c) {
    for (d=0; d<dim; ++d) f0[c] += u[d]*u_x[c*dim+d];
  }
  f0[0] -= (t*(3*X[0]*X[0] + 3*X[1]*X[1]) + 3*X[0]*X[0]*X[0]*X[0]*X[0] + 6*X[0]*X[0]*X[0]*X[1]*X[1] - 6*X[0]*X[0]*X[1]*X[1]*X[1] - ( 6*X[0] + 6*X[1])*nu + 3*X[0] + 1);
  f0[1] -= (t*(3*X[0]*X[0] - 6*X[0]*X[1]) + 3*X[0]*X[0]*X[0]*X[0]*X[1] + 6*X[0]*X[0]*X[1]*X[1]*X[1] - 6*X[0]*X[1]*X[1]*X[1]*X[1] - (12*X[0] - 6*X[1])*nu + 3*X[1] + 1);
}

static void f0_cubic_w(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                       const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                       const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                       PetscReal t, const PetscReal X[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f0[])
{
  PetscInt              d;
  const PetscReal alpha = PetscRealPart(constants[1]);

  for (d = 0, f0[0] = 0; d < dim; ++d) f0[0] += u[uOff[0]+d]*u_x[uOff_x[2]+d];
  f0[0] += u_t[uOff[2]] - (X[0]*X[0]*X[0]*X[0] + 2.0*X[0]*X[0]*X[0]*X[1] - 3.0*X[0]*X[0]*X[1]*X[1] + X[0]*X[1]*X[1]*X[1] + X[0]*t + X[1]*t - 2.0*alpha + 1);
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
  const PetscReal nu = PetscRealPart(constants[0]);
  PetscInt        Nc = dim;
  PetscInt        c, d;

  for (d = 0; d < dim; ++d) f0[d] = u_t[uOff[0]+d];

  for (c = 0; c < Nc; ++c) {
    for (d = 0; d < dim; ++d) f0[c] += u[d]*u_x[c*dim+d];
  }
  f0[0] -= 100.*PetscCosReal(t)*(3*X[0]*X[0])               + 100.*PetscSinReal(t)*(3*X[1]*X[1] - 1.) + 3*X[0]*X[0]*X[0]*X[0]*X[0] + 6*X[0]*X[0]*X[0]*X[1]*X[1] - 6*X[0]*X[0]*X[1]*X[1]*X[1] - ( 6*X[0] + 6*X[1])*nu + 3*X[0];
  f0[1] -= 100.*PetscCosReal(t)*(6*X[0]*X[0] - 6*X[0]*X[1]) - 100.*PetscSinReal(t)*(3*X[0]*X[0])      + 3*X[0]*X[0]*X[0]*X[0]*X[1] + 6*X[0]*X[0]*X[1]*X[1]*X[1] - 6*X[0]*X[1]*X[1]*X[1]*X[1] - (12*X[0] - 6*X[1])*nu + 3*X[1];
}

static void f0_cubic_trig_w(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                            const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                            const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                            PetscReal t, const PetscReal X[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f0[])
{
  const PetscReal alpha = PetscRealPart(constants[1]);
  PetscInt        d;

  for (d = 0, f0[0] = 0; d < dim; ++d) f0[0] += u[uOff[0]+d]*u_x[uOff_x[2]+d];
  f0[0] += u_t[uOff[2]] - (100.*PetscCosReal(t)*X[0] + 100.*PetscSinReal(t)*(X[1] - 1.) + X[0]*X[0]*X[0]*X[0] + 2.0*X[0]*X[0]*X[0]*X[1] - 3.0*X[0]*X[0]*X[1]*X[1] + X[0]*X[1]*X[1]*X[1] - 2.0*alpha);
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
      //f1[c*dim+d] = nu*u_x[c*dim+d];
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
  PetscInt       sol;
  PetscErrorCode ierr;


  PetscFunctionBeginUser;
  options->solType = SOL_QUADRATIC;

  ierr = PetscOptionsBegin(comm, "", "Low Mach flow Problem Options", "DMPLEX");CHKERRQ(ierr);
  sol = options->solType;
  ierr = PetscOptionsEList("-sol_type", "The solution type", "ex62.c", solTypes, NUM_SOL_TYPES, solTypes[options->solType], &sol, NULL);CHKERRQ(ierr);
  options->solType = (SolType) sol;
  ierr = PetscOptionsEnd();
  PetscFunctionReturn(0);
}

static PetscErrorCode SetupParameters(AppCtx *user)
{
  PetscBag       bag;
  Parameter     *p;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  /* setup PETSc parameter bag */
  ierr = PetscBagGetData(user->bag, (void **) &p);CHKERRQ(ierr);
  ierr = PetscBagSetName(user->bag, "par", "Low Mach flow parameters");CHKERRQ(ierr);
  bag  = user->bag;
  ierr = PetscBagRegisterReal(bag, &p->nu,    1.0, "nu",    "Kinematic viscosity");CHKERRQ(ierr);
  ierr = PetscBagRegisterReal(bag, &p->alpha, 1.0, "alpha", "Thermal diffusivity");CHKERRQ(ierr);
  ierr = PetscBagRegisterReal(bag, &p->T_in,  1.0, "T_in",  "Inlet temperature");CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode CreateMesh(MPI_Comm comm, AppCtx *user, DM *dm)
{
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = DMPlexCreateBoxMesh(comm, 2, PETSC_TRUE, NULL, NULL, NULL, NULL, PETSC_TRUE, dm);CHKERRQ(ierr);
  ierr = DMSetFromOptions(*dm);CHKERRQ(ierr);
  ierr = DMViewFromOptions(*dm, NULL, "-dm_view");CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode SetupProblem(DM dm, AppCtx *user)
{
  PetscErrorCode (*exactFuncs[3])(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nf, PetscScalar *u, void *ctx);
  PetscErrorCode (*exactFuncs_t[3])(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nf, PetscScalar *u, void *ctx);
  PetscDS          prob;
  Parameter       *ctx;
  PetscInt         id;
  PetscErrorCode   ierr;

  PetscFunctionBeginUser;
  ierr = DMGetDS(dm, &prob);CHKERRQ(ierr);
  switch(user->solType){
  case SOL_QUADRATIC:
    ierr = PetscDSSetResidual(prob, 0, f0_quadratic_v, f1_v);CHKERRQ(ierr);
    ierr = PetscDSSetResidual(prob, 2, f0_quadratic_w, f1_w);CHKERRQ(ierr);

    exactFuncs[0]   = quadratic_u;
    exactFuncs[1]   = quadratic_p;
    exactFuncs[2]   = quadratic_T;
    exactFuncs_t[0] = quadratic_u_t;
    exactFuncs_t[1] = NULL;
    exactFuncs_t[2] = quadratic_T_t;
    break;
  case SOL_CUBIC:
    ierr = PetscDSSetResidual(prob, 0, f0_cubic_v, f1_v);CHKERRQ(ierr);
    ierr = PetscDSSetResidual(prob, 2, f0_cubic_w, f1_w);CHKERRQ(ierr);

    exactFuncs[0]   = cubic_u;
    exactFuncs[1]   = cubic_p;
    exactFuncs[2]   = cubic_T;
    exactFuncs_t[0] = cubic_u_t;
    exactFuncs_t[1] = NULL;
    exactFuncs_t[2] = cubic_T_t;
    break;
  case SOL_CUBIC_TRIG:
    ierr = PetscDSSetResidual(prob, 0, f0_cubic_trig_v, f1_v);CHKERRQ(ierr);
    ierr = PetscDSSetResidual(prob, 2, f0_cubic_trig_w, f1_w);CHKERRQ(ierr);

    exactFuncs[0]   = cubic_trig_u;
    exactFuncs[1]   = cubic_trig_p;
    exactFuncs[2]   = cubic_trig_T;
    exactFuncs_t[0] = cubic_trig_u_t;
    exactFuncs_t[1] = NULL;
    exactFuncs_t[2] = cubic_trig_T_t;
    break;
   default: SETERRQ2(PetscObjectComm((PetscObject) prob), PETSC_ERR_ARG_WRONG, "Unsupported solution type: %s (%D)", solTypes[PetscMin(user->solType, NUM_SOL_TYPES)], user->solType);
  }

  ierr = PetscDSSetResidual(prob, 1, f0_q, NULL);CHKERRQ(ierr);

  ierr = PetscDSSetJacobian(prob, 0, 0, g0_vu, g1_vu,  NULL,  g3_vu);CHKERRQ(ierr);
  ierr = PetscDSSetJacobian(prob, 0, 1, NULL, NULL,  g2_vp, NULL);CHKERRQ(ierr);
  ierr = PetscDSSetJacobian(prob, 1, 0, NULL, g1_qu, NULL,  NULL);CHKERRQ(ierr);
  ierr = PetscDSSetJacobian(prob, 2, 0, g0_wu, NULL, NULL,  NULL);CHKERRQ(ierr);
  ierr = PetscDSSetJacobian(prob, 2, 2, g0_wT, g1_wT, NULL,  g3_wT);CHKERRQ(ierr);
  /* Setup constants */
  {
    Parameter  *param;
    PetscScalar constants[3];

    ierr = PetscBagGetData(user->bag, (void **) &param);CHKERRQ(ierr);

    constants[0] = param->nu;
    constants[1] = param->alpha;
    constants[2] = param->T_in;
    ierr = PetscDSSetConstants(prob, 3, constants);CHKERRQ(ierr);
  }
  /* Setup Boundary Conditions */
  ierr = PetscBagGetData(user->bag, (void **) &ctx);CHKERRQ(ierr);
  id   = 3;
  ierr = PetscDSAddBoundary(prob, DM_BC_ESSENTIAL, "top wall velocity",    "marker", 0, 0, NULL, (void (*)(void)) exactFuncs[0], (void (*)(void)) exactFuncs_t[0], 1, &id, ctx);CHKERRQ(ierr);
  id   = 1;
  ierr = PetscDSAddBoundary(prob, DM_BC_ESSENTIAL, "bottom wall velocity", "marker", 0, 0, NULL, (void (*)(void)) exactFuncs[0], (void (*)(void)) exactFuncs_t[0], 1, &id, ctx);CHKERRQ(ierr);
  id   = 2;
  ierr = PetscDSAddBoundary(prob, DM_BC_ESSENTIAL, "right wall velocity",  "marker", 0, 0, NULL, (void (*)(void)) exactFuncs[0], (void (*)(void)) exactFuncs_t[0], 1, &id, ctx);CHKERRQ(ierr);
  id   = 4;
  ierr = PetscDSAddBoundary(prob, DM_BC_ESSENTIAL, "left wall velocity",   "marker", 0, 0, NULL, (void (*)(void)) exactFuncs[0], (void (*)(void)) exactFuncs_t[0], 1, &id, ctx);CHKERRQ(ierr);
  id   = 3;
  ierr = PetscDSAddBoundary(prob, DM_BC_ESSENTIAL, "top wall temp",    "marker", 2, 0, NULL, (void (*)(void)) exactFuncs[2], (void (*)(void)) exactFuncs_t[2], 1, &id, ctx);CHKERRQ(ierr);
  id   = 1;
  ierr = PetscDSAddBoundary(prob, DM_BC_ESSENTIAL, "bottom wall temp", "marker", 2, 0, NULL, (void (*)(void)) exactFuncs[2], (void (*)(void)) exactFuncs_t[2], 1, &id, ctx);CHKERRQ(ierr);
  id   = 2;
  ierr = PetscDSAddBoundary(prob, DM_BC_ESSENTIAL, "right wall temp",  "marker", 2, 0, NULL, (void (*)(void)) exactFuncs[2], (void (*)(void)) exactFuncs_t[2], 1, &id, ctx);CHKERRQ(ierr);
  id   = 4;
  ierr = PetscDSAddBoundary(prob, DM_BC_ESSENTIAL, "left wall temp",   "marker", 2, 0, NULL, (void (*)(void)) exactFuncs[2], (void (*)(void)) exactFuncs_t[2], 1, &id, ctx);CHKERRQ(ierr);

  /*setup exact solution.*/
  ierr = PetscDSSetExactSolution(prob, 0, exactFuncs[0], ctx);CHKERRQ(ierr);
  ierr = PetscDSSetExactSolution(prob, 1, exactFuncs[1], ctx);CHKERRQ(ierr);
  ierr = PetscDSSetExactSolution(prob, 2, exactFuncs[2], ctx);CHKERRQ(ierr);
  ierr = PetscDSSetExactSolutionTimeDerivative(prob, 0, exactFuncs_t[0], ctx);CHKERRQ(ierr);
  ierr = PetscDSSetExactSolutionTimeDerivative(prob, 1, exactFuncs_t[1], ctx);CHKERRQ(ierr);
  ierr = PetscDSSetExactSolutionTimeDerivative(prob, 2, exactFuncs_t[2], ctx);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode SetupDiscretization(DM dm, AppCtx *user)
{
  DM              cdm   = dm;
  PetscFE         fe[3];
  Parameter      *param;
  MPI_Comm        comm;
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
  ierr = PetscObjectGetComm((PetscObject) dm, &comm);CHKERRQ(ierr);
  ierr = PetscFECreateDefault(comm, dim, dim, simplex, "vel_", PETSC_DEFAULT, &fe[0]);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) fe[0], "velocity");CHKERRQ(ierr);

  ierr = PetscFECreateDefault(comm, dim, 1, simplex, "pres_", PETSC_DEFAULT, &fe[1]);CHKERRQ(ierr);
  ierr = PetscFECopyQuadrature(fe[0], fe[1]);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) fe[1], "pressure");CHKERRQ(ierr);

  ierr = PetscFECreateDefault(comm, dim, 1, simplex, "temp_", PETSC_DEFAULT, &fe[2]);CHKERRQ(ierr);
  ierr = PetscFECopyQuadrature(fe[0], fe[2]);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) fe[2], "temperature");CHKERRQ(ierr);

  /* Set discretization and boundary conditions for each mesh */
  ierr = DMSetField(dm, 0, NULL, (PetscObject) fe[0]);CHKERRQ(ierr);
  ierr = DMSetField(dm, 1, NULL, (PetscObject) fe[1]);CHKERRQ(ierr);
  ierr = DMSetField(dm, 2, NULL, (PetscObject) fe[2]);CHKERRQ(ierr);
  ierr = DMCreateDS(dm);CHKERRQ(ierr);
  ierr = SetupProblem(dm, user);CHKERRQ(ierr);
  ierr = PetscBagGetData(user->bag, (void **) &param);CHKERRQ(ierr);
  while (cdm) {
    ierr = DMCopyDisc(dm, cdm);CHKERRQ(ierr);
    ierr = DMGetCoarseDM(cdm, &cdm);CHKERRQ(ierr);
  }
  ierr = PetscFEDestroy(&fe[0]);CHKERRQ(ierr);
  ierr = PetscFEDestroy(&fe[1]);CHKERRQ(ierr);
  ierr = PetscFEDestroy(&fe[2]);CHKERRQ(ierr);

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

static PetscErrorCode CreatePressureNullSpace(DM dm, PetscInt ofield, PetscInt nfield, MatNullSpace *nullSpace)
{
  Vec              vec;
  PetscErrorCode (*funcs[3])(PetscInt, PetscReal, const PetscReal[], PetscInt, PetscScalar *, void *) = {zero, zero, zero};
  PetscErrorCode   ierr;

  PetscFunctionBeginUser;
  if (ofield != 1) SETERRQ1(PetscObjectComm((PetscObject) dm), PETSC_ERR_ARG_WRONG, "Nullspace must be for pressure field at index 1, not %D", ofield);
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
  MatNullSpace   nullsp;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = TSGetDM(ts, &dm);CHKERRQ(ierr);
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

static PetscErrorCode SetInitialConditions(TS ts, Vec u)
{
  DM             dm;
  PetscReal      t;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = TSGetDM(ts, &dm);CHKERRQ(ierr);
  ierr = TSGetTime(ts, &t);CHKERRQ(ierr);
  ierr = DMComputeExactSolution(dm, t, u, NULL);CHKERRQ(ierr);
  ierr = RemoveDiscretePressureNullspace_Private(ts, u);CHKERRQ(ierr);
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
  PetscInt         f;
  PetscErrorCode   ierr;

  PetscFunctionBeginUser;
  ierr = TSGetDM(ts, &dm);CHKERRQ(ierr);
  ierr = DMGetDS(dm, &ds);CHKERRQ(ierr);

  for (f = 0; f < 3; ++f) {ierr = PetscDSGetExactSolution(ds, f, &exactFuncs[f], &ctxs[f]);CHKERRQ(ierr);}
  ierr = DMComputeL2FieldDiff(dm, crtime, exactFuncs, ctxs, u, ferrors);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD, "Timestep: %04d time = %-8.4g \t L_2 Error: [%2.3g, %2.3g, %2.3g]\n", (int) step, (double) crtime, (double) ferrors[0], (double) ferrors[1], (double) ferrors[2]);CHKERRQ(ierr);

  ierr = DMGetGlobalVector(dm, &u);CHKERRQ(ierr);
  //ierr = TSGetSolution(ts, &u);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) u, "Numerical Solution");CHKERRQ(ierr);
  ierr = VecViewFromOptions(u, NULL, "-sol_vec_view");CHKERRQ(ierr);
  ierr = DMRestoreGlobalVector(dm, &u);CHKERRQ(ierr);

  ierr = DMGetGlobalVector(dm, &v);CHKERRQ(ierr);
  // ierr = VecSet(v, 0.0);CHKERRQ(ierr);
  ierr = DMProjectFunction(dm, 0.0, exactFuncs, ctxs, INSERT_ALL_VALUES, v);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) v, "Exact Solution");CHKERRQ(ierr);
  ierr = VecViewFromOptions(v, NULL, "-exact_vec_view");CHKERRQ(ierr);
  ierr = DMRestoreGlobalVector(dm, &v);CHKERRQ(ierr);

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
  ierr = SetupParameters(&user);CHKERRQ(ierr);
  ierr = TSCreate(PETSC_COMM_WORLD, &ts);CHKERRQ(ierr);
  ierr = CreateMesh(PETSC_COMM_WORLD, &user, &dm);CHKERRQ(ierr);
  ierr = TSSetDM(ts, dm);CHKERRQ(ierr);
  ierr = DMSetApplicationContext(dm, &user);CHKERRQ(ierr);
  /* Setup problem */
  ierr = SetupDiscretization(dm, &user);CHKERRQ(ierr);
  ierr = DMPlexCreateClosureIndex(dm, NULL);CHKERRQ(ierr);

  ierr = DMCreateGlobalVector(dm, &u);CHKERRQ(ierr);
  ierr = DMSetNullSpaceConstructor(dm, 1, CreatePressureNullSpace);CHKERRQ(ierr);

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
  ierr = PetscObjectSetName((PetscObject) u, "Numerical Solution");CHKERRQ(ierr);

  ierr = VecDestroy(&u);CHKERRQ(ierr);
  ierr = DMDestroy(&dm);CHKERRQ(ierr);
  ierr = TSDestroy(&ts);CHKERRQ(ierr);
  ierr = PetscBagDestroy(&user.bag);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

  test:
    suffix: 2d_tri_p2_p1_p1
    requires: triangle !single
    args: -dm_plex_separate_marker -sol_type quadratic -dm_refine 0 \
      -vel_petscspace_degree 2 -pres_petscspace_degree 1 -temp_petscspace_degree 1 \
      -dmts_check .001 -ts_max_steps 4 -ts_dt 0.1 \
      -ksp_type fgmres -ksp_gmres_restart 10 -ksp_rtol 1.0e-9 -ksp_error_if_not_converged \
      -pc_type fieldsplit -pc_fieldsplit_0_fields 0,2 -pc_fieldsplit_1_fields 1 -pc_fieldsplit_type schur -pc_fieldsplit_schur_factorization_type full \
        -fieldsplit_0_pc_type lu \
        -fieldsplit_pressure_ksp_rtol 1e-10 -fieldsplit_pressure_pc_type jacobi

  # TODO Need trig t for convergence in time, also need to refine in space
  test:
    # Using -dm_refine 5 -convest_num_refine 2 gives L_2 convergence rate: [0.89, 0.011, 1.0]
    suffix: 2d_tri_p2_p1_p1_tconv
    requires: triangle !single
    args: -dm_plex_separate_marker -sol_type cubic_trig -dm_refine 0 \
      -vel_petscspace_degree 2 -pres_petscspace_degree 1 -temp_petscspace_degree 1 \
      -ts_max_steps 4 -ts_dt 0.1 -ts_convergence_estimate -convest_num_refine 1 \
      -snes_error_if_not_converged -snes_convergence_test correct_pressure \
      -ksp_type fgmres -ksp_gmres_restart 10 -ksp_rtol 1.0e-9 -ksp_error_if_not_converged \
      -pc_type fieldsplit -pc_fieldsplit_0_fields 0,2 -pc_fieldsplit_1_fields 1 -pc_fieldsplit_type schur -pc_fieldsplit_schur_factorization_type full \
        -fieldsplit_0_pc_type lu \
        -fieldsplit_pressure_ksp_rtol 1e-10 -fieldsplit_pressure_pc_type jacobi

  test:
    # Using -dm_refine 3 -convest_num_refine 3 gives L_2 convergence rate: [3.0, 2.5, 1.9]
    suffix: 2d_tri_p2_p1_p1_sconv
    requires: triangle !single
    args: -dm_plex_separate_marker -sol_type cubic -dm_refine 0 \
      -vel_petscspace_degree 2 -pres_petscspace_degree 1 -temp_petscspace_degree 1 \
      -ts_max_steps 1 -ts_dt 1e-4 -ts_convergence_estimate -ts_convergence_temporal 0 -convest_num_refine 1 \
      -snes_error_if_not_converged -snes_convergence_test correct_pressure \
      -ksp_type fgmres -ksp_gmres_restart 10 -ksp_rtol 1.0e-9 -ksp_atol 1e-16 -ksp_error_if_not_converged \
      -pc_type fieldsplit -pc_fieldsplit_0_fields 0,2 -pc_fieldsplit_1_fields 1 -pc_fieldsplit_type schur -pc_fieldsplit_schur_factorization_type full \
        -fieldsplit_0_pc_type lu \
        -fieldsplit_pressure_ksp_rtol 1e-10 -fieldsplit_pressure_pc_type jacobi

  test:
    suffix: 2d_tri_p3_p2_p2
    requires: triangle !single
    args: -dm_plex_separate_marker -sol_type cubic -dm_refine 0 \
      -vel_petscspace_degree 3 -pres_petscspace_degree 2 -temp_petscspace_degree 2 \
      -dmts_check .001 -ts_max_steps 4 -ts_dt 0.1 \
      -snes_convergence_test correct_pressure \
      -ksp_type fgmres -ksp_gmres_restart 10 -ksp_rtol 1.0e-9 -ksp_error_if_not_converged \
      -pc_type fieldsplit -pc_fieldsplit_0_fields 0,2 -pc_fieldsplit_1_fields 1 -pc_fieldsplit_type schur -pc_fieldsplit_schur_factorization_type full \
        -fieldsplit_0_pc_type lu \
        -fieldsplit_pressure_ksp_rtol 1e-10 -fieldsplit_pressure_pc_type jacobi

TEST*/
