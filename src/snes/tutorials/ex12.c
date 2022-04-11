static char help[] = "Poisson Problem in 2d and 3d with simplicial finite elements.\n\
We solve the Poisson problem in a rectangular\n\
domain, using a parallel unstructured mesh (DMPLEX) to discretize it.\n\
This example supports discretized auxiliary fields (conductivity) as well as\n\
multilevel nonlinear solvers.\n\n\n";

/*
A visualization of the adaptation can be accomplished using:

  -dm_adapt_view hdf5:$PWD/adapt.h5 -sol_adapt_view hdf5:$PWD/adapt.h5::append -dm_adapt_pre_view hdf5:$PWD/orig.h5 -sol_adapt_pre_view hdf5:$PWD/orig.h5::append

Information on refinement:

   -info :~sys,vec,is,mat,ksp,snes,ts
*/

#include <petscdmplex.h>
#include <petscdmadaptor.h>
#include <petscsnes.h>
#include <petscds.h>
#include <petscviewerhdf5.h>

typedef enum {NEUMANN, DIRICHLET, NONE} BCType;
typedef enum {RUN_FULL, RUN_EXACT, RUN_TEST, RUN_PERF} RunType;
typedef enum {COEFF_NONE, COEFF_ANALYTIC, COEFF_FIELD, COEFF_NONLINEAR, COEFF_BALL, COEFF_CROSS, COEFF_CHECKERBOARD_0, COEFF_CHECKERBOARD_1} CoeffType;

typedef struct {
  RunType        runType;           /* Whether to run tests, or solve the full problem */
  PetscBool      jacobianMF;        /* Whether to calculate the Jacobian action on the fly */
  PetscBool      showInitial, showSolution, restart, quiet, nonzInit;
  /* Problem definition */
  BCType         bcType;
  CoeffType      variableCoefficient;
  PetscErrorCode (**exactFuncs)(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nc, PetscScalar *u, void *ctx);
  PetscBool      fieldBC;
  void           (**exactFields)(PetscInt, PetscInt, PetscInt,
                                 const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[],
                                 const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[],
                                 PetscReal, const PetscReal[], PetscInt, const PetscScalar[], PetscScalar[]);
  PetscBool      bdIntegral;        /* Compute the integral of the solution on the boundary */
  /* Reproducing tests from SISC 40(3), pp. A1473-A1493, 2018 */
  PetscInt       div;               /* Number of divisions */
  PetscInt       k;                 /* Parameter for checkerboard coefficient */
  PetscInt      *kgrid;             /* Random parameter grid */
  PetscBool      rand;              /* Make random assignments */
  /* Solver */
  PC             pcmg;              /* This is needed for error monitoring */
  PetscBool      checkksp;          /* Whether to check the KSPSolve for runType == RUN_TEST */
} AppCtx;

static PetscErrorCode zero(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nc, PetscScalar *u, void *ctx)
{
  u[0] = 0.0;
  return 0;
}

static PetscErrorCode ecks(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nc, PetscScalar *u, void *ctx)
{
  u[0] = x[0];
  return 0;
}

/*
  In 2D for Dirichlet conditions, we use exact solution:

    u = x^2 + y^2
    f = 4

  so that

    -\Delta u + f = -4 + 4 = 0

  For Neumann conditions, we have

    -\nabla u \cdot -\hat y |_{y=0} =  (2y)|_{y=0} =  0 (bottom)
    -\nabla u \cdot  \hat y |_{y=1} = -(2y)|_{y=1} = -2 (top)
    -\nabla u \cdot -\hat x |_{x=0} =  (2x)|_{x=0} =  0 (left)
    -\nabla u \cdot  \hat x |_{x=1} = -(2x)|_{x=1} = -2 (right)

  Which we can express as

    \nabla u \cdot  \hat n|_\Gamma = {2 x, 2 y} \cdot \hat n = 2 (x + y)

  The boundary integral of this solution is (assuming we are not orienting the edges)

    \int^1_0 x^2 dx + \int^1_0 (1 + y^2) dy + \int^1_0 (x^2 + 1) dx + \int^1_0 y^2 dy = 1/3 + 4/3 + 4/3 + 1/3 = 3 1/3
*/
static PetscErrorCode quadratic_u_2d(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nc, PetscScalar *u, void *ctx)
{
  *u = x[0]*x[0] + x[1]*x[1];
  return 0;
}

static void quadratic_u_field_2d(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                                 const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                                 const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                                 PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar uexact[])
{
  uexact[0] = a[0];
}

static PetscErrorCode ball_u_2d(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nc, PetscScalar *u, void *ctx)
{
  const PetscReal alpha   = 500.;
  const PetscReal radius2 = PetscSqr(0.15);
  const PetscReal r2      = PetscSqr(x[0] - 0.5) + PetscSqr(x[1] - 0.5);
  const PetscReal xi      = alpha*(radius2 - r2);

  *u = PetscTanhScalar(xi) + 1.0;
  return 0;
}

static PetscErrorCode cross_u_2d(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nc, PetscScalar *u, void *ctx)
{
  const PetscReal alpha = 50*4;
  const PetscReal xy    = (x[0]-0.5)*(x[1]-0.5);

  *u = PetscSinReal(alpha*xy) * (alpha*PetscAbsReal(xy) < 2*PETSC_PI ? 1 : 0.01);
  return 0;
}

static void f0_u(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                 const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                 const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                 PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f0[])
{
  f0[0] = 4.0;
}

static void f0_ball_u(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                      const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                      const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                      PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f0[])
{
  PetscInt        d;
  const PetscReal alpha = 500., radius2 = PetscSqr(0.15);
  PetscReal       r2, xi;

  for (d = 0, r2 = 0.0; d < dim; ++d) r2 += PetscSqr(x[d] - 0.5);
  xi = alpha*(radius2 - r2);
  f0[0] = (-2.0*dim*alpha - 8.0*PetscSqr(alpha)*r2*PetscTanhReal(xi)) * PetscSqr(1.0/PetscCoshReal(xi));
}

static void f0_cross_u_2d(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                          const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                          const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                          PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f0[])
{
  const PetscReal alpha = 50*4;
  const PetscReal xy    = (x[0]-0.5)*(x[1]-0.5);

  f0[0] = PetscSinReal(alpha*xy) * (alpha*PetscAbsReal(xy) < 2*PETSC_PI ? 1 : 0.01);
}

static void f0_checkerboard_0_u(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                                const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                                const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                                PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f0[])
{
  f0[0] = -20.0*PetscExpReal(-(PetscSqr(x[0] - 0.5) + PetscSqr(x[1] - 0.5)));
}

static void f0_bd_u(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                    const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                    const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                    PetscReal t, const PetscReal x[], const PetscReal n[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f0[])
{
  PetscInt d;
  for (d = 0, f0[0] = 0.0; d < dim; ++d) f0[0] += -n[d]*2.0*x[d];
}

/* gradU[comp*dim+d] = {u_x, u_y} or {u_x, u_y, u_z} */
static void f1_u(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                 const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                 const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                 PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f1[])
{
  PetscInt d;
  for (d = 0; d < dim; ++d) f1[d] = u_x[d];
}

/* < \nabla v, \nabla u + {\nabla u}^T >
   This just gives \nabla u, give the perdiagonal for the transpose */
static void g3_uu(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                  const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                  const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                  PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g3[])
{
  PetscInt d;
  for (d = 0; d < dim; ++d) g3[d*dim+d] = 1.0;
}

/*
  In 2D for x periodicity and y Dirichlet conditions, we use exact solution:

    u = sin(2 pi x)
    f = -4 pi^2 sin(2 pi x)

  so that

    -\Delta u + f = 4 pi^2 sin(2 pi x) - 4 pi^2 sin(2 pi x) = 0
*/
static PetscErrorCode xtrig_u_2d(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nc, PetscScalar *u, void *ctx)
{
  *u = PetscSinReal(2.0*PETSC_PI*x[0]);
  return 0;
}

static void f0_xtrig_u(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                       const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                       const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                       PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f0[])
{
  f0[0] = -4.0*PetscSqr(PETSC_PI)*PetscSinReal(2.0*PETSC_PI*x[0]);
}

/*
  In 2D for x-y periodicity, we use exact solution:

    u = sin(2 pi x) sin(2 pi y)
    f = -8 pi^2 sin(2 pi x)

  so that

    -\Delta u + f = 4 pi^2 sin(2 pi x) sin(2 pi y) + 4 pi^2 sin(2 pi x) sin(2 pi y) - 8 pi^2 sin(2 pi x) = 0
*/
static PetscErrorCode xytrig_u_2d(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nc, PetscScalar *u, void *ctx)
{
  *u = PetscSinReal(2.0*PETSC_PI*x[0])*PetscSinReal(2.0*PETSC_PI*x[1]);
  return 0;
}

static void f0_xytrig_u(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                        const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                        const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                        PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f0[])
{
  f0[0] = -8.0*PetscSqr(PETSC_PI)*PetscSinReal(2.0*PETSC_PI*x[0]);
}

/*
  In 2D for Dirichlet conditions with a variable coefficient, we use exact solution:

    u  = x^2 + y^2
    f  = 6 (x + y)
    nu = (x + y)

  so that

    -\div \nu \grad u + f = -6 (x + y) + 6 (x + y) = 0
*/
static PetscErrorCode nu_2d(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nc, PetscScalar *u, void *ctx)
{
  *u = x[0] + x[1];
  return 0;
}

static PetscErrorCode checkerboardCoeff(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nc, PetscScalar *u, void *ctx)
{
  AppCtx  *user = (AppCtx *) ctx;
  PetscInt div  = user->div;
  PetscInt k    = user->k;
  PetscInt mask = 0, ind = 0, d;

  PetscFunctionBeginUser;
  for (d = 0; d < dim; ++d) mask = (mask + (PetscInt) (x[d]*div)) % 2;
  if (user->kgrid) {
    for (d = 0; d < dim; ++d) {
      if (d > 0) ind *= dim;
      ind += (PetscInt) (x[d]*div);
    }
    k = user->kgrid[ind];
  }
  u[0] = mask ? 1.0 : PetscPowRealInt(10.0, -k);
  PetscFunctionReturn(0);
}

void f0_analytic_u(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                   const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                   const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                   PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f0[])
{
  f0[0] = 6.0*(x[0] + x[1]);
}

/* gradU[comp*dim+d] = {u_x, u_y} or {u_x, u_y, u_z} */
void f1_analytic_u(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                   const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                   const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                   PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f1[])
{
  PetscInt d;
  for (d = 0; d < dim; ++d) f1[d] = (x[0] + x[1])*u_x[d];
}

void f1_field_u(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f1[])
{
  PetscInt d;
  for (d = 0; d < dim; ++d) f1[d] = a[0]*u_x[d];
}

/* < \nabla v, \nabla u + {\nabla u}^T >
   This just gives \nabla u, give the perdiagonal for the transpose */
void g3_analytic_uu(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                    const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                    const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                    PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g3[])
{
  PetscInt d;
  for (d = 0; d < dim; ++d) g3[d*dim+d] = x[0] + x[1];
}

void g3_field_uu(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                 const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                 const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                 PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g3[])
{
  PetscInt d;
  for (d = 0; d < dim; ++d) g3[d*dim+d] = a[0];
}

/*
  In 2D for Dirichlet conditions with a nonlinear coefficient (p-Laplacian with p = 4), we use exact solution:

    u  = x^2 + y^2
    f  = 16 (x^2 + y^2)
    nu = 1/2 |grad u|^2

  so that

    -\div \nu \grad u + f = -16 (x^2 + y^2) + 16 (x^2 + y^2) = 0
*/
void f0_analytic_nonlinear_u(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                             const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                             const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                             PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f0[])
{
  f0[0] = 16.0*(x[0]*x[0] + x[1]*x[1]);
}

/* gradU[comp*dim+d] = {u_x, u_y} or {u_x, u_y, u_z} */
void f1_analytic_nonlinear_u(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                             const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                             const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                             PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f1[])
{
  PetscScalar nu = 0.0;
  PetscInt    d;
  for (d = 0; d < dim; ++d) nu += u_x[d]*u_x[d];
  for (d = 0; d < dim; ++d) f1[d] = 0.5*nu*u_x[d];
}

/*
  grad (u + eps w) - grad u = eps grad w

  1/2 |grad (u + eps w)|^2 grad (u + eps w) - 1/2 |grad u|^2 grad u
= 1/2 (|grad u|^2 + 2 eps <grad u,grad w>) (grad u + eps grad w) - 1/2 |grad u|^2 grad u
= 1/2 (eps |grad u|^2 grad w + 2 eps <grad u,grad w> grad u)
= eps (1/2 |grad u|^2 grad w + grad u <grad u,grad w>)
*/
void g3_analytic_nonlinear_uu(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                              const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                              const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                              PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g3[])
{
  PetscScalar nu = 0.0;
  PetscInt    d, e;
  for (d = 0; d < dim; ++d) nu += u_x[d]*u_x[d];
  for (d = 0; d < dim; ++d) {
    g3[d*dim+d] = 0.5*nu;
    for (e = 0; e < dim; ++e) {
      g3[d*dim+e] += u_x[d]*u_x[e];
    }
  }
}

/*
  In 3D for Dirichlet conditions we use exact solution:

    u = 2/3 (x^2 + y^2 + z^2)
    f = 4

  so that

    -\Delta u + f = -2/3 * 6 + 4 = 0

  For Neumann conditions, we have

    -\nabla u \cdot -\hat z |_{z=0} =  (2z)|_{z=0} =  0 (bottom)
    -\nabla u \cdot  \hat z |_{z=1} = -(2z)|_{z=1} = -2 (top)
    -\nabla u \cdot -\hat y |_{y=0} =  (2y)|_{y=0} =  0 (front)
    -\nabla u \cdot  \hat y |_{y=1} = -(2y)|_{y=1} = -2 (back)
    -\nabla u \cdot -\hat x |_{x=0} =  (2x)|_{x=0} =  0 (left)
    -\nabla u \cdot  \hat x |_{x=1} = -(2x)|_{x=1} = -2 (right)

  Which we can express as

    \nabla u \cdot  \hat n|_\Gamma = {2 x, 2 y, 2z} \cdot \hat n = 2 (x + y + z)
*/
static PetscErrorCode quadratic_u_3d(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nc, PetscScalar *u, void *ctx)
{
  *u = 2.0*(x[0]*x[0] + x[1]*x[1] + x[2]*x[2])/3.0;
  return 0;
}

static PetscErrorCode ball_u_3d(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nc, PetscScalar *u, void *ctx)
{
  const PetscReal alpha   = 500.;
  const PetscReal radius2 = PetscSqr(0.15);
  const PetscReal r2      = PetscSqr(x[0] - 0.5) + PetscSqr(x[1] - 0.5) + PetscSqr(x[2] - 0.5);
  const PetscReal xi      = alpha*(radius2 - r2);

  *u = PetscTanhScalar(xi) + 1.0;
  return 0;
}

static void quadratic_u_field_3d(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                                 const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                                 const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                                 PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar uexact[])
{
  uexact[0] = a[0];
}

static PetscErrorCode cross_u_3d(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nc, PetscScalar *u, void *ctx)
{
  const PetscReal alpha = 50*4;
  const PetscReal xyz   = (x[0]-0.5)*(x[1]-0.5)*(x[2]-0.5);

  *u = PetscSinReal(alpha*xyz) * (alpha*PetscAbsReal(xyz) < 2*PETSC_PI ? (alpha*PetscAbsReal(xyz) > -2*PETSC_PI ? 1.0 : 0.01) : 0.01);
  return 0;
}

static void f0_cross_u_3d(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                          const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                          const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                          PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f0[])
{
  const PetscReal alpha = 50*4;
  const PetscReal xyz   = (x[0]-0.5)*(x[1]-0.5)*(x[2]-0.5);

  f0[0] = PetscSinReal(alpha*xyz) * (alpha*PetscAbsReal(xyz) < 2*PETSC_PI ? (alpha*PetscAbsReal(xyz) > -2*PETSC_PI ? 1.0 : 0.01) : 0.01);
}

static void bd_integral_2d(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                           const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                           const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                           PetscReal t, const PetscReal x[], const PetscReal n[], PetscInt numConstants, const PetscScalar constants[], PetscScalar *uint)
{
  uint[0] = u[0];
}

static PetscErrorCode ProcessOptions(MPI_Comm comm, AppCtx *options)
{
  const char    *bcTypes[3]  = {"neumann", "dirichlet", "none"};
  const char    *runTypes[4] = {"full", "exact", "test", "perf"};
  const char    *coeffTypes[8] = {"none", "analytic", "field", "nonlinear", "ball", "cross", "checkerboard_0", "checkerboard_1"};
  PetscInt       bc, run, coeff;

  PetscFunctionBeginUser;
  options->runType             = RUN_FULL;
  options->bcType              = DIRICHLET;
  options->variableCoefficient = COEFF_NONE;
  options->fieldBC             = PETSC_FALSE;
  options->jacobianMF          = PETSC_FALSE;
  options->showInitial         = PETSC_FALSE;
  options->showSolution        = PETSC_FALSE;
  options->restart             = PETSC_FALSE;
  options->quiet               = PETSC_FALSE;
  options->nonzInit            = PETSC_FALSE;
  options->bdIntegral          = PETSC_FALSE;
  options->checkksp            = PETSC_FALSE;
  options->div                 = 4;
  options->k                   = 1;
  options->kgrid               = NULL;
  options->rand                = PETSC_FALSE;

  PetscOptionsBegin(comm, "", "Poisson Problem Options", "DMPLEX");
  run  = options->runType;
  PetscCall(PetscOptionsEList("-run_type", "The run type", "ex12.c", runTypes, 4, runTypes[options->runType], &run, NULL));
  options->runType = (RunType) run;
  bc   = options->bcType;
  PetscCall(PetscOptionsEList("-bc_type","Type of boundary condition","ex12.c",bcTypes,3,bcTypes[options->bcType],&bc,NULL));
  options->bcType = (BCType) bc;
  coeff = options->variableCoefficient;
  PetscCall(PetscOptionsEList("-variable_coefficient","Type of variable coefficent","ex12.c",coeffTypes,8,coeffTypes[options->variableCoefficient],&coeff,NULL));
  options->variableCoefficient = (CoeffType) coeff;

  PetscCall(PetscOptionsBool("-field_bc", "Use a field representation for the BC", "ex12.c", options->fieldBC, &options->fieldBC, NULL));
  PetscCall(PetscOptionsBool("-jacobian_mf", "Calculate the action of the Jacobian on the fly", "ex12.c", options->jacobianMF, &options->jacobianMF, NULL));
  PetscCall(PetscOptionsBool("-show_initial", "Output the initial guess for verification", "ex12.c", options->showInitial, &options->showInitial, NULL));
  PetscCall(PetscOptionsBool("-show_solution", "Output the solution for verification", "ex12.c", options->showSolution, &options->showSolution, NULL));
  PetscCall(PetscOptionsBool("-restart", "Read in the mesh and solution from a file", "ex12.c", options->restart, &options->restart, NULL));
  PetscCall(PetscOptionsBool("-quiet", "Don't print any vecs", "ex12.c", options->quiet, &options->quiet, NULL));
  PetscCall(PetscOptionsBool("-nonzero_initial_guess", "nonzero initial guess", "ex12.c", options->nonzInit, &options->nonzInit, NULL));
  PetscCall(PetscOptionsBool("-bd_integral", "Compute the integral of the solution on the boundary", "ex12.c", options->bdIntegral, &options->bdIntegral, NULL));
  if (options->runType == RUN_TEST) {
    PetscCall(PetscOptionsBool("-run_test_check_ksp", "Check solution of KSP", "ex12.c", options->checkksp, &options->checkksp, NULL));
  }
  PetscCall(PetscOptionsInt("-div", "The number of division for the checkerboard coefficient", "ex12.c", options->div, &options->div, NULL));
  PetscCall(PetscOptionsInt("-k", "The exponent for the checkerboard coefficient", "ex12.c", options->k, &options->k, NULL));
  PetscCall(PetscOptionsBool("-k_random", "Assign random k values to checkerboard", "ex12.c", options->rand, &options->rand, NULL));
  PetscOptionsEnd();
  PetscFunctionReturn(0);
}

static PetscErrorCode CreateBCLabel(DM dm, const char name[])
{
  DM             plex;
  DMLabel        label;

  PetscFunctionBeginUser;
  PetscCall(DMCreateLabel(dm, name));
  PetscCall(DMGetLabel(dm, name, &label));
  PetscCall(DMConvert(dm, DMPLEX, &plex));
  PetscCall(DMPlexMarkBoundaryFaces(plex, 1, label));
  PetscCall(DMDestroy(&plex));
  PetscFunctionReturn(0);
}

static PetscErrorCode CreateMesh(MPI_Comm comm, AppCtx *user, DM *dm)
{
  PetscFunctionBeginUser;
  PetscCall(DMCreate(comm, dm));
  PetscCall(DMSetType(*dm, DMPLEX));
  PetscCall(DMSetFromOptions(*dm));
  {
    char      convType[256];
    PetscBool flg;

    PetscOptionsBegin(comm, "", "Mesh conversion options", "DMPLEX");
    PetscCall(PetscOptionsFList("-dm_plex_convert_type","Convert DMPlex to another format","ex12",DMList,DMPLEX,convType,256,&flg));
    PetscOptionsEnd();
    if (flg) {
      DM dmConv;

      PetscCall(DMConvert(*dm,convType,&dmConv));
      if (dmConv) {
        PetscCall(DMDestroy(dm));
        *dm  = dmConv;
      }
      PetscCall(DMSetFromOptions(*dm));
      PetscCall(DMSetUp(*dm));
    }
  }
  PetscCall(DMViewFromOptions(*dm, NULL, "-dm_view"));
  if (user->rand) {
    PetscRandom r;
    PetscReal   val;
    PetscInt    dim, N, i;

    PetscCall(DMGetDimension(*dm, &dim));
    N    = PetscPowInt(user->div, dim);
    PetscCall(PetscMalloc1(N, &user->kgrid));
    PetscCall(PetscRandomCreate(PETSC_COMM_SELF, &r));
    PetscCall(PetscRandomSetFromOptions(r));
    PetscCall(PetscRandomSetInterval(r, 0.0, user->k));
    PetscCall(PetscRandomSetSeed(r, 1973));
    PetscCall(PetscRandomSeed(r));
    for (i = 0; i < N; ++i) {
      PetscCall(PetscRandomGetValueReal(r, &val));
      user->kgrid[i] = 1 + (PetscInt) val;
    }
    PetscCall(PetscRandomDestroy(&r));
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode SetupProblem(DM dm, AppCtx *user)
{
  PetscDS         ds;
  DMLabel         label;
  PetscWeakForm   wf;
  const DMBoundaryType *periodicity;
  const PetscInt  id = 1;
  PetscInt        bd, dim;

  PetscFunctionBeginUser;
  PetscCall(DMGetDS(dm, &ds));
  PetscCall(DMGetDimension(dm, &dim));
  PetscCall(DMGetPeriodicity(dm, NULL, NULL, NULL, &periodicity));
  switch (user->variableCoefficient) {
  case COEFF_NONE:
    if (periodicity && periodicity[0]) {
      if (periodicity && periodicity[1]) {
        PetscCall(PetscDSSetResidual(ds, 0, f0_xytrig_u, f1_u));
        PetscCall(PetscDSSetJacobian(ds, 0, 0, NULL, NULL, NULL, g3_uu));
      } else {
        PetscCall(PetscDSSetResidual(ds, 0, f0_xtrig_u,  f1_u));
        PetscCall(PetscDSSetJacobian(ds, 0, 0, NULL, NULL, NULL, g3_uu));
      }
    } else {
      PetscCall(PetscDSSetResidual(ds, 0, f0_u, f1_u));
      PetscCall(PetscDSSetJacobian(ds, 0, 0, NULL, NULL, NULL, g3_uu));
    }
    break;
  case COEFF_ANALYTIC:
    PetscCall(PetscDSSetResidual(ds, 0, f0_analytic_u, f1_analytic_u));
    PetscCall(PetscDSSetJacobian(ds, 0, 0, NULL, NULL, NULL, g3_analytic_uu));
    break;
  case COEFF_FIELD:
    PetscCall(PetscDSSetResidual(ds, 0, f0_analytic_u, f1_field_u));
    PetscCall(PetscDSSetJacobian(ds, 0, 0, NULL, NULL, NULL, g3_field_uu));
    break;
  case COEFF_NONLINEAR:
    PetscCall(PetscDSSetResidual(ds, 0, f0_analytic_nonlinear_u, f1_analytic_nonlinear_u));
    PetscCall(PetscDSSetJacobian(ds, 0, 0, NULL, NULL, NULL, g3_analytic_nonlinear_uu));
    break;
  case COEFF_BALL:
    PetscCall(PetscDSSetResidual(ds, 0, f0_ball_u, f1_u));
    PetscCall(PetscDSSetJacobian(ds, 0, 0, NULL, NULL, NULL, g3_uu));
    break;
  case COEFF_CROSS:
    switch (dim) {
    case 2:
      PetscCall(PetscDSSetResidual(ds, 0, f0_cross_u_2d, f1_u));
      break;
    case 3:
      PetscCall(PetscDSSetResidual(ds, 0, f0_cross_u_3d, f1_u));
      break;
    default:
      SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE, "Invalid dimension %d", dim);
    }
    PetscCall(PetscDSSetJacobian(ds, 0, 0, NULL, NULL, NULL, g3_uu));
    break;
  case COEFF_CHECKERBOARD_0:
    PetscCall(PetscDSSetResidual(ds, 0, f0_checkerboard_0_u, f1_field_u));
    PetscCall(PetscDSSetJacobian(ds, 0, 0, NULL, NULL, NULL, g3_field_uu));
    break;
  default: SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Invalid variable coefficient type %d", user->variableCoefficient);
  }
  switch (dim) {
  case 2:
    switch (user->variableCoefficient) {
    case COEFF_BALL:
      user->exactFuncs[0]  = ball_u_2d;break;
    case COEFF_CROSS:
      user->exactFuncs[0]  = cross_u_2d;break;
    case COEFF_CHECKERBOARD_0:
      user->exactFuncs[0]  = zero;break;
    default:
      if (periodicity && periodicity[0]) {
        if (periodicity && periodicity[1]) {
          user->exactFuncs[0] = xytrig_u_2d;
        } else {
          user->exactFuncs[0] = xtrig_u_2d;
        }
      } else {
        user->exactFuncs[0]  = quadratic_u_2d;
        user->exactFields[0] = quadratic_u_field_2d;
      }
    }
    if (user->bcType == NEUMANN) {
      PetscCall(DMGetLabel(dm, "boundary", &label));
      PetscCall(DMAddBoundary(dm, DM_BC_NATURAL, "wall", label, 1, &id, 0, 0, NULL, NULL, NULL, user, &bd));
      PetscCall(PetscDSGetBoundary(ds, bd, &wf, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL));
      PetscCall(PetscWeakFormSetIndexBdResidual(wf, label, id, 0, 0, 0, f0_bd_u, 0, NULL));
    }
    break;
  case 3:
    switch (user->variableCoefficient) {
    case COEFF_BALL:
      user->exactFuncs[0]  = ball_u_3d;break;
    case COEFF_CROSS:
      user->exactFuncs[0]  = cross_u_3d;break;
    default:
      user->exactFuncs[0]  = quadratic_u_3d;
      user->exactFields[0] = quadratic_u_field_3d;
    }
    if (user->bcType == NEUMANN) {
      PetscCall(DMGetLabel(dm, "boundary", &label));
      PetscCall(DMAddBoundary(dm, DM_BC_NATURAL, "wall", label, 1, &id, 0, 0, NULL, NULL, NULL, user, &bd));
      PetscCall(PetscDSGetBoundary(ds, bd, &wf, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL));
      PetscCall(PetscWeakFormSetIndexBdResidual(wf, label, id, 0, 0, 0, f0_bd_u, 0, NULL));
    }
    break;
  default:
    SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE, "Invalid dimension %d", dim);
  }
  /* Setup constants */
  switch (user->variableCoefficient) {
  case COEFF_CHECKERBOARD_0:
  {
    PetscScalar constants[2];

    constants[0] = user->div;
    constants[1] = user->k;
    PetscCall(PetscDSSetConstants(ds, 2, constants));
  }
  break;
  default: break;
  }
  PetscCall(PetscDSSetExactSolution(ds, 0, user->exactFuncs[0], user));
  /* Setup Boundary Conditions */
  if (user->bcType == DIRICHLET) {
    PetscCall(DMGetLabel(dm, "marker", &label));
    if (!label) {
      /* Right now, p4est cannot create labels immediately */
      PetscCall(PetscDSAddBoundaryByName(ds, user->fieldBC ? DM_BC_ESSENTIAL_FIELD : DM_BC_ESSENTIAL, "wall", "marker", 1, &id, 0, 0, NULL, user->fieldBC ? (void (*)(void)) user->exactFields[0] : (void (*)(void)) user->exactFuncs[0], NULL, user, NULL));
    } else {
      PetscCall(DMAddBoundary(dm, user->fieldBC ? DM_BC_ESSENTIAL_FIELD : DM_BC_ESSENTIAL, "wall", label, 1, &id, 0, 0, NULL, user->fieldBC ? (void (*)(void)) user->exactFields[0] : (void (*)(void)) user->exactFuncs[0], NULL, user, NULL));
    }
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode SetupMaterial(DM dm, DM dmAux, AppCtx *user)
{
  PetscErrorCode (*matFuncs[1])(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nc, PetscScalar u[], void *ctx) = {nu_2d};
  void            *ctx[1];
  Vec              nu;

  PetscFunctionBegin;
  ctx[0] = user;
  if (user->variableCoefficient == COEFF_CHECKERBOARD_0) {matFuncs[0] = checkerboardCoeff;}
  PetscCall(DMCreateLocalVector(dmAux, &nu));
  PetscCall(PetscObjectSetName((PetscObject) nu, "Coefficient"));
  PetscCall(DMProjectFunctionLocal(dmAux, 0.0, matFuncs, ctx, INSERT_ALL_VALUES, nu));
  PetscCall(DMSetAuxiliaryVec(dm, NULL, 0, 0, nu));
  PetscCall(VecDestroy(&nu));
  PetscFunctionReturn(0);
}

static PetscErrorCode SetupBC(DM dm, DM dmAux, AppCtx *user)
{
  PetscErrorCode (*bcFuncs[1])(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nc, PetscScalar u[], void *ctx);
  Vec            uexact;
  PetscInt       dim;

  PetscFunctionBegin;
  PetscCall(DMGetDimension(dm, &dim));
  if (dim == 2) bcFuncs[0] = quadratic_u_2d;
  else          bcFuncs[0] = quadratic_u_3d;
  PetscCall(DMCreateLocalVector(dmAux, &uexact));
  PetscCall(DMProjectFunctionLocal(dmAux, 0.0, bcFuncs, NULL, INSERT_ALL_VALUES, uexact));
  PetscCall(DMSetAuxiliaryVec(dm, NULL, 0, 0, uexact));
  PetscCall(VecDestroy(&uexact));
  PetscFunctionReturn(0);
}

static PetscErrorCode SetupAuxDM(DM dm, PetscFE feAux, AppCtx *user)
{
  DM             dmAux, coordDM;

  PetscFunctionBegin;
  /* MUST call DMGetCoordinateDM() in order to get p4est setup if present */
  PetscCall(DMGetCoordinateDM(dm, &coordDM));
  if (!feAux) PetscFunctionReturn(0);
  PetscCall(DMClone(dm, &dmAux));
  PetscCall(DMSetCoordinateDM(dmAux, coordDM));
  PetscCall(DMSetField(dmAux, 0, NULL, (PetscObject) feAux));
  PetscCall(DMCreateDS(dmAux));
  if (user->fieldBC) PetscCall(SetupBC(dm, dmAux, user));
  else               PetscCall(SetupMaterial(dm, dmAux, user));
  PetscCall(DMDestroy(&dmAux));
  PetscFunctionReturn(0);
}

static PetscErrorCode SetupDiscretization(DM dm, AppCtx *user)
{
  DM             plex, cdm = dm;
  PetscFE        fe, feAux = NULL;
  PetscBool      simplex;
  PetscInt       dim;
  MPI_Comm       comm;

  PetscFunctionBeginUser;
  PetscCall(DMGetDimension(dm, &dim));
  PetscCall(DMConvert(dm, DMPLEX, &plex));
  PetscCall(DMPlexIsSimplex(plex, &simplex));
  PetscCall(DMDestroy(&plex));
  PetscCall(PetscObjectGetComm((PetscObject) dm, &comm));
  PetscCall(PetscFECreateDefault(PETSC_COMM_SELF, dim, 1, simplex, NULL, -1, &fe));
  PetscCall(PetscObjectSetName((PetscObject) fe, "potential"));
  if (user->variableCoefficient == COEFF_FIELD || user->variableCoefficient == COEFF_CHECKERBOARD_0) {
    PetscCall(PetscFECreateDefault(PETSC_COMM_SELF, dim, 1, simplex, "mat_", -1, &feAux));
    PetscCall(PetscObjectSetName((PetscObject) feAux, "coefficient"));
    PetscCall(PetscFECopyQuadrature(fe, feAux));
  } else if (user->fieldBC) {
    PetscCall(PetscFECreateDefault(PETSC_COMM_SELF, dim, 1, simplex, "bc_", -1, &feAux));
    PetscCall(PetscFECopyQuadrature(fe, feAux));
  }
  /* Set discretization and boundary conditions for each mesh */
  PetscCall(DMSetField(dm, 0, NULL, (PetscObject) fe));
  PetscCall(DMCreateDS(dm));
  PetscCall(SetupProblem(dm, user));
  while (cdm) {
    PetscCall(SetupAuxDM(cdm, feAux, user));
    if (user->bcType == DIRICHLET) {
      PetscBool hasLabel;

      PetscCall(DMHasLabel(cdm, "marker", &hasLabel));
      if (!hasLabel) PetscCall(CreateBCLabel(cdm, "marker"));
    }
    PetscCall(DMCopyDisc(dm, cdm));
    PetscCall(DMGetCoarseDM(cdm, &cdm));
  }
  PetscCall(PetscFEDestroy(&fe));
  PetscCall(PetscFEDestroy(&feAux));
  PetscFunctionReturn(0);
}

int main(int argc, char **argv)
{
  DM             dm;          /* Problem specification */
  SNES           snes;        /* nonlinear solver */
  Vec            u;           /* solution vector */
  Mat            A,J;         /* Jacobian matrix */
  MatNullSpace   nullSpace;   /* May be necessary for Neumann conditions */
  AppCtx         user;        /* user-defined work context */
  JacActionCtx   userJ;       /* context for Jacobian MF action */
  PetscReal      error = 0.0; /* L_2 error in the solution */

  PetscCall(PetscInitialize(&argc, &argv, NULL,help));
  PetscCall(ProcessOptions(PETSC_COMM_WORLD, &user));
  PetscCall(SNESCreate(PETSC_COMM_WORLD, &snes));
  PetscCall(CreateMesh(PETSC_COMM_WORLD, &user, &dm));
  PetscCall(SNESSetDM(snes, dm));
  PetscCall(DMSetApplicationContext(dm, &user));

  PetscCall(PetscMalloc2(1, &user.exactFuncs, 1, &user.exactFields));
  PetscCall(SetupDiscretization(dm, &user));

  PetscCall(DMCreateGlobalVector(dm, &u));
  PetscCall(PetscObjectSetName((PetscObject) u, "potential"));

  PetscCall(DMCreateMatrix(dm, &J));
  if (user.jacobianMF) {
    PetscInt M, m, N, n;

    PetscCall(MatGetSize(J, &M, &N));
    PetscCall(MatGetLocalSize(J, &m, &n));
    PetscCall(MatCreate(PETSC_COMM_WORLD, &A));
    PetscCall(MatSetSizes(A, m, n, M, N));
    PetscCall(MatSetType(A, MATSHELL));
    PetscCall(MatSetUp(A));
#if 0
    PetscCall(MatShellSetOperation(A, MATOP_MULT, (void (*)(void))FormJacobianAction));
#endif

    userJ.dm   = dm;
    userJ.J    = J;
    userJ.user = &user;

    PetscCall(DMCreateLocalVector(dm, &userJ.u));
    if (user.fieldBC) PetscCall(DMProjectFieldLocal(dm, 0.0, userJ.u, user.exactFields, INSERT_BC_VALUES, userJ.u));
    else              PetscCall(DMProjectFunctionLocal(dm, 0.0, user.exactFuncs, NULL, INSERT_BC_VALUES, userJ.u));
    PetscCall(MatShellSetContext(A, &userJ));
  } else {
    A = J;
  }

  nullSpace = NULL;
  if (user.bcType != DIRICHLET) {
    PetscCall(MatNullSpaceCreate(PetscObjectComm((PetscObject) dm), PETSC_TRUE, 0, NULL, &nullSpace));
    PetscCall(MatSetNullSpace(A, nullSpace));
  }

  PetscCall(DMPlexSetSNESLocalFEM(dm,&user,&user,&user));
  PetscCall(SNESSetJacobian(snes, A, J, NULL, NULL));

  PetscCall(SNESSetFromOptions(snes));

  if (user.fieldBC) PetscCall(DMProjectField(dm, 0.0, u, user.exactFields, INSERT_ALL_VALUES, u));
  else              PetscCall(DMProjectFunction(dm, 0.0, user.exactFuncs, NULL, INSERT_ALL_VALUES, u));
  if (user.restart) {
#if defined(PETSC_HAVE_HDF5)
    PetscViewer viewer;
    char        filename[PETSC_MAX_PATH_LEN];

    PetscCall(PetscOptionsGetString(NULL, NULL, "-dm_plex_filename", filename, sizeof(filename), NULL));
    PetscCall(PetscViewerCreate(PETSC_COMM_WORLD, &viewer));
    PetscCall(PetscViewerSetType(viewer, PETSCVIEWERHDF5));
    PetscCall(PetscViewerFileSetMode(viewer, FILE_MODE_READ));
    PetscCall(PetscViewerFileSetName(viewer, filename));
    PetscCall(PetscViewerHDF5PushGroup(viewer, "/fields"));
    PetscCall(VecLoad(u, viewer));
    PetscCall(PetscViewerHDF5PopGroup(viewer));
    PetscCall(PetscViewerDestroy(&viewer));
#endif
  }
  if (user.showInitial) {
    Vec lv;
    PetscCall(DMGetLocalVector(dm, &lv));
    PetscCall(DMGlobalToLocalBegin(dm, u, INSERT_VALUES, lv));
    PetscCall(DMGlobalToLocalEnd(dm, u, INSERT_VALUES, lv));
    PetscCall(DMPrintLocalVec(dm, "Local function", 1.0e-10, lv));
    PetscCall(DMRestoreLocalVector(dm, &lv));
  }
  if (user.runType == RUN_FULL || user.runType == RUN_EXACT) {
    PetscErrorCode (*initialGuess[1])(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nc, PetscScalar u[], void *ctx) = {zero};

    if (user.nonzInit) initialGuess[0] = ecks;
    if (user.runType == RUN_FULL) {
      PetscCall(DMProjectFunction(dm, 0.0, initialGuess, NULL, INSERT_VALUES, u));
    }
    PetscCall(VecViewFromOptions(u, NULL, "-guess_vec_view"));
    PetscCall(SNESSolve(snes, NULL, u));
    PetscCall(SNESGetSolution(snes, &u));
    PetscCall(SNESGetDM(snes, &dm));

    if (user.showSolution) {
      PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Solution\n"));
      PetscCall(VecChop(u, 3.0e-9));
      PetscCall(VecView(u, PETSC_VIEWER_STDOUT_WORLD));
    }
  } else if (user.runType == RUN_PERF) {
    Vec       r;
    PetscReal res = 0.0;

    PetscCall(SNESGetFunction(snes, &r, NULL, NULL));
    PetscCall(SNESComputeFunction(snes, u, r));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Initial Residual\n"));
    PetscCall(VecChop(r, 1.0e-10));
    PetscCall(VecNorm(r, NORM_2, &res));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "L_2 Residual: %g\n", (double)res));
  } else {
    Vec       r;
    PetscReal res = 0.0, tol = 1.0e-11;

    /* Check discretization error */
    PetscCall(SNESGetFunction(snes, &r, NULL, NULL));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Initial guess\n"));
    if (!user.quiet) PetscCall(VecView(u, PETSC_VIEWER_STDOUT_WORLD));
    PetscCall(DMComputeL2Diff(dm, 0.0, user.exactFuncs, NULL, u, &error));
    if (error < tol) PetscCall(PetscPrintf(PETSC_COMM_WORLD, "L_2 Error: < %2.1e\n", (double)tol));
    else             PetscCall(PetscPrintf(PETSC_COMM_WORLD, "L_2 Error: %g\n", (double)error));
    /* Check residual */
    PetscCall(SNESComputeFunction(snes, u, r));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Initial Residual\n"));
    PetscCall(VecChop(r, 1.0e-10));
    if (!user.quiet) PetscCall(VecView(r, PETSC_VIEWER_STDOUT_WORLD));
    PetscCall(VecNorm(r, NORM_2, &res));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "L_2 Residual: %g\n", (double)res));
    /* Check Jacobian */
    {
      Vec b;

      PetscCall(SNESComputeJacobian(snes, u, A, A));
      PetscCall(VecDuplicate(u, &b));
      PetscCall(VecSet(r, 0.0));
      PetscCall(SNESComputeFunction(snes, r, b));
      PetscCall(MatMult(A, u, r));
      PetscCall(VecAXPY(r, 1.0, b));
      PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Au - b = Au + F(0)\n"));
      PetscCall(VecChop(r, 1.0e-10));
      if (!user.quiet) PetscCall(VecView(r, PETSC_VIEWER_STDOUT_WORLD));
      PetscCall(VecNorm(r, NORM_2, &res));
      PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Linear L_2 Residual: %g\n", (double)res));
      /* check solver */
      if (user.checkksp) {
        KSP ksp;

        if (nullSpace) {
          PetscCall(MatNullSpaceRemove(nullSpace, u));
        }
        PetscCall(SNESComputeJacobian(snes, u, A, J));
        PetscCall(MatMult(A, u, b));
        PetscCall(SNESGetKSP(snes, &ksp));
        PetscCall(KSPSetOperators(ksp, A, J));
        PetscCall(KSPSolve(ksp, b, r));
        PetscCall(VecAXPY(r, -1.0, u));
        PetscCall(VecNorm(r, NORM_2, &res));
        PetscCall(PetscPrintf(PETSC_COMM_WORLD, "KSP Error: %g\n", (double)res));
      }
      PetscCall(VecDestroy(&b));
    }
  }
  PetscCall(VecViewFromOptions(u, NULL, "-vec_view"));
  {
    Vec nu;

    PetscCall(DMGetAuxiliaryVec(dm, NULL, 0, 0, &nu));
    if (nu) PetscCall(VecViewFromOptions(nu, NULL, "-coeff_view"));
  }

  if (user.bdIntegral) {
    DMLabel   label;
    PetscInt  id = 1;
    PetscScalar bdInt = 0.0;
    PetscReal   exact = 3.3333333333;

    PetscCall(DMGetLabel(dm, "marker", &label));
    PetscCall(DMPlexComputeBdIntegral(dm, u, label, 1, &id, bd_integral_2d, &bdInt, NULL));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Solution boundary integral: %.4g\n", (double) PetscAbsScalar(bdInt)));
    PetscCheckFalse(PetscAbsReal(PetscAbsScalar(bdInt) - exact) > PETSC_SQRT_MACHINE_EPSILON,PETSC_COMM_WORLD, PETSC_ERR_PLIB, "Invalid boundary integral %g != %g", (double) PetscAbsScalar(bdInt), (double)exact);
  }

  PetscCall(MatNullSpaceDestroy(&nullSpace));
  if (user.jacobianMF) PetscCall(VecDestroy(&userJ.u));
  if (A != J) PetscCall(MatDestroy(&A));
  PetscCall(MatDestroy(&J));
  PetscCall(VecDestroy(&u));
  PetscCall(SNESDestroy(&snes));
  PetscCall(DMDestroy(&dm));
  PetscCall(PetscFree2(user.exactFuncs, user.exactFields));
  PetscCall(PetscFree(user.kgrid));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST
  # 2D serial P1 test 0-4
  test:
    suffix: 2d_p1_0
    requires: triangle
    args: -run_type test -bc_type dirichlet -dm_plex_interpolate 0 -petscspace_degree 1 -show_initial -dm_plex_print_fem 1

  test:
    suffix: 2d_p1_1
    requires: triangle
    args: -run_type test -bc_type dirichlet -petscspace_degree 1 -show_initial -dm_plex_print_fem 1

  test:
    suffix: 2d_p1_2
    requires: triangle
    args: -run_type test -dm_refine_volume_limit_pre 0.0625 -bc_type dirichlet -petscspace_degree 1 -show_initial -dm_plex_print_fem 1

  test:
    suffix: 2d_p1_neumann_0
    requires: triangle
    args: -dm_coord_space 0 -run_type test -bc_type neumann -dm_plex_boundary_label boundary -petscspace_degree 1 -show_initial -dm_plex_print_fem 1 -dm_view ascii::ascii_info_detail

  test:
    suffix: 2d_p1_neumann_1
    requires: triangle
    args: -run_type test -dm_refine_volume_limit_pre 0.0625 -bc_type neumann -dm_plex_boundary_label boundary -petscspace_degree 1 -show_initial -dm_plex_print_fem 1

  # 2D serial P2 test 5-8
  test:
    suffix: 2d_p2_0
    requires: triangle
    args: -run_type test -bc_type dirichlet -petscspace_degree 2 -show_initial -dm_plex_print_fem 1

  test:
    suffix: 2d_p2_1
    requires: triangle
    args: -run_type test -dm_refine_volume_limit_pre 0.0625 -bc_type dirichlet -petscspace_degree 2 -show_initial -dm_plex_print_fem 1

  test:
    suffix: 2d_p2_neumann_0
    requires: triangle
    args: -dm_coord_space 0 -run_type test -bc_type neumann -dm_plex_boundary_label boundary -petscspace_degree 2 -show_initial -dm_plex_print_fem 1 -dm_view ascii::ascii_info_detail

  test:
    suffix: 2d_p2_neumann_1
    requires: triangle
    args: -dm_coord_space 0 -run_type test -dm_refine_volume_limit_pre 0.0625 -bc_type neumann -dm_plex_boundary_label boundary -petscspace_degree 2 -show_initial -dm_plex_print_fem 1 -dm_view ascii::ascii_info_detail

  test:
    suffix: bd_int_0
    requires: triangle
    args: -run_type test -bc_type dirichlet -petscspace_degree 2 -bd_integral -dm_view -quiet

  test:
    suffix: bd_int_1
    requires: triangle
    args: -run_type test -dm_refine 2 -bc_type dirichlet -petscspace_degree 2 -bd_integral -dm_view -quiet

  # 3D serial P1 test 9-12
  test:
    suffix: 3d_p1_0
    requires: ctetgen
    args: -run_type test -dm_plex_dim 3 -bc_type dirichlet -dm_plex_interpolate 0 -petscspace_degree 1 -show_initial -dm_plex_print_fem 1 -dm_view

  test:
    suffix: 3d_p1_1
    requires: ctetgen
    args: -run_type test -dm_plex_dim 3 -bc_type dirichlet -petscspace_degree 1 -show_initial -dm_plex_print_fem 1 -dm_view

  test:
    suffix: 3d_p1_2
    requires: ctetgen
    args: -run_type test -dm_plex_dim 3 -dm_refine_volume_limit_pre 0.0125 -bc_type dirichlet -petscspace_degree 1 -show_initial -dm_plex_print_fem 1 -dm_view

  test:
    suffix: 3d_p1_neumann_0
    requires: ctetgen
    args: -run_type test -dm_plex_dim 3 -bc_type neumann -dm_plex_boundary_label boundary -petscspace_degree 1 -snes_fd -show_initial -dm_plex_print_fem 1 -dm_view

  # Analytic variable coefficient 13-20
  test:
    suffix: 13
    requires: triangle
    args: -run_type test -variable_coefficient analytic -petscspace_degree 1 -show_initial -dm_plex_print_fem 1
  test:
    suffix: 14
    requires: triangle
    args: -run_type test -dm_refine_volume_limit_pre 0.0625 -variable_coefficient analytic -petscspace_degree 1 -show_initial -dm_plex_print_fem 1
  test:
    suffix: 15
    requires: triangle
    args: -run_type test -variable_coefficient analytic -petscspace_degree 2 -show_initial -dm_plex_print_fem 1
  test:
    suffix: 16
    requires: triangle
    args: -run_type test -dm_refine_volume_limit_pre 0.0625 -variable_coefficient analytic -petscspace_degree 2 -show_initial -dm_plex_print_fem 1
  test:
    suffix: 17
    requires: ctetgen
    args: -run_type test -dm_plex_dim 3 -variable_coefficient analytic -petscspace_degree 1 -show_initial -dm_plex_print_fem 1

  test:
    suffix: 18
    requires: ctetgen
    args: -run_type test -dm_plex_dim 3 -dm_refine_volume_limit_pre 0.0125 -variable_coefficient analytic -petscspace_degree 1 -show_initial -dm_plex_print_fem 1

  test:
    suffix: 19
    requires: ctetgen
    args: -run_type test -dm_plex_dim 3 -variable_coefficient analytic -petscspace_degree 2 -show_initial -dm_plex_print_fem 1

  test:
    suffix: 20
    requires: ctetgen
    args: -run_type test -dm_plex_dim 3 -dm_refine_volume_limit_pre 0.0125 -variable_coefficient analytic -petscspace_degree 2 -show_initial -dm_plex_print_fem 1

  # P1 variable coefficient 21-28
  test:
    suffix: 21
    requires: triangle
    args: -run_type test -variable_coefficient field -petscspace_degree 1 -mat_petscspace_degree 1 -show_initial -dm_plex_print_fem 1

  test:
    suffix: 22
    requires: triangle
    args: -run_type test -dm_refine_volume_limit_pre 0.0625 -variable_coefficient field -petscspace_degree 1 -mat_petscspace_degree 1 -show_initial -dm_plex_print_fem 1

  test:
    suffix: 23
    requires: triangle
    args: -run_type test -variable_coefficient field -petscspace_degree 2 -mat_petscspace_degree 1 -show_initial -dm_plex_print_fem 1

  test:
    suffix: 24
    requires: triangle
    args: -run_type test -dm_refine_volume_limit_pre 0.0625 -variable_coefficient field -petscspace_degree 2 -mat_petscspace_degree 1 -show_initial -dm_plex_print_fem 1

  test:
    suffix: 25
    requires: ctetgen
    args: -run_type test -dm_plex_dim 3 -variable_coefficient field -petscspace_degree 1 -mat_petscspace_degree 1 -show_initial -dm_plex_print_fem 1

  test:
    suffix: 26
    requires: ctetgen
    args: -run_type test -dm_plex_dim 3 -dm_refine_volume_limit_pre 0.0125 -variable_coefficient field -petscspace_degree 1 -mat_petscspace_degree 1 -show_initial -dm_plex_print_fem 1

  test:
    suffix: 27
    requires: ctetgen
    args: -run_type test -dm_plex_dim 3 -variable_coefficient field -petscspace_degree 2 -mat_petscspace_degree 1 -show_initial -dm_plex_print_fem 1

  test:
    suffix: 28
    requires: ctetgen
    args: -run_type test -dm_plex_dim 3 -dm_refine_volume_limit_pre 0.0125 -variable_coefficient field -petscspace_degree 2 -mat_petscspace_degree 1 -show_initial -dm_plex_print_fem 1

  # P0 variable coefficient 29-36
  test:
    suffix: 29
    requires: triangle
    args: -run_type test -variable_coefficient field -petscspace_degree 1 -show_initial -dm_plex_print_fem 1

  test:
    suffix: 30
    requires: triangle
    args: -run_type test -dm_refine_volume_limit_pre 0.0625 -variable_coefficient field -petscspace_degree 1 -show_initial -dm_plex_print_fem 1

  test:
    suffix: 31
    requires: triangle
    args: -run_type test -variable_coefficient field -petscspace_degree 2 -show_initial -dm_plex_print_fem 1

  test:
    requires: triangle
    suffix: 32
    args: -run_type test -dm_refine_volume_limit_pre 0.0625 -variable_coefficient field -petscspace_degree 2 -show_initial -dm_plex_print_fem 1

  test:
    requires: ctetgen
    suffix: 33
    args: -run_type test -dm_plex_dim 3 -variable_coefficient field -petscspace_degree 1 -show_initial -dm_plex_print_fem 1

  test:
    suffix: 34
    requires: ctetgen
    args: -run_type test -dm_plex_dim 3 -dm_refine_volume_limit_pre 0.0125 -variable_coefficient field -petscspace_degree 1 -show_initial -dm_plex_print_fem 1

  test:
    suffix: 35
    requires: ctetgen
    args: -run_type test -dm_plex_dim 3 -variable_coefficient field -petscspace_degree 2 -show_initial -dm_plex_print_fem 1

  test:
    suffix: 36
    requires: ctetgen
    args: -run_type test -dm_plex_dim 3 -dm_refine_volume_limit_pre 0.0125 -variable_coefficient field -petscspace_degree 2 -show_initial -dm_plex_print_fem 1

  # Full solve 39-44
  test:
    suffix: 39
    requires: triangle !single
    args: -run_type full -dm_refine_volume_limit_pre 0.015625 -petscspace_degree 2 -pc_type gamg -pc_gamg_esteig_ksp_type cg -pc_gamg_esteig_ksp_max_it 10 -ksp_rtol 1.0e-10 -ksp_monitor_short -ksp_converged_reason -snes_monitor_short -snes_converged_reason ::ascii_info_detail
  test:
    suffix: 40
    requires: triangle !single
    args: -run_type full -dm_refine_volume_limit_pre 0.015625 -variable_coefficient nonlinear -petscspace_degree 2 -pc_type svd -ksp_rtol 1.0e-10 -snes_monitor_short -snes_converged_reason ::ascii_info_detail
  test:
    suffix: 41
    requires: triangle !single
    args: -run_type full -dm_refine_volume_limit_pre 0.03125 -variable_coefficient nonlinear -petscspace_degree 1 -snes_type fas -snes_fas_levels 2 -fas_coarse_pc_type svd -fas_coarse_ksp_rtol 1.0e-10 -fas_coarse_snes_monitor_short -snes_monitor_short -fas_coarse_snes_linesearch_type basic -snes_converged_reason ::ascii_info_detail -dm_refine_hierarchy 1 -snes_view -fas_levels_1_snes_type newtonls -fas_levels_1_pc_type svd -fas_levels_1_ksp_rtol 1.0e-10 -fas_levels_1_snes_monitor_short
  test:
    suffix: 42
    requires: triangle !single
    args: -run_type full -dm_refine_volume_limit_pre 0.0625 -variable_coefficient nonlinear -petscspace_degree 1 -snes_type fas -snes_fas_levels 3 -fas_coarse_pc_type svd -fas_coarse_ksp_rtol 1.0e-10 -fas_coarse_snes_monitor_short -snes_monitor_short -fas_coarse_snes_linesearch_type basic -snes_converged_reason ::ascii_info_detail -dm_refine_hierarchy 2 -snes_view -fas_levels_1_snes_type newtonls -fas_levels_1_pc_type svd -fas_levels_1_ksp_rtol 1.0e-10 -fas_levels_1_snes_monitor_short -fas_levels_2_snes_type newtonls -fas_levels_2_pc_type svd -fas_levels_2_ksp_rtol 1.0e-10 -fas_levels_2_snes_atol 1.0e-11 -fas_levels_2_snes_monitor_short
  test:
    suffix: 43
    requires: triangle !single
    nsize: 2
    args: -run_type full -dm_refine_volume_limit_pre 0.03125 -variable_coefficient nonlinear -petscspace_degree 1 -snes_type fas -snes_fas_levels 2 -fas_coarse_pc_type svd -fas_coarse_ksp_rtol 1.0e-10 -fas_coarse_snes_monitor_short -snes_monitor_short -fas_coarse_snes_linesearch_type basic -snes_converged_reason ::ascii_info_detail -dm_refine_hierarchy 1 -snes_view -fas_levels_1_snes_type newtonls -fas_levels_1_pc_type svd -fas_levels_1_ksp_rtol 1.0e-10 -fas_levels_1_snes_monitor_short

  test:
    suffix: 44
    requires: triangle !single
    nsize: 2
    args: -run_type full -dm_refine_volume_limit_pre 0.0625 -variable_coefficient nonlinear -petscspace_degree 1 -snes_type fas -snes_fas_levels 3 -fas_coarse_pc_type svd -fas_coarse_ksp_rtol 1.0e-10 -fas_coarse_snes_monitor_short -snes_monitor_short  -fas_coarse_snes_linesearch_type basic -snes_converged_reason ::ascii_info_detail -dm_refine_hierarchy 2 -dm_plex_print_fem 0 -snes_view -fas_levels_1_snes_type newtonls -fas_levels_1_pc_type svd -fas_levels_1_ksp_rtol 1.0e-10 -fas_levels_1_snes_monitor_short -fas_levels_2_snes_type newtonls -fas_levels_2_pc_type svd -fas_levels_2_ksp_rtol 1.0e-10 -fas_levels_2_snes_atol 1.0e-11 -fas_levels_2_snes_monitor_short

  # These tests use a loose tolerance just to exercise the PtAP operations for MATIS and multiple PCBDDC setup calls inside PCMG
  testset:
    requires: triangle !single
    nsize: 3
    args: -run_type full -petscspace_degree 1 -dm_mat_type is -pc_type mg -pc_mg_levels 2 -mg_coarse_pc_type bddc -pc_mg_galerkin pmat -ksp_rtol 1.0e-2 -snes_converged_reason -dm_refine_hierarchy 2 -snes_max_it 4
    test:
      suffix: gmg_bddc
      filter: sed -e "s/CONVERGED_FNORM_RELATIVE iterations 3/CONVERGED_FNORM_RELATIVE iterations 4/g"
      args: -mg_levels_pc_type jacobi
    test:
      filter: sed -e "s/iterations [0-4]/iterations 4/g"
      suffix: gmg_bddc_lev
      args: -mg_levels_pc_type bddc

  # Restarting
  testset:
    suffix: restart
    requires: hdf5 triangle !complex
    args: -run_type test -bc_type dirichlet -petscspace_degree 1
    test:
      args: -dm_view hdf5:sol.h5 -vec_view hdf5:sol.h5::append
    test:
      args: -dm_plex_filename sol.h5 -dm_plex_name box -restart

  # Periodicity
  test:
    suffix: periodic_0
    requires: triangle
    args: -run_type full -bc_type dirichlet -petscspace_degree 1 -snes_converged_reason ::ascii_info_detail

  test:
    requires: !complex
    suffix: periodic_1
    args: -quiet -run_type test -dm_plex_simplex 0 -dm_plex_box_faces 3,3 -dm_plex_box_bd periodic,periodic -vec_view vtk:test.vtu:vtk_vtu -petscspace_degree 1 -dm_refine 1

  # 2D serial P1 test with field bc
  test:
    suffix: field_bc_2d_p1_0
    requires: triangle
    args: -run_type test -bc_type dirichlet -field_bc -petscspace_degree 1 -bc_petscspace_degree 2 -show_initial -dm_plex_print_fem 1

  test:
    suffix: field_bc_2d_p1_1
    requires: triangle
    args: -run_type test -dm_refine 1 -bc_type dirichlet -field_bc -petscspace_degree 1 -bc_petscspace_degree 2 -show_initial -dm_plex_print_fem 1

  test:
    suffix: field_bc_2d_p1_neumann_0
    requires: triangle
    args: -run_type test -bc_type neumann -dm_plex_boundary_label boundary -field_bc -petscspace_degree 1 -bc_petscspace_degree 2 -show_initial -dm_plex_print_fem 1

  test:
    suffix: field_bc_2d_p1_neumann_1
    requires: triangle
    args: -run_type test -dm_refine 1 -bc_type neumann -dm_plex_boundary_label boundary -field_bc -petscspace_degree 1 -bc_petscspace_degree 2 -show_initial -dm_plex_print_fem 1

  # 3D serial P1 test with field bc
  test:
    suffix: field_bc_3d_p1_0
    requires: ctetgen
    args: -run_type test -dm_plex_dim 3 -bc_type dirichlet -field_bc -petscspace_degree 1 -bc_petscspace_degree 2 -show_initial -dm_plex_print_fem 1

  test:
    suffix: field_bc_3d_p1_1
    requires: ctetgen
    args: -run_type test -dm_plex_dim 3 -dm_refine 1 -bc_type dirichlet -field_bc -petscspace_degree 1 -bc_petscspace_degree 2 -show_initial -dm_plex_print_fem 1

  test:
    suffix: field_bc_3d_p1_neumann_0
    requires: ctetgen
    args: -run_type test -dm_plex_dim 3 -bc_type neumann -dm_plex_boundary_label boundary -field_bc -petscspace_degree 1 -bc_petscspace_degree 2 -show_initial -dm_plex_print_fem 1

  test:
    suffix: field_bc_3d_p1_neumann_1
    requires: ctetgen
    args: -run_type test -dm_plex_dim 3 -dm_refine 1 -bc_type neumann -dm_plex_boundary_label boundary -field_bc -petscspace_degree 1 -bc_petscspace_degree 2 -show_initial -dm_plex_print_fem 1

  # 2D serial P2 test with field bc
  test:
    suffix: field_bc_2d_p2_0
    requires: triangle
    args: -run_type test -bc_type dirichlet -field_bc -petscspace_degree 2 -bc_petscspace_degree 2 -show_initial -dm_plex_print_fem 1

  test:
    suffix: field_bc_2d_p2_1
    requires: triangle
    args: -run_type test -dm_refine 1 -bc_type dirichlet -field_bc -petscspace_degree 2 -bc_petscspace_degree 2 -show_initial -dm_plex_print_fem 1

  test:
    suffix: field_bc_2d_p2_neumann_0
    requires: triangle
    args: -run_type test -bc_type neumann -dm_plex_boundary_label boundary -field_bc -petscspace_degree 2 -bc_petscspace_degree 2 -show_initial -dm_plex_print_fem 1

  test:
    suffix: field_bc_2d_p2_neumann_1
    requires: triangle
    args: -run_type test -dm_refine 1 -bc_type neumann -dm_plex_boundary_label boundary -field_bc -petscspace_degree 2 -bc_petscspace_degree 2 -show_initial -dm_plex_print_fem 1

  # 3D serial P2 test with field bc
  test:
    suffix: field_bc_3d_p2_0
    requires: ctetgen
    args: -run_type test -dm_plex_dim 3 -bc_type dirichlet -field_bc -petscspace_degree 2 -bc_petscspace_degree 2 -show_initial -dm_plex_print_fem 1

  test:
    suffix: field_bc_3d_p2_1
    requires: ctetgen
    args: -run_type test -dm_plex_dim 3 -dm_refine 1 -bc_type dirichlet -field_bc -petscspace_degree 2 -bc_petscspace_degree 2 -show_initial -dm_plex_print_fem 1

  test:
    suffix: field_bc_3d_p2_neumann_0
    requires: ctetgen
    args: -run_type test -dm_plex_dim 3 -bc_type neumann -dm_plex_boundary_label boundary -field_bc -petscspace_degree 2 -bc_petscspace_degree 2 -show_initial -dm_plex_print_fem 1

  test:
    suffix: field_bc_3d_p2_neumann_1
    requires: ctetgen
    args: -run_type test -dm_plex_dim 3 -dm_refine 1 -bc_type neumann -dm_plex_boundary_label boundary -field_bc -petscspace_degree 2 -bc_petscspace_degree 2 -show_initial -dm_plex_print_fem 1

  # Full solve simplex: Convergence
  test:
    suffix: 3d_p1_conv
    requires: ctetgen
    args: -run_type full -dm_plex_dim 3 -dm_refine 1 -bc_type dirichlet -petscspace_degree 1 \
      -snes_convergence_estimate -convest_num_refine 1 -pc_type lu

  # Full solve simplex: PCBDDC
  test:
    suffix: tri_bddc
    requires: triangle !single
    nsize: 5
    args: -run_type full -petscpartitioner_type simple -dm_refine 2 -bc_type dirichlet -petscspace_degree 1 -ksp_type gmres -ksp_gmres_restart 100 -ksp_rtol 1.0e-9 -dm_mat_type is -pc_type bddc -snes_monitor_short -ksp_monitor_short -snes_converged_reason ::ascii_info_detail -ksp_converged_reason -snes_view -show_solution 0

  # Full solve simplex: PCBDDC
  test:
    suffix: tri_parmetis_bddc
    requires: triangle !single parmetis
    nsize: 4
    args: -run_type full -petscpartitioner_type parmetis -dm_refine 2 -bc_type dirichlet -petscspace_degree 1 -ksp_type gmres -ksp_gmres_restart 100 -ksp_rtol 1.0e-9 -dm_mat_type is -pc_type bddc -snes_monitor_short -ksp_monitor_short -snes_converged_reason ::ascii_info_detail -ksp_converged_reason -snes_view -show_solution 0

  testset:
    args: -run_type full -dm_plex_simplex 0 -dm_plex_box_faces 3,3 -petscpartitioner_type simple -dm_refine 2 -bc_type dirichlet -petscspace_degree 2 -dm_mat_type is -pc_type bddc -ksp_type gmres -snes_monitor_short -ksp_monitor_short -snes_view -petscspace_poly_tensor -pc_bddc_corner_selection -ksp_rtol 1.e-9 -pc_bddc_use_edges 0
    nsize: 5
    output_file: output/ex12_quad_bddc.out
    filter: sed -e "s/aijcusparse/aij/g" -e "s/aijviennacl/aij/g" -e "s/factorization: cusparse/factorization: petsc/g"
    test:
      requires: !single
      suffix: quad_bddc
    test:
      requires: !single cuda
      suffix: quad_bddc_cuda
      args: -matis_localmat_type aijcusparse -pc_bddc_dirichlet_pc_factor_mat_solver_type cusparse -pc_bddc_neumann_pc_factor_mat_solver_type cusparse
    test:
      requires: !single viennacl
      suffix: quad_bddc_viennacl
      args: -matis_localmat_type aijviennacl

  # Full solve simplex: ASM
  test:
    suffix: tri_q2q1_asm_lu
    requires: triangle !single
    args: -run_type full -dm_refine 3 -bc_type dirichlet -petscspace_degree 1 -ksp_type gmres -ksp_gmres_restart 100 -ksp_rtol 1.0e-9 -pc_type asm -pc_asm_type restrict -pc_asm_blocks 4 -sub_pc_type lu -snes_monitor_short -ksp_monitor_short -snes_converged_reason ::ascii_info_detail -ksp_converged_reason -snes_view -show_solution 0

  test:
    suffix: tri_q2q1_msm_lu
    requires: triangle !single
    args: -run_type full -dm_refine 3 -bc_type dirichlet -petscspace_degree 1 -ksp_type gmres -ksp_gmres_restart 100 -ksp_rtol 1.0e-9 -pc_type asm -pc_asm_type restrict -pc_asm_local_type multiplicative -pc_asm_blocks 4 -sub_pc_type lu -snes_monitor_short -ksp_monitor_short -snes_converged_reason ::ascii_info_detail -ksp_converged_reason -snes_view -show_solution 0

  test:
    suffix: tri_q2q1_asm_sor
    requires: triangle !single
    args: -run_type full -dm_refine 3 -bc_type dirichlet -petscspace_degree 1 -ksp_type gmres -ksp_gmres_restart 100 -ksp_rtol 1.0e-9 -pc_type asm -pc_asm_type restrict -pc_asm_blocks 4 -sub_pc_type sor -snes_monitor_short -ksp_monitor_short -snes_converged_reason ::ascii_info_detail -ksp_converged_reason -snes_view -show_solution 0

  test:
    suffix: tri_q2q1_msm_sor
    requires: triangle !single
    args: -run_type full -dm_refine 3 -bc_type dirichlet -petscspace_degree 1 -ksp_type gmres -ksp_gmres_restart 100 -ksp_rtol 1.0e-9 -pc_type asm -pc_asm_type restrict -pc_asm_local_type multiplicative -pc_asm_blocks 4 -sub_pc_type sor -snes_monitor_short -ksp_monitor_short -snes_converged_reason ::ascii_info_detail -ksp_converged_reason -snes_view -show_solution 0

  # Full solve simplex: FAS
  test:
    suffix: fas_newton_0
    requires: triangle !single
    args: -run_type full -variable_coefficient nonlinear -petscspace_degree 1 -snes_type fas -snes_fas_levels 2 -fas_coarse_pc_type svd -fas_coarse_ksp_rtol 1.0e-10 -fas_coarse_snes_monitor_short -snes_monitor_short -fas_coarse_snes_linesearch_type basic -snes_converged_reason ::ascii_info_detail -dm_refine_hierarchy 1 -snes_view -fas_levels_1_snes_type newtonls -fas_levels_1_pc_type svd -fas_levels_1_ksp_rtol 1.0e-10 -fas_levels_1_snes_monitor_short

  test:
    suffix: fas_newton_1
    requires: triangle !single
    args: -run_type full -dm_refine_hierarchy 3 -petscspace_degree 1 -snes_type fas -snes_fas_levels 3 -fas_coarse_pc_type lu -fas_coarse_snes_monitor_short -snes_monitor_short -fas_coarse_snes_linesearch_type basic -snes_converged_reason ::ascii_info_detail -snes_view -fas_levels_snes_type newtonls -fas_levels_snes_linesearch_type basic -fas_levels_ksp_rtol 1.0e-10 -fas_levels_snes_monitor_short
    filter: sed -e "s/total number of linear solver iterations=14/total number of linear solver iterations=15/g"

  test:
    suffix: fas_ngs_0
    requires: triangle !single
    args: -run_type full -variable_coefficient nonlinear -petscspace_degree 1 -snes_type fas -snes_fas_levels 2 -fas_coarse_pc_type svd -fas_coarse_ksp_rtol 1.0e-10 -fas_coarse_snes_monitor_short -snes_monitor_short -fas_coarse_snes_linesearch_type basic -snes_converged_reason ::ascii_info_detail -dm_refine_hierarchy 1 -snes_view -fas_levels_1_snes_type ngs -fas_levels_1_snes_monitor_short

  # These two tests are broken because DMPlexComputeInjectorFEM() only works for regularly refined meshes
  test:
    suffix: fas_newton_coarse_0
    requires: pragmatic triangle
    TODO: broken
    args: -run_type full -variable_coefficient nonlinear -petscspace_degree 1 \
          -dm_refine 2 -dm_coarsen_hierarchy 1 -dm_plex_hash_location -dm_adaptor pragmatic \
          -snes_type fas -snes_fas_levels 2 -snes_converged_reason ::ascii_info_detail -snes_monitor_short -snes_view \
            -fas_coarse_pc_type svd -fas_coarse_ksp_rtol 1.0e-10 -fas_coarse_snes_monitor_short -fas_coarse_snes_linesearch_type basic \
            -fas_levels_1_snes_type newtonls -fas_levels_1_pc_type svd -fas_levels_1_ksp_rtol 1.0e-10 -fas_levels_1_snes_monitor_short

  test:
    suffix: mg_newton_coarse_0
    requires: triangle pragmatic
    TODO: broken
    args: -run_type full -petscspace_degree 1 \
          -dm_refine 3 -dm_coarsen_hierarchy 3 -dm_plex_hash_location -dm_adaptor pragmatic \
          -snes_atol 1.0e-8 -snes_rtol 0.0 -snes_monitor_short -snes_converged_reason ::ascii_info_detail -snes_view \
            -ksp_type richardson -ksp_atol 1.0e-8 -ksp_rtol 0.0 -ksp_norm_type unpreconditioned -ksp_monitor_true_residual \
              -pc_type mg -pc_mg_levels 4 \
              -mg_levels_ksp_type gmres -mg_levels_pc_type ilu -mg_levels_ksp_max_it 10

  # Full solve tensor
  test:
    suffix: tensor_plex_2d
    args: -run_type test -dm_plex_simplex 0 -bc_type dirichlet -petscspace_degree 1 -dm_refine_hierarchy 2

  test:
    suffix: tensor_p4est_2d
    requires: p4est
    args: -run_type test -dm_plex_simplex 0 -bc_type dirichlet -petscspace_degree 1 -dm_forest_initial_refinement 2 -dm_forest_minimum_refinement 0 -dm_plex_convert_type p4est

  test:
    suffix: tensor_plex_3d
    args: -run_type test -dm_plex_simplex 0 -bc_type dirichlet -petscspace_degree 1 -dm_plex_dim 3 -dm_refine_hierarchy 1 -dm_plex_box_faces 2,2,2

  test:
    suffix: tensor_p4est_3d
    requires: p4est
    args: -run_type test -dm_plex_simplex 0 -bc_type dirichlet -petscspace_degree 1 -dm_forest_initial_refinement 1 -dm_forest_minimum_refinement 0 -dm_plex_dim 3 -dm_plex_convert_type p8est -dm_plex_box_faces 2,2,2

  test:
    suffix: p4est_test_q2_conformal_serial
    requires: p4est
    args: -run_type test -petscspace_degree 2 -dm_plex_simplex 0 -dm_plex_convert_type p4est -dm_forest_minimum_refinement 0 -dm_forest_initial_refinement 2

  test:
    suffix: p4est_test_q2_conformal_parallel
    requires: p4est
    nsize: 7
    args: -run_type test -petscspace_degree 2 -dm_plex_simplex 0 -dm_plex_convert_type p4est -dm_forest_minimum_refinement 0 -dm_forest_initial_refinement 2 -petscpartitioner_type simple

  test:
    suffix: p4est_test_q2_conformal_parallel_parmetis
    requires: parmetis p4est
    nsize: 4
    args: -run_type test -petscspace_degree 2 -dm_plex_simplex 0 -dm_plex_convert_type p4est -dm_forest_minimum_refinement 0 -dm_forest_initial_refinement 2 -petscpartitioner_type parmetis

  test:
    suffix: p4est_test_q2_nonconformal_serial
    requires: p4est
    filter: grep -v "CG or CGNE: variant"
    args: -run_type test -petscspace_degree 2 -dm_plex_simplex 0 -dm_plex_convert_type p4est -dm_forest_minimum_refinement 0 -dm_forest_initial_refinement 2 -dm_forest_maximum_refinement 4 -dm_p4est_refine_pattern hash

  test:
    suffix: p4est_test_q2_nonconformal_parallel
    requires: p4est
    filter: grep -v "CG or CGNE: variant"
    nsize: 7
    args: -run_type test -petscspace_degree 2 -dm_plex_simplex 0 -dm_plex_convert_type p4est -dm_forest_minimum_refinement 0 -dm_forest_initial_refinement 2 -dm_forest_maximum_refinement 4 -dm_p4est_refine_pattern hash -petscpartitioner_type simple

  test:
    suffix: p4est_test_q2_nonconformal_parallel_parmetis
    requires: parmetis p4est
    nsize: 4
    args: -run_type test -petscspace_degree 2 -dm_plex_simplex 0 -dm_plex_convert_type p4est -dm_forest_minimum_refinement 0 -dm_forest_initial_refinement 2 -dm_forest_maximum_refinement 4 -dm_p4est_refine_pattern hash -petscpartitioner_type parmetis

  test:
    suffix: p4est_exact_q2_conformal_serial
    requires: p4est !single !complex !__float128
    args: -run_type exact -petscspace_degree 2 -fas_levels_snes_atol 1.e-10 -snes_max_it 1 -snes_type fas -snes_fas_levels 3 -fas_coarse_pc_type none -fas_coarse_ksp_type preonly -fas_coarse_snes_monitor_short -snes_monitor_short -fas_coarse_snes_linesearch_type basic -snes_converged_reason ::ascii_info_detail -snes_view -fas_levels_snes_type newtonls -fas_levels_pc_type none -fas_levels_ksp_type preonly -fas_levels_snes_monitor_short -dm_plex_simplex 0 -dm_plex_convert_type p4est -dm_forest_minimum_refinement 0 -dm_forest_initial_refinement 2

  test:
    suffix: p4est_exact_q2_conformal_parallel
    requires: p4est !single !complex !__float128
    nsize: 4
    args: -run_type exact -petscspace_degree 2 -fas_levels_snes_atol 1.e-10 -snes_max_it 1 -snes_type fas -snes_fas_levels 3 -fas_coarse_pc_type none -fas_coarse_ksp_type preonly -fas_coarse_snes_monitor_short -snes_monitor_short -fas_coarse_snes_linesearch_type basic -snes_converged_reason ::ascii_info_detail -snes_view -fas_levels_snes_type newtonls -fas_levels_pc_type none -fas_levels_ksp_type preonly -fas_levels_snes_monitor_short -dm_plex_simplex 0 -dm_plex_convert_type p4est -dm_forest_minimum_refinement 0 -dm_forest_initial_refinement 2

  test:
    suffix: p4est_exact_q2_conformal_parallel_parmetis
    requires: parmetis p4est !single
    nsize: 4
    args: -run_type exact -petscspace_degree 2 -fas_levels_snes_atol 1.e-10 -snes_max_it 1 -snes_type fas -snes_fas_levels 3 -fas_coarse_pc_type none -fas_coarse_ksp_type preonly -fas_coarse_snes_monitor_short -snes_monitor_short -fas_coarse_snes_linesearch_type basic -snes_converged_reason ::ascii_info_detail -snes_view -fas_levels_snes_type newtonls -fas_levels_pc_type none -fas_levels_ksp_type preonly -fas_levels_snes_monitor_short -dm_plex_simplex 0 -dm_plex_convert_type p4est -dm_forest_minimum_refinement 0 -dm_forest_initial_refinement 2 -petscpartitioner_type parmetis

  test:
    suffix: p4est_exact_q2_nonconformal_serial
    requires: p4est
    args: -run_type exact -petscspace_degree 2 -fas_levels_snes_atol 1.e-10 -snes_max_it 1 -snes_type fas -snes_fas_levels 3 -fas_coarse_pc_type none -fas_coarse_ksp_type preonly -fas_coarse_snes_monitor_short -snes_monitor_short -fas_coarse_snes_linesearch_type basic -snes_converged_reason ::ascii_info_detail -snes_view -fas_levels_snes_type newtonls -fas_levels_pc_type none -fas_levels_ksp_type preonly -fas_levels_snes_monitor_short -dm_plex_simplex 0 -dm_plex_convert_type p4est -dm_forest_minimum_refinement 0 -dm_forest_initial_refinement 2 -dm_forest_maximum_refinement 4 -dm_p4est_refine_pattern hash

  test:
    suffix: p4est_exact_q2_nonconformal_parallel
    requires: p4est
    nsize: 7
    args: -run_type exact -petscspace_degree 2 -fas_levels_snes_atol 1.e-10 -snes_max_it 1 -snes_type fas -snes_fas_levels 3 -fas_coarse_pc_type none -fas_coarse_ksp_type preonly -fas_coarse_snes_monitor_short -snes_monitor_short -fas_coarse_snes_linesearch_type basic -snes_converged_reason ::ascii_info_detail -snes_view -fas_levels_snes_type newtonls -fas_levels_pc_type none -fas_levels_ksp_type preonly -fas_levels_snes_monitor_short -dm_plex_simplex 0 -dm_plex_convert_type p4est -dm_forest_minimum_refinement 0 -dm_forest_initial_refinement 2 -dm_forest_maximum_refinement 4 -dm_p4est_refine_pattern hash -petscpartitioner_type simple

  test:
    suffix: p4est_exact_q2_nonconformal_parallel_parmetis
    requires: parmetis p4est
    nsize: 4
    args: -run_type exact -petscspace_degree 2 -fas_levels_snes_atol 1.e-10 -snes_max_it 1 -snes_type fas -snes_fas_levels 3 -fas_coarse_pc_type none -fas_coarse_ksp_type preonly -fas_coarse_snes_monitor_short -snes_monitor_short -fas_coarse_snes_linesearch_type basic -snes_converged_reason ::ascii_info_detail -snes_view -fas_levels_snes_type newtonls -fas_levels_pc_type none -fas_levels_ksp_type preonly -fas_levels_snes_monitor_short -dm_plex_simplex 0 -dm_plex_convert_type p4est -dm_forest_minimum_refinement 0 -dm_forest_initial_refinement 2 -dm_forest_maximum_refinement 4 -dm_p4est_refine_pattern hash -petscpartitioner_type parmetis

  test:
    suffix: p4est_full_q2_nonconformal_serial
    requires: p4est !single
    filter: grep -v "variant HERMITIAN"
    args: -run_type full -petscspace_degree 2 -snes_max_it 20 -snes_type fas -snes_fas_levels 3 -fas_coarse_pc_type jacobi -fas_coarse_ksp_type cg -fas_coarse_snes_monitor_short -snes_monitor_short -fas_coarse_snes_linesearch_type basic -snes_converged_reason ::ascii_info_detail -snes_view -fas_levels_snes_type newtonls -fas_levels_pc_type jacobi -fas_levels_ksp_type cg -fas_levels_snes_monitor_short -dm_plex_simplex 0 -dm_plex_convert_type p4est -dm_forest_minimum_refinement 0 -dm_forest_initial_refinement 2 -dm_forest_maximum_refinement 4 -dm_p4est_refine_pattern hash

  test:
    suffix: p4est_full_q2_nonconformal_parallel
    requires: p4est !single
    filter: grep -v "variant HERMITIAN"
    nsize: 7
    args: -run_type full -petscspace_degree 2 -snes_max_it 20 -snes_type fas -snes_fas_levels 3 -fas_coarse_pc_type jacobi -fas_coarse_ksp_type cg -fas_coarse_snes_monitor_short -snes_monitor_short -fas_coarse_snes_linesearch_type basic -snes_converged_reason ::ascii_info_detail -snes_view -fas_levels_snes_type newtonls -fas_levels_pc_type jacobi -fas_levels_ksp_type cg -fas_levels_snes_monitor_short -dm_plex_simplex 0 -dm_plex_convert_type p4est -dm_forest_minimum_refinement 0 -dm_forest_initial_refinement 2 -dm_forest_maximum_refinement 4 -dm_p4est_refine_pattern hash -petscpartitioner_type simple

  test:
    suffix: p4est_full_q2_nonconformal_parallel_bddcfas
    requires: p4est !single
    filter: grep -v "variant HERMITIAN"
    nsize: 7
    args: -run_type full -petscspace_degree 2 -snes_max_it 20 -snes_type fas -snes_fas_levels 3 -dm_mat_type is -fas_coarse_pc_type bddc -fas_coarse_ksp_type cg -fas_coarse_snes_monitor_short -snes_monitor_short -fas_coarse_snes_linesearch_type basic -snes_converged_reason ::ascii_info_detail -snes_view -fas_levels_snes_type newtonls -fas_levels_pc_type bddc -fas_levels_ksp_type cg -fas_levels_snes_monitor_short -dm_plex_simplex 0 -dm_plex_convert_type p4est -dm_forest_minimum_refinement 0 -dm_forest_initial_refinement 2 -dm_forest_maximum_refinement 4 -dm_p4est_refine_pattern hash -petscpartitioner_type simple

  test:
    suffix: p4est_full_q2_nonconformal_parallel_bddc
    requires: p4est !single
    filter: grep -v "variant HERMITIAN"
    nsize: 7
    args: -run_type full -petscspace_degree 2 -snes_max_it 20 -snes_type newtonls -dm_mat_type is -pc_type bddc -ksp_type cg -snes_monitor_short -snes_linesearch_type basic -snes_converged_reason ::ascii_info_detail -snes_view -dm_plex_simplex 0 -dm_plex_convert_type p4est -dm_forest_minimum_refinement 0 -dm_forest_initial_refinement 2 -dm_forest_maximum_refinement 4 -dm_p4est_refine_pattern hash -petscpartitioner_type simple

  test:
    TODO: broken
    suffix: p4est_fas_q2_conformal_serial
    requires: p4est !complex !__float128
    args: -run_type full -variable_coefficient nonlinear -petscspace_degree 2 -snes_max_it 20 -snes_type fas -snes_fas_levels 3 -pc_type jacobi -ksp_type gmres -fas_coarse_pc_type svd -fas_coarse_ksp_type gmres -fas_coarse_snes_monitor_short -snes_monitor_short -fas_coarse_snes_linesearch_type basic -snes_converged_reason ::ascii_info_detail -snes_view -fas_levels_snes_type newtonls -fas_levels_pc_type svd -fas_levels_ksp_type gmres -fas_levels_snes_monitor_short -dm_plex_simplex 0 -dm_refine_hierarchy 3

  test:
    TODO: broken
    suffix: p4est_fas_q2_nonconformal_serial
    requires: p4est
    args: -run_type full -variable_coefficient nonlinear -petscspace_degree 2 -snes_max_it 20 -snes_type fas -snes_fas_levels 3 -pc_type jacobi -ksp_type gmres -fas_coarse_pc_type jacobi -fas_coarse_ksp_type gmres -fas_coarse_ksp_monitor_true_residual -fas_coarse_snes_monitor_short -snes_monitor_short -fas_coarse_snes_linesearch_type basic -snes_converged_reason ::ascii_info_detail -snes_view -fas_levels_snes_type newtonls -fas_levels_pc_type jacobi -fas_levels_ksp_type gmres -fas_levels_snes_monitor_short -dm_plex_simplex 0 -dm_plex_convert_type p4est -dm_forest_minimum_refinement 0 -dm_forest_initial_refinement 2 -dm_forest_maximum_refinement 4 -dm_p4est_refine_pattern hash

  test:
    suffix: fas_newton_0_p4est
    requires: p4est !single !__float128
    args: -run_type full -variable_coefficient nonlinear -petscspace_degree 1 -snes_type fas -snes_fas_levels 2 -fas_coarse_pc_type svd -fas_coarse_ksp_rtol 1.0e-10 -fas_coarse_snes_monitor_short -snes_monitor_short -fas_coarse_snes_linesearch_type basic -snes_converged_reason ::ascii_info_detail -snes_view -fas_levels_1_snes_type newtonls -fas_levels_1_pc_type svd -fas_levels_1_ksp_rtol 1.0e-10 -fas_levels_1_snes_monitor_short -dm_plex_simplex 0 -dm_plex_convert_type p4est -dm_forest_minimum_refinement 0 -dm_forest_initial_refinement 2 -dm_forest_maximum_refinement 4 -dm_p4est_refine_pattern hash

  # Full solve simplicial AMR
  test:
    suffix: tri_p1_adapt_init_pragmatic
    requires: pragmatic
    args: -run_type exact -dm_refine 5 -bc_type dirichlet -petscspace_degree 1 -variable_coefficient ball -snes_converged_reason ::ascii_info_detail -pc_type lu -snes_adapt_initial 1 -adaptor_target_num 4000 -dm_plex_metric_h_max 0.5 -dm_adaptor pragmatic

  test:
    suffix: tri_p2_adapt_init_pragmatic
    requires: pragmatic
    args: -run_type exact -dm_refine 5 -bc_type dirichlet -petscspace_degree 2 -variable_coefficient ball -snes_converged_reason ::ascii_info_detail -pc_type lu -snes_adapt_initial 1 -adaptor_target_num 4000 -dm_plex_metric_h_max 0.5 -dm_adaptor pragmatic

  test:
    suffix: tri_p1_adapt_init_mmg
    requires: mmg
    args: -run_type exact -dm_refine 5 -bc_type dirichlet -petscspace_degree 1 -variable_coefficient ball -snes_converged_reason ::ascii_info_detail -pc_type lu -snes_adapt_initial 1 -adaptor_target_num 4000 -dm_plex_metric_h_max 0.5 -dm_adaptor mmg

  test:
    suffix: tri_p2_adapt_init_mmg
    requires: mmg
    args: -run_type exact -dm_refine 5 -bc_type dirichlet -petscspace_degree 2 -variable_coefficient ball -snes_converged_reason ::ascii_info_detail -pc_type lu -snes_adapt_initial 1 -adaptor_target_num 4000 -dm_plex_metric_h_max 0.5 -dm_adaptor mmg

  test:
    suffix: tri_p1_adapt_seq_pragmatic
    requires: pragmatic
    args: -run_type exact -dm_refine 5 -bc_type dirichlet -petscspace_degree 1 -variable_coefficient ball -snes_converged_reason ::ascii_info_detail -pc_type lu -snes_adapt_sequence 2 -adaptor_target_num 4000 -dm_plex_metric_h_max 0.5 -dm_adaptor pragmatic

  test:
    suffix: tri_p2_adapt_seq_pragmatic
    requires: pragmatic
    args: -run_type exact -dm_refine 5 -bc_type dirichlet -petscspace_degree 2 -variable_coefficient ball -snes_converged_reason ::ascii_info_detail -pc_type lu -snes_adapt_sequence 2 -adaptor_target_num 4000 -dm_plex_metric_h_max 0.5 -dm_adaptor pragmatic

  test:
    suffix: tri_p1_adapt_seq_mmg
    requires: mmg
    args: -run_type exact -dm_refine 5 -bc_type dirichlet -petscspace_degree 1 -variable_coefficient ball -snes_converged_reason ::ascii_info_detail -pc_type lu -snes_adapt_sequence 2 -adaptor_target_num 4000 -dm_plex_metric_h_max 0.5 -dm_adaptor mmg

  test:
    suffix: tri_p2_adapt_seq_mmg
    requires: mmg
    args: -run_type exact -dm_refine 5 -bc_type dirichlet -petscspace_degree 2 -variable_coefficient ball -snes_converged_reason ::ascii_info_detail -pc_type lu -snes_adapt_sequence 2 -adaptor_target_num 4000 -dm_plex_metric_h_max 0.5 -dm_adaptor mmg

  test:
    suffix: tri_p1_adapt_analytic_pragmatic
    requires: pragmatic
    args: -run_type exact -dm_refine 3 -bc_type dirichlet -petscspace_degree 1 -variable_coefficient cross -snes_adapt_initial 4 -adaptor_target_num 500 -adaptor_monitor -dm_plex_metric_h_min 0.0001 -dm_plex_metric_h_max 0.05 -dm_adaptor pragmatic

  test:
    suffix: tri_p2_adapt_analytic_pragmatic
    requires: pragmatic
    args: -run_type exact -dm_refine 3 -bc_type dirichlet -petscspace_degree 2 -variable_coefficient cross -snes_adapt_initial 4 -adaptor_target_num 500 -adaptor_monitor -dm_plex_metric_h_min 0.0001 -dm_plex_metric_h_max 0.05 -dm_adaptor pragmatic

  test:
    suffix: tri_p1_adapt_analytic_mmg
    requires: mmg
    args: -run_type exact -dm_refine 3 -bc_type dirichlet -petscspace_degree 1 -variable_coefficient cross -snes_adapt_initial 4 -adaptor_target_num 500 -adaptor_monitor -dm_plex_metric_h_max 0.5 -dm_adaptor mmg

  test:
    suffix: tri_p2_adapt_analytic_mmg
    requires: mmg
    args: -run_type exact -dm_refine 3 -bc_type dirichlet -petscspace_degree 2 -variable_coefficient cross -snes_adapt_initial 4 -adaptor_target_num 500 -adaptor_monitor -dm_plex_metric_h_max 0.5 -dm_adaptor mmg

  test:
    suffix: tri_p1_adapt_uniform_pragmatic
    requires: pragmatic tetgen
    nsize: 2
    args: -run_type full -dm_plex_box_faces 8,8,8 -bc_type dirichlet -petscspace_degree 1 -variable_coefficient none -snes_converged_reason ::ascii_info_detail -ksp_type cg -pc_type sor -snes_adapt_sequence 3 -adaptor_target_num 400 -dm_plex_metric_h_max 0.5 -dm_plex_dim 3 -dm_adaptor pragmatic
    timeoutfactor: 2

  test:
    suffix: tri_p2_adapt_uniform_pragmatic
    requires: pragmatic tetgen
    nsize: 2
    args: -run_type full -dm_plex_box_faces 8,8,8 -bc_type dirichlet -petscspace_degree 2 -variable_coefficient none -snes_converged_reason ::ascii_info_detail -ksp_type cg -pc_type sor -snes_adapt_sequence 1 -adaptor_target_num 400 -dm_plex_metric_h_max 0.5 -dm_plex_dim 3 -dm_adaptor pragmatic
    timeoutfactor: 1

  test:
    suffix: tri_p1_adapt_uniform_mmg
    requires: mmg tetgen
    args: -run_type full -dm_plex_box_faces 4,4,4 -bc_type dirichlet -petscspace_degree 1 -variable_coefficient none -snes_converged_reason ::ascii_info_detail -ksp_type cg -pc_type sor -snes_adapt_sequence 3 -adaptor_target_num 400 -dm_plex_metric_h_max 0.5 -dm_plex_dim 3 -dm_adaptor mmg
    timeoutfactor: 2

  test:
    suffix: tri_p2_adapt_uniform_mmg
    requires: mmg tetgen
    args: -run_type full -dm_plex_box_faces 4,4,4 -bc_type dirichlet -petscspace_degree 2 -variable_coefficient none -snes_converged_reason ::ascii_info_detail -ksp_type cg -pc_type sor -snes_adapt_sequence 1 -adaptor_target_num 400 -dm_plex_metric_h_max 0.5 -dm_plex_dim 3 -dm_adaptor mmg
    timeoutfactor: 1

  test:
    suffix: tri_p1_adapt_uniform_parmmg
    requires: parmmg tetgen
    nsize: 2
    args: -run_type full -dm_plex_box_faces 8,8,8 -bc_type dirichlet -petscspace_degree 1 -variable_coefficient none -snes_converged_reason ::ascii_info_detail -ksp_type cg -pc_type sor -snes_adapt_sequence 3 -adaptor_target_num 400 -dm_plex_metric_h_max 0.5 -dm_plex_dim 3 -dm_adaptor parmmg
    timeoutfactor: 2

  test:
    suffix: tri_p2_adapt_uniform_parmmg
    requires: parmmg tetgen
    nsize: 2
    args: -run_type full -dm_plex_box_faces 8,8,8 -bc_type dirichlet -petscspace_degree 2 -variable_coefficient none -snes_converged_reason ::ascii_info_detail -ksp_type cg -pc_type sor -snes_adapt_sequence 1 -adaptor_target_num 400 -dm_plex_metric_h_max 0.5 -dm_plex_dim 3 -dm_adaptor parmmg
    timeoutfactor: 1

  # Full solve tensor AMR
  test:
    suffix: quad_q1_adapt_0
    requires: p4est
    args: -run_type exact -dm_plex_simplex 0 -dm_plex_convert_type p4est -bc_type dirichlet -petscspace_degree 1 -variable_coefficient ball -snes_converged_reason ::ascii_info_detail -pc_type lu -dm_forest_initial_refinement 4 -snes_adapt_initial 1 -dm_view
    filter: grep -v DM_

  test:
    suffix: amr_0
    nsize: 5
    args: -run_type test -petscpartitioner_type simple -dm_plex_simplex 0 -bc_type dirichlet -petscspace_degree 1 -dm_refine 1

  test:
    suffix: amr_1
    requires: p4est !complex
    args: -run_type test -dm_plex_simplex 0 -bc_type dirichlet -petscspace_degree 1 -dm_plex_convert_type p4est -dm_p4est_refine_pattern center -dm_forest_maximum_refinement 5 -dm_view vtk:amr.vtu:vtk_vtu -vec_view vtk:amr.vtu:vtk_vtu:append

  test:
    suffix: p4est_solve_bddc
    requires: p4est !complex
    args: -run_type full -variable_coefficient nonlinear -nonzero_initial_guess 1 -petscspace_degree 2 -snes_max_it 20 -snes_type newtonls -dm_mat_type is -pc_type bddc -ksp_type cg -snes_monitor_short -ksp_monitor -snes_linesearch_type bt -snes_converged_reason -snes_view -dm_plex_simplex 0 -petscspace_poly_tensor -dm_plex_convert_type p4est -dm_forest_minimum_refinement 0 -dm_forest_initial_refinement 2 -dm_forest_maximum_refinement 4 -dm_p4est_refine_pattern hash -petscpartitioner_type simple -pc_bddc_detect_disconnected
    nsize: 4

  test:
    suffix: p4est_solve_fas
    requires: p4est
    args: -run_type full -variable_coefficient nonlinear -nonzero_initial_guess 1 -petscspace_degree 2 -snes_max_it 10 -snes_type fas -snes_linesearch_type bt -snes_fas_levels 3 -fas_coarse_snes_type newtonls -fas_coarse_snes_linesearch_type basic -fas_coarse_ksp_type cg -fas_coarse_pc_type jacobi -fas_coarse_snes_monitor_short -fas_levels_snes_max_it 4 -fas_levels_snes_type newtonls -fas_levels_snes_linesearch_type bt -fas_levels_ksp_type cg -fas_levels_pc_type jacobi -fas_levels_snes_monitor_short -fas_levels_cycle_snes_linesearch_type bt -snes_monitor_short -snes_converged_reason -snes_view -dm_plex_simplex 0 -petscspace_poly_tensor -dm_plex_convert_type p4est -dm_forest_minimum_refinement 0 -dm_forest_initial_refinement 2 -dm_forest_maximum_refinement 4 -dm_p4est_refine_pattern hash
    nsize: 4
    TODO: identical machine two runs produce slightly different solver trackers

  test:
    suffix: p4est_convergence_test_1
    requires: p4est
    args:  -quiet -run_type test -petscspace_degree 1 -dm_plex_simplex 0 -petscspace_poly_tensor -dm_plex_convert_type p4est -dm_forest_minimum_refinement 2 -dm_forest_initial_refinement 2 -dm_forest_maximum_refinement 4 -dm_p4est_refine_pattern hash
    nsize: 4

  test:
    suffix: p4est_convergence_test_2
    requires: p4est
    args: -quiet -run_type test -petscspace_degree 1 -dm_plex_simplex 0 -petscspace_poly_tensor -dm_plex_convert_type p4est -dm_forest_minimum_refinement 3 -dm_forest_initial_refinement 3 -dm_forest_maximum_refinement 5 -dm_p4est_refine_pattern hash

  test:
    suffix: p4est_convergence_test_3
    requires: p4est
    args: -quiet -run_type test -petscspace_degree 1 -dm_plex_simplex 0 -petscspace_poly_tensor -dm_plex_convert_type p4est -dm_forest_minimum_refinement 4 -dm_forest_initial_refinement 4 -dm_forest_maximum_refinement 6 -dm_p4est_refine_pattern hash

  test:
    suffix: p4est_convergence_test_4
    requires: p4est
    args: -quiet -run_type test -petscspace_degree 1 -dm_plex_simplex 0 -petscspace_poly_tensor -dm_plex_convert_type p4est -dm_forest_minimum_refinement 5 -dm_forest_initial_refinement 5 -dm_forest_maximum_refinement 7 -dm_p4est_refine_pattern hash
    timeoutfactor: 5

  # Serial tests with GLVis visualization
  test:
    suffix: glvis_2d_tet_p1
    args: -quiet -run_type test -bc_type dirichlet -petscspace_degree 1 -vec_view glvis: -dm_plex_filename ${wPETSC_DIR}/share/petsc/datafiles/meshes/square_periodic.msh -dm_plex_boundary_label marker -dm_plex_gmsh_periodic 0 -dm_coord_space 0
  test:
    suffix: glvis_2d_tet_p2
    args: -quiet -run_type test -bc_type dirichlet -petscspace_degree 2 -vec_view glvis: -dm_plex_filename ${wPETSC_DIR}/share/petsc/datafiles/meshes/square_periodic.msh -dm_plex_boundary_label marker -dm_plex_gmsh_periodic 0 -dm_coord_space 0
  test:
    suffix: glvis_2d_hex_p1
    args: -quiet -run_type test -bc_type dirichlet -petscspace_degree 1 -vec_view glvis: -dm_plex_simplex 0 -dm_refine 1 -dm_coord_space 0
  test:
    suffix: glvis_2d_hex_p2
    args: -quiet -run_type test -bc_type dirichlet -petscspace_degree 2 -vec_view glvis: -dm_plex_simplex 0 -dm_refine 1 -dm_coord_space 0
  test:
    suffix: glvis_2d_hex_p2_p4est
    requires: p4est
    args: -quiet -run_type test -bc_type dirichlet -petscspace_degree 2 -vec_view glvis: -dm_plex_simplex 0 -dm_plex_convert_type p4est -dm_forest_minimum_refinement 0 -dm_forest_initial_refinement 1 -dm_forest_maximum_refinement 4 -dm_p4est_refine_pattern hash -viewer_glvis_dm_plex_enable_ncmesh
  test:
    suffix: glvis_2d_tet_p0
    args: -run_type exact -guess_vec_view glvis: -nonzero_initial_guess 1 -dm_plex_filename ${wPETSC_DIR}/share/petsc/datafiles/meshes/square_periodic.msh -dm_plex_boundary_label marker -petscspace_degree 0 -dm_coord_space 0
  test:
    suffix: glvis_2d_hex_p0
    args: -run_type exact -guess_vec_view glvis: -nonzero_initial_guess 1 -dm_plex_box_faces 5,7 -dm_plex_simplex 0 -petscspace_degree 0 -dm_coord_space 0

  # PCHPDDM tests
  testset:
    nsize: 4
    requires: hpddm slepc !single defined(PETSC_HAVE_DYNAMIC_LIBRARIES) defined(PETSC_USE_SHARED_LIBRARIES)
    args: -run_type test -run_test_check_ksp -quiet -petscspace_degree 1 -petscpartitioner_type simple -bc_type none -dm_plex_simplex 0 -pc_type hpddm -pc_hpddm_levels_1_sub_pc_type lu -pc_hpddm_levels_1_eps_nev 2 -pc_hpddm_coarse_p 1 -pc_hpddm_coarse_pc_type svd -ksp_rtol 1.e-10 -pc_hpddm_levels_1_st_pc_factor_shift_type INBLOCKS -ksp_converged_reason
    test:
      suffix: quad_singular_hpddm
      args: -dm_plex_box_faces 6,7
    test:
      requires: p4est
      suffix: p4est_singular_2d_hpddm
      args: -dm_plex_convert_type p4est -dm_forest_minimum_refinement 1 -dm_forest_initial_refinement 3 -dm_forest_maximum_refinement 3
    test:
      requires: p4est
      suffix: p4est_nc_singular_2d_hpddm
      args: -dm_plex_convert_type p4est -dm_forest_minimum_refinement 1 -dm_forest_initial_refinement 1 -dm_forest_maximum_refinement 3 -dm_p4est_refine_pattern hash
  testset:
    nsize: 4
    requires: hpddm slepc triangle !single defined(PETSC_HAVE_DYNAMIC_LIBRARIES) defined(PETSC_USE_SHARED_LIBRARIES)
    args: -run_type full -petscpartitioner_type simple -dm_refine 2 -bc_type dirichlet -petscspace_degree 2 -ksp_type gmres -ksp_gmres_restart 100 -pc_type hpddm -snes_monitor_short -ksp_monitor_short -snes_converged_reason ::ascii_info_detail -ksp_converged_reason -snes_view -show_solution 0 -pc_type hpddm -pc_hpddm_levels_1_sub_pc_type lu -pc_hpddm_levels_1_eps_nev 4 -pc_hpddm_coarse_p 2 -pc_hpddm_coarse_pc_type redundant -ksp_rtol 1.e-1
    test:
      args: -pc_hpddm_coarse_mat_type baij -options_left no
      suffix: tri_hpddm_reuse_baij
    test:
      requires: !complex
      suffix: tri_hpddm_reuse
  testset:
    nsize: 4
    requires: hpddm slepc !single defined(PETSC_HAVE_DYNAMIC_LIBRARIES) defined(PETSC_USE_SHARED_LIBRARIES)
    args: -run_type full -petscpartitioner_type simple -dm_plex_box_faces 7,5 -dm_refine 2 -dm_plex_simplex 0 -bc_type dirichlet -petscspace_degree 2 -ksp_type gmres -ksp_gmres_restart 100 -pc_type hpddm -snes_monitor_short -ksp_monitor_short -snes_converged_reason ::ascii_info_detail -ksp_converged_reason -snes_view -show_solution 0 -pc_type hpddm -pc_hpddm_levels_1_sub_pc_type lu -pc_hpddm_levels_1_eps_nev 4 -pc_hpddm_coarse_p 2 -pc_hpddm_coarse_pc_type redundant -ksp_rtol 1.e-1
    test:
      args: -pc_hpddm_coarse_mat_type baij -options_left no
      suffix: quad_hpddm_reuse_baij
    test:
      requires: !complex
      suffix: quad_hpddm_reuse
  testset:
    nsize: 4
    requires: hpddm slepc !single defined(PETSC_HAVE_DYNAMIC_LIBRARIES) defined(PETSC_USE_SHARED_LIBRARIES)
    args: -run_type full -petscpartitioner_type simple -dm_plex_box_faces 7,5 -dm_refine 2 -dm_plex_simplex 0 -bc_type dirichlet -petscspace_degree 1 -ksp_type gmres -ksp_gmres_restart 100 -pc_type hpddm -snes_monitor_short -ksp_monitor_short -snes_converged_reason ::ascii_info_detail -ksp_converged_reason -snes_view -show_solution 0 -pc_type hpddm -pc_hpddm_levels_1_sub_pc_type lu -pc_hpddm_levels_1_eps_threshold 0.1 -pc_hpddm_coarse_p 2 -pc_hpddm_coarse_pc_type redundant -ksp_rtol 1.e-1
    test:
      args: -pc_hpddm_coarse_mat_type baij -options_left no
      suffix: quad_hpddm_reuse_threshold_baij
    test:
      requires: !complex
      suffix: quad_hpddm_reuse_threshold
  testset:
    nsize: 4
    requires: hpddm slepc parmetis !single defined(PETSC_HAVE_DYNAMIC_LIBRARIES) defined(PETSC_USE_SHARED_LIBRARIES)
    filter: sed -e "s/linear solver iterations=17/linear solver iterations=16/g"
    args: -run_type full -petscpartitioner_type parmetis -dm_refine 3 -bc_type dirichlet -petscspace_degree 1 -ksp_type gmres -ksp_gmres_restart 100 -pc_type hpddm -snes_monitor_short -snes_converged_reason ::ascii_info_detail -snes_view -show_solution 0 -pc_type hpddm -pc_hpddm_levels_1_sub_pc_type icc -pc_hpddm_levels_1_eps_nev 20 -pc_hpddm_coarse_p 2 -pc_hpddm_coarse_pc_type redundant -ksp_rtol 1.e-10 -dm_plex_filename ${PETSC_DIR}/share/petsc/datafiles/meshes/square_periodic.msh -dm_plex_boundary_label marker -pc_hpddm_levels_1_sub_pc_factor_levels 3 -variable_coefficient ball -dm_plex_gmsh_periodic 0
    test:
      args: -pc_hpddm_coarse_mat_type baij -options_left no
      filter: grep -v "      total: nonzeros=" | grep -v "      rows=" | sed -e "s/total number of linear solver iterations=1[5-7]/total number of linear solver iterations=16/g"
      suffix: tri_parmetis_hpddm_baij
    test:
      filter: grep -v "      total: nonzeros=" | grep -v "      rows=" | sed -e "s/total number of linear solver iterations=1[5-7]/total number of linear solver iterations=16/g"
      requires: !complex
      suffix: tri_parmetis_hpddm

  # 2D serial P1 tests for adaptive MG
  test:
    suffix: 2d_p1_adaptmg_0
    requires: triangle bamg
    args: -dm_refine_hierarchy 3 -dm_plex_box_faces 4,4 -bc_type dirichlet -petscspace_degree 1 \
          -variable_coefficient checkerboard_0 -mat_petscspace_degree 0 -div 16 -k 3 \
          -snes_max_it 1 -ksp_converged_reason \
          -ksp_rtol 1e-8 -pc_type mg
  # -ksp_monitor_true_residual -ksp_converged_reason -mg_levels_ksp_monitor_true_residual -pc_mg_mesp_monitor -dm_adapt_interp_view_fine draw -dm_adapt_interp_view_coarse draw -draw_pause 1
  test:
    suffix: 2d_p1_adaptmg_1
    requires: triangle bamg
    args: -dm_refine_hierarchy 3 -dm_plex_box_faces 4,4 -bc_type dirichlet -petscspace_degree 1 \
          -variable_coefficient checkerboard_0 -mat_petscspace_degree 0 -div 16 -k 3 \
          -snes_max_it 1 -ksp_converged_reason \
          -ksp_rtol 1e-8 -pc_type mg -pc_mg_galerkin -pc_mg_adapt_interp -pc_mg_adapt_interp_coarse_space eigenvector -pc_mg_adapt_interp_n 1 \
            -pc_mg_mesp_ksp_type richardson -pc_mg_mesp_ksp_richardson_self_scale -pc_mg_mesp_ksp_max_it 100 -pc_mg_mesp_pc_type none

TEST*/
