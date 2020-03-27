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
typedef enum {COEFF_NONE, COEFF_ANALYTIC, COEFF_FIELD, COEFF_NONLINEAR, COEFF_CIRCLE, COEFF_CROSS} CoeffType;

typedef struct {
  PetscInt       debug;             /* The debugging level */
  RunType        runType;           /* Whether to run tests, or solve the full problem */
  PetscBool      jacobianMF;        /* Whether to calculate the Jacobian action on the fly */
  PetscLogEvent  createMeshEvent;
  PetscBool      showInitial, showSolution, restart, quiet, nonzInit;
  /* Domain and mesh definition */
  PetscInt       dim;               /* The topological mesh dimension */
  DMBoundaryType periodicity[3];    /* The domain periodicity */
  PetscInt       cells[3];          /* The initial domain division */
  char           filename[2048];    /* The optional mesh file */
  PetscBool      interpolate;       /* Generate intermediate mesh elements */
  PetscReal      refinementLimit;   /* The largest allowable cell volume */
  PetscBool      viewHierarchy;     /* Whether to view the hierarchy */
  PetscBool      simplex;           /* Simplicial mesh */
  /* Problem definition */
  BCType         bcType;
  CoeffType      variableCoefficient;
  PetscErrorCode (**exactFuncs)(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nc, PetscScalar *u, void *ctx);
  PetscBool      fieldBC;
  void           (**exactFields)(PetscInt, PetscInt, PetscInt,
                                 const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[],
                                 const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[],
                                 PetscReal, const PetscReal[], PetscInt, const PetscScalar[], PetscScalar[]);
  PetscBool      bdIntegral;       /* Compute the integral of the solution on the boundary */
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

static PetscErrorCode circle_u_2d(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nc, PetscScalar *u, void *ctx)
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

static void f0_circle_u(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                        const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                        const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                        PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f0[])
{
  const PetscReal alpha   = 500.;
  const PetscReal radius2 = PetscSqr(0.15);
  const PetscReal r2      = PetscSqr(x[0] - 0.5) + PetscSqr(x[1] - 0.5);
  const PetscReal xi      = alpha*(radius2 - r2);

  f0[0] = (-4.0*alpha - 8.0*PetscSqr(alpha)*r2*PetscTanhReal(xi)) * PetscSqr(1.0/PetscCoshReal(xi));
}

static void f0_cross_u(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                       const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                       const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                       PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f0[])
{
  const PetscReal alpha = 50*4;
  const PetscReal xy    = (x[0]-0.5)*(x[1]-0.5);

  f0[0] = PetscSinReal(alpha*xy) * (alpha*PetscAbsReal(xy) < 2*PETSC_PI ? 1 : 0.01);
}

static void f0_bd_u(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                    const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                    const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                    PetscReal t, const PetscReal x[], const PetscReal n[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f0[])
{
  PetscInt d;
  for (d = 0, f0[0] = 0.0; d < dim; ++d) f0[0] += -n[d]*2.0*x[d];
}

static void f1_bd_zero(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                       const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                       const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                       PetscReal t, const PetscReal x[], const PetscReal n[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f1[])
{
  PetscInt comp;
  for (comp = 0; comp < dim; ++comp) f1[comp] = 0.0;
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

static void quadratic_u_field_3d(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                                 const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                                 const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                                 PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar uexact[])
{
  uexact[0] = a[0];
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
  const char    *coeffTypes[6] = {"none", "analytic", "field", "nonlinear", "circle", "cross"};
  PetscInt       bd, bc, run, coeff, n;
  PetscBool      flg;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  options->debug               = 0;
  options->runType             = RUN_FULL;
  options->dim                 = 2;
  options->periodicity[0]      = DM_BOUNDARY_NONE;
  options->periodicity[1]      = DM_BOUNDARY_NONE;
  options->periodicity[2]      = DM_BOUNDARY_NONE;
  options->cells[0]            = 2;
  options->cells[1]            = 2;
  options->cells[2]            = 2;
  options->filename[0]         = '\0';
  options->interpolate         = PETSC_TRUE;
  options->refinementLimit     = 0.0;
  options->bcType              = DIRICHLET;
  options->variableCoefficient = COEFF_NONE;
  options->fieldBC             = PETSC_FALSE;
  options->jacobianMF          = PETSC_FALSE;
  options->showInitial         = PETSC_FALSE;
  options->showSolution        = PETSC_FALSE;
  options->restart             = PETSC_FALSE;
  options->viewHierarchy       = PETSC_FALSE;
  options->simplex             = PETSC_TRUE;
  options->quiet               = PETSC_FALSE;
  options->nonzInit            = PETSC_FALSE;
  options->bdIntegral          = PETSC_FALSE;
  options->checkksp            = PETSC_FALSE;

  ierr = PetscOptionsBegin(comm, "", "Poisson Problem Options", "DMPLEX");CHKERRQ(ierr);
  ierr = PetscOptionsInt("-debug", "The debugging level", "ex12.c", options->debug, &options->debug, NULL);CHKERRQ(ierr);
  run  = options->runType;
  ierr = PetscOptionsEList("-run_type", "The run type", "ex12.c", runTypes, 4, runTypes[options->runType], &run, NULL);CHKERRQ(ierr);

  options->runType = (RunType) run;

  ierr = PetscOptionsInt("-dim", "The topological mesh dimension", "ex12.c", options->dim, &options->dim, NULL);CHKERRQ(ierr);
  bd = options->periodicity[0];
  ierr = PetscOptionsEList("-x_periodicity", "The x-boundary periodicity", "ex12.c", DMBoundaryTypes, 5, DMBoundaryTypes[options->periodicity[0]], &bd, NULL);CHKERRQ(ierr);
  options->periodicity[0] = (DMBoundaryType) bd;
  bd = options->periodicity[1];
  ierr = PetscOptionsEList("-y_periodicity", "The y-boundary periodicity", "ex12.c", DMBoundaryTypes, 5, DMBoundaryTypes[options->periodicity[1]], &bd, NULL);CHKERRQ(ierr);
  options->periodicity[1] = (DMBoundaryType) bd;
  bd = options->periodicity[2];
  ierr = PetscOptionsEList("-z_periodicity", "The z-boundary periodicity", "ex12.c", DMBoundaryTypes, 5, DMBoundaryTypes[options->periodicity[2]], &bd, NULL);CHKERRQ(ierr);
  options->periodicity[2] = (DMBoundaryType) bd;
  n = 3;
  ierr = PetscOptionsIntArray("-cells", "The initial mesh division", "ex12.c", options->cells, &n, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsString("-f", "Mesh filename to read", "ex12.c", options->filename, options->filename, sizeof(options->filename), &flg);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-interpolate", "Generate intermediate mesh elements", "ex12.c", options->interpolate, &options->interpolate, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-refinement_limit", "The largest allowable cell volume", "ex12.c", options->refinementLimit, &options->refinementLimit, NULL);CHKERRQ(ierr);
  bc   = options->bcType;
  ierr = PetscOptionsEList("-bc_type","Type of boundary condition","ex12.c",bcTypes,3,bcTypes[options->bcType],&bc,NULL);CHKERRQ(ierr);
  options->bcType = (BCType) bc;
  coeff = options->variableCoefficient;
  ierr = PetscOptionsEList("-variable_coefficient","Type of variable coefficent","ex12.c",coeffTypes,6,coeffTypes[options->variableCoefficient],&coeff,NULL);CHKERRQ(ierr);
  options->variableCoefficient = (CoeffType) coeff;

  ierr = PetscOptionsBool("-field_bc", "Use a field representation for the BC", "ex12.c", options->fieldBC, &options->fieldBC, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-jacobian_mf", "Calculate the action of the Jacobian on the fly", "ex12.c", options->jacobianMF, &options->jacobianMF, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-show_initial", "Output the initial guess for verification", "ex12.c", options->showInitial, &options->showInitial, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-show_solution", "Output the solution for verification", "ex12.c", options->showSolution, &options->showSolution, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-restart", "Read in the mesh and solution from a file", "ex12.c", options->restart, &options->restart, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-dm_view_hierarchy", "View the coarsened hierarchy", "ex12.c", options->viewHierarchy, &options->viewHierarchy, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-simplex", "Simplicial (true) or tensor (false) mesh", "ex12.c", options->simplex, &options->simplex, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-quiet", "Don't print any vecs", "ex12.c", options->quiet, &options->quiet, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-nonzero_initial_guess", "nonzero intial guess", "ex12.c", options->nonzInit, &options->nonzInit, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-bd_integral", "Compute the integral of the solution on the boundary", "ex12.c", options->bdIntegral, &options->bdIntegral, NULL);CHKERRQ(ierr);
  if (options->runType == RUN_TEST) {
    ierr = PetscOptionsBool("-run_test_check_ksp", "Check solution of KSP", "ex12.c", options->checkksp, &options->checkksp, NULL);CHKERRQ(ierr);
  }
  ierr = PetscOptionsEnd();
  ierr = PetscLogEventRegister("CreateMesh", DM_CLASSID, &options->createMeshEvent);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode CreateBCLabel(DM dm, const char name[])
{
  DM             plex;
  DMLabel        label;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = DMCreateLabel(dm, name);CHKERRQ(ierr);
  ierr = DMGetLabel(dm, name, &label);CHKERRQ(ierr);
  ierr = DMConvert(dm, DMPLEX, &plex);CHKERRQ(ierr);
  ierr = DMPlexMarkBoundaryFaces(plex, 1, label);CHKERRQ(ierr);
  ierr = DMDestroy(&plex);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode CreateMesh(MPI_Comm comm, AppCtx *user, DM *dm)
{
  PetscInt       dim             = user->dim;
  const char    *filename        = user->filename;
  PetscBool      interpolate     = user->interpolate;
  PetscReal      refinementLimit = user->refinementLimit;
  size_t         len;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = PetscLogEventBegin(user->createMeshEvent,0,0,0,0);CHKERRQ(ierr);
  ierr = PetscStrlen(filename, &len);CHKERRQ(ierr);
  if (!len) {
    PetscInt d;

    if (user->periodicity[0] || user->periodicity[1] || user->periodicity[2]) for (d = 0; d < dim; ++d) user->cells[d] = PetscMax(user->cells[d], 3);
    ierr = DMPlexCreateBoxMesh(comm, dim, user->simplex, user->cells, NULL, NULL, user->periodicity, interpolate, dm);CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject) *dm, "Mesh");CHKERRQ(ierr);
  } else {
    ierr = DMPlexCreateFromFile(comm, filename, interpolate, dm);CHKERRQ(ierr);
    ierr = DMPlexSetRefinementUniform(*dm, PETSC_FALSE);CHKERRQ(ierr);
  }
  {
    PetscPartitioner part;
    DM               refinedMesh     = NULL;
    DM               distributedMesh = NULL;

    /* Refine mesh using a volume constraint */
    if (refinementLimit > 0.0) {
      ierr = DMPlexSetRefinementLimit(*dm, refinementLimit);CHKERRQ(ierr);
      ierr = DMRefine(*dm, comm, &refinedMesh);CHKERRQ(ierr);
      if (refinedMesh) {
        const char *name;

        ierr = PetscObjectGetName((PetscObject) *dm,         &name);CHKERRQ(ierr);
        ierr = PetscObjectSetName((PetscObject) refinedMesh,  name);CHKERRQ(ierr);
        ierr = DMDestroy(dm);CHKERRQ(ierr);
        *dm  = refinedMesh;
      }
    }
    /* Distribute mesh over processes */
    ierr = DMPlexGetPartitioner(*dm,&part);CHKERRQ(ierr);
    ierr = PetscPartitionerSetFromOptions(part);CHKERRQ(ierr);
    ierr = DMPlexDistribute(*dm, 0, NULL, &distributedMesh);CHKERRQ(ierr);
    if (distributedMesh) {
      ierr = DMDestroy(dm);CHKERRQ(ierr);
      *dm  = distributedMesh;
    }
  }
  if (interpolate) {
    if (user->bcType == NEUMANN) {
      DMLabel   label;

      ierr = DMCreateLabel(*dm, "boundary");CHKERRQ(ierr);
      ierr = DMGetLabel(*dm, "boundary", &label);CHKERRQ(ierr);
      ierr = DMPlexMarkBoundaryFaces(*dm, 1, label);CHKERRQ(ierr);
    } else if (user->bcType == DIRICHLET) {
      PetscBool hasLabel;

      ierr = DMHasLabel(*dm,"marker",&hasLabel);CHKERRQ(ierr);
      if (!hasLabel) {ierr = CreateBCLabel(*dm, "marker");CHKERRQ(ierr);}
    }
  }
  {
    char      convType[256];
    PetscBool flg;

    ierr = PetscOptionsBegin(comm, "", "Mesh conversion options", "DMPLEX");CHKERRQ(ierr);
    ierr = PetscOptionsFList("-dm_plex_convert_type","Convert DMPlex to another format","ex12",DMList,DMPLEX,convType,256,&flg);CHKERRQ(ierr);
    ierr = PetscOptionsEnd();
    if (flg) {
      DM dmConv;

      ierr = DMConvert(*dm,convType,&dmConv);CHKERRQ(ierr);
      if (dmConv) {
        ierr = DMDestroy(dm);CHKERRQ(ierr);
        *dm  = dmConv;
      }
    }
  }
  ierr = DMLocalizeCoordinates(*dm);CHKERRQ(ierr); /* needed for periodic */
  ierr = DMSetFromOptions(*dm);CHKERRQ(ierr);
  ierr = DMViewFromOptions(*dm, NULL, "-dm_view");CHKERRQ(ierr);
  if (user->viewHierarchy) {
    DM       cdm = *dm;
    PetscInt i   = 0;
    char     buf[256];

    while (cdm) {
      ierr = DMSetUp(cdm);CHKERRQ(ierr);
      ierr = DMGetCoarseDM(cdm, &cdm);CHKERRQ(ierr);
      ++i;
    }
    cdm = *dm;
    while (cdm) {
      PetscViewer       viewer;
      PetscBool   isHDF5, isVTK;

      --i;
      ierr = PetscViewerCreate(comm,&viewer);CHKERRQ(ierr);
      ierr = PetscViewerSetType(viewer,PETSCVIEWERHDF5);CHKERRQ(ierr);
      ierr = PetscViewerSetOptionsPrefix(viewer,"hierarchy_");CHKERRQ(ierr);
      ierr = PetscViewerSetFromOptions(viewer);CHKERRQ(ierr);
      ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERHDF5,&isHDF5);CHKERRQ(ierr);
      ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERVTK,&isVTK);CHKERRQ(ierr);
      if (isHDF5) {
        ierr = PetscSNPrintf(buf, 256, "ex12-%d.h5", i);CHKERRQ(ierr);
      } else if (isVTK) {
        ierr = PetscSNPrintf(buf, 256, "ex12-%d.vtu", i);CHKERRQ(ierr);
        ierr = PetscViewerPushFormat(viewer,PETSC_VIEWER_VTK_VTU);CHKERRQ(ierr);
      } else {
        ierr = PetscSNPrintf(buf, 256, "ex12-%d", i);CHKERRQ(ierr);
      }
      ierr = PetscViewerFileSetMode(viewer,FILE_MODE_WRITE);CHKERRQ(ierr);
      ierr = PetscViewerFileSetName(viewer,buf);CHKERRQ(ierr);
      ierr = DMView(cdm, viewer);CHKERRQ(ierr);
      ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
      ierr = DMGetCoarseDM(cdm, &cdm);CHKERRQ(ierr);
    }
  }
  ierr = PetscLogEventEnd(user->createMeshEvent,0,0,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode SetupProblem(DM dm, AppCtx *user)
{
  PetscDS        prob;
  const PetscInt id = 1;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = DMGetDS(dm, &prob);CHKERRQ(ierr);
  switch (user->variableCoefficient) {
  case COEFF_NONE:
    if (user->periodicity[0]) {
      if (user->periodicity[1]) {
        ierr = PetscDSSetResidual(prob, 0, f0_xytrig_u, f1_u);CHKERRQ(ierr);
        ierr = PetscDSSetJacobian(prob, 0, 0, NULL, NULL, NULL, g3_uu);CHKERRQ(ierr);
      } else {
        ierr = PetscDSSetResidual(prob, 0, f0_xtrig_u,  f1_u);CHKERRQ(ierr);
        ierr = PetscDSSetJacobian(prob, 0, 0, NULL, NULL, NULL, g3_uu);CHKERRQ(ierr);
      }
    } else {
      ierr = PetscDSSetResidual(prob, 0, f0_u, f1_u);CHKERRQ(ierr);
      ierr = PetscDSSetJacobian(prob, 0, 0, NULL, NULL, NULL, g3_uu);CHKERRQ(ierr);
    }
    break;
  case COEFF_ANALYTIC:
    ierr = PetscDSSetResidual(prob, 0, f0_analytic_u, f1_analytic_u);CHKERRQ(ierr);
    ierr = PetscDSSetJacobian(prob, 0, 0, NULL, NULL, NULL, g3_analytic_uu);CHKERRQ(ierr);
    break;
  case COEFF_FIELD:
    ierr = PetscDSSetResidual(prob, 0, f0_analytic_u, f1_field_u);CHKERRQ(ierr);
    ierr = PetscDSSetJacobian(prob, 0, 0, NULL, NULL, NULL, g3_field_uu);CHKERRQ(ierr);
    break;
  case COEFF_NONLINEAR:
    ierr = PetscDSSetResidual(prob, 0, f0_analytic_nonlinear_u, f1_analytic_nonlinear_u);CHKERRQ(ierr);
    ierr = PetscDSSetJacobian(prob, 0, 0, NULL, NULL, NULL, g3_analytic_nonlinear_uu);CHKERRQ(ierr);
    break;
  case COEFF_CIRCLE:
    ierr = PetscDSSetResidual(prob, 0, f0_circle_u, f1_u);CHKERRQ(ierr);
    ierr = PetscDSSetJacobian(prob, 0, 0, NULL, NULL, NULL, g3_uu);CHKERRQ(ierr);
    break;
  case COEFF_CROSS:
    ierr = PetscDSSetResidual(prob, 0, f0_cross_u, f1_u);CHKERRQ(ierr);
    ierr = PetscDSSetJacobian(prob, 0, 0, NULL, NULL, NULL, g3_uu);CHKERRQ(ierr);
    break;
  default: SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Invalid variable coefficient type %d", user->variableCoefficient);
  }
  switch (user->dim) {
  case 2:
    switch (user->variableCoefficient) {
    case COEFF_CIRCLE:
      user->exactFuncs[0]  = circle_u_2d;break;
    case COEFF_CROSS:
      user->exactFuncs[0]  = cross_u_2d;break;
    default:
      if (user->periodicity[0]) {
        if (user->periodicity[1]) {
          user->exactFuncs[0] = xytrig_u_2d;
        } else {
          user->exactFuncs[0] = xtrig_u_2d;
        }
      } else {
        user->exactFuncs[0]  = quadratic_u_2d;
        user->exactFields[0] = quadratic_u_field_2d;
      }
    }
    if (user->bcType == NEUMANN) {ierr = PetscDSSetBdResidual(prob, 0, f0_bd_u, f1_bd_zero);CHKERRQ(ierr);}
    break;
  case 3:
    user->exactFuncs[0]  = quadratic_u_3d;
    user->exactFields[0] = quadratic_u_field_3d;
    if (user->bcType == NEUMANN) {ierr = PetscDSSetBdResidual(prob, 0, f0_bd_u, f1_bd_zero);CHKERRQ(ierr);}
    break;
  default:
    SETERRQ1(PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE, "Invalid dimension %d", user->dim);
  }
  ierr = PetscDSSetExactSolution(prob, 0, user->exactFuncs[0], user);CHKERRQ(ierr);
  if (user->bcType != NONE) {
    ierr = DMAddBoundary(dm, user->bcType == DIRICHLET ? (user->fieldBC ? DM_BC_ESSENTIAL_FIELD : DM_BC_ESSENTIAL) : DM_BC_NATURAL,
                         "wall", user->bcType == DIRICHLET ? "marker" : "boundary", 0, 0, NULL,
                         user->fieldBC ? (void (*)(void)) user->exactFields[0] : (void (*)(void)) user->exactFuncs[0], 1, &id, user);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode SetupMaterial(DM dm, DM dmAux, AppCtx *user)
{
  PetscErrorCode (*matFuncs[1])(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nc, PetscScalar u[], void *ctx) = {nu_2d};
  Vec            nu;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMCreateLocalVector(dmAux, &nu);CHKERRQ(ierr);
  ierr = DMProjectFunctionLocal(dmAux, 0.0, matFuncs, NULL, INSERT_ALL_VALUES, nu);CHKERRQ(ierr);
  ierr = PetscObjectCompose((PetscObject) dm, "A", (PetscObject) nu);CHKERRQ(ierr);
  ierr = VecDestroy(&nu);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode SetupBC(DM dm, DM dmAux, AppCtx *user)
{
  PetscErrorCode (*bcFuncs[1])(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nc, PetscScalar u[], void *ctx);
  Vec            uexact;
  PetscInt       dim;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);
  if (dim == 2) bcFuncs[0] = quadratic_u_2d;
  else          bcFuncs[0] = quadratic_u_3d;
  ierr = DMCreateLocalVector(dmAux, &uexact);CHKERRQ(ierr);
  ierr = DMProjectFunctionLocal(dmAux, 0.0, bcFuncs, NULL, INSERT_ALL_VALUES, uexact);CHKERRQ(ierr);
  ierr = PetscObjectCompose((PetscObject) dm, "A", (PetscObject) uexact);CHKERRQ(ierr);
  ierr = VecDestroy(&uexact);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode SetupAuxDM(DM dm, PetscFE feAux, AppCtx *user)
{
  DM             dmAux, coordDM;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /* MUST call DMGetCoordinateDM() in order to get p4est setup if present */
  ierr = DMGetCoordinateDM(dm, &coordDM);CHKERRQ(ierr);
  if (!feAux) PetscFunctionReturn(0);
  ierr = DMClone(dm, &dmAux);CHKERRQ(ierr);
  ierr = PetscObjectCompose((PetscObject) dm, "dmAux", (PetscObject) dmAux);CHKERRQ(ierr);
  ierr = DMSetCoordinateDM(dmAux, coordDM);CHKERRQ(ierr);
  ierr = DMSetField(dmAux, 0, NULL, (PetscObject) feAux);CHKERRQ(ierr);
  ierr = DMCreateDS(dmAux);CHKERRQ(ierr);
  if (user->fieldBC) {ierr = SetupBC(dm, dmAux, user);CHKERRQ(ierr);}
  else               {ierr = SetupMaterial(dm, dmAux, user);CHKERRQ(ierr);}
  ierr = DMDestroy(&dmAux);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode SetupDiscretization(DM dm, AppCtx *user)
{
  DM             cdm = dm;
  const PetscInt dim = user->dim;
  PetscFE        fe, feAux = NULL;
  PetscBool      simplex   = user->simplex;
  MPI_Comm       comm;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  /* Create finite element for each field and auxiliary field */
  ierr = PetscObjectGetComm((PetscObject) dm, &comm);CHKERRQ(ierr);
  ierr = PetscFECreateDefault(comm, dim, 1, simplex, NULL, -1, &fe);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) fe, "potential");CHKERRQ(ierr);
  if (user->variableCoefficient == COEFF_FIELD) {
    ierr = PetscFECreateDefault(comm, dim, 1, simplex, "mat_", -1, &feAux);CHKERRQ(ierr);
    ierr = PetscFECopyQuadrature(fe, feAux);CHKERRQ(ierr);
  } else if (user->fieldBC) {
    ierr = PetscFECreateDefault(comm, dim, 1, simplex, "bc_", -1, &feAux);CHKERRQ(ierr);
    ierr = PetscFECopyQuadrature(fe, feAux);CHKERRQ(ierr);
  }
  /* Set discretization and boundary conditions for each mesh */
  ierr = DMSetField(dm, 0, NULL, (PetscObject) fe);CHKERRQ(ierr);
  ierr = DMCreateDS(dm);CHKERRQ(ierr);
  ierr = SetupProblem(dm, user);CHKERRQ(ierr);
  while (cdm) {
    ierr = SetupAuxDM(cdm, feAux, user);CHKERRQ(ierr);
    if (user->bcType == DIRICHLET && user->interpolate) {
      PetscBool hasLabel;

      ierr = DMHasLabel(cdm, "marker", &hasLabel);CHKERRQ(ierr);
      if (!hasLabel) {ierr = CreateBCLabel(cdm, "marker");CHKERRQ(ierr);}
    }
    ierr = DMCopyDisc(dm, cdm);CHKERRQ(ierr);
    ierr = DMGetCoarseDM(cdm, &cdm);CHKERRQ(ierr);
  }
  ierr = PetscFEDestroy(&fe);CHKERRQ(ierr);
  ierr = PetscFEDestroy(&feAux);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#include "petsc/private/petscimpl.h"

/*@C
  KSPMonitorError - Outputs the error at each iteration of an iterative solver.

  Collective on KSP

  Input Parameters:
+ ksp   - the KSP
. its   - iteration number
. rnorm - 2-norm, preconditioned residual value (may be estimated).
- ctx   - monitor context

  Level: intermediate

.seealso: KSPMonitorSet(), KSPMonitorTrueResidualNorm(), KSPMonitorDefault()
@*/
static PetscErrorCode KSPMonitorError(KSP ksp, PetscInt its, PetscReal rnorm, void *ctx)
{
  AppCtx        *user = (AppCtx *) ctx;
  DM             dm;
  Vec            du = NULL, r;
  PetscInt       level = 0;
  PetscBool      hasLevel;
#if defined(PETSC_HAVE_HDF5)
  PetscViewer    viewer;
  char           buf[256];
#endif
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = KSPGetDM(ksp, &dm);CHKERRQ(ierr);
  /* Calculate solution */
  {
    PC        pc = user->pcmg; /* The MG PC */
    DM        fdm = NULL,  cdm = NULL;
    KSP       fksp, cksp;
    Vec       fu,   cu = NULL;
    PetscInt  levels, l;

    ierr = KSPBuildSolution(ksp, NULL, &du);CHKERRQ(ierr);
    ierr = PetscObjectComposedDataGetInt((PetscObject) ksp, PetscMGLevelId, level, hasLevel);CHKERRQ(ierr);
    ierr = PCMGGetLevels(pc, &levels);CHKERRQ(ierr);
    ierr = PCMGGetSmoother(pc, levels-1, &fksp);CHKERRQ(ierr);
    ierr = KSPBuildSolution(fksp, NULL, &fu);CHKERRQ(ierr);
    for (l = levels-1; l > level; --l) {
      Mat R;
      Vec s;

      ierr = PCMGGetSmoother(pc, l-1, &cksp);CHKERRQ(ierr);
      ierr = KSPGetDM(cksp, &cdm);CHKERRQ(ierr);
      ierr = DMGetGlobalVector(cdm, &cu);CHKERRQ(ierr);
      ierr = PCMGGetRestriction(pc, l, &R);CHKERRQ(ierr);
      ierr = PCMGGetRScale(pc, l, &s);CHKERRQ(ierr);
      ierr = MatRestrict(R, fu, cu);CHKERRQ(ierr);
      ierr = VecPointwiseMult(cu, cu, s);CHKERRQ(ierr);
      if (l < levels-1) {ierr = DMRestoreGlobalVector(fdm, &fu);CHKERRQ(ierr);}
      fdm  = cdm;
      fu   = cu;
    }
    if (levels-1 > level) {
      ierr = VecAXPY(du, 1.0, cu);CHKERRQ(ierr);
      ierr = DMRestoreGlobalVector(cdm, &cu);CHKERRQ(ierr);
    }
  }
  /* Calculate error */
  ierr = DMGetGlobalVector(dm, &r);CHKERRQ(ierr);
  ierr = DMProjectFunction(dm, 0.0, user->exactFuncs, NULL, INSERT_ALL_VALUES, r);CHKERRQ(ierr);
  ierr = VecAXPY(r,-1.0,du);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) r, "solution error");CHKERRQ(ierr);
  /* View error */
#if defined(PETSC_HAVE_HDF5)
  ierr = PetscSNPrintf(buf, 256, "ex12-%D.h5", level);CHKERRQ(ierr);
  ierr = PetscViewerHDF5Open(PETSC_COMM_WORLD, buf, FILE_MODE_APPEND, &viewer);CHKERRQ(ierr);
  ierr = VecView(r, viewer);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
#endif
  ierr = DMRestoreGlobalVector(dm, &r);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
  SNESMonitorError - Outputs the error at each iteration of an iterative solver.

  Collective on SNES

  Input Parameters:
+ snes  - the SNES
. its   - iteration number
. rnorm - 2-norm of residual
- ctx   - user context

  Level: intermediate

.seealso: SNESMonitorDefault(), SNESMonitorSet(), SNESMonitorSolution()
@*/
static PetscErrorCode SNESMonitorError(SNES snes, PetscInt its, PetscReal rnorm, void *ctx)
{
  AppCtx        *user = (AppCtx *) ctx;
  DM             dm;
  Vec            u, r;
  PetscInt       level = -1;
  PetscBool      hasLevel;
#if defined(PETSC_HAVE_HDF5)
  PetscViewer    viewer;
#endif
  char           buf[256];
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = SNESGetDM(snes, &dm);CHKERRQ(ierr);
  /* Calculate error */
  ierr = SNESGetSolution(snes, &u);CHKERRQ(ierr);
  ierr = DMGetGlobalVector(dm, &r);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) r, "solution error");CHKERRQ(ierr);
  ierr = DMProjectFunction(dm, 0.0, user->exactFuncs, NULL, INSERT_ALL_VALUES, r);CHKERRQ(ierr);
  ierr = VecAXPY(r, -1.0, u);CHKERRQ(ierr);
  /* View error */
  ierr = PetscObjectComposedDataGetInt((PetscObject) snes, PetscMGLevelId, level, hasLevel);CHKERRQ(ierr);
  ierr = PetscSNPrintf(buf, 256, "ex12-%D.h5", level);CHKERRQ(ierr);
#if defined(PETSC_HAVE_HDF5)
  ierr = PetscViewerHDF5Open(PETSC_COMM_WORLD, buf, FILE_MODE_APPEND, &viewer);CHKERRQ(ierr);
  ierr = VecView(r, viewer);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
  /* Cleanup */
  ierr = DMRestoreGlobalVector(dm, &r);CHKERRQ(ierr);
  PetscFunctionReturn(0);
#else
  SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP,"You need to configure with --download-hdf5");
#endif
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
  PetscBool      isFAS;
  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc, &argv, NULL,help);if (ierr) return ierr;
  ierr = ProcessOptions(PETSC_COMM_WORLD, &user);CHKERRQ(ierr);
  ierr = SNESCreate(PETSC_COMM_WORLD, &snes);CHKERRQ(ierr);
  ierr = CreateMesh(PETSC_COMM_WORLD, &user, &dm);CHKERRQ(ierr);
  ierr = SNESSetDM(snes, dm);CHKERRQ(ierr);
  ierr = DMSetApplicationContext(dm, &user);CHKERRQ(ierr);

  ierr = PetscMalloc2(1, &user.exactFuncs, 1, &user.exactFields);CHKERRQ(ierr);
  ierr = SetupDiscretization(dm, &user);CHKERRQ(ierr);

  ierr = DMCreateGlobalVector(dm, &u);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) u, "potential");CHKERRQ(ierr);

  ierr = DMCreateMatrix(dm, &J);CHKERRQ(ierr);
  if (user.jacobianMF) {
    PetscInt M, m, N, n;

    ierr = MatGetSize(J, &M, &N);CHKERRQ(ierr);
    ierr = MatGetLocalSize(J, &m, &n);CHKERRQ(ierr);
    ierr = MatCreate(PETSC_COMM_WORLD, &A);CHKERRQ(ierr);
    ierr = MatSetSizes(A, m, n, M, N);CHKERRQ(ierr);
    ierr = MatSetType(A, MATSHELL);CHKERRQ(ierr);
    ierr = MatSetUp(A);CHKERRQ(ierr);
#if 0
    ierr = MatShellSetOperation(A, MATOP_MULT, (void (*)(void))FormJacobianAction);CHKERRQ(ierr);
#endif

    userJ.dm   = dm;
    userJ.J    = J;
    userJ.user = &user;

    ierr = DMCreateLocalVector(dm, &userJ.u);CHKERRQ(ierr);
    if (user.fieldBC) {ierr = DMProjectFieldLocal(dm, 0.0, userJ.u, user.exactFields, INSERT_BC_VALUES, userJ.u);CHKERRQ(ierr);}
    else              {ierr = DMProjectFunctionLocal(dm, 0.0, user.exactFuncs, NULL, INSERT_BC_VALUES, userJ.u);CHKERRQ(ierr);}
    ierr = MatShellSetContext(A, &userJ);CHKERRQ(ierr);
  } else {
    A = J;
  }

  nullSpace = NULL;
  if (user.bcType != DIRICHLET) {
    ierr = MatNullSpaceCreate(PetscObjectComm((PetscObject) dm), PETSC_TRUE, 0, NULL, &nullSpace);CHKERRQ(ierr);
    ierr = MatSetNullSpace(A, nullSpace);CHKERRQ(ierr);
  }

  ierr = DMPlexSetSNESLocalFEM(dm,&user,&user,&user);CHKERRQ(ierr);
  ierr = SNESSetJacobian(snes, A, J, NULL, NULL);CHKERRQ(ierr);

  ierr = SNESSetFromOptions(snes);CHKERRQ(ierr);

  if (user.fieldBC) {ierr = DMProjectField(dm, 0.0, u, user.exactFields, INSERT_ALL_VALUES, u);CHKERRQ(ierr);}
  else              {ierr = DMProjectFunction(dm, 0.0, user.exactFuncs, NULL, INSERT_ALL_VALUES, u);CHKERRQ(ierr);}
  if (user.restart) {
#if defined(PETSC_HAVE_HDF5)
    PetscViewer viewer;

    ierr = PetscViewerCreate(PETSC_COMM_WORLD, &viewer);CHKERRQ(ierr);
    ierr = PetscViewerSetType(viewer, PETSCVIEWERHDF5);CHKERRQ(ierr);
    ierr = PetscViewerFileSetMode(viewer, FILE_MODE_READ);CHKERRQ(ierr);
    ierr = PetscViewerFileSetName(viewer, user.filename);CHKERRQ(ierr);
    ierr = PetscViewerHDF5PushGroup(viewer, "/fields");CHKERRQ(ierr);
    ierr = VecLoad(u, viewer);CHKERRQ(ierr);
    ierr = PetscViewerHDF5PopGroup(viewer);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
#endif
  }
  if (user.showInitial) {
    Vec lv;
    ierr = DMGetLocalVector(dm, &lv);CHKERRQ(ierr);
    ierr = DMGlobalToLocalBegin(dm, u, INSERT_VALUES, lv);CHKERRQ(ierr);
    ierr = DMGlobalToLocalEnd(dm, u, INSERT_VALUES, lv);CHKERRQ(ierr);
    ierr = DMPrintLocalVec(dm, "Local function", 1.0e-10, lv);CHKERRQ(ierr);
    ierr = DMRestoreLocalVector(dm, &lv);CHKERRQ(ierr);
  }
  if (user.viewHierarchy) {
    SNES      lsnes;
    KSP       ksp;
    PC        pc;
    PetscInt  numLevels, l;
    PetscBool isMG;

    ierr = PetscObjectTypeCompare((PetscObject) snes, SNESFAS, &isFAS);CHKERRQ(ierr);
    if (isFAS) {
      ierr = SNESFASGetLevels(snes, &numLevels);CHKERRQ(ierr);
      for (l = 0; l < numLevels; ++l) {
        ierr = SNESFASGetCycleSNES(snes, l, &lsnes);CHKERRQ(ierr);
        ierr = SNESMonitorSet(lsnes, SNESMonitorError, &user, NULL);CHKERRQ(ierr);
      }
    } else {
      ierr = SNESGetKSP(snes, &ksp);CHKERRQ(ierr);
      ierr = KSPGetPC(ksp, &pc);CHKERRQ(ierr);
      ierr = PetscObjectTypeCompare((PetscObject) pc, PCMG, &isMG);CHKERRQ(ierr);
      if (isMG) {
        user.pcmg = pc;
        ierr = PCMGGetLevels(pc, &numLevels);CHKERRQ(ierr);
        for (l = 0; l < numLevels; ++l) {
          ierr = PCMGGetSmootherDown(pc, l, &ksp);CHKERRQ(ierr);
          ierr = KSPMonitorSet(ksp, KSPMonitorError, &user, NULL);CHKERRQ(ierr);
        }
      }
    }
  }
  if (user.runType == RUN_FULL || user.runType == RUN_EXACT) {
    PetscErrorCode (*initialGuess[1])(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nc, PetscScalar u[], void *ctx) = {zero};

    if (user.nonzInit) initialGuess[0] = ecks;
    if (user.runType == RUN_FULL) {
      ierr = DMProjectFunction(dm, 0.0, initialGuess, NULL, INSERT_VALUES, u);CHKERRQ(ierr);
    }
    if (user.debug) {
      ierr = PetscPrintf(PETSC_COMM_WORLD, "Initial guess\n");CHKERRQ(ierr);
      ierr = VecView(u, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
    }
    ierr = VecViewFromOptions(u, NULL, "-guess_vec_view");CHKERRQ(ierr);
    ierr = SNESSolve(snes, NULL, u);CHKERRQ(ierr);
    ierr = SNESGetSolution(snes, &u);CHKERRQ(ierr);
    ierr = SNESGetDM(snes, &dm);CHKERRQ(ierr);

    if (user.showSolution) {
      ierr = PetscPrintf(PETSC_COMM_WORLD, "Solution\n");CHKERRQ(ierr);
      ierr = VecChop(u, 3.0e-9);CHKERRQ(ierr);
      ierr = VecView(u, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
    }
    ierr = VecViewFromOptions(u, NULL, "-vec_view");CHKERRQ(ierr);
  } else if (user.runType == RUN_PERF) {
    Vec       r;
    PetscReal res = 0.0;

    ierr = SNESGetFunction(snes, &r, NULL, NULL);CHKERRQ(ierr);
    ierr = SNESComputeFunction(snes, u, r);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD, "Initial Residual\n");CHKERRQ(ierr);
    ierr = VecChop(r, 1.0e-10);CHKERRQ(ierr);
    ierr = VecNorm(r, NORM_2, &res);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD, "L_2 Residual: %g\n", (double)res);CHKERRQ(ierr);
  } else {
    Vec       r;
    PetscReal res = 0.0, tol = 1.0e-11;

    /* Check discretization error */
    ierr = SNESGetFunction(snes, &r, NULL, NULL);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD, "Initial guess\n");CHKERRQ(ierr);
    if (!user.quiet) {ierr = VecView(u, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);}
    ierr = DMComputeL2Diff(dm, 0.0, user.exactFuncs, NULL, u, &error);CHKERRQ(ierr);
    if (error < tol) {ierr = PetscPrintf(PETSC_COMM_WORLD, "L_2 Error: < %2.1e\n", (double)tol);CHKERRQ(ierr);}
    else             {ierr = PetscPrintf(PETSC_COMM_WORLD, "L_2 Error: %g\n", (double)error);CHKERRQ(ierr);}
    /* Check residual */
    ierr = SNESComputeFunction(snes, u, r);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD, "Initial Residual\n");CHKERRQ(ierr);
    ierr = VecChop(r, 1.0e-10);CHKERRQ(ierr);
    if (!user.quiet) {ierr = VecView(r, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);}
    ierr = VecNorm(r, NORM_2, &res);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD, "L_2 Residual: %g\n", (double)res);CHKERRQ(ierr);
    /* Check Jacobian */
    {
      Vec b;

      ierr = SNESComputeJacobian(snes, u, A, A);CHKERRQ(ierr);
      ierr = VecDuplicate(u, &b);CHKERRQ(ierr);
      ierr = VecSet(r, 0.0);CHKERRQ(ierr);
      ierr = SNESComputeFunction(snes, r, b);CHKERRQ(ierr);
      ierr = MatMult(A, u, r);CHKERRQ(ierr);
      ierr = VecAXPY(r, 1.0, b);CHKERRQ(ierr);
      ierr = PetscPrintf(PETSC_COMM_WORLD, "Au - b = Au + F(0)\n");CHKERRQ(ierr);
      ierr = VecChop(r, 1.0e-10);CHKERRQ(ierr);
      if (!user.quiet) {ierr = VecView(r, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);}
      ierr = VecNorm(r, NORM_2, &res);CHKERRQ(ierr);
      ierr = PetscPrintf(PETSC_COMM_WORLD, "Linear L_2 Residual: %g\n", (double)res);CHKERRQ(ierr);
      /* check solver */
      if (user.checkksp) {
        KSP ksp;

        if (nullSpace) {
          ierr = MatNullSpaceRemove(nullSpace, u);CHKERRQ(ierr);
        }
        ierr = SNESComputeJacobian(snes, u, A, J);CHKERRQ(ierr);
        ierr = MatMult(A, u, b);CHKERRQ(ierr);
        ierr = SNESGetKSP(snes, &ksp);CHKERRQ(ierr);
        ierr = KSPSetOperators(ksp, A, J);CHKERRQ(ierr);
        ierr = KSPSolve(ksp, b, r);CHKERRQ(ierr);
        ierr = VecAXPY(r, -1.0, u);CHKERRQ(ierr);
        ierr = VecNorm(r, NORM_2, &res);CHKERRQ(ierr);
        ierr = PetscPrintf(PETSC_COMM_WORLD, "KSP Error: %g\n", (double)res);CHKERRQ(ierr);
      }
      ierr = VecDestroy(&b);CHKERRQ(ierr);
    }
  }
  ierr = VecViewFromOptions(u, NULL, "-vec_view");CHKERRQ(ierr);

  if (user.bdIntegral) {
    DMLabel   label;
    PetscInt  id = 1;
    PetscScalar bdInt = 0.0;
    PetscReal   exact = 3.3333333333;

    ierr = DMGetLabel(dm, "marker", &label);CHKERRQ(ierr);
    ierr = DMPlexComputeBdIntegral(dm, u, label, 1, &id, bd_integral_2d, &bdInt, NULL);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD, "Solution boundary integral: %.4g\n", (double) PetscAbsScalar(bdInt));CHKERRQ(ierr);
    if (PetscAbsReal(PetscAbsScalar(bdInt) - exact) > PETSC_SQRT_MACHINE_EPSILON) SETERRQ2(PETSC_COMM_WORLD, PETSC_ERR_PLIB, "Invalid boundary integral %g != %g", (double) PetscAbsScalar(bdInt), (double)exact);
  }

  ierr = MatNullSpaceDestroy(&nullSpace);CHKERRQ(ierr);
  if (user.jacobianMF) {ierr = VecDestroy(&userJ.u);CHKERRQ(ierr);}
  if (A != J) {ierr = MatDestroy(&A);CHKERRQ(ierr);}
  ierr = MatDestroy(&J);CHKERRQ(ierr);
  ierr = VecDestroy(&u);CHKERRQ(ierr);
  ierr = SNESDestroy(&snes);CHKERRQ(ierr);
  ierr = DMDestroy(&dm);CHKERRQ(ierr);
  ierr = PetscFree2(user.exactFuncs, user.exactFields);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

/*TEST
  # 2D serial P1 test 0-4
  test:
    suffix: 2d_p1_0
    requires: triangle
    args: -run_type test -refinement_limit 0.0    -bc_type dirichlet -interpolate 0 -petscspace_degree 1 -show_initial -dm_plex_print_fem 1

  test:
    suffix: 2d_p1_1
    requires: triangle
    args: -run_type test -refinement_limit 0.0    -bc_type dirichlet -interpolate 1 -petscspace_degree 1 -show_initial -dm_plex_print_fem 1

  test:
    suffix: 2d_p1_2
    requires: triangle
    args: -run_type test -refinement_limit 0.0625 -bc_type dirichlet -interpolate 1 -petscspace_degree 1 -show_initial -dm_plex_print_fem 1

  test:
    suffix: 2d_p1_neumann_0
    requires: triangle
    args: -run_type test -refinement_limit 0.0    -bc_type neumann   -interpolate 1 -petscspace_degree 1 -show_initial -dm_plex_print_fem 1 -dm_view ascii::ascii_info_detail

  test:
    suffix: 2d_p1_neumann_1
    requires: triangle
    args: -run_type test -refinement_limit 0.0625 -bc_type neumann   -interpolate 1 -petscspace_degree 1 -show_initial -dm_plex_print_fem 1

  # 2D serial P2 test 5-8
  test:
    suffix: 2d_p2_0
    requires: triangle
    args: -run_type test -refinement_limit 0.0    -bc_type dirichlet -interpolate 1 -petscspace_degree 2 -show_initial -dm_plex_print_fem 1

  test:
    suffix: 2d_p2_1
    requires: triangle
    args: -run_type test -refinement_limit 0.0625 -bc_type dirichlet -interpolate 1 -petscspace_degree 2 -show_initial -dm_plex_print_fem 1

  test:
    suffix: 2d_p2_neumann_0
    requires: triangle
    args: -run_type test -refinement_limit 0.0    -bc_type neumann   -interpolate 1 -petscspace_degree 2 -show_initial -dm_plex_print_fem 1 -dm_view ascii::ascii_info_detail

  test:
    suffix: 2d_p2_neumann_1
    requires: triangle
    args: -run_type test -refinement_limit 0.0625 -bc_type neumann   -interpolate 1 -petscspace_degree 2 -show_initial -dm_plex_print_fem 1 -dm_view ascii::ascii_info_detail

  test:
    suffix: bd_int_0
    requires: triangle
    args: -run_type test -bc_type dirichlet -interpolate 1 -petscspace_degree 2 -bd_integral -dm_view -quiet

  test:
    suffix: bd_int_1
    requires: triangle
    args: -run_type test -dm_refine 2 -bc_type dirichlet -interpolate 1 -petscspace_degree 2 -bd_integral -dm_view -quiet

  # 3D serial P1 test 9-12
  test:
    suffix: 3d_p1_0
    requires: ctetgen
    args: -run_type test -dim 3 -refinement_limit 0.0    -bc_type dirichlet -interpolate 0 -petscspace_degree 1 -show_initial -dm_plex_print_fem 1 -dm_view -cells 1,1,1

  test:
    suffix: 3d_p1_1
    requires: ctetgen
    args: -run_type test -dim 3 -refinement_limit 0.0    -bc_type dirichlet -interpolate 1 -petscspace_degree 1 -show_initial -dm_plex_print_fem 1 -dm_view -cells 1,1,1

  test:
    suffix: 3d_p1_2
    requires: ctetgen
    args: -run_type test -dim 3 -refinement_limit 0.0125 -bc_type dirichlet -interpolate 1 -petscspace_degree 1 -show_initial -dm_plex_print_fem 1 -dm_view -cells 1,1,1

  test:
    suffix: 3d_p1_neumann_0
    requires: ctetgen
    args: -run_type test -dim 3 -bc_type neumann   -interpolate 1 -petscspace_degree 1 -snes_fd -show_initial -dm_plex_print_fem 1 -dm_view -cells 1,1,1

  # Analytic variable coefficient 13-20
  test:
    suffix: 13
    requires: triangle
    args: -run_type test -refinement_limit 0.0    -variable_coefficient analytic -interpolate 1 -petscspace_degree 1 -show_initial -dm_plex_print_fem 1
  test:
    suffix: 14
    requires: triangle
    args: -run_type test -refinement_limit 0.0625 -variable_coefficient analytic -interpolate 1 -petscspace_degree 1 -show_initial -dm_plex_print_fem 1
  test:
    suffix: 15
    requires: triangle
    args: -run_type test -refinement_limit 0.0    -variable_coefficient analytic -interpolate 1 -petscspace_degree 2 -show_initial -dm_plex_print_fem 1
  test:
    suffix: 16
    requires: triangle
    args: -run_type test -refinement_limit 0.0625 -variable_coefficient analytic -interpolate 1 -petscspace_degree 2 -show_initial -dm_plex_print_fem 1
  test:
    suffix: 17
    requires: ctetgen
    args: -run_type test -dim 3 -refinement_limit 0.0    -variable_coefficient analytic -interpolate 1 -petscspace_degree 1 -show_initial -dm_plex_print_fem 1 -cells 1,1,1

  test:
    suffix: 18
    requires: ctetgen
    args: -run_type test -dim 3 -refinement_limit 0.0125 -variable_coefficient analytic -interpolate 1 -petscspace_degree 1 -show_initial -dm_plex_print_fem 1 -cells 1,1,1

  test:
    suffix: 19
    requires: ctetgen
    args: -run_type test -dim 3 -refinement_limit 0.0    -variable_coefficient analytic -interpolate 1 -petscspace_degree 2 -show_initial -dm_plex_print_fem 1 -cells 1,1,1

  test:
    suffix: 20
    requires: ctetgen
    args: -run_type test -dim 3 -refinement_limit 0.0125 -variable_coefficient analytic -interpolate 1 -petscspace_degree 2 -show_initial -dm_plex_print_fem 1 -cells 1,1,1

  # P1 variable coefficient 21-28
  test:
    suffix: 21
    requires: triangle
    args: -run_type test -refinement_limit 0.0    -variable_coefficient field    -interpolate 1 -petscspace_degree 1 -mat_petscspace_degree 1 -show_initial -dm_plex_print_fem 1

  test:
    suffix: 22
    requires: triangle
    args: -run_type test -refinement_limit 0.0625 -variable_coefficient field    -interpolate 1 -petscspace_degree 1 -mat_petscspace_degree 1 -show_initial -dm_plex_print_fem 1

  test:
    suffix: 23
    requires: triangle
    args: -run_type test -refinement_limit 0.0    -variable_coefficient field    -interpolate 1 -petscspace_degree 2 -mat_petscspace_degree 1 -show_initial -dm_plex_print_fem 1

  test:
    suffix: 24
    requires: triangle
    args: -run_type test -refinement_limit 0.0625 -variable_coefficient field    -interpolate 1 -petscspace_degree 2 -mat_petscspace_degree 1 -show_initial -dm_plex_print_fem 1

  test:
    suffix: 25
    requires: ctetgen
    args: -run_type test -dim 3 -refinement_limit 0.0    -variable_coefficient field    -interpolate 1 -petscspace_degree 1 -mat_petscspace_degree 1 -show_initial -dm_plex_print_fem 1 -cells 1,1,1

  test:
    suffix: 26
    requires: ctetgen
    args: -run_type test -dim 3 -refinement_limit 0.0125 -variable_coefficient field    -interpolate 1 -petscspace_degree 1 -mat_petscspace_degree 1 -show_initial -dm_plex_print_fem 1 -cells 1,1,1

  test:
    suffix: 27
    requires: ctetgen
    args: -run_type test -dim 3 -refinement_limit 0.0    -variable_coefficient field    -interpolate 1 -petscspace_degree 2 -mat_petscspace_degree 1 -show_initial -dm_plex_print_fem 1 -cells 1,1,1

  test:
    suffix: 28
    requires: ctetgen
    args: -run_type test -dim 3 -refinement_limit 0.0125 -variable_coefficient field    -interpolate 1 -petscspace_degree 2 -mat_petscspace_degree 1 -show_initial -dm_plex_print_fem 1 -cells 1,1,1

  # P0 variable coefficient 29-36
  test:
    suffix: 29
    requires: triangle
    args: -run_type test -refinement_limit 0.0    -variable_coefficient field    -interpolate 1 -petscspace_degree 1 -show_initial -dm_plex_print_fem 1

  test:
    suffix: 30
    requires: triangle
    args: -run_type test -refinement_limit 0.0625 -variable_coefficient field    -interpolate 1 -petscspace_degree 1 -show_initial -dm_plex_print_fem 1

  test:
    suffix: 31
    requires: triangle
    args: -run_type test -refinement_limit 0.0    -variable_coefficient field    -interpolate 1 -petscspace_degree 2 -show_initial -dm_plex_print_fem 1

  test:
    requires: triangle
    suffix: 32
    args: -run_type test -refinement_limit 0.0625 -variable_coefficient field    -interpolate 1 -petscspace_degree 2 -show_initial -dm_plex_print_fem 1

  test:
    requires: ctetgen
    suffix: 33
    args: -run_type test -dim 3 -refinement_limit 0.0    -variable_coefficient field    -interpolate 1 -petscspace_degree 1 -show_initial -dm_plex_print_fem 1 -cells 1,1,1

  test:
    suffix: 34
    requires: ctetgen
    args: -run_type test -dim 3 -refinement_limit 0.0125 -variable_coefficient field    -interpolate 1 -petscspace_degree 1 -show_initial -dm_plex_print_fem 1 -cells 1,1,1

  test:
    suffix: 35
    requires: ctetgen
    args: -run_type test -dim 3 -refinement_limit 0.0    -variable_coefficient field    -interpolate 1 -petscspace_degree 2 -show_initial -dm_plex_print_fem 1 -cells 1,1,1

  test:
    suffix: 36
    requires: ctetgen
    args: -run_type test -dim 3 -refinement_limit 0.0125 -variable_coefficient field    -interpolate 1 -petscspace_degree 2 -show_initial -dm_plex_print_fem 1 -cells 1,1,1

  # Full solve 39-44
  test:
    suffix: 39
    requires: triangle !single
    args: -run_type full -refinement_limit 0.015625 -interpolate 1 -petscspace_degree 2 -pc_type gamg -ksp_rtol 1.0e-10 -ksp_monitor_short -ksp_converged_reason -snes_monitor_short -snes_converged_reason ::ascii_info_detail
  test:
    suffix: 40
    requires: triangle !single
    args: -run_type full -refinement_limit 0.015625 -variable_coefficient nonlinear -interpolate 1 -petscspace_degree 2 -pc_type svd -ksp_rtol 1.0e-10 -snes_monitor_short -snes_converged_reason ::ascii_info_detail
  test:
    suffix: 41
    requires: triangle !single
    args: -run_type full -refinement_limit 0.03125 -variable_coefficient nonlinear -interpolate 1 -petscspace_degree 1 -snes_type fas -snes_fas_levels 2 -fas_coarse_pc_type svd -fas_coarse_ksp_rtol 1.0e-10 -fas_coarse_snes_monitor_short -snes_monitor_short -fas_coarse_snes_linesearch_type basic -snes_converged_reason ::ascii_info_detail -dm_refine_hierarchy 1 -snes_view -fas_levels_1_snes_type newtonls -fas_levels_1_pc_type svd -fas_levels_1_ksp_rtol 1.0e-10 -fas_levels_1_snes_monitor_short
  test:
    suffix: 42
    requires: triangle !single
    args: -run_type full -refinement_limit 0.0625 -variable_coefficient nonlinear -interpolate 1 -petscspace_degree 1 -snes_type fas -snes_fas_levels 3 -fas_coarse_pc_type svd -fas_coarse_ksp_rtol 1.0e-10 -fas_coarse_snes_monitor_short -snes_monitor_short -fas_coarse_snes_linesearch_type basic -snes_converged_reason ::ascii_info_detail -dm_refine_hierarchy 2 -dm_plex_print_fem 0 -snes_view -fas_levels_1_snes_type newtonls -fas_levels_1_pc_type svd -fas_levels_1_ksp_rtol 1.0e-10 -fas_levels_1_snes_monitor_short -fas_levels_2_snes_type newtonls -fas_levels_2_pc_type svd -fas_levels_2_ksp_rtol 1.0e-10 -fas_levels_2_snes_atol 1.0e-11 -fas_levels_2_snes_monitor_short
  test:
    suffix: 43
    requires: triangle !single
    nsize: 2
    args: -run_type full -refinement_limit 0.03125 -variable_coefficient nonlinear -interpolate 1 -petscspace_degree 1 -snes_type fas -snes_fas_levels 2 -fas_coarse_pc_type svd -fas_coarse_ksp_rtol 1.0e-10 -fas_coarse_snes_monitor_short -snes_monitor_short -fas_coarse_snes_linesearch_type basic -snes_converged_reason ::ascii_info_detail -dm_refine_hierarchy 1 -snes_view -fas_levels_1_snes_type newtonls -fas_levels_1_pc_type svd -fas_levels_1_ksp_rtol 1.0e-10 -fas_levels_1_snes_monitor_short

  test:
    suffix: 44
    requires: triangle !single
    nsize: 2
    args: -run_type full -refinement_limit 0.0625 -variable_coefficient nonlinear -interpolate 1 -petscspace_degree 1 -snes_type fas -snes_fas_levels 3 -fas_coarse_pc_type svd -fas_coarse_ksp_rtol 1.0e-10 -fas_coarse_snes_monitor_short -snes_monitor_short  -fas_coarse_snes_linesearch_type basic -snes_converged_reason ::ascii_info_detail -dm_refine_hierarchy 2 -dm_plex_print_fem 0 -snes_view -fas_levels_1_snes_type newtonls -fas_levels_1_pc_type svd -fas_levels_1_ksp_rtol 1.0e-10 -fas_levels_1_snes_monitor_short -fas_levels_2_snes_type newtonls -fas_levels_2_pc_type svd -fas_levels_2_ksp_rtol 1.0e-10 -fas_levels_2_snes_atol 1.0e-11 -fas_levels_2_snes_monitor_short

  # These tests use a loose tolerance just to exercise the PtAP operations for MATIS and multiple PCBDDC setup calls inside PCMG
  testset:
    requires: triangle !single
    nsize: 3
    args: -interpolate -run_type full -petscspace_degree 1 -dm_mat_type is -pc_type mg -pc_mg_levels 2 -mg_coarse_pc_type bddc -pc_mg_galerkin pmat -ksp_rtol 1.0e-2 -snes_converged_reason -dm_refine_hierarchy 2 -snes_max_it 4
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
    args: -run_type test -refinement_limit 0.0    -bc_type dirichlet -interpolate 1 -petscspace_degree 1
    test:
      args: -dm_view hdf5:sol.h5 -vec_view hdf5:sol.h5::append
    test:
      args: -f sol.h5 -restart

  # Periodicity
  test:
    suffix: periodic_0
    requires: triangle
    args: -run_type full -refinement_limit 0.0    -bc_type dirichlet -interpolate 1 -petscspace_degree 1 -snes_converged_reason ::ascii_info_detail

  test:
    requires: !complex
    suffix: periodic_1
    args: -quiet -run_type test -simplex 0 -x_periodicity periodic -y_periodicity periodic -vec_view vtk:test.vtu:vtk_vtu -interpolate 1 -petscspace_degree 1 -dm_refine 1

  # 2D serial P1 test with field bc
  test:
    suffix: field_bc_2d_p1_0
    requires: triangle
    args: -run_type test              -interpolate 1 -bc_type dirichlet -field_bc -petscspace_degree 1 -bc_petscspace_degree 2 -show_initial -dm_plex_print_fem 1

  test:
    suffix: field_bc_2d_p1_1
    requires: triangle
    args: -run_type test -dm_refine 1 -interpolate 1 -bc_type dirichlet -field_bc -petscspace_degree 1 -bc_petscspace_degree 2 -show_initial -dm_plex_print_fem 1

  test:
    suffix: field_bc_2d_p1_neumann_0
    requires: triangle
    args: -run_type test              -interpolate 1 -bc_type neumann   -field_bc -petscspace_degree 1 -bc_petscspace_degree 2 -show_initial -dm_plex_print_fem 1

  test:
    suffix: field_bc_2d_p1_neumann_1
    requires: triangle
    args: -run_type test -dm_refine 1 -interpolate 1 -bc_type neumann   -field_bc -petscspace_degree 1 -bc_petscspace_degree 2 -show_initial -dm_plex_print_fem 1

  # 3D serial P1 test with field bc
  test:
    suffix: field_bc_3d_p1_0
    requires: ctetgen
    args: -run_type test -dim 3              -interpolate 1 -bc_type dirichlet -field_bc -petscspace_degree 1 -bc_petscspace_degree 2 -show_initial -dm_plex_print_fem 1 -cells 1,1,1

  test:
    suffix: field_bc_3d_p1_1
    requires: ctetgen
    args: -run_type test -dim 3 -dm_refine 1 -interpolate 1 -bc_type dirichlet -field_bc -petscspace_degree 1 -bc_petscspace_degree 2 -show_initial -dm_plex_print_fem 1 -cells 1,1,1

  test:
    suffix: field_bc_3d_p1_neumann_0
    requires: ctetgen
    args: -run_type test -dim 3              -interpolate 1 -bc_type neumann   -field_bc -petscspace_degree 1 -bc_petscspace_degree 2 -show_initial -dm_plex_print_fem 1 -cells 1,1,1

  test:
    suffix: field_bc_3d_p1_neumann_1
    requires: ctetgen
    args: -run_type test -dim 3 -dm_refine 1 -interpolate 1 -bc_type neumann   -field_bc -petscspace_degree 1 -bc_petscspace_degree 2 -show_initial -dm_plex_print_fem 1 -cells 1,1,1

  # 2D serial P2 test with field bc
  test:
    suffix: field_bc_2d_p2_0
    requires: triangle
    args: -run_type test              -interpolate 1 -bc_type dirichlet -field_bc -petscspace_degree 2 -bc_petscspace_degree 2 -show_initial -dm_plex_print_fem 1

  test:
    suffix: field_bc_2d_p2_1
    requires: triangle
    args: -run_type test -dm_refine 1 -interpolate 1 -bc_type dirichlet -field_bc -petscspace_degree 2 -bc_petscspace_degree 2 -show_initial -dm_plex_print_fem 1

  test:
    suffix: field_bc_2d_p2_neumann_0
    requires: triangle
    args: -run_type test              -interpolate 1 -bc_type neumann   -field_bc -petscspace_degree 2 -bc_petscspace_degree 2 -show_initial -dm_plex_print_fem 1

  test:
    suffix: field_bc_2d_p2_neumann_1
    requires: triangle
    args: -run_type test -dm_refine 1 -interpolate 1 -bc_type neumann   -field_bc -petscspace_degree 2 -bc_petscspace_degree 2 -show_initial -dm_plex_print_fem 1

  # 3D serial P2 test with field bc
  test:
    suffix: field_bc_3d_p2_0
    requires: ctetgen
    args: -run_type test -dim 3              -interpolate 1 -bc_type dirichlet -field_bc -petscspace_degree 2 -bc_petscspace_degree 2 -show_initial -dm_plex_print_fem 1 -cells 1,1,1

  test:
    suffix: field_bc_3d_p2_1
    requires: ctetgen
    args: -run_type test -dim 3 -dm_refine 1 -interpolate 1 -bc_type dirichlet -field_bc -petscspace_degree 2 -bc_petscspace_degree 2 -show_initial -dm_plex_print_fem 1 -cells 1,1,1

  test:
    suffix: field_bc_3d_p2_neumann_0
    requires: ctetgen
    args: -run_type test -dim 3              -interpolate 1 -bc_type neumann   -field_bc -petscspace_degree 2 -bc_petscspace_degree 2 -show_initial -dm_plex_print_fem 1 -cells 1,1,1

  test:
    suffix: field_bc_3d_p2_neumann_1
    requires: ctetgen
    args: -run_type test -dim 3 -dm_refine 1 -interpolate 1 -bc_type neumann   -field_bc -petscspace_degree 2 -bc_petscspace_degree 2 -show_initial -dm_plex_print_fem 1 -cells 1,1,1

  # Full solve simplex: Convergence
  test:
    suffix: tet_conv_p1_r0
    requires: ctetgen
    args: -run_type full -dim 3 -dm_refine 0 -bc_type dirichlet -interpolate 1 -petscspace_degree 1 -dm_view -snes_converged_reason ::ascii_info_detail -pc_type lu -cells 1,1,1
  test:
    suffix: tet_conv_p1_r2
    requires: ctetgen
    args: -run_type full -dim 3 -dm_refine 2 -bc_type dirichlet -interpolate 1 -petscspace_degree 1 -dm_view -snes_converged_reason ::ascii_info_detail -pc_type lu -cells 1,1,1
  test:
    suffix: tet_conv_p1_r3
    requires: ctetgen
    args: -run_type full -dim 3 -dm_refine 3 -bc_type dirichlet -interpolate 1 -petscspace_degree 1 -dm_view -snes_converged_reason ::ascii_info_detail -pc_type lu -cells 1,1,1
  test:
    suffix: tet_conv_p2_r0
    requires: ctetgen
    args: -run_type full -dim 3 -dm_refine 0 -bc_type dirichlet -interpolate 1 -petscspace_degree 2 -dm_view -snes_converged_reason ::ascii_info_detail -pc_type lu -cells 1,1,1
  test:
    suffix: tet_conv_p2_r2
    requires: ctetgen
    args: -run_type full -dim 3 -dm_refine 2 -bc_type dirichlet -interpolate 1 -petscspace_degree 2 -dm_view -snes_converged_reason ::ascii_info_detail -pc_type lu -cells 1,1,1

  # Full solve simplex: PCBDDC
  test:
    suffix: tri_bddc
    requires: triangle !single
    nsize: 5
    args: -run_type full -petscpartitioner_type simple -dm_refine 2 -bc_type dirichlet -interpolate 1 -petscspace_degree 1 -ksp_type gmres -ksp_gmres_restart 100 -ksp_rtol 1.0e-9 -dm_mat_type is -pc_type bddc -snes_monitor_short -ksp_monitor_short -snes_converged_reason ::ascii_info_detail -ksp_converged_reason -snes_view -show_solution 0

  # Full solve simplex: PCBDDC
  test:
    suffix: tri_parmetis_bddc
    requires: triangle !single parmetis
    nsize: 4
    args: -run_type full -petscpartitioner_type parmetis -dm_refine 2 -bc_type dirichlet -interpolate 1 -petscspace_degree 1 -ksp_type gmres -ksp_gmres_restart 100 -ksp_rtol 1.0e-9 -dm_mat_type is -pc_type bddc -snes_monitor_short -ksp_monitor_short -snes_converged_reason ::ascii_info_detail -ksp_converged_reason -snes_view -show_solution 0

  testset:
    args: -run_type full -petscpartitioner_type simple -dm_refine 2 -bc_type dirichlet -interpolate 1 -petscspace_degree 2 -dm_mat_type is -pc_type bddc -ksp_type gmres -snes_monitor_short -ksp_monitor_short -snes_view -simplex 0 -petscspace_poly_tensor -pc_bddc_corner_selection -cells 3,3 -ksp_rtol 1.e-9 -pc_bddc_use_edges 0
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
    args: -run_type full -dm_refine 3 -bc_type dirichlet -interpolate 1 -petscspace_degree 1 -ksp_type gmres -ksp_gmres_restart 100 -ksp_rtol 1.0e-9 -pc_type asm -pc_asm_type restrict -pc_asm_blocks 4 -sub_pc_type lu -snes_monitor_short -ksp_monitor_short -snes_converged_reason ::ascii_info_detail -ksp_converged_reason -snes_view -show_solution 0

  test:
    suffix: tri_q2q1_msm_lu
    requires: triangle !single
    args: -run_type full -dm_refine 3 -bc_type dirichlet -interpolate 1 -petscspace_degree 1 -ksp_type gmres -ksp_gmres_restart 100 -ksp_rtol 1.0e-9 -pc_type asm -pc_asm_type restrict -pc_asm_local_type multiplicative -pc_asm_blocks 4 -sub_pc_type lu -snes_monitor_short -ksp_monitor_short -snes_converged_reason ::ascii_info_detail -ksp_converged_reason -snes_view -show_solution 0

  test:
    suffix: tri_q2q1_asm_sor
    requires: triangle !single
    args: -run_type full -dm_refine 3 -bc_type dirichlet -interpolate 1 -petscspace_degree 1 -ksp_type gmres -ksp_gmres_restart 100 -ksp_rtol 1.0e-9 -pc_type asm -pc_asm_type restrict -pc_asm_blocks 4 -sub_pc_type sor -snes_monitor_short -ksp_monitor_short -snes_converged_reason ::ascii_info_detail -ksp_converged_reason -snes_view -show_solution 0

  test:
    suffix: tri_q2q1_msm_sor
    requires: triangle !single
    args: -run_type full -dm_refine 3 -bc_type dirichlet -interpolate 1 -petscspace_degree 1 -ksp_type gmres -ksp_gmres_restart 100 -ksp_rtol 1.0e-9 -pc_type asm -pc_asm_type restrict -pc_asm_local_type multiplicative -pc_asm_blocks 4 -sub_pc_type sor -snes_monitor_short -ksp_monitor_short -snes_converged_reason ::ascii_info_detail -ksp_converged_reason -snes_view -show_solution 0

  # Full solve simplex: FAS
  test:
    suffix: fas_newton_0
    requires: triangle !single
    args: -run_type full -variable_coefficient nonlinear -interpolate 1 -petscspace_degree 1 -snes_type fas -snes_fas_levels 2 -fas_coarse_pc_type svd -fas_coarse_ksp_rtol 1.0e-10 -fas_coarse_snes_monitor_short -snes_monitor_short -fas_coarse_snes_linesearch_type basic -snes_converged_reason ::ascii_info_detail -dm_refine_hierarchy 1 -snes_view -fas_levels_1_snes_type newtonls -fas_levels_1_pc_type svd -fas_levels_1_ksp_rtol 1.0e-10 -fas_levels_1_snes_monitor_short

  test:
    suffix: fas_newton_1
    requires: triangle !single
    args: -run_type full -dm_refine_hierarchy 3 -interpolate 1 -petscspace_degree 1 -snes_type fas -snes_fas_levels 3 -fas_coarse_pc_type lu -fas_coarse_snes_monitor_short -snes_monitor_short -fas_coarse_snes_linesearch_type basic -snes_converged_reason ::ascii_info_detail -snes_view -fas_levels_snes_type newtonls -fas_levels_snes_linesearch_type basic -fas_levels_ksp_rtol 1.0e-10 -fas_levels_snes_monitor_short

  test:
    suffix: fas_ngs_0
    requires: triangle !single
    args: -run_type full -variable_coefficient nonlinear -interpolate 1 -petscspace_degree 1 -snes_type fas -snes_fas_levels 2 -fas_coarse_pc_type svd -fas_coarse_ksp_rtol 1.0e-10 -fas_coarse_snes_monitor_short -snes_monitor_short -fas_coarse_snes_linesearch_type basic -snes_converged_reason ::ascii_info_detail -dm_refine_hierarchy 1 -snes_view -fas_levels_1_snes_type ngs -fas_levels_1_snes_monitor_short

  test:
    suffix: fas_newton_coarse_0
    requires: pragmatic triangle
    TODO: broken
    args: -run_type full -dm_refine 2 -dm_plex_hash_location -variable_coefficient nonlinear -interpolate 1 -petscspace_degree 1 -snes_type fas -snes_fas_levels 2 -pc_type svd -ksp_rtol 1.0e-10 -fas_coarse_pc_type svd -fas_coarse_ksp_rtol 1.0e-10 -fas_coarse_snes_monitor_short -snes_monitor_short -fas_coarse_snes_linesearch_type basic -snes_converged_reason ::ascii_info_detail -dm_coarsen_hierarchy 1 -snes_view -fas_levels_1_snes_type newtonls -fas_levels_1_pc_type svd -fas_levels_1_ksp_rtol 1.0e-10 -fas_levels_1_snes_monitor_short

  test:
    suffix: mg_newton_coarse_0
    requires: triangle pragmatic
    TODO: broken
    args: -run_type full -dm_refine 3 -interpolate 1 -petscspace_degree 1 -snes_monitor_short -ksp_monitor_true_residual -snes_converged_reason ::ascii_info_detail -dm_coarsen_hierarchy 3 -dm_plex_hash_location -snes_view -dm_view -ksp_type richardson -pc_type mg  -pc_mg_levels 4 -snes_atol 1.0e-8 -ksp_atol 1.0e-8 -snes_rtol 0.0 -ksp_rtol 0.0 -ksp_norm_type unpreconditioned -mg_levels_ksp_type gmres -mg_levels_pc_type ilu -mg_levels_ksp_max_it 10

  test:
    suffix: mg_newton_coarse_1
    requires: triangle pragmatic
    TODO: broken
    args: -run_type full -dm_refine 5 -interpolate 1 -petscspace_degree 1 -dm_coarsen_hierarchy 5 -dm_plex_hash_location -dm_plex_separate_marker -dm_plex_coarsen_bd_label marker -dm_plex_remesh_bd -ksp_type richardson -ksp_rtol 1.0e-12 -pc_type mg -pc_mg_levels 3 -mg_levels_ksp_max_it 2 -snes_converged_reason ::ascii_info_detail -snes_monitor -ksp_monitor_true_residual -mg_levels_ksp_monitor_true_residual -dm_view -ksp_view

  test:
    suffix: mg_newton_coarse_2
    requires: triangle pragmatic
    TODO: broken
    args: -run_type full -dm_refine 5 -interpolate 1 -petscspace_degree 1 -dm_coarsen_hierarchy 5 -dm_plex_hash_location -dm_plex_separate_marker -dm_plex_remesh_bd -ksp_type richardson -ksp_rtol 1.0e-12 -pc_type mg -pc_mg_levels 3 -mg_levels_ksp_max_it 2 -snes_converged_reason ::ascii_info_detail -snes_monitor -ksp_monitor_true_residual -mg_levels_ksp_monitor_true_residual -dm_view -ksp_view

  # Full solve tensor
  test:
    suffix: tensor_plex_2d
    args: -run_type test -refinement_limit 0.0 -simplex 0 -interpolate -bc_type dirichlet -petscspace_degree 1 -dm_refine_hierarchy 2 -cells 2,2

  test:
    suffix: tensor_p4est_2d
    requires: p4est
    args: -run_type test -refinement_limit 0.0 -simplex 0 -interpolate -bc_type dirichlet -petscspace_degree 1 -dm_forest_initial_refinement 2 -dm_forest_minimum_refinement 0 -dm_plex_convert_type p4est -cells 2,2

  test:
    suffix: tensor_plex_3d
    args: -run_type test -refinement_limit 0.0 -simplex 0 -interpolate -bc_type dirichlet -petscspace_degree 1 -dim 3 -dm_refine_hierarchy 1 -cells 2,2,2

  test:
    suffix: tensor_p4est_3d
    requires: p4est
    args: -run_type test -refinement_limit 0.0 -simplex 0 -interpolate -bc_type dirichlet -petscspace_degree 1 -dm_forest_initial_refinement 1 -dm_forest_minimum_refinement 0 -dim 3 -dm_plex_convert_type p8est -cells 2,2,2

  test:
    suffix: p4est_test_q2_conformal_serial
    requires: p4est
    args: -run_type test -interpolate 1 -petscspace_degree 2 -simplex 0 -dm_plex_convert_type p4est -dm_forest_minimum_refinement 0 -dm_forest_initial_refinement 2 -cells 2,2

  test:
    suffix: p4est_test_q2_conformal_parallel
    requires: p4est
    nsize: 7
    args: -run_type test -interpolate 1 -petscspace_degree 2 -simplex 0 -dm_plex_convert_type p4est -dm_forest_minimum_refinement 0 -dm_forest_initial_refinement 2 -petscpartitioner_type simple -cells 2,2

  test:
    suffix: p4est_test_q2_conformal_parallel_parmetis
    requires: parmetis p4est
    nsize: 4
    args: -run_type test -interpolate 1 -petscspace_degree 2 -simplex 0 -dm_plex_convert_type p4est -dm_forest_minimum_refinement 0 -dm_forest_initial_refinement 2 -petscpartitioner_type parmetis -cells 2,2

  test:
    suffix: p4est_test_q2_nonconformal_serial
    requires: p4est
    filter: grep -v "CG or CGNE: variant"
    args: -run_type test -interpolate 1 -petscspace_degree 2 -simplex 0 -dm_plex_convert_type p4est -dm_forest_minimum_refinement 0 -dm_forest_initial_refinement 2 -dm_forest_maximum_refinement 4 -dm_p4est_refine_pattern hash -cells 2,2

  test:
    suffix: p4est_test_q2_nonconformal_parallel
    requires: p4est
    filter: grep -v "CG or CGNE: variant"
    nsize: 7
    args: -run_type test -interpolate 1 -petscspace_degree 2 -simplex 0 -dm_plex_convert_type p4est -dm_forest_minimum_refinement 0 -dm_forest_initial_refinement 2 -dm_forest_maximum_refinement 4 -dm_p4est_refine_pattern hash -petscpartitioner_type simple -cells 2,2

  test:
    suffix: p4est_test_q2_nonconformal_parallel_parmetis
    requires: parmetis p4est
    nsize: 4
    args: -run_type test -interpolate 1 -petscspace_degree 2 -simplex 0 -dm_plex_convert_type p4est -dm_forest_minimum_refinement 0 -dm_forest_initial_refinement 2 -dm_forest_maximum_refinement 4 -dm_p4est_refine_pattern hash -petscpartitioner_type parmetis -cells 2,2

  test:
    suffix: p4est_exact_q2_conformal_serial
    requires: p4est !single !complex !__float128
    args: -run_type exact -interpolate 1 -petscspace_degree 2 -fas_levels_snes_atol 1.e-10 -snes_max_it 1 -snes_type fas -snes_fas_levels 3 -fas_coarse_pc_type none -fas_coarse_ksp_type preonly -fas_coarse_snes_monitor_short -snes_monitor_short -fas_coarse_snes_linesearch_type basic -snes_converged_reason ::ascii_info_detail -snes_view -fas_levels_snes_type newtonls -fas_levels_pc_type none -fas_levels_ksp_type preonly -fas_levels_snes_monitor_short -simplex 0 -dm_plex_convert_type p4est -dm_forest_minimum_refinement 0 -dm_forest_initial_refinement 2 -cells 2,2

  test:
    suffix: p4est_exact_q2_conformal_parallel
    requires: p4est !single !complex !__float128
    nsize: 4
    args: -run_type exact -interpolate 1 -petscspace_degree 2 -fas_levels_snes_atol 1.e-10 -snes_max_it 1 -snes_type fas -snes_fas_levels 3 -fas_coarse_pc_type none -fas_coarse_ksp_type preonly -fas_coarse_snes_monitor_short -snes_monitor_short -fas_coarse_snes_linesearch_type basic -snes_converged_reason ::ascii_info_detail -snes_view -fas_levels_snes_type newtonls -fas_levels_pc_type none -fas_levels_ksp_type preonly -fas_levels_snes_monitor_short -simplex 0 -dm_plex_convert_type p4est -dm_forest_minimum_refinement 0 -dm_forest_initial_refinement 2 -cells 2,2

  test:
    suffix: p4est_exact_q2_conformal_parallel_parmetis
    requires: parmetis p4est !single
    nsize: 4
    args: -run_type exact -interpolate 1 -petscspace_degree 2 -fas_levels_snes_atol 1.e-10 -snes_max_it 1 -snes_type fas -snes_fas_levels 3 -fas_coarse_pc_type none -fas_coarse_ksp_type preonly -fas_coarse_snes_monitor_short -snes_monitor_short -fas_coarse_snes_linesearch_type basic -snes_converged_reason ::ascii_info_detail -snes_view -fas_levels_snes_type newtonls -fas_levels_pc_type none -fas_levels_ksp_type preonly -fas_levels_snes_monitor_short -simplex 0 -dm_plex_convert_type p4est -dm_forest_minimum_refinement 0 -dm_forest_initial_refinement 2 -petscpartitioner_type parmetis  -cells 2,2

  test:
    suffix: p4est_exact_q2_nonconformal_serial
    requires: p4est
    args: -run_type exact -interpolate 1 -petscspace_degree 2 -fas_levels_snes_atol 1.e-10 -snes_max_it 1 -snes_type fas -snes_fas_levels 3 -fas_coarse_pc_type none -fas_coarse_ksp_type preonly -fas_coarse_snes_monitor_short -snes_monitor_short -fas_coarse_snes_linesearch_type basic -snes_converged_reason ::ascii_info_detail -snes_view -fas_levels_snes_type newtonls -fas_levels_pc_type none -fas_levels_ksp_type preonly -fas_levels_snes_monitor_short -simplex 0 -dm_plex_convert_type p4est -dm_forest_minimum_refinement 0 -dm_forest_initial_refinement 2 -dm_forest_maximum_refinement 4 -dm_p4est_refine_pattern hash -cells 2,2

  test:
    suffix: p4est_exact_q2_nonconformal_parallel
    requires: p4est
    nsize: 7
    args: -run_type exact -interpolate 1 -petscspace_degree 2 -fas_levels_snes_atol 1.e-10 -snes_max_it 1 -snes_type fas -snes_fas_levels 3 -fas_coarse_pc_type none -fas_coarse_ksp_type preonly -fas_coarse_snes_monitor_short -snes_monitor_short -fas_coarse_snes_linesearch_type basic -snes_converged_reason ::ascii_info_detail -snes_view -fas_levels_snes_type newtonls -fas_levels_pc_type none -fas_levels_ksp_type preonly -fas_levels_snes_monitor_short -simplex 0 -dm_plex_convert_type p4est -dm_forest_minimum_refinement 0 -dm_forest_initial_refinement 2 -dm_forest_maximum_refinement 4 -dm_p4est_refine_pattern hash -petscpartitioner_type simple -cells 2,2

  test:
    suffix: p4est_exact_q2_nonconformal_parallel_parmetis
    requires: parmetis p4est
    nsize: 4
    args: -run_type exact -interpolate 1 -petscspace_degree 2 -fas_levels_snes_atol 1.e-10 -snes_max_it 1 -snes_type fas -snes_fas_levels 3 -fas_coarse_pc_type none -fas_coarse_ksp_type preonly -fas_coarse_snes_monitor_short -snes_monitor_short -fas_coarse_snes_linesearch_type basic -snes_converged_reason ::ascii_info_detail -snes_view -fas_levels_snes_type newtonls -fas_levels_pc_type none -fas_levels_ksp_type preonly -fas_levels_snes_monitor_short -simplex 0 -dm_plex_convert_type p4est -dm_forest_minimum_refinement 0 -dm_forest_initial_refinement 2 -dm_forest_maximum_refinement 4 -dm_p4est_refine_pattern hash -petscpartitioner_type parmetis -cells 2,2

  test:
    suffix: p4est_full_q2_nonconformal_serial
    requires: p4est !single
    filter: grep -v "variant HERMITIAN"
    args: -run_type full -interpolate 1 -petscspace_degree 2 -snes_max_it 20 -snes_type fas -snes_fas_levels 3 -fas_coarse_pc_type jacobi -fas_coarse_ksp_type cg -fas_coarse_snes_monitor_short -snes_monitor_short -fas_coarse_snes_linesearch_type basic -snes_converged_reason ::ascii_info_detail -snes_view -fas_levels_snes_type newtonls -fas_levels_pc_type jacobi -fas_levels_ksp_type cg -fas_levels_snes_monitor_short -simplex 0 -dm_plex_convert_type p4est -dm_forest_minimum_refinement 0 -dm_forest_initial_refinement 2 -dm_forest_maximum_refinement 4 -dm_p4est_refine_pattern hash -cells 2,2

  test:
    suffix: p4est_full_q2_nonconformal_parallel
    requires: p4est !single
    filter: grep -v "variant HERMITIAN"
    nsize: 7
    args: -run_type full -interpolate 1 -petscspace_degree 2 -snes_max_it 20 -snes_type fas -snes_fas_levels 3 -fas_coarse_pc_type jacobi -fas_coarse_ksp_type cg -fas_coarse_snes_monitor_short -snes_monitor_short -fas_coarse_snes_linesearch_type basic -snes_converged_reason ::ascii_info_detail -snes_view -fas_levels_snes_type newtonls -fas_levels_pc_type jacobi -fas_levels_ksp_type cg -fas_levels_snes_monitor_short -simplex 0 -dm_plex_convert_type p4est -dm_forest_minimum_refinement 0 -dm_forest_initial_refinement 2 -dm_forest_maximum_refinement 4 -dm_p4est_refine_pattern hash -petscpartitioner_type simple -cells 2,2

  test:
    suffix: p4est_full_q2_nonconformal_parallel_bddcfas
    requires: p4est !single
    filter: grep -v "variant HERMITIAN"
    nsize: 7
    args: -run_type full -interpolate 1 -petscspace_degree 2 -snes_max_it 20 -snes_type fas -snes_fas_levels 3 -dm_mat_type is -fas_coarse_pc_type bddc -fas_coarse_ksp_type cg -fas_coarse_snes_monitor_short -snes_monitor_short -fas_coarse_snes_linesearch_type basic -snes_converged_reason ::ascii_info_detail -snes_view -fas_levels_snes_type newtonls -fas_levels_pc_type bddc -fas_levels_ksp_type cg -fas_levels_snes_monitor_short -simplex 0 -dm_plex_convert_type p4est -dm_forest_minimum_refinement 0 -dm_forest_initial_refinement 2 -dm_forest_maximum_refinement 4 -dm_p4est_refine_pattern hash -petscpartitioner_type simple -cells 2,2

  test:
    suffix: p4est_full_q2_nonconformal_parallel_bddc
    requires: p4est !single
    filter: grep -v "variant HERMITIAN"
    nsize: 7
    args: -run_type full -interpolate 1 -petscspace_degree 2 -snes_max_it 20 -snes_type newtonls -dm_mat_type is -pc_type bddc -ksp_type cg -snes_monitor_short -snes_linesearch_type basic -snes_converged_reason ::ascii_info_detail -snes_view -simplex 0 -dm_plex_convert_type p4est -dm_forest_minimum_refinement 0 -dm_forest_initial_refinement 2 -dm_forest_maximum_refinement 4 -dm_p4est_refine_pattern hash -petscpartitioner_type simple -cells 2,2

  test:
    TODO: broken
    suffix: p4est_fas_q2_conformal_serial
    requires: p4est !complex !__float128
    args: -run_type full -variable_coefficient nonlinear -interpolate 1 -petscspace_degree 2 -snes_max_it 20 -snes_type fas -snes_fas_levels 3 -pc_type jacobi -ksp_type gmres -fas_coarse_pc_type svd -fas_coarse_ksp_type gmres -fas_coarse_snes_monitor_short -snes_monitor_short -fas_coarse_snes_linesearch_type basic -snes_converged_reason ::ascii_info_detail -snes_view -fas_levels_snes_type newtonls -fas_levels_pc_type svd -fas_levels_ksp_type gmres -fas_levels_snes_monitor_short -simplex 0 -dm_refine_hierarchy 3 -cells 2,2

  test:
    TODO: broken
    suffix: p4est_fas_q2_nonconformal_serial
    requires: p4est
    args: -run_type full -variable_coefficient nonlinear -interpolate 1 -petscspace_degree 2 -snes_max_it 20 -snes_type fas -snes_fas_levels 3 -pc_type jacobi -ksp_type gmres -fas_coarse_pc_type jacobi -fas_coarse_ksp_type gmres -fas_coarse_ksp_monitor_true_residual -fas_coarse_snes_monitor_short -snes_monitor_short -fas_coarse_snes_linesearch_type basic -snes_converged_reason ::ascii_info_detail -snes_view -fas_levels_snes_type newtonls -fas_levels_pc_type jacobi -fas_levels_ksp_type gmres -fas_levels_snes_monitor_short -simplex 0 -dm_plex_convert_type p4est -dm_forest_minimum_refinement 0 -dm_forest_initial_refinement 2 -dm_forest_maximum_refinement 4 -dm_p4est_refine_pattern hash -cells 2,2

  test:
    suffix: fas_newton_0_p4est
    requires: p4est !single !__float128
    args: -run_type full -variable_coefficient nonlinear -interpolate 1 -petscspace_degree 1 -snes_type fas -snes_fas_levels 2 -fas_coarse_pc_type svd -fas_coarse_ksp_rtol 1.0e-10 -fas_coarse_snes_monitor_short -snes_monitor_short -fas_coarse_snes_linesearch_type basic -snes_converged_reason ::ascii_info_detail -snes_view -fas_levels_1_snes_type newtonls -fas_levels_1_pc_type svd -fas_levels_1_ksp_rtol 1.0e-10 -fas_levels_1_snes_monitor_short -simplex 0 -dm_plex_convert_type p4est -dm_forest_minimum_refinement 0 -dm_forest_initial_refinement 2 -dm_forest_maximum_refinement 4 -dm_p4est_refine_pattern hash -cells 2,2

  # Full solve simplicial AMR
  test:
    suffix: tri_p1_adapt_0
    requires: pragmatic
    TODO: broken
    args: -run_type exact -dim 2 -dm_refine 5 -bc_type dirichlet -interpolate 1 -petscspace_degree 1 -variable_coefficient circle -snes_converged_reason ::ascii_info_detail -pc_type lu -adaptor_refinement_factor 1.0 -dm_view -dm_adapt_view -snes_adapt_initial 1

  test:
    suffix: tri_p1_adapt_1
    requires: pragmatic
    TODO: broken
    args: -run_type exact -dim 2 -dm_refine 5 -bc_type dirichlet -interpolate 1 -petscspace_degree 1 -variable_coefficient circle -snes_converged_reason ::ascii_info_detail -pc_type lu -adaptor_refinement_factor 1.0 -dm_view -dm_adapt_iter_view -dm_adapt_view -snes_adapt_sequence 2

  test:
    suffix: tri_p1_adapt_analytic_0
    requires: pragmatic
    TODO: broken
    args: -run_type exact -dim 2 -dm_refine 3 -bc_type dirichlet -interpolate 1 -petscspace_degree 1 -variable_coefficient cross -snes_adapt_initial 4 -adaptor_target_num 500 -adaptor_monitor -dm_view -dm_adapt_iter_view

  # Full solve tensor AMR
  test:
    suffix: quad_q1_adapt_0
    requires: p4est
    args: -run_type exact -dim 2 -simplex 0 -dm_plex_convert_type p4est -bc_type dirichlet -interpolate 1 -petscspace_degree 1 -variable_coefficient circle -snes_converged_reason ::ascii_info_detail -pc_type lu -dm_forest_initial_refinement 4   -snes_adapt_initial 1 -dm_view
    filter: grep -v DM_

  test:
    suffix: amr_0
    nsize: 5
    args: -run_type test -petscpartitioner_type simple -refinement_limit 0.0 -simplex 0 -interpolate -bc_type dirichlet -petscspace_degree 1 -dm_refine 1 -cells 2,2

  test:
    suffix: amr_1
    requires: p4est !complex
    args: -run_type test -refinement_limit 0.0 -simplex 0 -interpolate -bc_type dirichlet -petscspace_degree 1 -dm_plex_convert_type p4est -dm_p4est_refine_pattern center -dm_forest_maximum_refinement 5 -dm_view vtk:amr.vtu:vtk_vtu -vec_view vtk:amr.vtu:vtk_vtu:append -cells 2,2

  test:
    suffix: p4est_solve_bddc
    requires: p4est !complex
    args: -run_type full -variable_coefficient nonlinear -nonzero_initial_guess 1 -interpolate 1 -petscspace_degree 2 -snes_max_it 20 -snes_type newtonls -dm_mat_type is -pc_type bddc -ksp_type cg -snes_monitor_short -ksp_monitor -snes_linesearch_type bt -snes_converged_reason -snes_view -simplex 0 -petscspace_poly_tensor -dm_plex_convert_type p4est -dm_forest_minimum_refinement 0 -dm_forest_initial_refinement 2 -dm_forest_maximum_refinement 4 -dm_p4est_refine_pattern hash -petscpartitioner_type simple -pc_bddc_detect_disconnected
    nsize: 4

  test:
    suffix: p4est_solve_fas
    requires: p4est
    args: -run_type full -variable_coefficient nonlinear -nonzero_initial_guess 1 -interpolate 1 -petscspace_degree 2 -snes_max_it 10 -snes_type fas -snes_linesearch_type bt -snes_fas_levels 3 -fas_coarse_snes_type newtonls -fas_coarse_snes_linesearch_type basic -fas_coarse_ksp_type cg -fas_coarse_pc_type jacobi -fas_coarse_snes_monitor_short -fas_levels_snes_max_it 4 -fas_levels_snes_type newtonls -fas_levels_snes_linesearch_type bt -fas_levels_ksp_type cg -fas_levels_pc_type jacobi -fas_levels_snes_monitor_short -fas_levels_cycle_snes_linesearch_type bt -snes_monitor_short -snes_converged_reason -snes_view -simplex 0 -petscspace_poly_tensor -dm_plex_convert_type p4est -dm_forest_minimum_refinement 0 -dm_forest_initial_refinement 2 -dm_forest_maximum_refinement 4 -dm_p4est_refine_pattern hash
    nsize: 4
    TODO: identical machine two runs produce slightly different solver trackers

  test:
    suffix: p4est_convergence_test_1
    requires: p4est
    args:  -quiet -run_type test -interpolate 1 -petscspace_degree 1 -simplex 0 -petscspace_poly_tensor -dm_plex_convert_type p4est -dm_forest_minimum_refinement 2 -dm_forest_initial_refinement 2 -dm_forest_maximum_refinement 4 -dm_p4est_refine_pattern hash
    nsize: 4

  test:
    suffix: p4est_convergence_test_2
    requires: p4est
    args: -quiet -run_type test -interpolate 1 -petscspace_degree 1 -simplex 0 -petscspace_poly_tensor -dm_plex_convert_type p4est -dm_forest_minimum_refinement 3 -dm_forest_initial_refinement 3 -dm_forest_maximum_refinement 5 -dm_p4est_refine_pattern hash

  test:
    suffix: p4est_convergence_test_3
    requires: p4est
    args: -quiet -run_type test -interpolate 1 -petscspace_degree 1 -simplex 0 -petscspace_poly_tensor -dm_plex_convert_type p4est -dm_forest_minimum_refinement 4 -dm_forest_initial_refinement 4 -dm_forest_maximum_refinement 6 -dm_p4est_refine_pattern hash

  test:
    suffix: p4est_convergence_test_4
    requires: p4est
    args: -quiet -run_type test -interpolate 1 -petscspace_degree 1 -simplex 0 -petscspace_poly_tensor -dm_plex_convert_type p4est -dm_forest_minimum_refinement 5 -dm_forest_initial_refinement 5 -dm_forest_maximum_refinement 7 -dm_p4est_refine_pattern hash
    timeoutfactor: 5

  # Serial tests with GLVis visualization
  test:
    suffix: glvis_2d_tet_p1
    args: -quiet -run_type test -interpolate 1 -bc_type dirichlet -petscspace_degree 1 -vec_view glvis: -f ${wPETSC_DIR}/share/petsc/datafiles/meshes/square_periodic.msh -dm_plex_gmsh_periodic 0
  test:
    suffix: glvis_2d_tet_p2
    args: -quiet -run_type test -interpolate 1 -bc_type dirichlet -petscspace_degree 2 -vec_view glvis: -f ${wPETSC_DIR}/share/petsc/datafiles/meshes/square_periodic.msh -dm_plex_gmsh_periodic 0
  test:
    suffix: glvis_2d_hex_p1
    args: -quiet -run_type test -interpolate 1 -bc_type dirichlet -petscspace_degree 1 -vec_view glvis: -simplex 0 -dm_refine 1
  test:
    suffix: glvis_2d_hex_p2
    args: -quiet -run_type test -interpolate 1 -bc_type dirichlet -petscspace_degree 2 -vec_view glvis: -simplex 0 -dm_refine 1
  test:
    suffix: glvis_2d_hex_p2_p4est
    requires: p4est
    args: -quiet -run_type test -interpolate 1 -bc_type dirichlet -petscspace_degree 2 -vec_view glvis: -simplex 0 -dm_plex_convert_type p4est -dm_forest_minimum_refinement 0 -dm_forest_initial_refinement 1 -dm_forest_maximum_refinement 4 -dm_p4est_refine_pattern hash -cells 2,2 -viewer_glvis_dm_plex_enable_ncmesh
  test:
    suffix: glvis_2d_tet_p0
    args: -run_type exact  -interpolate 1 -guess_vec_view glvis: -nonzero_initial_guess 1 -f ${wPETSC_DIR}/share/petsc/datafiles/meshes/square_periodic.msh -petscspace_degree 0
  test:
    suffix: glvis_2d_hex_p0
    args: -run_type exact  -interpolate 1 -guess_vec_view glvis: -nonzero_initial_guess 1 -cells 5,7  -simplex 0 -petscspace_degree 0

  # PCHPDDM tests
  testset:
    nsize: 4
    requires: hpddm slepc !single
    args: -run_type test -run_test_check_ksp -quiet -petscspace_degree 1 -interpolate 1 -petscpartitioner_type simple -bc_type none -simplex 0 -pc_type hpddm -pc_hpddm_levels_1_sub_pc_type lu -pc_hpddm_levels_1_eps_nev 2 -pc_hpddm_coarse_p 1 -pc_hpddm_coarse_pc_type svd -ksp_rtol 1.e-10 -pc_hpddm_levels_1_st_pc_factor_shift_type INBLOCKS -ksp_converged_reason
    test:
      suffix: quad_singular_hpddm
      args: -cells 6,7
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
    requires: hpddm slepc triangle !single
    args: -run_type full -petscpartitioner_type simple -dm_refine 2 -bc_type dirichlet -interpolate 1 -petscspace_degree 2 -ksp_type gmres -ksp_gmres_restart 100 -pc_type hpddm -snes_monitor_short -ksp_monitor_short -snes_converged_reason ::ascii_info_detail -ksp_converged_reason -snes_view -show_solution 0 -pc_type hpddm -pc_hpddm_levels_1_sub_pc_type lu -pc_hpddm_levels_1_eps_nev 4 -pc_hpddm_coarse_p 2 -pc_hpddm_coarse_pc_type redundant -ksp_rtol 1.e-1
    test:
      args: -pc_hpddm_coarse_mat_type baij -options_left no
      suffix: tri_hpddm_reuse_baij
    test:
      requires: !complex
      suffix: tri_hpddm_reuse
  testset:
    nsize: 4
    requires: hpddm slepc !single
    args: -run_type full -petscpartitioner_type simple -cells 7,5 -dm_refine 2 -simplex 0 -bc_type dirichlet -interpolate 1 -petscspace_degree 2 -ksp_type gmres -ksp_gmres_restart 100 -pc_type hpddm -snes_monitor_short -ksp_monitor_short -snes_converged_reason ::ascii_info_detail -ksp_converged_reason -snes_view -show_solution 0 -pc_type hpddm -pc_hpddm_levels_1_sub_pc_type lu -pc_hpddm_levels_1_eps_nev 4 -pc_hpddm_coarse_p 2 -pc_hpddm_coarse_pc_type redundant -ksp_rtol 1.e-1
    test:
      args: -pc_hpddm_coarse_mat_type baij -options_left no
      suffix: quad_hpddm_reuse_baij
    test:
      requires: !complex
      suffix: quad_hpddm_reuse
  testset:
    nsize: 4
    requires: hpddm slepc !single
    args: -run_type full -petscpartitioner_type simple -cells 7,5 -dm_refine 2 -simplex 0 -bc_type dirichlet -interpolate 1 -petscspace_degree 1 -ksp_type gmres -ksp_gmres_restart 100 -pc_type hpddm -snes_monitor_short -ksp_monitor_short -snes_converged_reason ::ascii_info_detail -ksp_converged_reason -snes_view -show_solution 0 -pc_type hpddm -pc_hpddm_levels_1_sub_pc_type lu -pc_hpddm_levels_1_eps_threshold 0.1 -pc_hpddm_coarse_p 2 -pc_hpddm_coarse_pc_type redundant -ksp_rtol 1.e-1
    test:
      args: -pc_hpddm_coarse_mat_type baij -options_left no
      suffix: quad_hpddm_reuse_threshold_baij
    test:
      requires: !complex
      suffix: quad_hpddm_reuse_threshold
  testset:
    nsize: 4
    requires: hpddm slepc parmetis !single
    args: -run_type full -petscpartitioner_type parmetis -dm_refine 3 -bc_type dirichlet -interpolate 1 -petscspace_degree 1 -ksp_type gmres -ksp_gmres_restart 100 -pc_type hpddm -snes_monitor_short -ksp_monitor_short -snes_converged_reason ::ascii_info_detail -ksp_converged_reason -snes_view -show_solution 0 -pc_type hpddm -pc_hpddm_levels_1_sub_pc_type icc -pc_hpddm_levels_1_eps_nev 20 -pc_hpddm_coarse_p 2 -pc_hpddm_coarse_pc_type redundant -ksp_rtol 1.e-10 -f ${PETSC_DIR}/share/petsc/datafiles/meshes/square_periodic.msh -pc_hpddm_levels_1_sub_pc_factor_levels 3 -variable_coefficient circle -dm_plex_gmsh_periodic 0
    test:
      args: -pc_hpddm_coarse_mat_type baij -options_left no
      suffix: tri_parmetis_hpddm_baij
    test:
      requires: !complex
      suffix: tri_parmetis_hpddm
TEST*/
