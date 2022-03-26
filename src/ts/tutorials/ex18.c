static char help[] = "Hybrid Finite Element-Finite Volume Example.\n";
/*F
  Here we are advecting a passive tracer in a harmonic velocity field, defined by
a forcing function $f$:
\begin{align}
  -\Delta \mathbf{u} + f &= 0 \\
  \frac{\partial\phi}{\partial t} + \nabla\cdot \phi \mathbf{u} &= 0
\end{align}
F*/

#include <petscdmplex.h>
#include <petscds.h>
#include <petscts.h>

#include <petsc/private/dmpleximpl.h> /* For DotD */

#define ALEN(a) (sizeof(a)/sizeof((a)[0]))

typedef enum {VEL_ZERO, VEL_CONSTANT, VEL_HARMONIC, VEL_SHEAR} VelocityDistribution;

typedef enum {ZERO, CONSTANT, GAUSSIAN, TILTED, DELTA} PorosityDistribution;

static PetscErrorCode constant_u_2d(PetscInt, PetscReal, const PetscReal[], PetscInt, PetscScalar *, void *);

/*
  FunctionalFunc - Calculates the value of a functional of the solution at a point

  Input Parameters:
+ dm   - The DM
. time - The TS time
. x    - The coordinates of the evaluation point
. u    - The field values at point x
- ctx  - A user context, or NULL

  Output Parameter:
. f    - The value of the functional at point x

*/
typedef PetscErrorCode (*FunctionalFunc)(DM, PetscReal, const PetscReal *, const PetscScalar *, PetscReal *, void *);

typedef struct _n_Functional *Functional;
struct _n_Functional {
  char          *name;
  FunctionalFunc func;
  void          *ctx;
  PetscInt       offset;
  Functional     next;
};

typedef struct {
  /* Problem definition */
  PetscBool      useFV;             /* Use a finite volume scheme for advection */
  PetscErrorCode (*initialGuess[2])(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nf, PetscScalar *u, void *ctx);
  VelocityDistribution velocityDist;
  PorosityDistribution porosityDist;
  PetscReal            inflowState;
  PetscReal            source[3];
  /* Monitoring */
  PetscInt       numMonitorFuncs, maxMonitorFunc;
  Functional    *monitorFuncs;
  PetscInt       errorFunctional;
  Functional     functionalRegistry;
} AppCtx;

static  AppCtx *globalUser;

static PetscErrorCode ProcessOptions(MPI_Comm comm, AppCtx *options)
{
  const char    *velocityDist[4]  = {"zero", "constant", "harmonic", "shear"};
  const char    *porosityDist[5]  = {"zero", "constant", "gaussian", "tilted", "delta"};
  PetscInt       vd, pd, d;
  PetscBool      flg;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  options->useFV        = PETSC_FALSE;
  options->velocityDist = VEL_HARMONIC;
  options->porosityDist = ZERO;
  options->inflowState  = -2.0;
  options->numMonitorFuncs = 0;
  options->source[0]    = 0.5;
  options->source[1]    = 0.5;
  options->source[2]    = 0.5;

  ierr = PetscOptionsBegin(comm, "", "Magma Dynamics Options", "DMPLEX");PetscCall(ierr);
  PetscCall(PetscOptionsBool("-use_fv", "Use the finite volume method for advection", "ex18.c", options->useFV, &options->useFV, NULL));
  vd   = options->velocityDist;
  PetscCall(PetscOptionsEList("-velocity_dist","Velocity distribution type","ex18.c",velocityDist,4,velocityDist[options->velocityDist],&vd,NULL));
  options->velocityDist = (VelocityDistribution) vd;
  pd   = options->porosityDist;
  PetscCall(PetscOptionsEList("-porosity_dist","Initial porosity distribution type","ex18.c",porosityDist,5,porosityDist[options->porosityDist],&pd,NULL));
  options->porosityDist = (PorosityDistribution) pd;
  PetscCall(PetscOptionsReal("-inflow_state", "The inflow state", "ex18.c", options->inflowState, &options->inflowState, NULL));
  d    = 2;
  PetscCall(PetscOptionsRealArray("-source_loc", "The source location", "ex18.c", options->source, &d, &flg));
  PetscCheck(!flg || d == 2,PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Must give dim coordinates for the source location, not %d", d);
  ierr = PetscOptionsEnd();PetscCall(ierr);

  PetscFunctionReturn(0);
}

static PetscErrorCode ProcessMonitorOptions(MPI_Comm comm, AppCtx *options)
{
  Functional     func;
  char          *names[256];
  PetscInt       f;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = PetscOptionsBegin(comm, "", "Simulation Monitor Options", "DMPLEX");PetscCall(ierr);
  options->numMonitorFuncs = ALEN(names);
  PetscCall(PetscOptionsStringArray("-monitor", "List of functionals to monitor", "", names, &options->numMonitorFuncs, NULL));
  PetscCall(PetscMalloc1(options->numMonitorFuncs, &options->monitorFuncs));
  for (f = 0; f < options->numMonitorFuncs; ++f) {
    for (func = options->functionalRegistry; func; func = func->next) {
      PetscBool match;

      PetscCall(PetscStrcasecmp(names[f], func->name, &match));
      if (match) break;
    }
    PetscCheck(func,comm, PETSC_ERR_USER, "No known functional '%s'", names[f]);
    options->monitorFuncs[f] = func;
    /* Jed inserts a de-duplication of functionals here */
    PetscCall(PetscFree(names[f]));
  }
  /* Find out the maximum index of any functional computed by a function we will be calling (even if we are not using it) */
  options->maxMonitorFunc = -1;
  for (func = options->functionalRegistry; func; func = func->next) {
    for (f = 0; f < options->numMonitorFuncs; ++f) {
      Functional call = options->monitorFuncs[f];

      if (func->func == call->func && func->ctx == call->ctx) options->maxMonitorFunc = PetscMax(options->maxMonitorFunc, func->offset);
    }
  }
  ierr = PetscOptionsEnd();PetscCall(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode FunctionalRegister(Functional *functionalRegistry, const char name[], PetscInt *offset, FunctionalFunc func, void *ctx)
{
  Functional    *ptr, f;
  PetscInt       lastoffset = -1;

  PetscFunctionBeginUser;
  for (ptr = functionalRegistry; *ptr; ptr = &(*ptr)->next) lastoffset = (*ptr)->offset;
  PetscCall(PetscNew(&f));
  PetscCall(PetscStrallocpy(name, &f->name));
  f->offset = lastoffset + 1;
  f->func   = func;
  f->ctx    = ctx;
  f->next   = NULL;
  *ptr      = f;
  *offset   = f->offset;
  PetscFunctionReturn(0);
}

static PetscErrorCode FunctionalDestroy(Functional *link)
{
  Functional     next, l;

  PetscFunctionBeginUser;
  if (!link) PetscFunctionReturn(0);
  l     = *link;
  *link = NULL;
  for (; l; l=next) {
    next = l->next;
    PetscCall(PetscFree(l->name));
    PetscCall(PetscFree(l));
  }
  PetscFunctionReturn(0);
}

static void f0_zero_u(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                      const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                      const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                      PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f0[])
{
  PetscInt comp;
  for (comp = 0; comp < dim; ++comp) f0[comp] = u[comp];
}

static void f0_constant_u(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                          const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                          const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                          PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f0[])
{
  PetscScalar wind[3] = {0.0, 0.0, 0.0};
  PetscInt    comp;

  constant_u_2d(dim, t, x, Nf, wind, NULL);
  for (comp = 0; comp < dim && comp < 3; ++comp) f0[comp] = u[comp] - wind[comp];
}

static void f1_constant_u(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                          const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                          const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                          PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f1[])
{
  PetscInt comp;
  for (comp = 0; comp < dim*dim; ++comp) f1[comp] = 0.0;
}

static void g0_constant_uu(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                           const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                           const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                           PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g0[])
{
  PetscInt d;
  for (d = 0; d < dim; ++d) g0[d*dim+d] = 1.0;
}

static void g0_constant_pp(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                           const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                           const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                           PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g0[])
{
  g0[0] = 1.0;
}

static void f0_lap_u(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                     const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                     const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                     PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f0[])
{
  PetscInt comp;
  for (comp = 0; comp < dim; ++comp) f0[comp] = 4.0;
}

static void f1_lap_u(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                     const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                     const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                     PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f1[])
{
  PetscInt comp, d;
  for (comp = 0; comp < dim; ++comp) {
    for (d = 0; d < dim; ++d) {
      f1[comp*dim+d] = u_x[comp*dim+d];
    }
  }
}

static void f0_lap_periodic_u(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                              const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                              const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                              PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f0[])
{
  f0[0] = -PetscSinReal(2.0*PETSC_PI*x[0]);
  f0[1] = 2.0*PETSC_PI*x[1]*PetscCosReal(2.0*PETSC_PI*x[0]);
}

static void f0_lap_doubly_periodic_u(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                                     const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                                     const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                                     PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f0[])
{
  f0[0] = -2.0*PetscSinReal(2.0*PETSC_PI*x[0])*PetscCosReal(2.0*PETSC_PI*x[1]);
  f0[1] =  2.0*PetscSinReal(2.0*PETSC_PI*x[1])*PetscCosReal(2.0*PETSC_PI*x[0]);
}

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

/* \frac{\partial\phi}{\partial t} + \nabla\phi \cdot \mathbf{u} + \phi \nabla \cdot \mathbf{u} = 0 */
static void f0_advection(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                         const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                         const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                         PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f0[])
{
  PetscInt d;
  f0[0] = u_t[dim];
  for (d = 0; d < dim; ++d) f0[0] += u[dim]*u_x[d*dim+d] + u_x[dim*dim+d]*u[d];
}

static void f1_advection(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                         const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                         const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                         PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f1[])
{
  PetscInt d;
  for (d = 0; d < dim; ++d) f1[0] = 0.0;
}

void g0_adv_pp(PetscInt dim, PetscInt Nf, PetscInt NfAux,
               const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
               const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
               PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g0[])
{
  PetscInt d;
  g0[0] = u_tShift;
  for (d = 0; d < dim; ++d) g0[0] += u_x[d*dim+d];
}

void g1_adv_pp(PetscInt dim, PetscInt Nf, PetscInt NfAux,
               const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
               const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
               PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g1[])
{
  PetscInt d;
  for (d = 0; d < dim; ++d) g1[d] = u[d];
}

void g0_adv_pu(PetscInt dim, PetscInt Nf, PetscInt NfAux,
               const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
               const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
               PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g0[])
{
  PetscInt d;
  for (d = 0; d < dim; ++d) g0[0] += u_x[dim*dim+d];
}

void g1_adv_pu(PetscInt dim, PetscInt Nf, PetscInt NfAux,
               const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
               const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
               PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g1[])
{
  PetscInt d;
  for (d = 0; d < dim; ++d) g1[d*dim+d] = u[dim];
}

static void riemann_advection(PetscInt dim, PetscInt Nf, const PetscReal *qp, const PetscReal *n, const PetscScalar *uL, const PetscScalar *uR, PetscInt numConstants, const PetscScalar constants[], PetscScalar *flux, void *ctx)
{
  PetscReal wind[3] = {0.0, 1.0, 0.0};
  PetscReal wn = DMPlex_DotRealD_Internal(PetscMin(dim,3), wind, n);

  flux[0] = (wn > 0 ? uL[dim] : uR[dim]) * wn;
}

static void riemann_coupled_advection(PetscInt dim, PetscInt Nf, const PetscReal *qp, const PetscReal *n, const PetscScalar *uL, const PetscScalar *uR, PetscInt numConstants, const PetscScalar constants[], PetscScalar *flux, void *ctx)
{
  PetscReal wn = DMPlex_DotD_Internal(dim, uL, n);

#if 1
  flux[0] = (wn > 0 ? uL[dim] : uR[dim]) * wn;
#else
  /* if (fabs(uL[0] - wind[0]) > 1.0e-7 || fabs(uL[1] - wind[1]) > 1.0e-7) PetscPrintf(PETSC_COMM_SELF, "wind (%g, %g) uL (%g, %g) uR (%g, %g)\n", wind[0], wind[1], uL[0], uL[1], uR[0], uR[1]); */
  /* Smear it out */
  flux[0] = 0.5*((uL[dim] + uR[dim]) + (uL[dim] - uR[dim])*tanh(1.0e5*wn)) * wn;
#endif
}

static PetscErrorCode zero_u_2d(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nf, PetscScalar *u, void *ctx)
{
  u[0] = 0.0;
  u[1] = 0.0;
  return 0;
}

static PetscErrorCode constant_u_2d(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nf, PetscScalar *u, void *ctx)
{
  u[0] = 0.0;
  u[1] = 1.0;
  return 0;
}

/* Coordinates of the point which was at x at t = 0 */
static PetscErrorCode constant_x_2d(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nf, PetscScalar *u, void *ctx)
{
  const PetscReal t = *((PetscReal *) ctx);
  u[0] = x[0];
  u[1] = x[1] + t;
#if 0
  PetscErrorCode  ierr;
  PetscCall(DMLocalizeCoordinate(globalUser->dm, u, PETSC_FALSE, u));
#else
  u[1] = u[1] - (int) PetscRealPart(u[1]);
#endif
  return 0;
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
    h_t(x + y + (u + v) t) - u . grad phi - phi div u
  = u h' + v h'              - u h_x - v h_y
  = 0

We will conserve phi since

    \nabla \cdot u = 2x - 2x = 0

Also try h((x + ut)^2 + (y + vt)^2), so that

    h_t((x + ut)^2 + (y + vt)^2) - u . grad phi - phi div u
  = 2 h' (u (x + ut) + v (y + vt)) - u h_x - v h_y
  = 2 h' (u (x + ut) + v (y + vt)) - u h' 2 (x + u t) - v h' 2 (y + vt)
  = 2 h' (u (x + ut) + v (y + vt)  - u (x + u t) - v (y + vt))
  = 0

*/
static PetscErrorCode quadratic_u_2d(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nf, PetscScalar *u, void *ctx)
{
  u[0] = x[0]*x[0] + x[1]*x[1];
  u[1] = 2.0*x[0]*x[0] - 2.0*x[0]*x[1];
  return 0;
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
static PetscErrorCode periodic_u_2d(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nf, PetscScalar *u, void *ctx)
{
  u[0] = PetscSinReal(2.0*PETSC_PI*x[0])/PetscSqr(2.0*PETSC_PI);
  u[1] = -x[1]*PetscCosReal(2.0*PETSC_PI*x[0])/(2.0*PETSC_PI);
  return 0;
}

/*
  In 2D we use the exact, doubly periodic solution:

    u   =  sin(2 pi x) cos(2 pi y)/4 pi^2
    v   = -sin(2 pi y) cos(2 pi x)/4 pi^2
    phi = h(x + y + (u + v) t)
    f_x = -2sin(2 pi x) cos(2 pi y)
    f_y =  2sin(2 pi y) cos(2 pi x)

  so that

    -\Delta u + f = <2 sin(2pi x) cos(2pi y),  -2 sin(2pi y) cos(2pi x)> + <-2 sin(2pi x) cos(2pi y), 2 sin(2pi y) cos(2pi x)> = 0

We will conserve phi since

    \nabla \cdot u = cos(2pi x) cos(2pi y)/2pi - cos(2pi y) cos(2pi x)/2pi = 0
*/
static PetscErrorCode doubly_periodic_u_2d(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nf, PetscScalar *u, void *ctx)
{
  u[0] =  PetscSinReal(2.0*PETSC_PI*x[0])*PetscCosReal(2.0*PETSC_PI*x[1])/PetscSqr(2.0*PETSC_PI);
  u[1] = -PetscSinReal(2.0*PETSC_PI*x[1])*PetscCosReal(2.0*PETSC_PI*x[0])/PetscSqr(2.0*PETSC_PI);
  return 0;
}

static PetscErrorCode shear_bc(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nf, PetscScalar *u, void *ctx)
{
  u[0] = x[1] - 0.5;
  u[1] = 0.0;
  return 0;
}

static PetscErrorCode initialVelocity(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nf, PetscScalar *u, void *ctx)
{
  PetscInt d;
  for (d = 0; d < dim; ++d) u[d] = 0.0;
  return 0;
}

static PetscErrorCode zero_phi(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nf, PetscScalar *u, void *ctx)
{
  u[0] = 0.0;
  return 0;
}

static PetscErrorCode constant_phi(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nf, PetscScalar *u, void *ctx)
{
  u[0] = 1.0;
  return 0;
}

static PetscErrorCode delta_phi_2d(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nf, PetscScalar *u, void *ctx)
{
  PetscReal   x0[2];
  PetscScalar xn[2];

  x0[0] = globalUser->source[0];
  x0[1] = globalUser->source[1];
  constant_x_2d(dim, time, x0, Nf, xn, ctx);
  {
    const PetscReal xi  = x[0] - PetscRealPart(xn[0]);
    const PetscReal eta = x[1] - PetscRealPart(xn[1]);
    const PetscReal r2  = xi*xi + eta*eta;

    u[0] = r2 < 1.0e-7 ? 1.0 : 0.0;
  }
  return 0;
}

/*
  Gaussian blob, initially centered on (0.5, 0.5)

  xi = x(t) - x0, eta = y(t) - y0

where x(t), y(t) are the integral curves of v(t),

  dx/dt . grad f = v . f

Check: constant v(t) = {v0, w0}, x(t) = {x0 + v0 t, y0 + w0 t}

  v0 f_x + w0 f_y = v . f
*/
static PetscErrorCode gaussian_phi_2d(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nf, PetscScalar *u, void *ctx)
{
  const PetscReal x0[2] = {0.5, 0.5};
  const PetscReal sigma = 1.0/6.0;
  PetscScalar     xn[2];

  constant_x_2d(dim, time, x0, Nf, xn, ctx);
  {
    /* const PetscReal xi  = x[0] + (sin(2.0*PETSC_PI*x[0])/(4.0*PETSC_PI*PETSC_PI))*t - x0[0]; */
    /* const PetscReal eta = x[1] + (-x[1]*cos(2.0*PETSC_PI*x[0])/(2.0*PETSC_PI))*t - x0[1]; */
    const PetscReal xi  = x[0] - PetscRealPart(xn[0]);
    const PetscReal eta = x[1] - PetscRealPart(xn[1]);
    const PetscReal r2  = xi*xi + eta*eta;

    u[0] = PetscExpReal(-r2/(2.0*sigma*sigma))/(sigma*PetscSqrtReal(2.0*PETSC_PI));
  }
  return 0;
}

static PetscErrorCode tilted_phi_2d(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nf, PetscScalar *u, void *ctx)
{
  PetscReal       x0[3];
  const PetscReal wind[3] = {0.0, 1.0, 0.0};
  const PetscReal t       = *((PetscReal *) ctx);

  DMPlex_WaxpyD_Internal(2, -t, wind, x, x0);
  if (x0[1] > 0) u[0] =  1.0*x[0] + 3.0*x[1];
  else           u[0] = -2.0; /* Inflow state */
  return 0;
}

static PetscErrorCode tilted_phi_coupled_2d(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nf, PetscScalar *u, void *ctx)
{
  PetscReal       ur[3];
  PetscReal       x0[3];
  const PetscReal t = *((PetscReal *) ctx);

  ur[0] = PetscRealPart(u[0]); ur[1] = PetscRealPart(u[1]); ur[2] = PetscRealPart(u[2]);
  DMPlex_WaxpyD_Internal(2, -t, ur, x, x0);
  if (x0[1] > 0) u[0] =  1.0*x[0] + 3.0*x[1];
  else           u[0] = -2.0; /* Inflow state */
  return 0;
}

static PetscErrorCode advect_inflow(PetscReal time, const PetscReal *c, const PetscReal *n, const PetscScalar *xI, PetscScalar *xG, void *ctx)
{
  AppCtx *user = (AppCtx *) ctx;

  PetscFunctionBeginUser;
  xG[0] = user->inflowState;
  PetscFunctionReturn(0);
}

static PetscErrorCode advect_outflow(PetscReal time, const PetscReal *c, const PetscReal *n, const PetscScalar *xI, PetscScalar *xG, void *ctx)
{
  PetscFunctionBeginUser;
  //xG[0] = xI[dim];
  xG[0] = xI[2];
  PetscFunctionReturn(0);
}

static PetscErrorCode ExactSolution(DM dm, PetscReal time, const PetscReal *x, PetscScalar *u, void *ctx)
{
  AppCtx        *user = (AppCtx *) ctx;
  PetscInt       dim;

  PetscFunctionBeginUser;
  PetscCall(DMGetDimension(dm, &dim));
  switch (user->porosityDist) {
  case TILTED:
    if (user->velocityDist == VEL_ZERO) tilted_phi_2d(dim, time, x, 2, u, (void *) &time);
    else                                tilted_phi_coupled_2d(dim, time, x, 2, u, (void *) &time);
    break;
  case GAUSSIAN:
    gaussian_phi_2d(dim, time, x, 2, u, (void *) &time);
    break;
  case DELTA:
    delta_phi_2d(dim, time, x, 2, u, (void *) &time);
    break;
  default: SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Unknown solution type");
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode Functional_Error(DM dm, PetscReal time, const PetscReal *x, const PetscScalar *y, PetscReal *f, void *ctx)
{
  AppCtx        *user = (AppCtx *) ctx;
  PetscScalar    yexact[3]={0,0,0};

  PetscFunctionBeginUser;
  PetscCall(ExactSolution(dm, time, x, yexact, ctx));
  f[user->errorFunctional] = PetscAbsScalar(y[0] - yexact[0]);
  PetscFunctionReturn(0);
}

static PetscErrorCode CreateMesh(MPI_Comm comm, AppCtx *user, DM *dm)
{
  PetscFunctionBeginUser;
  PetscCall(DMCreate(comm, dm));
  PetscCall(DMSetType(*dm, DMPLEX));
#if 0
  PetscBool       periodic = user->bd[0] == DM_BOUNDARY_PERIODIC || user->bd[0] == DM_BOUNDARY_TWIST || user->bd[1] == DM_BOUNDARY_PERIODIC || user->bd[1] == DM_BOUNDARY_TWIST ? PETSC_TRUE : PETSC_FALSE;
  const PetscReal L[3]     = {1.0, 1.0, 1.0};
  PetscReal       maxCell[3];
  PetscInt        d;

  if (periodic) {for (d = 0; d < 3; ++d) maxCell[d] = 1.1*(L[d]/cells[d]); PetscCall(DMSetPeriodicity(*dm, PETSC_TRUE, maxCell, L, user->bd));}
#endif
  PetscCall(DMSetFromOptions(*dm));
  PetscCall(DMViewFromOptions(*dm, NULL, "-orig_dm_view"));
  PetscFunctionReturn(0);
}

static PetscErrorCode SetupBC(DM dm, AppCtx *user)
{
  PetscErrorCode (*exactFuncs[2])(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nf, PetscScalar *u, void *ctx);
  DMBoundaryType bdt[3] = {DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE};
  PetscDS        prob;
  DMLabel        label;
  PetscBool      check;
  PetscInt       dim, n = 3;
  const char    *prefix;

  PetscFunctionBeginUser;
  PetscCall(PetscObjectGetOptionsPrefix((PetscObject) dm, &prefix));
  PetscCall(PetscOptionsGetEnumArray(NULL, prefix, "-dm_plex_box_bd", DMBoundaryTypes, (PetscEnum *) bdt, &n, NULL));
  PetscCall(DMGetDimension(dm, &dim));
  /* Set initial guesses and exact solutions */
  switch (dim) {
    case 2:
      user->initialGuess[0] = initialVelocity;
      switch(user->porosityDist) {
        case ZERO:     user->initialGuess[1] = zero_phi;break;
        case CONSTANT: user->initialGuess[1] = constant_phi;break;
        case GAUSSIAN: user->initialGuess[1] = gaussian_phi_2d;break;
        case DELTA:    user->initialGuess[1] = delta_phi_2d;break;
        case TILTED:
        if (user->velocityDist == VEL_ZERO) user->initialGuess[1] = tilted_phi_2d;
        else                                user->initialGuess[1] = tilted_phi_coupled_2d;
        break;
      }
      break;
    default: SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_SUP, "Dimension %D not supported", dim);
  }
  exactFuncs[0] = user->initialGuess[0];
  exactFuncs[1] = user->initialGuess[1];
  switch (dim) {
    case 2:
      switch (user->velocityDist) {
        case VEL_ZERO:
          exactFuncs[0] = zero_u_2d; break;
        case VEL_CONSTANT:
          exactFuncs[0] = constant_u_2d; break;
        case VEL_HARMONIC:
          switch (bdt[0]) {
            case DM_BOUNDARY_PERIODIC:
              switch (bdt[1]) {
                case DM_BOUNDARY_PERIODIC:
                  exactFuncs[0] = doubly_periodic_u_2d; break;
                default:
                  exactFuncs[0] = periodic_u_2d; break;
              }
              break;
            default:
              exactFuncs[0] = quadratic_u_2d; break;
          }
          break;
        case VEL_SHEAR:
          exactFuncs[0] = shear_bc; break;
        default: SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE, "Invalid dimension %d", dim);
      }
      break;
    default: SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_SUP, "Dimension %D not supported", dim);
  }
  {
    PetscBool isImplicit = PETSC_FALSE;

    PetscCall(PetscOptionsHasName(NULL,"", "-use_implicit", &isImplicit));
    if (user->velocityDist == VEL_CONSTANT && !isImplicit) user->initialGuess[0] = exactFuncs[0];
  }
  PetscCall(PetscOptionsHasName(NULL,NULL, "-dmts_check", &check));
  if (check) {
    user->initialGuess[0] = exactFuncs[0];
    user->initialGuess[1] = exactFuncs[1];
  }
  /* Set BC */
  PetscCall(DMGetDS(dm, &prob));
  PetscCall(DMGetLabel(dm, "marker", &label));
  PetscCall(PetscDSSetExactSolution(prob, 0, exactFuncs[0], user));
  PetscCall(PetscDSSetExactSolution(prob, 1, exactFuncs[1], user));
  if (label) {
    const PetscInt id = 1;

    PetscCall(DMAddBoundary(dm, DM_BC_ESSENTIAL, "wall", label, 1, &id, 0, 0, NULL, (void (*)(void)) exactFuncs[0], NULL, user, NULL));
  }
  PetscCall(DMGetLabel(dm, "Face Sets", &label));
  if (label && user->useFV) {
    const PetscInt inflowids[] = {100,200,300}, outflowids[] = {101};

    PetscCall(DMAddBoundary(dm, DM_BC_NATURAL_RIEMANN, "inflow",  label,  ALEN(inflowids),  inflowids, 1, 0, NULL, (void (*)(void)) advect_inflow, NULL, user, NULL));
    PetscCall(DMAddBoundary(dm, DM_BC_NATURAL_RIEMANN, "outflow", label, ALEN(outflowids), outflowids, 1, 0, NULL, (void (*)(void)) advect_outflow, NULL, user, NULL));
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode SetupProblem(DM dm, AppCtx *user)
{
  DMBoundaryType bdt[3] = {DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE};
  PetscDS        prob;
  PetscInt       n = 3;
  const char    *prefix;

  PetscFunctionBeginUser;
  PetscCall(PetscObjectGetOptionsPrefix((PetscObject) dm, &prefix));
  PetscCall(PetscOptionsGetEnumArray(NULL, prefix, "-dm_plex_box_bd", DMBoundaryTypes, (PetscEnum *) bdt, &n, NULL));
  PetscCall(DMGetDS(dm, &prob));
  switch (user->velocityDist) {
  case VEL_ZERO:
    PetscCall(PetscDSSetResidual(prob, 0, f0_zero_u, f1_constant_u));
    break;
  case VEL_CONSTANT:
    PetscCall(PetscDSSetResidual(prob, 0, f0_constant_u, f1_constant_u));
    PetscCall(PetscDSSetJacobian(prob, 0, 0, g0_constant_uu, NULL, NULL, NULL));
    PetscCall(PetscDSSetJacobian(prob, 1, 1, g0_constant_pp, NULL, NULL, NULL));
    break;
  case VEL_HARMONIC:
    switch (bdt[0]) {
    case DM_BOUNDARY_PERIODIC:
      switch (bdt[1]) {
      case DM_BOUNDARY_PERIODIC:
        PetscCall(PetscDSSetResidual(prob, 0, f0_lap_doubly_periodic_u, f1_lap_u));
        break;
      default:
        PetscCall(PetscDSSetResidual(prob, 0, f0_lap_periodic_u, f1_lap_u));
        break;
      }
      break;
    default:
      PetscCall(PetscDSSetResidual(prob, 0, f0_lap_u, f1_lap_u));
      break;
    }
    PetscCall(PetscDSSetJacobian(prob, 0, 0, NULL, NULL, NULL, g3_uu));
    break;
  case VEL_SHEAR:
    PetscCall(PetscDSSetResidual(prob, 0, f0_zero_u, f1_lap_u));
    PetscCall(PetscDSSetJacobian(prob, 0, 0, NULL, NULL, NULL, g3_uu));
    break;
  }
  PetscCall(PetscDSSetResidual(prob, 1, f0_advection, f1_advection));
  PetscCall(PetscDSSetJacobian(prob, 1, 1, g0_adv_pp, g1_adv_pp, NULL, NULL));
  PetscCall(PetscDSSetJacobian(prob, 1, 0, g0_adv_pu, g1_adv_pu, NULL, NULL));
  if (user->velocityDist == VEL_ZERO) PetscCall(PetscDSSetRiemannSolver(prob, 1, riemann_advection));
  else                                PetscCall(PetscDSSetRiemannSolver(prob, 1, riemann_coupled_advection));

  PetscCall(FunctionalRegister(&user->functionalRegistry, "Error", &user->errorFunctional, Functional_Error, user));
  PetscFunctionReturn(0);
}

static PetscErrorCode SetupDiscretization(DM dm, AppCtx *user)
{
  DM              cdm = dm;
  PetscQuadrature q;
  PetscFE         fe[2];
  PetscFV         fv;
  MPI_Comm        comm;
  PetscInt        dim;

  PetscFunctionBeginUser;
  /* Create finite element */
  PetscCall(PetscObjectGetComm((PetscObject) dm, &comm));
  PetscCall(DMGetDimension(dm, &dim));
  PetscCall(PetscFECreateDefault(comm, dim, dim, PETSC_FALSE, "velocity_", PETSC_DEFAULT, &fe[0]));
  PetscCall(PetscObjectSetName((PetscObject) fe[0], "velocity"));
  PetscCall(PetscFECreateDefault(comm, dim, 1, PETSC_FALSE, "porosity_", PETSC_DEFAULT, &fe[1]));
  PetscCall(PetscFECopyQuadrature(fe[0], fe[1]));
  PetscCall(PetscObjectSetName((PetscObject) fe[1], "porosity"));

  PetscCall(PetscFVCreate(PetscObjectComm((PetscObject) dm), &fv));
  PetscCall(PetscObjectSetName((PetscObject) fv, "porosity"));
  PetscCall(PetscFVSetFromOptions(fv));
  PetscCall(PetscFVSetNumComponents(fv, 1));
  PetscCall(PetscFVSetSpatialDimension(fv, dim));
  PetscCall(PetscFEGetQuadrature(fe[0], &q));
  PetscCall(PetscFVSetQuadrature(fv, q));

  PetscCall(DMSetField(dm, 0, NULL, (PetscObject) fe[0]));
  if (user->useFV) PetscCall(DMSetField(dm, 1, NULL, (PetscObject) fv));
  else             PetscCall(DMSetField(dm, 1, NULL, (PetscObject) fe[1]));
  PetscCall(DMCreateDS(dm));
  PetscCall(SetupProblem(dm, user));

  /* Set discretization and boundary conditions for each mesh */
  while (cdm) {
    PetscCall(DMCopyDisc(dm, cdm));
    PetscCall(DMGetCoarseDM(cdm, &cdm));
    /* Coordinates were never localized for coarse meshes */
    if (cdm) PetscCall(DMLocalizeCoordinates(cdm));
  }
  PetscCall(PetscFEDestroy(&fe[0]));
  PetscCall(PetscFEDestroy(&fe[1]));
  PetscCall(PetscFVDestroy(&fv));
  PetscFunctionReturn(0);
}

static PetscErrorCode CreateDM(MPI_Comm comm, AppCtx *user, DM *dm)
{
  PetscFunctionBeginUser;
  PetscCall(CreateMesh(comm, user, dm));
  /* Handle refinement, etc. */
  PetscCall(DMSetFromOptions(*dm));
  /* Construct ghost cells */
  if (user->useFV) {
    DM gdm;

    PetscCall(DMPlexConstructGhostCells(*dm, NULL, NULL, &gdm));
    PetscCall(DMDestroy(dm));
    *dm  = gdm;
  }
  /* Localize coordinates */
  PetscCall(DMLocalizeCoordinates(*dm));
  PetscCall(PetscObjectSetName((PetscObject)(*dm),"Mesh"));
  PetscCall(DMViewFromOptions(*dm, NULL, "-dm_view"));
  /* Setup problem */
  PetscCall(SetupDiscretization(*dm, user));
  /* Setup BC */
  PetscCall(SetupBC(*dm, user));
  PetscFunctionReturn(0);
}

static PetscErrorCode SetInitialConditionFVM(DM dm, Vec X, PetscInt field, PetscErrorCode (*func)(PetscInt, PetscReal, const PetscReal [], PetscInt, PetscScalar *, void *), void *ctx)
{
  PetscDS            prob;
  DM                 dmCell;
  Vec                cellgeom;
  const PetscScalar *cgeom;
  PetscScalar       *x;
  PetscInt           dim, Nf, cStart, cEnd, c;

  PetscFunctionBeginUser;
  PetscCall(DMGetDS(dm, &prob));
  PetscCall(DMGetDimension(dm, &dim));
  PetscCall(PetscDSGetNumFields(prob, &Nf));
  PetscCall(DMPlexGetGeometryFVM(dm, NULL, &cellgeom, NULL));
  PetscCall(VecGetDM(cellgeom, &dmCell));
  PetscCall(DMPlexGetSimplexOrBoxCells(dm, 0, &cStart, &cEnd));
  PetscCall(VecGetArrayRead(cellgeom, &cgeom));
  PetscCall(VecGetArray(X, &x));
  for (c = cStart; c < cEnd; ++c) {
    PetscFVCellGeom       *cg;
    PetscScalar           *xc;

    PetscCall(DMPlexPointLocalRead(dmCell, c, cgeom, &cg));
    PetscCall(DMPlexPointGlobalFieldRef(dm, c, field, x, &xc));
    if (xc) PetscCall((*func)(dim, 0.0, cg->centroid, Nf, xc, ctx));
  }
  PetscCall(VecRestoreArrayRead(cellgeom, &cgeom));
  PetscCall(VecRestoreArray(X, &x));
  PetscFunctionReturn(0);
}

static PetscErrorCode MonitorFunctionals(TS ts, PetscInt stepnum, PetscReal time, Vec X, void *ctx)
{
  AppCtx            *user   = (AppCtx *) ctx;
  char              *ftable = NULL;
  DM                 dm;
  PetscSection       s;
  Vec                cellgeom;
  const PetscScalar *x;
  PetscScalar       *a;
  PetscReal         *xnorms;
  PetscInt           pStart, pEnd, p, Nf, f;

  PetscFunctionBeginUser;
  PetscCall(VecViewFromOptions(X, (PetscObject) ts, "-view_solution"));
  PetscCall(VecGetDM(X, &dm));
  PetscCall(DMPlexGetGeometryFVM(dm, NULL, &cellgeom, NULL));
  PetscCall(DMGetLocalSection(dm, &s));
  PetscCall(PetscSectionGetNumFields(s, &Nf));
  PetscCall(PetscSectionGetChart(s, &pStart, &pEnd));
  PetscCall(PetscCalloc1(Nf*2, &xnorms));
  PetscCall(VecGetArrayRead(X, &x));
  for (p = pStart; p < pEnd; ++p) {
    for (f = 0; f < Nf; ++f) {
      PetscInt dof, cdof, d;

      PetscCall(PetscSectionGetFieldDof(s, p, f, &dof));
      PetscCall(PetscSectionGetFieldConstraintDof(s, p, f, &cdof));
      PetscCall(DMPlexPointGlobalFieldRead(dm, p, f, x, &a));
      /* TODO Use constrained indices here */
      for (d = 0; d < dof-cdof; ++d) xnorms[f*2+0]  = PetscMax(xnorms[f*2+0], PetscAbsScalar(a[d]));
      for (d = 0; d < dof-cdof; ++d) xnorms[f*2+1] += PetscAbsScalar(a[d]);
    }
  }
  PetscCall(VecRestoreArrayRead(X, &x));
  if (stepnum >= 0) { /* No summary for final time */
    DM                 dmCell, *fdm;
    Vec               *fv;
    const PetscScalar *cgeom;
    PetscScalar      **fx;
    PetscReal         *fmin, *fmax, *fint, *ftmp, t;
    PetscInt           cStart, cEnd, c, fcount, f, num;

    size_t             ftableused,ftablealloc;

    /* Functionals have indices after registering, this is an upper bound */
    fcount = user->numMonitorFuncs;
    PetscCall(PetscMalloc4(fcount,&fmin,fcount,&fmax,fcount,&fint,fcount,&ftmp));
    PetscCall(PetscMalloc3(fcount,&fdm,fcount,&fv,fcount,&fx));
    for (f = 0; f < fcount; ++f) {
      PetscSection fs;
      const char  *name = user->monitorFuncs[f]->name;

      fmin[f] = PETSC_MAX_REAL;
      fmax[f] = PETSC_MIN_REAL;
      fint[f] = 0;
      /* Make monitor vecs */
      PetscCall(DMClone(dm, &fdm[f]));
      PetscCall(DMGetOutputSequenceNumber(dm, &num, &t));
      PetscCall(DMSetOutputSequenceNumber(fdm[f], num, t));
      PetscCall(PetscSectionClone(s, &fs));
      PetscCall(PetscSectionSetFieldName(fs, 0, NULL));
      PetscCall(PetscSectionSetFieldName(fs, 1, name));
      PetscCall(DMSetLocalSection(fdm[f], fs));
      PetscCall(PetscSectionDestroy(&fs));
      PetscCall(DMGetGlobalVector(fdm[f], &fv[f]));
      PetscCall(PetscObjectSetName((PetscObject) fv[f], name));
      PetscCall(VecGetArray(fv[f], &fx[f]));
    }
    PetscCall(DMPlexGetSimplexOrBoxCells(dm, 0, &cStart, &cEnd));
    PetscCall(VecGetDM(cellgeom, &dmCell));
    PetscCall(VecGetArrayRead(cellgeom, &cgeom));
    PetscCall(VecGetArrayRead(X, &x));
    for (c = cStart; c < cEnd; ++c) {
      PetscFVCellGeom *cg;
      PetscScalar     *cx;

      PetscCall(DMPlexPointLocalRead(dmCell, c, cgeom, &cg));
      PetscCall(DMPlexPointGlobalFieldRead(dm, c, 1, x, &cx));
      if (!cx) continue;        /* not a global cell */
      for (f = 0;  f < user->numMonitorFuncs; ++f) {
        Functional   func = user->monitorFuncs[f];
        PetscScalar *fxc;

        PetscCall(DMPlexPointGlobalFieldRef(dm, c, 1, fx[f], &fxc));
        /* I need to make it easier to get interpolated values here */
        PetscCall((*func->func)(dm, time, cg->centroid, cx, ftmp, func->ctx));
        fxc[0] = ftmp[user->monitorFuncs[f]->offset];
      }
      for (f = 0; f < fcount; ++f) {
        fmin[f]  = PetscMin(fmin[f], ftmp[f]);
        fmax[f]  = PetscMax(fmax[f], ftmp[f]);
        fint[f] += cg->volume * ftmp[f];
      }
    }
    PetscCall(VecRestoreArrayRead(cellgeom, &cgeom));
    PetscCall(VecRestoreArrayRead(X, &x));
    PetscCallMPI(MPI_Allreduce(MPI_IN_PLACE, fmin, fcount, MPIU_REAL, MPIU_MIN, PetscObjectComm((PetscObject)ts)));
    PetscCallMPI(MPI_Allreduce(MPI_IN_PLACE, fmax, fcount, MPIU_REAL, MPIU_MAX, PetscObjectComm((PetscObject)ts)));
    PetscCallMPI(MPI_Allreduce(MPI_IN_PLACE, fint, fcount, MPIU_REAL, MPIU_SUM, PetscObjectComm((PetscObject)ts)));
    /* Output functional data */
    ftablealloc = fcount * 100;
    ftableused  = 0;
    PetscCall(PetscCalloc1(ftablealloc, &ftable));
    for (f = 0; f < user->numMonitorFuncs; ++f) {
      Functional func      = user->monitorFuncs[f];
      PetscInt   id        = func->offset;
      char       newline[] = "\n";
      char       buffer[256], *p, *prefix;
      size_t     countused, len;

      /* Create string with functional outputs */
      if (f % 3) {
        PetscCall(PetscArraycpy(buffer, "  ", 2));
        p    = buffer + 2;
      } else if (f) {
        PetscCall(PetscArraycpy(buffer, newline, sizeof(newline)-1));
        p    = buffer + sizeof(newline) - 1;
      } else {
        p = buffer;
      }
      PetscCall(PetscSNPrintfCount(p, sizeof buffer-(p-buffer), "%12s [%12.6g,%12.6g] int %12.6g", &countused, func->name, (double) fmin[id], (double) fmax[id], (double) fint[id]));
      countused += p - buffer;
      /* reallocate */
      if (countused > ftablealloc-ftableused-1) {
        char *ftablenew;

        ftablealloc = 2*ftablealloc + countused;
        PetscCall(PetscMalloc1(ftablealloc, &ftablenew));
        PetscCall(PetscArraycpy(ftablenew, ftable, ftableused));
        PetscCall(PetscFree(ftable));
        ftable = ftablenew;
      }
      PetscCall(PetscArraycpy(ftable+ftableused, buffer, countused));
      ftableused += countused;
      ftable[ftableused] = 0;
      /* Output vecs */
      PetscCall(VecRestoreArray(fv[f], &fx[f]));
      PetscCall(PetscStrlen(func->name, &len));
      PetscCall(PetscMalloc1(len+2,&prefix));
      PetscCall(PetscStrcpy(prefix, func->name));
      PetscCall(PetscStrcat(prefix, "_"));
      PetscCall(PetscObjectSetOptionsPrefix((PetscObject)fv[f], prefix));
      PetscCall(VecViewFromOptions(fv[f], NULL, "-vec_view"));
      PetscCall(PetscFree(prefix));
      PetscCall(DMRestoreGlobalVector(fdm[f], &fv[f]));
      PetscCall(DMDestroy(&fdm[f]));
    }
    PetscCall(PetscFree4(fmin, fmax, fint, ftmp));
    PetscCall(PetscFree3(fdm, fv, fx));
    PetscCall(PetscPrintf(PetscObjectComm((PetscObject) ts), "% 3D  time %8.4g  |x| (", stepnum, (double) time));
    for (f = 0; f < Nf; ++f) {
      if (f > 0) PetscCall(PetscPrintf(PetscObjectComm((PetscObject) ts), ", "));
      PetscCall(PetscPrintf(PetscObjectComm((PetscObject) ts), "%8.4g", (double) xnorms[f*2+0]));
    }
    PetscCall(PetscPrintf(PetscObjectComm((PetscObject) ts), ") |x|_1 ("));
    for (f = 0; f < Nf; ++f) {
      if (f > 0) PetscCall(PetscPrintf(PetscObjectComm((PetscObject) ts), ", "));
      PetscCall(PetscPrintf(PetscObjectComm((PetscObject) ts), "%8.4g", (double) xnorms[f*2+1]));
    }
    PetscCall(PetscPrintf(PetscObjectComm((PetscObject) ts), ")  %s\n", ftable ? ftable : ""));
    PetscCall(PetscFree(ftable));
  }
  PetscCall(PetscFree(xnorms));
  PetscFunctionReturn(0);
}

int main(int argc, char **argv)
{
  MPI_Comm       comm;
  TS             ts;
  DM             dm;
  Vec            u;
  AppCtx         user;
  PetscReal      t0, t = 0.0;
  void          *ctxs[2];

  ctxs[0] = &t;
  ctxs[1] = &t;
  PetscCall(PetscInitialize(&argc, &argv, (char*) 0, help));
  comm = PETSC_COMM_WORLD;
  user.functionalRegistry = NULL;
  globalUser = &user;
  PetscCall(ProcessOptions(comm, &user));
  PetscCall(TSCreate(comm, &ts));
  PetscCall(TSSetType(ts, TSBEULER));
  PetscCall(CreateDM(comm, &user, &dm));
  PetscCall(TSSetDM(ts, dm));
  PetscCall(ProcessMonitorOptions(comm, &user));

  PetscCall(DMCreateGlobalVector(dm, &u));
  PetscCall(PetscObjectSetName((PetscObject) u, "solution"));
  if (user.useFV) {
    PetscBool isImplicit = PETSC_FALSE;

    PetscCall(PetscOptionsHasName(NULL,"", "-use_implicit", &isImplicit));
    if (isImplicit) {
      PetscCall(DMTSSetIFunctionLocal(dm, DMPlexTSComputeIFunctionFEM, &user));
      PetscCall(DMTSSetIJacobianLocal(dm, DMPlexTSComputeIJacobianFEM, &user));
    }
    PetscCall(DMTSSetBoundaryLocal(dm, DMPlexTSComputeBoundary, &user));
    PetscCall(DMTSSetRHSFunctionLocal(dm, DMPlexTSComputeRHSFunctionFVM, &user));
  } else {
    PetscCall(DMTSSetBoundaryLocal(dm, DMPlexTSComputeBoundary, &user));
    PetscCall(DMTSSetIFunctionLocal(dm, DMPlexTSComputeIFunctionFEM, &user));
    PetscCall(DMTSSetIJacobianLocal(dm, DMPlexTSComputeIJacobianFEM, &user));
  }
  if (user.useFV) PetscCall(TSMonitorSet(ts, MonitorFunctionals, &user, NULL));
  PetscCall(TSSetMaxSteps(ts, 1));
  PetscCall(TSSetMaxTime(ts, 2.0));
  PetscCall(TSSetTimeStep(ts,0.01));
  PetscCall(TSSetExactFinalTime(ts,TS_EXACTFINALTIME_STEPOVER));
  PetscCall(TSSetFromOptions(ts));

  PetscCall(DMProjectFunction(dm, 0.0, user.initialGuess, ctxs, INSERT_VALUES, u));
  if (user.useFV) PetscCall(SetInitialConditionFVM(dm, u, 1, user.initialGuess[1], ctxs[1]));
  PetscCall(VecViewFromOptions(u, NULL, "-init_vec_view"));
  PetscCall(TSGetTime(ts, &t));
  t0   = t;
  PetscCall(DMTSCheckFromOptions(ts, u));
  PetscCall(TSSolve(ts, u));
  PetscCall(TSGetTime(ts, &t));
  if (t > t0) PetscCall(DMTSCheckFromOptions(ts, u));
  PetscCall(VecViewFromOptions(u, NULL, "-sol_vec_view"));
  {
    PetscReal ftime;
    PetscInt  nsteps;
    TSConvergedReason reason;

    PetscCall(TSGetSolveTime(ts, &ftime));
    PetscCall(TSGetStepNumber(ts, &nsteps));
    PetscCall(TSGetConvergedReason(ts, &reason));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "%s at time %g after %D steps\n", TSConvergedReasons[reason], (double) ftime, nsteps));
  }

  PetscCall(VecDestroy(&u));
  PetscCall(DMDestroy(&dm));
  PetscCall(TSDestroy(&ts));
  PetscCall(PetscFree(user.monitorFuncs));
  PetscCall(FunctionalDestroy(&user.functionalRegistry));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

  testset:
    args: -dm_plex_simplex 0 -dm_plex_box_faces 3,3,3

    # 2D harmonic velocity, no porosity
    test:
      suffix: p1p1
      requires: !complex !single
      args: -velocity_petscspace_degree 1 -porosity_petscspace_degree 1 -snes_fd_color -snes_fd_color_use_mat -mat_coloring_type greedy -pc_factor_shift_type nonzero -ts_monitor -snes_error_if_not_converged -ksp_error_if_not_converged -dmts_check
    test:
      suffix: p1p1_xper
      requires: !complex !single
      args: -dm_refine 1 -dm_plex_box_bd periodic,none -velocity_petscspace_degree 1 -porosity_petscspace_degree 1 -snes_fd_color -snes_fd_color_use_mat -mat_coloring_type greedy -pc_type lu -pc_factor_shift_type nonzero -ksp_rtol 1.0e-8 -ts_monitor -snes_error_if_not_converged -ksp_error_if_not_converged -dmts_check
    test:
      suffix: p1p1_xper_ref
      requires: !complex !single
      args: -dm_refine 2 -dm_plex_box_bd periodic,none -velocity_petscspace_degree 1 -porosity_petscspace_degree 1 -snes_fd_color -snes_fd_color_use_mat -mat_coloring_type greedy -pc_type lu -pc_factor_shift_type nonzero -ksp_rtol 1.0e-8 -ts_monitor -snes_error_if_not_converged -ksp_error_if_not_converged -dmts_check
    test:
      suffix: p1p1_xyper
      requires: !complex !single
      args: -dm_refine 1 -dm_plex_box_bd periodic,periodic -velocity_petscspace_degree 1 -porosity_petscspace_degree 1 -snes_fd_color -snes_fd_color_use_mat -mat_coloring_type greedy -pc_type lu -pc_factor_shift_type nonzero -ksp_rtol 1.0e-8 -ts_monitor -snes_error_if_not_converged -ksp_error_if_not_converged -dmts_check
    test:
      suffix: p1p1_xyper_ref
      requires: !complex !single
      args: -dm_refine 2 -dm_plex_box_bd periodic,periodic -velocity_petscspace_degree 1 -porosity_petscspace_degree 1 -snes_fd_color -snes_fd_color_use_mat -mat_coloring_type greedy -pc_type lu -pc_factor_shift_type nonzero -ksp_rtol 1.0e-8 -ts_monitor -snes_error_if_not_converged -ksp_error_if_not_converged -dmts_check
    test:
      suffix: p2p1
      requires: !complex !single
      args: -velocity_petscspace_degree 2 -porosity_petscspace_degree 1 -snes_fd_color -snes_fd_color_use_mat -mat_coloring_type greedy -ts_monitor   -snes_error_if_not_converged -ksp_error_if_not_converged -dmts_check
    test:
      suffix: p2p1_xyper
      requires: !complex !single
      args: -dm_refine 1 -dm_plex_box_bd periodic,periodic -velocity_petscspace_degree 2 -porosity_petscspace_degree 1 -snes_fd_color -snes_fd_color_use_mat -mat_coloring_type greedy -pc_type lu -pc_factor_shift_type nonzero -ksp_rtol 1.0e-8 -ts_monitor -snes_error_if_not_converged -ksp_error_if_not_converged -dmts_check

    test:
      suffix: adv_1
      requires: !complex !single
      args: -use_fv -velocity_dist zero -porosity_dist tilted -ts_type ssp -ts_max_time 2.0 -ts_max_steps 1000 -ts_dt 0.993392 -bc_inflow 1,2,4 -bc_outflow 3 -ts_view -dm_view

    test:
      suffix: adv_2
      requires: !complex
      TODO: broken memory corruption
      args: -use_fv -velocity_dist zero -porosity_dist tilted -ts_type beuler -ts_max_time 2.0 -ts_max_steps 1000 -ts_dt 0.993392 -bc_inflow 3,4 -bc_outflow 1,2 -snes_fd_color -snes_fd_color_use_mat -mat_coloring_type greedy -ksp_max_it 100 -ts_view -dm_view -snes_converged_reason -ksp_converged_reason

    test:
      suffix: adv_3
      requires: !complex
      TODO: broken memory corruption
      args: -dm_plex_box_bd periodic,none -use_fv -velocity_dist zero -porosity_dist tilted -ts_type beuler -ts_max_time 2.0 -ts_max_steps 1000 -ts_dt 0.993392 -bc_inflow 3 -bc_outflow 1 -snes_fd_color -snes_fd_color_use_mat -mat_coloring_type greedy -ksp_max_it 100 -ts_view -dm_view -snes_converged_reason

    test:
      suffix: adv_3_ex
      requires: !complex
      args: -dm_plex_box_bd periodic,none -use_fv -velocity_dist zero -porosity_dist tilted -ts_type ssp -ts_max_time 2.0 -ts_max_steps 1000 -ts_dt   0.1 -bc_inflow 3 -bc_outflow 1 -snes_fd_color -ksp_max_it 100 -ts_view -dm_view

    test:
      suffix: adv_4
      requires: !complex
      TODO: broken memory corruption
      args: -use_fv -velocity_dist zero -porosity_dist tilted -ts_type beuler -ts_max_time 2.0 -ts_max_steps 1000 -ts_dt 0.993392 -bc_inflow 3 -bc_outflow 1 -snes_fd_color -snes_fd_color_use_mat -mat_coloring_type greedy -ksp_max_it 100 -ts_view -dm_view -snes_converged_reason

    # 2D Advection, box, delta
    test:
      suffix: adv_delta_yper_0
      requires: !complex
      TODO: broken
      args: -dm_plex_box_bd none,periodic -use_fv -velocity_dist constant -porosity_dist delta -inflow_state 0.0 -ts_type euler -ts_max_time 5.0 -ts_max_steps 20 -ts_dt 0.333333 -bc_inflow 2 -bc_outflow 4 -ts_view -dm_view -monitor Error

    test:
      suffix: adv_delta_yper_1
      requires: !complex
      TODO: broken
      args: -dm_plex_box_bd none,periodic -use_fv -velocity_dist constant -porosity_dist delta -inflow_state 0.0 -ts_type euler -ts_max_time 5.0 -ts_max_steps 40 -ts_dt 0.166666 -bc_inflow 2 -bc_outflow 4 -ts_view -dm_view -monitor Error -dm_refine 1 -source_loc 0.416666,0.416666

    test:
      suffix: adv_delta_yper_2
      requires: !complex
      TODO: broken
      args: -dm_plex_box_bd none,periodic -use_fv -velocity_dist constant -porosity_dist delta -inflow_state 0.0 -ts_type euler -ts_max_time 5.0 -ts_max_steps 80 -ts_dt 0.083333 -bc_inflow 2 -bc_outflow 4 -ts_view -dm_view -monitor Error -dm_refine 2 -source_loc 0.458333,0.458333

    test:
      suffix: adv_delta_yper_fim_0
      requires: !complex
      TODO: broken
      args: -dm_plex_box_bd none,periodic -use_fv -use_implicit -velocity_petscspace_degree 0 -velocity_dist constant -porosity_dist delta -inflow_state 0.0 -ts_type mimex -ts_max_time 5.0 -ts_max_steps 20 -ts_dt 0.333333 -bc_inflow 2 -bc_outflow 4 -ts_view -monitor Error -dm_view   -snes_fd_color -snes_fd_color_use_mat -mat_coloring_type greedy -mat_coloring_greedy_symmetric 0 -snes_rtol 1.0e-7 -pc_type lu -snes_converged_reason

    test:
      suffix: adv_delta_yper_fim_1
      requires: !complex
      TODO: broken
      args: -dm_plex_box_bd none,periodic -use_fv -use_implicit -velocity_petscspace_degree 1 -velocity_dist constant -porosity_dist delta -inflow_state 0.0 -ts_type mimex -ts_max_time 5.0 -ts_max_steps 20 -ts_dt 0.333333 -bc_inflow 2 -bc_outflow 4 -ts_view -monitor Error -dm_view   -snes_fd_color -snes_fd_color_use_mat -mat_coloring_type greedy -mat_coloring_greedy_symmetric 0 -snes_rtol 1.0e-7 -pc_type lu -snes_converged_reason -snes_linesearch_type basic

    test:
      suffix: adv_delta_yper_fim_2
      requires: !complex
      TODO: broken
      args: -dm_plex_box_bd none,periodic -use_fv -use_implicit -velocity_petscspace_degree 2 -velocity_dist constant -porosity_dist delta -inflow_state 0.0 -ts_type mimex -ts_max_time 5.0 -ts_max_steps 20 -ts_dt 0.333333 -bc_inflow 2 -bc_outflow 4 -ts_view -monitor Error -dm_view   -snes_fd_color -snes_fd_color_use_mat -mat_coloring_type greedy -mat_coloring_greedy_symmetric 0 -snes_rtol 1.0e-7 -pc_type lu -snes_converged_reason -snes_linesearch_type basic

    test:
      suffix: adv_delta_yper_im_0
      requires: !complex
      TODO: broken
      args: -dm_plex_box_bd none,periodic -use_fv -use_implicit -velocity_petscspace_degree 0 -velocity_dist constant -porosity_dist delta -inflow_state 0.0 -ts_type mimex -ts_mimex_version 0 -ts_max_time 5.0 -ts_max_steps 20 -ts_dt 0.333333 -bc_inflow 2 -bc_outflow 4 -ts_view -monitor Error -dm_view -snes_fd_color -snes_fd_color_use_mat -mat_coloring_type greedy -snes_rtol 1.0e-7 -pc_type lu -snes_converged_reason

    test:
      suffix: adv_delta_yper_im_1
      requires: !complex
      TODO: broken
      args: -dm_plex_box_bd none,periodic -use_fv -use_implicit -velocity_petscspace_degree 0 -velocity_dist constant -porosity_dist delta -inflow_state 0.0 -ts_type mimex -ts_mimex_version 0 -ts_max_time 5.0 -ts_max_steps 40 -ts_dt 0.166666 -bc_inflow 2 -bc_outflow 4 -ts_view -monitor Error -dm_view -snes_fd_color -snes_fd_color_use_mat -mat_coloring_type greedy -snes_rtol 1.0e-7 -pc_type lu -snes_converged_reason -dm_refine 1 -source_loc 0.416666,0.416666

    test:
      suffix: adv_delta_yper_im_2
      requires: !complex
      TODO: broken
      args: -dm_plex_box_bd none,periodic -use_fv -use_implicit -velocity_petscspace_degree 0 -velocity_dist constant -porosity_dist delta -inflow_state 0.0 -ts_type mimex -ts_mimex_version 0 -ts_max_time 5.0 -ts_max_steps 80 -ts_dt 0.083333 -bc_inflow 2 -bc_outflow 4 -ts_view -monitor Error -dm_view -snes_fd_color -snes_fd_color_use_mat -mat_coloring_type greedy -snes_rtol 1.0e-7 -pc_type lu -snes_converged_reason -dm_refine 2 -source_loc 0.458333,0.458333

    test:
      suffix: adv_delta_yper_im_3
      requires: !complex
      TODO: broken
      args: -dm_plex_box_bd none,periodic -use_fv -use_implicit -velocity_petscspace_degree 1 -velocity_dist constant -porosity_dist delta -inflow_state 0.0 -ts_type mimex -ts_mimex_version 0 -ts_max_time 5.0 -ts_max_steps 20 -ts_dt 0.333333 -bc_inflow 2 -bc_outflow 4 -ts_view -monitor Error -dm_view -snes_fd_color -snes_fd_color_use_mat -mat_coloring_type greedy -snes_rtol 1.0e-7 -pc_type lu -snes_converged_reason

    #    I believe the nullspace is sin(pi y)
    test:
      suffix: adv_delta_yper_im_4
      requires: !complex
      TODO: broken
      args: -dm_plex_box_bd none,periodic -use_fv -use_implicit -velocity_petscspace_degree 1 -velocity_dist constant -porosity_dist delta -inflow_state 0.0 -ts_type mimex -ts_mimex_version 0 -ts_max_time 5.0 -ts_max_steps 40 -ts_dt 0.166666 -bc_inflow 2 -bc_outflow 4 -ts_view -monitor Error -dm_view -snes_fd_color -snes_fd_color_use_mat -mat_coloring_type greedy -snes_rtol 1.0e-7 -pc_type lu -snes_converged_reason -dm_refine 1 -source_loc 0.416666,0.416666

    test:
      suffix: adv_delta_yper_im_5
      requires: !complex
      TODO: broken
      args: -dm_plex_box_bd none,periodic -use_fv -use_implicit -velocity_petscspace_degree 1 -velocity_dist constant -porosity_dist delta -inflow_state 0.0 -ts_type mimex -ts_mimex_version 0 -ts_max_time 5.0 -ts_max_steps 80 -ts_dt 0.083333 -bc_inflow 2 -bc_outflow 4 -ts_view -monitor Error -dm_view -snes_fd_color -snes_fd_color_use_mat -mat_coloring_type greedy -snes_rtol 1.0e-7 -pc_type lu -snes_converged_reason -dm_refine 2 -source_loc 0.458333,0.458333

    test:
      suffix: adv_delta_yper_im_6
      requires: !complex
      TODO: broken
      args: -dm_plex_box_bd none,periodic -use_fv -use_implicit -velocity_petscspace_degree 2 -velocity_dist constant -porosity_dist delta -inflow_state 0.0 -ts_type mimex -ts_max_time 5.0 -ts_max_steps 20 -ts_dt 0.333333 -bc_inflow 2 -bc_outflow 4 -ts_view -monitor Error -dm_view   -snes_fd_color -snes_fd_color_use_mat -mat_coloring_type greedy -snes_rtol 1.0e-7 -pc_type svd -snes_converged_reason
    # 2D Advection, magma benchmark 1

    test:
      suffix: adv_delta_shear_im_0
      requires: !complex
      TODO: broken
      args: -dm_plex_box_bd periodic,none -dm_refine 2 -use_fv -use_implicit -velocity_petscspace_degree 1 -velocity_dist shear -porosity_dist   delta -inflow_state 0.0 -ts_type mimex -ts_max_time 5.0 -ts_max_steps 20 -ts_dt 0.333333 -bc_inflow 1,3 -ts_view -dm_view -snes_fd_color -snes_fd_color_use_mat -mat_coloring_type greedy -snes_rtol 1.0e-7 -pc_type lu -snes_converged_reason -source_loc 0.458333,0.708333
    # 2D Advection, box, gaussian

    test:
      suffix: adv_gauss
      requires: !complex
      TODO: broken
      args: -use_fv -velocity_dist constant -porosity_dist gaussian -inflow_state 0.0 -ts_type ssp -ts_max_time 2.0 -ts_max_steps 100 -ts_dt 0.01 -bc_inflow 1 -bc_outflow 3 -ts_view -dm_view

    test:
      suffix: adv_gauss_im
      requires: !complex
      TODO: broken
      args: -use_fv -use_implicit -velocity_dist constant -porosity_dist gaussian -inflow_state 0.0 -ts_type beuler -ts_max_time 2.0 -ts_max_steps   100 -ts_dt 0.01 -bc_inflow 1 -bc_outflow 3 -ts_view -dm_view -snes_fd_color -snes_fd_color_use_mat -mat_coloring_type greedy -snes_rtol 1.0e-7

    test:
      suffix: adv_gauss_im_1
      requires: !complex
      TODO: broken
      args: -use_fv -use_implicit -velocity_petscspace_degree 1 -velocity_dist constant -porosity_dist gaussian -inflow_state 0.0 -ts_type beuler -ts_max_time 2.0 -ts_max_steps 100 -ts_dt 0.01 -bc_inflow 1 -bc_outflow 3 -ts_view -dm_view -snes_fd_color -snes_fd_color_use_mat -mat_coloring_type greedy -snes_rtol 1.0e-7

    test:
      suffix: adv_gauss_im_2
      requires: !complex
      TODO: broken
      args: -use_fv -use_implicit -velocity_petscspace_degree 2 -velocity_dist constant -porosity_dist gaussian -inflow_state 0.0 -ts_type beuler -ts_max_time 2.0 -ts_max_steps 100 -ts_dt 0.01 -bc_inflow 1 -bc_outflow 3 -ts_view -dm_view -snes_fd_color -snes_fd_color_use_mat -mat_coloring_type greedy -snes_rtol 1.0e-7

    test:
      suffix: adv_gauss_corner
      requires: !complex
      TODO: broken
      args: -use_fv -velocity_dist constant -porosity_dist gaussian -inflow_state 0.0 -ts_type ssp -ts_max_time 2.0 -ts_max_steps 100 -ts_dt 0.01 -bc_inflow 1 -bc_outflow 2 -ts_view -dm_view

    # 2D Advection+Harmonic 12-
    test:
      suffix: adv_harm_0
      requires: !complex
      TODO: broken memory corruption
      args: -velocity_petscspace_degree 2 -use_fv -velocity_dist harmonic -porosity_dist gaussian -ts_type beuler -ts_max_time 2.0 -ts_max_steps   1000 -ts_dt 0.993392 -bc_inflow 1,2,4 -bc_outflow 3 -use_implicit -snes_fd_color -snes_fd_color_use_mat -mat_coloring_type greedy -ksp_max_it   100 -ts_view -dm_view -snes_converged_reason -ksp_converged_reason -snes_monitor -dmts_check

  #   Must check that FV BCs propagate to coarse meshes
  #   Must check that FV BC ids propagate to coarse meshes
  #   Must check that FE+FV BCs work at the same time
  # 2D Advection, matching wind in ex11 8-11
  #   NOTE implicit solves are limited by accuracy of FD Jacobian
  test:
    suffix: adv_0
    requires: !complex !single exodusii
    args: -dm_plex_filename ${wPETSC_DIR}/share/petsc/datafiles/meshes/sevenside-quad.exo -use_fv -velocity_dist zero -porosity_dist tilted -ts_type ssp -ts_max_time 2.0 -ts_max_steps 1000 -ts_dt 0.993392 -ts_view -dm_view

  test:
    suffix: adv_0_im
    requires: !complex exodusii
    TODO: broken  memory corruption
    args: -dm_plex_filename ${wPETSC_DIR}/share/petsc/datafiles/meshes/sevenside-quad.exo -use_fv -use_implicit -velocity_dist zero -porosity_dist tilted -ts_type beuler -ts_max_time 2.0 -ts_max_steps 1000 -ts_dt 0.993392 -ts_view -dm_view -snes_fd_color -snes_fd_color_use_mat -mat_coloring_type greedy -pc_type lu

  test:
    suffix: adv_0_im_2
    requires: !complex exodusii
    TODO: broken
    args: -dm_plex_filename ${wPETSC_DIR}/share/petsc/datafiles/meshes/sevenside-quad.exo -use_fv -use_implicit -velocity_dist constant -porosity_dist tilted -ts_type beuler -ts_max_time 2.0 -ts_max_steps 1000 -ts_dt 0.993392 -ts_view -dm_view -snes_fd_color -snes_fd_color_use_mat -mat_coloring_type greedy -pc_type lu -snes_rtol 1.0e-7

  test:
    suffix: adv_0_im_3
    requires: !complex exodusii
    TODO: broken
    args: -dm_plex_filename ${wPETSC_DIR}/share/petsc/datafiles/meshes/sevenside-quad.exo -use_fv -use_implicit -velocity_petscspace_degree 1 -velocity_dist constant -porosity_dist tilted -ts_type beuler -ts_max_time 2.0 -ts_max_steps 1000 -ts_dt 0.993392 -ts_view -dm_view -snes_fd_color -snes_fd_color_use_mat -mat_coloring_type greedy -pc_type svd -snes_rtol 1.0e-7

  test:
    suffix: adv_0_im_4
    requires: !complex exodusii
    TODO: broken
    args: -dm_plex_filename ${wPETSC_DIR}/share/petsc/datafiles/meshes/sevenside-quad.exo -use_fv -use_implicit -velocity_petscspace_degree 2 -velocity_dist constant -porosity_dist tilted -ts_type beuler -ts_max_time 2.0 -ts_max_steps 1000 -ts_dt 0.993392 -ts_view -dm_view -snes_fd_color -snes_fd_color_use_mat -mat_coloring_type greedy -pc_type svd -snes_rtol 1.0e-7
  # 2D Advection, misc

TEST*/
