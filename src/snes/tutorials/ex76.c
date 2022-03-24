static char help[] = "Low Mach Flow in 2d and 3d channels with finite elements.\n\
We solve the Low Mach flow problem in a rectangular\n\
domain, using a parallel unstructured mesh (DMPLEX) to discretize it.\n\n\n";

/*F
This Low Mach flow is a steady-state isoviscous Navier-Stokes flow. We discretize using the
finite element method on an unstructured mesh. The weak form equations are

\begin{align*}
    < q, \nabla\cdot u >                                                                                     = 0
    <v, u \cdot \nabla u> + < \nabla v, \nu (\nabla u + {\nabla u}^T) > - < \nabla\cdot v, p >  - < v, f  >  = 0
    < w, u \cdot \nabla T > - < \nabla w, \alpha \nabla T > - < w, Q >                                       = 0
\end{align*}

where $\nu$ is the kinematic viscosity and $\alpha$ is thermal diffusivity.

For visualization, use

  -dm_view hdf5:$PWD/sol.h5 -sol_vec_view hdf5:$PWD/sol.h5::append -exact_vec_view hdf5:$PWD/sol.h5::append
F*/

#include <petscdmplex.h>
#include <petscsnes.h>
#include <petscds.h>
#include <petscbag.h>

typedef enum {SOL_QUADRATIC, SOL_CUBIC, NUM_SOL_TYPES} SolType;
const char *solTypes[NUM_SOL_TYPES+1] = {"quadratic", "cubic",  "unknown"};

typedef struct {
  PetscReal nu;      /* Kinematic viscosity */
  PetscReal theta;   /* Angle of pipe wall to x-axis */
  PetscReal alpha;   /* Thermal diffusivity */
  PetscReal T_in;    /* Inlet temperature*/
} Parameter;

typedef struct {
  PetscBool showError;
  PetscBag  bag;
  SolType   solType;
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

    u = x^2 + y^2
    v = 2x^2 - 2xy
    p = x + y - 1
    T = x + y
    f = <2x^3 + 4x^2y - 2xy^2 -4\nu + 1,  4xy^2 + 2x^2y - 2y^3 -4\nu + 1>
    Q = 3x^2 + y^2 - 2xy

  so that

(1)  \nabla \cdot u  = 2x - 2x = 0

(2)  u \cdot \nabla u - \nu \Delta u + \nabla p - f
     = <2x^3 + 4x^2y -2xy^2, 4xy^2 + 2x^2y - 2y^3> -\nu <4, 4> + <1, 1> - <2x^3 + 4x^2y - 2xy^2 -4\nu + 1,  4xy^2 + 2x^2y - 2y^3 -         4\nu + 1>  = 0

(3) u \cdot \nabla T - \alpha \Delta T - Q = 3x^2 + y^2 - 2xy - \alpha*0 - 3x^2 - y^2 + 2xy = 0
*/

static PetscErrorCode quadratic_u(PetscInt Dim, PetscReal time, const PetscReal X[], PetscInt Nf, PetscScalar *u, void *ctx)
{
  u[0] = X[0]*X[0] + X[1]*X[1];
  u[1] = 2.0*X[0]*X[0] - 2.0*X[0]*X[1];
  return 0;
}

static PetscErrorCode linear_p(PetscInt Dim, PetscReal time, const PetscReal X[], PetscInt Nf, PetscScalar *p, void *ctx)
{
  p[0] = X[0] + X[1] - 1.0;
  return 0;
}

static PetscErrorCode linear_T(PetscInt Dim, PetscReal time, const PetscReal X[], PetscInt Nf, PetscScalar *T, void *ctx)
{
  T[0] = X[0] + X[1];
  return 0;
}

static void f0_quadratic_v(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                           const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                           const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                           PetscReal t, const PetscReal X[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f0[])
{
  PetscInt                   c, d;
  PetscInt                   Nc = dim;
  const PetscReal    nu = PetscRealPart(constants[0]);

  for (c=0; c<Nc; ++c) {
    for (d=0; d<dim; ++d) f0[c] += u[d]*u_x[c*dim+d];
  }
  f0[0] -= (2*X[0]*X[0]*X[0] + 4*X[0]*X[0]*X[1] - 2*X[0]*X[1]*X[1] - 4.0*nu + 1);
  f0[1] -= (4*X[0]*X[1]*X[1] + 2*X[0]*X[0]*X[1] - 2*X[1]*X[1]*X[1] - 4.0*nu + 1);
}

static void f0_quadratic_w(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                           const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                           const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                           PetscReal t, const PetscReal X[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f0[])
{
  PetscInt d;
  for (d = 0, f0[0] = 0; d < dim; ++d) f0[0] += u[uOff[0]+d]*u_x[uOff_x[2]+d];
  f0[0] -= (3*X[0]*X[0] + X[1]*X[1] - 2*X[0]*X[1]);
}

/*
  CASE: cubic
  In 2D we use exact solution:

    u = x^3 + y^3
    v = 2x^3 - 3x^2y
    p = 3/2 x^2 + 3/2 y^2 - 1
    T = 1/2 x^2 + 1/2 y^2
    f = <3x^5 + 6x^3y^2 - 6x^2y^3 - \nu(6x + 6y), 6x^2y^3 + 3x^4y - 6xy^4 - \nu(12x - 6y) + 3y>
    Q = x^4 + xy^3 + 2x^3y - 3x^2y^2 - 2

  so that

  \nabla \cdot u = 3x^2 - 3x^2 = 0

  u \cdot \nabla u - \nu \Delta u + \nabla p - f
  = <3x^5 + 6x^3y^2 - 6x^2y^3, 6x^2y^3 + 3x^4y - 6xy^4> - \nu<6x + 6y, 12x - 6y> + <3x, 3y> - <3x^5 + 6x^3y^2 - 6x^2y^3 - \nu(6x + 6y), 6x^2y^3 + 3x^4y - 6xy^4 - \nu(12x - 6y) + 3y> = 0

  u \cdot \nabla T - \alpha\Delta T - Q = (x^3 + y^3) x + (2x^3 - 3x^2y) y - 2*\alpha - (x^4 + xy^3 + 2x^3y - 3x^2y^2 - 2)   = 0
*/

static PetscErrorCode cubic_u(PetscInt Dim, PetscReal time, const PetscReal X[], PetscInt Nf, PetscScalar *u, void *ctx)
{
  u[0] = X[0]*X[0]*X[0] + X[1]*X[1]*X[1];
  u[1] = 2.0*X[0]*X[0]*X[0] - 3.0*X[0]*X[0]*X[1];
  return 0;
}

static PetscErrorCode quadratic_p(PetscInt Dim, PetscReal time, const PetscReal X[], PetscInt Nf, PetscScalar *p, void *ctx)
{
  p[0] = 3.0*X[0]*X[0]/2.0 + 3.0*X[1]*X[1]/2.0 - 1.0;
  return 0;
}

static PetscErrorCode quadratic_T(PetscInt Dim, PetscReal time, const PetscReal X[], PetscInt Nf, PetscScalar *T, void *ctx)
{
  T[0] = X[0]*X[0]/2.0 + X[1]*X[1]/2.0;
  return 0;
}

static void f0_cubic_v(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                       const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                       const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                       PetscReal t, const PetscReal X[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f0[])
{
  PetscInt                   c, d;
  PetscInt                   Nc = dim;
  const PetscReal    nu = PetscRealPart(constants[0]);

  for (c=0; c<Nc; ++c) {
    for (d=0; d<dim; ++d) f0[c] += u[d]*u_x[c*dim+d];
  }
  f0[0] -= (3*X[0]*X[0]*X[0]*X[0]*X[0] + 6*X[0]*X[0]*X[0]*X[1]*X[1] - 6*X[0]*X[0]*X[1]*X[1]*X[1] - (6*X[0]+6*X[1])*nu + 3*X[0]);
  f0[1] -= (6*X[0]*X[0]*X[1]*X[1]*X[1] + 3*X[0]*X[0]*X[0]*X[0]*X[1] - 6*X[0]*X[1]*X[1]*X[1]*X[1] - (12*X[0] - 6*X[1])*nu + 3*X[1]);
}

static void f0_cubic_w(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                       const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                       const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                       PetscReal t, const PetscReal X[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f0[])
{
  const PetscReal alpha = PetscRealPart(constants[1]);
  PetscInt        d;

  for (d = 0, f0[0] = 0; d < dim; ++d) f0[0] += u[uOff[0]+d]*u_x[uOff_x[2]+d];
  f0[0] -= (X[0]*X[0]*X[0]*X[0] + X[0]*X[1]*X[1]*X[1] + 2.0*X[0]*X[0]*X[0]*X[1] - 3.0*X[0]*X[0]*X[1]*X[1] - 2.0*alpha);
}

static void f0_q(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                 const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                 const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                 PetscReal t, const PetscReal X[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f0[])
{
  PetscInt d;
  for (d = 0, f0[0] = 0.0; d < dim; ++d) f0[0] += u_x[d*dim+d];
}

static void f1_v(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                 const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                 const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                 PetscReal t, const PetscReal X[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f1[])
{
  const PetscReal nu = PetscRealPart(constants[0]);
  const PetscInt  Nc = dim;
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
  const PetscInt  Nc = dim;
   PetscInt            c, d;

  for (c = 0; c < Nc; ++c) {
    for (d = 0; d < dim; ++d) {
      g0[c*Nc+d] = u_x[ c*Nc+d];
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
          g1[(c*NcJ+d)*dim+e] = u[e];
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
   const PetscReal      nu = PetscRealPart(constants[0]);
   const PetscInt         Nc = dim;
   PetscInt                     c, d;

  for (c = 0; c < Nc; ++c) {
    for (d = 0; d < dim; ++d) {
      g3[((c*Nc+c)*dim+d)*dim+d] += nu; // gradU
      g3[((c*Nc+d)*dim+d)*dim+c] += nu; // gradU transpose
    }
  }
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
  PetscInt        d;

  for (d = 0; d < dim; ++d) g3[d*dim+d] = alpha;
}

static PetscErrorCode ProcessOptions(MPI_Comm comm, AppCtx *options)
{
  PetscInt       sol;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  options->solType   = SOL_QUADRATIC;
  options->showError = PETSC_FALSE;

  ierr = PetscOptionsBegin(comm, "", "Stokes Problem Options", "DMPLEX");CHKERRQ(ierr);
  sol = options->solType;
  CHKERRQ(PetscOptionsEList("-sol_type", "The solution type", "ex62.c", solTypes, NUM_SOL_TYPES, solTypes[options->solType], &sol, NULL));
  options->solType = (SolType) sol;
  CHKERRQ(PetscOptionsBool("-show_error", "Output the error for verification", "ex62.c", options->showError, &options->showError, NULL));
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
  CHKERRQ(PetscBagSetName(user->bag, "par", "Poiseuille flow parameters"));
  bag  = user->bag;
  CHKERRQ(PetscBagRegisterReal(bag, &p->nu,    1.0,   "nu",      "Kinematic viscosity"));
  CHKERRQ(PetscBagRegisterReal(bag, &p->alpha, 1.0,   "alpha",   "Thermal diffusivity"));
  CHKERRQ(PetscBagRegisterReal(bag, &p->theta, 0.0,   "theta",   "Angle of pipe wall to x-axis"));
  PetscFunctionReturn(0);
}

static PetscErrorCode CreateMesh(MPI_Comm comm, AppCtx *user, DM *dm)
{
  PetscFunctionBeginUser;
  CHKERRQ(DMCreate(comm, dm));
  CHKERRQ(DMSetType(*dm, DMPLEX));
  CHKERRQ(DMSetFromOptions(*dm));
  {
    Parameter   *param;
    Vec          coordinates;
    PetscScalar *coords;
    PetscReal    theta;
    PetscInt     cdim, N, bs, i;

    CHKERRQ(DMGetCoordinateDim(*dm, &cdim));
    CHKERRQ(DMGetCoordinates(*dm, &coordinates));
    CHKERRQ(VecGetLocalSize(coordinates, &N));
    CHKERRQ(VecGetBlockSize(coordinates, &bs));
    PetscCheckFalse(bs != cdim,comm, PETSC_ERR_ARG_WRONG, "Invalid coordinate blocksize %D != embedding dimension %D", bs, cdim);
    CHKERRQ(VecGetArray(coordinates, &coords));
    CHKERRQ(PetscBagGetData(user->bag, (void **) &param));
    theta = param->theta;
    for (i = 0; i < N; i += cdim) {
      PetscScalar x = coords[i+0];
      PetscScalar y = coords[i+1];

      coords[i+0] = PetscCosReal(theta)*x - PetscSinReal(theta)*y;
      coords[i+1] = PetscSinReal(theta)*x + PetscCosReal(theta)*y;
    }
    CHKERRQ(VecRestoreArray(coordinates, &coords));
    CHKERRQ(DMSetCoordinates(*dm, coordinates));
  }
  CHKERRQ(DMViewFromOptions(*dm, NULL, "-dm_view"));
  PetscFunctionReturn(0);
}

static PetscErrorCode SetupProblem(DM dm, AppCtx *user)
{
  PetscErrorCode (*exactFuncs[3])(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nf, PetscScalar *u, void *ctx);
  PetscDS          prob;
  DMLabel          label;
  Parameter       *ctx;
  PetscInt         id;

  PetscFunctionBeginUser;
  CHKERRQ(DMGetLabel(dm, "marker", &label));
  CHKERRQ(DMGetDS(dm, &prob));
  switch(user->solType) {
  case SOL_QUADRATIC:
    CHKERRQ(PetscDSSetResidual(prob, 0, f0_quadratic_v, f1_v));
    CHKERRQ(PetscDSSetResidual(prob, 2, f0_quadratic_w, f1_w));

    exactFuncs[0] = quadratic_u;
    exactFuncs[1] = linear_p;
    exactFuncs[2] = linear_T;
    break;
  case SOL_CUBIC:
    CHKERRQ(PetscDSSetResidual(prob, 0, f0_cubic_v, f1_v));
    CHKERRQ(PetscDSSetResidual(prob, 2, f0_cubic_w, f1_w));

    exactFuncs[0] = cubic_u;
    exactFuncs[1] = quadratic_p;
    exactFuncs[2] = quadratic_T;
    break;
   default: SETERRQ(PetscObjectComm((PetscObject) prob), PETSC_ERR_ARG_WRONG, "Unsupported solution type: %s (%D)", solTypes[PetscMin(user->solType, NUM_SOL_TYPES)], user->solType);
  }

  CHKERRQ(PetscDSSetResidual(prob, 1, f0_q, NULL));

  CHKERRQ(PetscDSSetJacobian(prob, 0, 0, g0_vu, g1_vu,  NULL,  g3_vu));
  CHKERRQ(PetscDSSetJacobian(prob, 0, 1, NULL, NULL,  g2_vp, NULL));
  CHKERRQ(PetscDSSetJacobian(prob, 1, 0, NULL, g1_qu, NULL,  NULL));
  CHKERRQ(PetscDSSetJacobian(prob, 2, 0, g0_wu, NULL, NULL,  NULL));
  CHKERRQ(PetscDSSetJacobian(prob, 2, 2, NULL, g1_wT, NULL,  g3_wT));
  /* Setup constants */
  {
    Parameter  *param;
    PetscScalar constants[3];

    CHKERRQ(PetscBagGetData(user->bag, (void **) &param));

    constants[0] = param->nu;
    constants[1] = param->alpha;
    constants[2] = param->theta;
    CHKERRQ(PetscDSSetConstants(prob, 3, constants));
  }
  /* Setup Boundary Conditions */
  CHKERRQ(PetscBagGetData(user->bag, (void **) &ctx));
  id   = 3;
  CHKERRQ(PetscDSAddBoundary(prob, DM_BC_ESSENTIAL, "top wall velocity",    label, 1, &id, 0, 0, NULL, (void (*)(void)) exactFuncs[0], NULL, ctx, NULL));
  id   = 1;
  CHKERRQ(PetscDSAddBoundary(prob, DM_BC_ESSENTIAL, "bottom wall velocity", label, 1, &id, 0, 0, NULL, (void (*)(void)) exactFuncs[0], NULL, ctx, NULL));
  id   = 2;
  CHKERRQ(PetscDSAddBoundary(prob, DM_BC_ESSENTIAL, "right wall velocity",  label, 1, &id, 0, 0, NULL, (void (*)(void)) exactFuncs[0], NULL, ctx, NULL));
  id   = 4;
  CHKERRQ(PetscDSAddBoundary(prob, DM_BC_ESSENTIAL, "left wall velocity",   label, 1, &id, 0, 0, NULL, (void (*)(void)) exactFuncs[0], NULL, ctx, NULL));
  id   = 3;
  CHKERRQ(PetscDSAddBoundary(prob, DM_BC_ESSENTIAL, "top wall temp",    label, 1, &id, 2, 0, NULL, (void (*)(void)) exactFuncs[2], NULL, ctx, NULL));
  id   = 1;
  CHKERRQ(PetscDSAddBoundary(prob, DM_BC_ESSENTIAL, "bottom wall temp", label, 1, &id, 2, 0, NULL, (void (*)(void)) exactFuncs[2], NULL, ctx, NULL));
  id   = 2;
  CHKERRQ(PetscDSAddBoundary(prob, DM_BC_ESSENTIAL, "right wall temp",  label, 1, &id, 2, 0, NULL, (void (*)(void)) exactFuncs[2], NULL, ctx, NULL));
  id   = 4;
  CHKERRQ(PetscDSAddBoundary(prob, DM_BC_ESSENTIAL, "left wall temp",   label, 1, &id, 2, 0, NULL, (void (*)(void)) exactFuncs[2], NULL, ctx, NULL));

  /*setup exact solution.*/
  CHKERRQ(PetscDSSetExactSolution(prob, 0, exactFuncs[0], ctx));
  CHKERRQ(PetscDSSetExactSolution(prob, 1, exactFuncs[1], ctx));
  CHKERRQ(PetscDSSetExactSolution(prob, 2, exactFuncs[2], ctx));
  PetscFunctionReturn(0);
}

static PetscErrorCode SetupDiscretization(DM dm, AppCtx *user)
{
  DM              cdm   = dm;
  PetscFE         fe[3];
  Parameter      *param;
  MPI_Comm        comm;
  PetscInt        dim;
  PetscBool       simplex;

  PetscFunctionBeginUser;
  CHKERRQ(DMGetDimension(dm, &dim));
  CHKERRQ(DMPlexIsSimplex(dm, &simplex));
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
    CHKERRQ(DMPlexCreateBasisRotation(cdm, param->theta, 0.0, 0.0));
    CHKERRQ(DMGetCoarseDM(cdm, &cdm));
  }
  CHKERRQ(PetscFEDestroy(&fe[0]));
  CHKERRQ(PetscFEDestroy(&fe[1]));
  CHKERRQ(PetscFEDestroy(&fe[2]));
  PetscFunctionReturn(0);
}

static PetscErrorCode CreatePressureNullSpace(DM dm, PetscInt ofield, PetscInt nfield, MatNullSpace *nullSpace)
{
  Vec              vec;
  PetscErrorCode (*funcs[3])(PetscInt, PetscReal, const PetscReal[], PetscInt, PetscScalar *, void *) = {zero, zero, zero};

  PetscFunctionBeginUser;
  PetscCheckFalse(ofield != 1,PetscObjectComm((PetscObject) dm), PETSC_ERR_ARG_WRONG, "Nullspace must be for pressure field at index 1, not %D", ofield);
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

int main(int argc, char **argv)
{
  SNES            snes;                 /* nonlinear solver */
  DM              dm;                   /* problem definition */
  Vec             u, r;                 /* solution, residual vectors */
  AppCtx          user;                 /* user-defined work context */

  CHKERRQ(PetscInitialize(&argc, &argv, NULL,help));
  CHKERRQ(ProcessOptions(PETSC_COMM_WORLD, &user));
  CHKERRQ(PetscBagCreate(PETSC_COMM_WORLD, sizeof(Parameter), &user.bag));
  CHKERRQ(SetupParameters(&user));
  CHKERRQ(SNESCreate(PETSC_COMM_WORLD, &snes));
  CHKERRQ(CreateMesh(PETSC_COMM_WORLD, &user, &dm));
  CHKERRQ(SNESSetDM(snes, dm));
  CHKERRQ(DMSetApplicationContext(dm, &user));
  /* Setup problem */
  CHKERRQ(SetupDiscretization(dm, &user));
  CHKERRQ(DMPlexCreateClosureIndex(dm, NULL));

  CHKERRQ(DMCreateGlobalVector(dm, &u));
  CHKERRQ(PetscObjectSetName((PetscObject) u, "Solution"));
  CHKERRQ(VecDuplicate(u, &r));

  CHKERRQ(DMSetNullSpaceConstructor(dm, 1, CreatePressureNullSpace));
  CHKERRQ(DMPlexSetSNESLocalFEM(dm,&user,&user,&user));

  CHKERRQ(SNESSetFromOptions(snes));
  {
    PetscDS          ds;
    PetscErrorCode (*exactFuncs[3])(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nf, PetscScalar *u, void *ctx);
    void            *ctxs[3];

    CHKERRQ(DMGetDS(dm, &ds));
    CHKERRQ(PetscDSGetExactSolution(ds, 0, &exactFuncs[0], &ctxs[0]));
    CHKERRQ(PetscDSGetExactSolution(ds, 1, &exactFuncs[1], &ctxs[1]));
    CHKERRQ(PetscDSGetExactSolution(ds, 2, &exactFuncs[2], &ctxs[2]));
    CHKERRQ(DMProjectFunction(dm, 0.0, exactFuncs, ctxs, INSERT_ALL_VALUES, u));
    CHKERRQ(PetscObjectSetName((PetscObject) u, "Exact Solution"));
    CHKERRQ(VecViewFromOptions(u, NULL, "-exact_vec_view"));
  }
  CHKERRQ(DMSNESCheckFromOptions(snes, u));
  CHKERRQ(VecSet(u, 0.0));
  CHKERRQ(SNESSolve(snes, NULL, u));

  if (user.showError) {
    PetscDS          ds;
    Vec              r;
    PetscErrorCode (*exactFuncs[3])(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nf, PetscScalar *u, void *ctx);
    void            *ctxs[3];

    CHKERRQ(DMGetDS(dm, &ds));
    CHKERRQ(PetscDSGetExactSolution(ds, 0, &exactFuncs[0], &ctxs[0]));
    CHKERRQ(PetscDSGetExactSolution(ds, 1, &exactFuncs[1], &ctxs[1]));
    CHKERRQ(PetscDSGetExactSolution(ds, 2, &exactFuncs[2], &ctxs[2]));
    CHKERRQ(DMGetGlobalVector(dm, &r));
    CHKERRQ(DMProjectFunction(dm, 0.0, exactFuncs, ctxs, INSERT_ALL_VALUES, r));
    CHKERRQ(VecAXPY(r, -1.0, u));
    CHKERRQ(PetscObjectSetName((PetscObject) r, "Solution Error"));
    CHKERRQ(VecViewFromOptions(r, NULL, "-error_vec_view"));
    CHKERRQ(DMRestoreGlobalVector(dm, &r));
  }
  CHKERRQ(PetscObjectSetName((PetscObject) u, "Numerical Solution"));
  CHKERRQ(VecViewFromOptions(u, NULL, "-sol_vec_view"));

  CHKERRQ(VecDestroy(&u));
  CHKERRQ(VecDestroy(&r));
  CHKERRQ(DMDestroy(&dm));
  CHKERRQ(SNESDestroy(&snes));
  CHKERRQ(PetscBagDestroy(&user.bag));
  CHKERRQ(PetscFinalize());
  return 0;
}

/*TEST

  test:
    suffix: 2d_tri_p2_p1_p1
    requires: triangle !single
    args: -dm_plex_separate_marker -sol_type quadratic -dm_refine 0 \
      -vel_petscspace_degree 2 -pres_petscspace_degree 1 -temp_petscspace_degree 1 \
      -dmsnes_check .001 -snes_error_if_not_converged \
      -ksp_type fgmres -ksp_gmres_restart 10 -ksp_rtol 1.0e-9 -ksp_error_if_not_converged \
      -pc_type fieldsplit -pc_fieldsplit_0_fields 0,2 -pc_fieldsplit_1_fields 1 -pc_fieldsplit_type schur -pc_fieldsplit_schur_factorization_type full \
        -fieldsplit_0_pc_type lu \
        -fieldsplit_pressure_ksp_rtol 1e-10 -fieldsplit_pressure_pc_type jacobi

  test:
    # Using -dm_refine 2 -convest_num_refine 3 gives L_2 convergence rate: [2.9, 2.3, 1.9]
    suffix: 2d_tri_p2_p1_p1_conv
    requires: triangle !single
    args: -dm_plex_separate_marker -sol_type cubic -dm_refine 0 \
      -vel_petscspace_degree 2 -pres_petscspace_degree 1 -temp_petscspace_degree 1 \
      -snes_error_if_not_converged -snes_convergence_test correct_pressure -snes_convergence_estimate -convest_num_refine 1 \
      -ksp_type fgmres -ksp_gmres_restart 10 -ksp_rtol 1.0e-9 -ksp_error_if_not_converged \
      -pc_type fieldsplit -pc_fieldsplit_0_fields 0,2 -pc_fieldsplit_1_fields 1 -pc_fieldsplit_type schur -pc_fieldsplit_schur_factorization_type full \
        -fieldsplit_0_pc_type lu \
        -fieldsplit_pressure_ksp_rtol 1e-10 -fieldsplit_pressure_pc_type jacobi

  test:
    suffix: 2d_tri_p3_p2_p2
    requires: triangle !single
    args: -dm_plex_separate_marker -sol_type cubic -dm_refine 0 \
      -vel_petscspace_degree 3 -pres_petscspace_degree 2 -temp_petscspace_degree 2 \
      -dmsnes_check .001 -snes_error_if_not_converged \
      -ksp_type fgmres -ksp_gmres_restart 10 -ksp_rtol 1.0e-9 -ksp_error_if_not_converged \
      -pc_type fieldsplit -pc_fieldsplit_0_fields 0,2 -pc_fieldsplit_1_fields 1 -pc_fieldsplit_type schur -pc_fieldsplit_schur_factorization_type full \
        -fieldsplit_0_pc_type lu \
        -fieldsplit_pressure_ksp_rtol 1e-10 -fieldsplit_pressure_pc_type jacobi

TEST*/
