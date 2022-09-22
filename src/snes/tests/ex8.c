static char help[] = "Test adaptive interpolation of functions of a given polynomial order\n\n";

#include <petscdmplex.h>
#include <petscsnes.h>

/*
  What properties does the adapted interpolator have?

1) If we adapt to quadratics, we can get lower interpolation error for quadratics (than local interpolation) when using a linear basis

$ ./ex8 -dm_refine 2 -petscspace_degree 1 -qorder 1 -dim 2 -porder 2 -K 2 -num_comp 1 -use_poly 1
Function tests FAIL for order 2 at tolerance 1e-10 error 0.00273757
Function tests FAIL for order 2 derivatives at tolerance 1e-10 error 0.0721688
Interpolation tests FAIL for order 2 at tolerance 1e-10 error 0.00284555
Interpolation tests FAIL for order 2 derivatives at tolerance 1e-10 error 0.0721688
 Adapting interpolator using polynomials
The number of input vectors 4 < 7 the maximum number of column entries
  Interpolation poly tests FAIL for order 2 at tolerance 1e-10 error 0.00659864
  Interpolation poly tests FAIL for order 2 derivatives at tolerance 1e-10 error 0.0836582
  Interpolation trig (0, 0) tests FAIL for order 1 at tolerance 1e-10 error 0.476194
  Interpolation trig (0, 0) tests FAIL for order 1 derivatives at tolerance 1e-10 error 2.22144
  Interpolation trig (0, 1) tests FAIL for order 1 at tolerance 1e-10 error 1.39768
  Interpolation trig (0, 1) tests FAIL for order 1 derivatives at tolerance 1e-10 error 2.22144
  Interpolation trig (1, 0) tests FAIL for order 2 at tolerance 1e-10 error 1.07315
  Interpolation trig (1, 0) tests FAIL for order 2 derivatives at tolerance 1e-10 error 4.55403
  Interpolation trig (1, 1) tests FAIL for order 2 at tolerance 1e-10 error 1.07315
  Interpolation trig (1, 1) tests FAIL for order 2 derivatives at tolerance 1e-10 error 4.55403

$ ./ex8 -dm_refine 2 -petscspace_degree 1 -qorder 1 -dim 2 -porder 2 -K 3 -num_comp 1 -use_poly 1
Function tests FAIL for order 2 at tolerance 1e-10 error 0.00273757
Function tests FAIL for order 2 derivatives at tolerance 1e-10 error 0.0721688
Interpolation tests FAIL for order 2 at tolerance 1e-10 error 0.00284555
Interpolation tests FAIL for order 2 derivatives at tolerance 1e-10 error 0.0721688
 Adapting interpolator using polynomials
The number of input vectors 6 < 7 the maximum number of column entries
  Interpolation poly tests FAIL for order 2 at tolerance 1e-10 error 0.00194055
  Interpolation poly tests FAIL for order 2 derivatives at tolerance 1e-10 error 0.0525591
  Interpolation trig (0, 0) tests FAIL for order 1 at tolerance 1e-10 error 0.476255
  Interpolation trig (0, 0) tests FAIL for order 1 derivatives at tolerance 1e-10 error 2.22132
  Interpolation trig (0, 1) tests FAIL for order 1 at tolerance 1e-10 error 1.39785
  Interpolation trig (0, 1) tests FAIL for order 1 derivatives at tolerance 1e-10 error 2.22119
  Interpolation trig (1, 0) tests FAIL for order 2 at tolerance 1e-10 error 1.0727
  Interpolation trig (1, 0) tests FAIL for order 2 derivatives at tolerance 1e-10 error 4.55364
  Interpolation trig (1, 1) tests FAIL for order 2 at tolerance 1e-10 error 1.0727
  Interpolation trig (1, 1) tests FAIL for order 2 derivatives at tolerance 1e-10 error 4.55364
  Interpolation trig (2, 0) tests FAIL for order 3 at tolerance 1e-10 error 0.705258
  Interpolation trig (2, 0) tests FAIL for order 3 derivatives at tolerance 1e-10 error 6.82037
  Interpolation trig (2, 1) tests FAIL for order 3 at tolerance 1e-10 error 0.705258
  Interpolation trig (2, 1) tests FAIL for order 3 derivatives at tolerance 1e-10 error 6.82037

2) We can more accurately capture low harmonics

If we adapt polynomials, we can be exact

$ ./ex8 -dm_refine 2 -petscspace_degree 1 -qorder 1 -dim 2 -porder 1 -K 2 -num_comp 1 -use_poly 1
Function tests pass for order 1 at tolerance 1e-10
Function tests pass for order 1 derivatives at tolerance 1e-10
Interpolation tests pass for order 1 at tolerance 1e-10
Interpolation tests pass for order 1 derivatives at tolerance 1e-10
 Adapting interpolator using polynomials
The number of input vectors 4 < 7 the maximum number of column entries
  Interpolation poly tests pass for order 1 at tolerance 1e-10
  Interpolation poly tests pass for order 1 derivatives at tolerance 1e-10
  Interpolation trig (0, 0) tests FAIL for order 1 at tolerance 1e-10 error 0.476194
  Interpolation trig (0, 0) tests FAIL for order 1 derivatives at tolerance 1e-10 error 2.22144
  Interpolation trig (0, 1) tests FAIL for order 1 at tolerance 1e-10 error 1.39768
  Interpolation trig (0, 1) tests FAIL for order 1 derivatives at tolerance 1e-10 error 2.22144
  Interpolation trig (1, 0) tests FAIL for order 2 at tolerance 1e-10 error 1.07315
  Interpolation trig (1, 0) tests FAIL for order 2 derivatives at tolerance 1e-10 error 4.55403
  Interpolation trig (1, 1) tests FAIL for order 2 at tolerance 1e-10 error 1.07315
  Interpolation trig (1, 1) tests FAIL for order 2 derivatives at tolerance 1e-10 error 4.55403

and least for small K,

$ ./ex8 -dm_refine 2 -petscspace_degree 1 -qorder 1 -dim 2 -porder 1 -K 4 -num_comp 1 -use_poly 1
Function tests pass for order 1 at tolerance 1e-10
Function tests pass for order 1 derivatives at tolerance 1e-10
Interpolation tests pass for order 1 at tolerance 1e-10
Interpolation tests pass for order 1 derivatives at tolerance 1e-10
 Adapting interpolator using polynomials
  Interpolation poly tests FAIL for order 1 at tolerance 1e-10 error 0.0015351
  Interpolation poly tests FAIL for order 1 derivatives at tolerance 1e-10 error 0.0427369
  Interpolation trig (0, 0) tests FAIL for order 1 at tolerance 1e-10 error 0.476359
  Interpolation trig (0, 0) tests FAIL for order 1 derivatives at tolerance 1e-10 error 2.22115
  Interpolation trig (0, 1) tests FAIL for order 1 at tolerance 1e-10 error 1.3981
  Interpolation trig (0, 1) tests FAIL for order 1 derivatives at tolerance 1e-10 error 2.22087
  Interpolation trig (1, 0) tests FAIL for order 2 at tolerance 1e-10 error 1.07228
  Interpolation trig (1, 0) tests FAIL for order 2 derivatives at tolerance 1e-10 error 4.55238
  Interpolation trig (1, 1) tests FAIL for order 2 at tolerance 1e-10 error 1.07228
  Interpolation trig (1, 1) tests FAIL for order 2 derivatives at tolerance 1e-10 error 4.55238
  Interpolation trig (2, 0) tests FAIL for order 3 at tolerance 1e-10 error 0.704947
  Interpolation trig (2, 0) tests FAIL for order 3 derivatives at tolerance 1e-10 error 6.82254
  Interpolation trig (2, 1) tests FAIL for order 3 at tolerance 1e-10 error 0.704948
  Interpolation trig (2, 1) tests FAIL for order 3 derivatives at tolerance 1e-10 error 6.82254
  Interpolation trig (3, 0) tests FAIL for order 4 at tolerance 1e-10 error 0.893279
  Interpolation trig (3, 0) tests FAIL for order 4 derivatives at tolerance 1e-10 error 8.93718
  Interpolation trig (3, 1) tests FAIL for order 4 at tolerance 1e-10 error 0.89328
  Interpolation trig (3, 1) tests FAIL for order 4 derivatives at tolerance 1e-10 error 8.93717

but adapting to harmonics gives alright polynomials errors and much better harmonics errors.

$ ./ex8 -dm_refine 2 -petscspace_degree 1 -qorder 1 -dim 2 -porder 1 -K 4 -num_comp 1 -use_poly 0
Function tests pass for order 1 at tolerance 1e-10
Function tests pass for order 1 derivatives at tolerance 1e-10
Interpolation tests pass for order 1 at tolerance 1e-10
Interpolation tests pass for order 1 derivatives at tolerance 1e-10
 Adapting interpolator using harmonics
  Interpolation poly tests FAIL for order 1 at tolerance 1e-10 error 0.0720606
  Interpolation poly tests FAIL for order 1 derivatives at tolerance 1e-10 error 1.97779
  Interpolation trig (0, 0) tests FAIL for order 1 at tolerance 1e-10 error 0.0398055
  Interpolation trig (0, 0) tests FAIL for order 1 derivatives at tolerance 1e-10 error 0.995963
  Interpolation trig (0, 1) tests FAIL for order 1 at tolerance 1e-10 error 0.0398051
  Interpolation trig (0, 1) tests FAIL for order 1 derivatives at tolerance 1e-10 error 0.995964
  Interpolation trig (1, 0) tests FAIL for order 2 at tolerance 1e-10 error 0.0238441
  Interpolation trig (1, 0) tests FAIL for order 2 derivatives at tolerance 1e-10 error 0.888611
  Interpolation trig (1, 1) tests FAIL for order 2 at tolerance 1e-10 error 0.0238346
  Interpolation trig (1, 1) tests FAIL for order 2 derivatives at tolerance 1e-10 error 0.888612
  Interpolation trig (2, 0) tests FAIL for order 3 at tolerance 1e-10 error 0.0537968
  Interpolation trig (2, 0) tests FAIL for order 3 derivatives at tolerance 1e-10 error 1.57665
  Interpolation trig (2, 1) tests FAIL for order 3 at tolerance 1e-10 error 0.0537779
  Interpolation trig (2, 1) tests FAIL for order 3 derivatives at tolerance 1e-10 error 1.57666
  Interpolation trig (3, 0) tests FAIL for order 4 at tolerance 1e-10 error 0.0775838
  Interpolation trig (3, 0) tests FAIL for order 4 derivatives at tolerance 1e-10 error 2.36926
  Interpolation trig (3, 1) tests FAIL for order 4 at tolerance 1e-10 error 0.0775464
  Interpolation trig (3, 1) tests FAIL for order 4 derivatives at tolerance 1e-10 error 2.36929
*/

typedef struct {
  /* Element definition */
  PetscInt qorder; /* Order of the quadrature */
  PetscInt Nc;     /* Number of field components */
  /* Testing space */
  PetscInt  porder;       /* Order of polynomials to test */
  PetscReal constants[3]; /* Constant values for each dimension */
  PetscInt  m;            /* The frequency of sinusoids to use */
  PetscInt  dir;          /* The direction of sinusoids to use */
  /* Adaptation */
  PetscInt  K;       /* Number of coarse modes used for optimization */
  PetscBool usePoly; /* Use polynomials, or harmonics, to adapt interpolator */
} AppCtx;

typedef enum {
  INTERPOLATION,
  RESTRICTION,
  INJECTION
} InterpType;

/* u = 1 */
PetscErrorCode constant(PetscInt dim, PetscReal time, const PetscReal coords[], PetscInt Nc, PetscScalar *u, void *ctx)
{
  AppCtx  *user = (AppCtx *)ctx;
  PetscInt d    = user->dir;

  if (Nc > 1) {
    for (d = 0; d < Nc; ++d) u[d] = user->constants[d];
  } else {
    u[0] = user->constants[d];
  }
  return 0;
}
PetscErrorCode constantDer(PetscInt dim, PetscReal time, const PetscReal coords[], const PetscReal n[], PetscInt Nc, PetscScalar *u, void *ctx)
{
  AppCtx  *user = (AppCtx *)ctx;
  PetscInt d    = user->dir;

  if (Nc > 1) {
    for (d = 0; d < Nc; ++d) u[d] = 0.0;
  } else {
    u[0] = user->constants[d];
  }
  return 0;
}

/* u = x */
PetscErrorCode linear(PetscInt dim, PetscReal time, const PetscReal coords[], PetscInt Nc, PetscScalar *u, void *ctx)
{
  AppCtx  *user = (AppCtx *)ctx;
  PetscInt d    = user->dir;

  if (Nc > 1) {
    for (d = 0; d < Nc; ++d) u[d] = coords[d];
  } else {
    u[0] = coords[d];
  }
  return 0;
}
PetscErrorCode linearDer(PetscInt dim, PetscReal time, const PetscReal coords[], const PetscReal n[], PetscInt Nc, PetscScalar *u, void *ctx)
{
  AppCtx  *user = (AppCtx *)ctx;
  PetscInt d    = user->dir;

  if (Nc > 1) {
    PetscInt e;
    for (d = 0; d < Nc; ++d) {
      u[d] = 0.0;
      for (e = 0; e < dim; ++e) u[d] += (d == e ? 1.0 : 0.0) * n[e];
    }
  } else {
    u[0] = n[d];
  }
  return 0;
}

/* u = x^2 or u = (x^2, xy) or u = (xy, yz, zx) */
PetscErrorCode quadratic(PetscInt dim, PetscReal time, const PetscReal coords[], PetscInt Nc, PetscScalar *u, void *ctx)
{
  AppCtx  *user = (AppCtx *)ctx;
  PetscInt d    = user->dir;

  if (Nc > 1) {
    if (Nc > 2) {
      u[0] = coords[0] * coords[1];
      u[1] = coords[1] * coords[2];
      u[2] = coords[2] * coords[0];
    } else {
      u[0] = coords[0] * coords[0];
      u[1] = coords[0] * coords[1];
    }
  } else {
    u[0] = coords[d] * coords[d];
  }
  return 0;
}
PetscErrorCode quadraticDer(PetscInt dim, PetscReal time, const PetscReal coords[], const PetscReal n[], PetscInt Nc, PetscScalar *u, void *ctx)
{
  AppCtx  *user = (AppCtx *)ctx;
  PetscInt d    = user->dir;

  if (Nc > 1) {
    if (Nc > 2) {
      u[0] = coords[1] * n[0] + coords[0] * n[1];
      u[1] = coords[2] * n[1] + coords[1] * n[2];
      u[2] = coords[2] * n[0] + coords[0] * n[2];
    } else {
      u[0] = 2.0 * coords[0] * n[0];
      u[1] = coords[1] * n[0] + coords[0] * n[1];
    }
  } else {
    u[0] = 2.0 * coords[d] * n[d];
  }
  return 0;
}

/* u = x^3 or u = (x^3, x^2y) or u = (x^2y, y^2z, z^2x) */
PetscErrorCode cubic(PetscInt dim, PetscReal time, const PetscReal coords[], PetscInt Nc, PetscScalar *u, void *ctx)
{
  AppCtx  *user = (AppCtx *)ctx;
  PetscInt d    = user->dir;

  if (Nc > 1) {
    if (Nc > 2) {
      u[0] = coords[0] * coords[0] * coords[1];
      u[1] = coords[1] * coords[1] * coords[2];
      u[2] = coords[2] * coords[2] * coords[0];
    } else {
      u[0] = coords[0] * coords[0] * coords[0];
      u[1] = coords[0] * coords[0] * coords[1];
    }
  } else {
    u[0] = coords[d] * coords[d] * coords[d];
  }
  return 0;
}
PetscErrorCode cubicDer(PetscInt dim, PetscReal time, const PetscReal coords[], const PetscReal n[], PetscInt Nc, PetscScalar *u, void *ctx)
{
  AppCtx  *user = (AppCtx *)ctx;
  PetscInt d    = user->dir;

  if (Nc > 1) {
    if (Nc > 2) {
      u[0] = 2.0 * coords[0] * coords[1] * n[0] + coords[0] * coords[0] * n[1];
      u[1] = 2.0 * coords[1] * coords[2] * n[1] + coords[1] * coords[1] * n[2];
      u[2] = 2.0 * coords[2] * coords[0] * n[2] + coords[2] * coords[2] * n[0];
    } else {
      u[0] = 3.0 * coords[0] * coords[0] * n[0];
      u[1] = 2.0 * coords[0] * coords[1] * n[0] + coords[0] * coords[0] * n[1];
    }
  } else {
    u[0] = 3.0 * coords[d] * coords[d] * n[d];
  }
  return 0;
}

/* u = x^4 or u = (x^4, x^2y^2) or u = (x^2y^2, y^2z^2, z^2x^2) */
PetscErrorCode quartic(PetscInt dim, PetscReal time, const PetscReal coords[], PetscInt Nc, PetscScalar *u, void *ctx)
{
  AppCtx  *user = (AppCtx *)ctx;
  PetscInt d    = user->dir;

  if (Nc > 1) {
    if (Nc > 2) {
      u[0] = coords[0] * coords[0] * coords[1] * coords[1];
      u[1] = coords[1] * coords[1] * coords[2] * coords[2];
      u[2] = coords[2] * coords[2] * coords[0] * coords[0];
    } else {
      u[0] = coords[0] * coords[0] * coords[0] * coords[0];
      u[1] = coords[0] * coords[0] * coords[1] * coords[1];
    }
  } else {
    u[0] = coords[d] * coords[d] * coords[d] * coords[d];
  }
  return 0;
}
PetscErrorCode quarticDer(PetscInt dim, PetscReal time, const PetscReal coords[], const PetscReal n[], PetscInt Nc, PetscScalar *u, void *ctx)
{
  AppCtx  *user = (AppCtx *)ctx;
  PetscInt d    = user->dir;

  if (Nc > 1) {
    if (Nc > 2) {
      u[0] = 2.0 * coords[0] * coords[1] * coords[1] * n[0] + 2.0 * coords[0] * coords[0] * coords[1] * n[1];
      u[1] = 2.0 * coords[1] * coords[2] * coords[2] * n[1] + 2.0 * coords[1] * coords[1] * coords[2] * n[2];
      u[2] = 2.0 * coords[2] * coords[0] * coords[0] * n[2] + 2.0 * coords[2] * coords[2] * coords[0] * n[0];
    } else {
      u[0] = 4.0 * coords[0] * coords[0] * coords[0] * n[0];
      u[1] = 2.0 * coords[0] * coords[1] * coords[1] * n[0] + 2.0 * coords[0] * coords[0] * coords[1] * n[1];
    }
  } else {
    u[0] = 4.0 * coords[d] * coords[d] * coords[d] * n[d];
  }
  return 0;
}

PetscErrorCode mytanh(PetscInt dim, PetscReal time, const PetscReal coords[], PetscInt Nc, PetscScalar *u, void *ctx)
{
  AppCtx  *user = (AppCtx *)ctx;
  PetscInt d    = user->dir;

  if (Nc > 1) {
    for (d = 0; d < Nc; ++d) u[d] = PetscTanhReal(coords[d] - 0.5);
  } else {
    u[0] = PetscTanhReal(coords[d] - 0.5);
  }
  return 0;
}
PetscErrorCode mytanhDer(PetscInt dim, PetscReal time, const PetscReal coords[], const PetscReal n[], PetscInt Nc, PetscScalar *u, void *ctx)
{
  AppCtx  *user = (AppCtx *)ctx;
  PetscInt d    = user->dir;

  if (Nc > 1) {
    for (d = 0; d < Nc; ++d) u[d] = 1.0 / PetscSqr(PetscCoshReal(coords[d] - 0.5)) * n[d];
  } else {
    u[0] = 1.0 / PetscSqr(PetscCoshReal(coords[d] - 0.5)) * n[d];
  }
  return 0;
}

PetscErrorCode trig(PetscInt dim, PetscReal time, const PetscReal coords[], PetscInt Nc, PetscScalar *u, void *ctx)
{
  AppCtx  *user = (AppCtx *)ctx;
  PetscInt m = user->m, d = user->dir;

  if (Nc > 1) {
    for (d = 0; d < Nc; ++d) u[d] = PetscSinReal(PETSC_PI * m * coords[d]);
  } else {
    u[0] = PetscSinReal(PETSC_PI * m * coords[d]);
  }
  return 0;
}
PetscErrorCode trigDer(PetscInt dim, PetscReal time, const PetscReal coords[], const PetscReal n[], PetscInt Nc, PetscScalar *u, void *ctx)
{
  AppCtx  *user = (AppCtx *)ctx;
  PetscInt m = user->m, d = user->dir;

  if (Nc > 1) {
    for (d = 0; d < Nc; ++d) u[d] = PETSC_PI * m * PetscCosReal(PETSC_PI * m * coords[d]) * n[d];
  } else {
    u[0] = PETSC_PI * m * PetscCosReal(PETSC_PI * m * coords[d]) * n[d];
  }
  return 0;
}

static PetscErrorCode ProcessOptions(MPI_Comm comm, AppCtx *options)
{
  PetscFunctionBeginUser;
  options->qorder  = 0;
  options->Nc      = PETSC_DEFAULT;
  options->porder  = 0;
  options->m       = 1;
  options->dir     = 0;
  options->K       = 0;
  options->usePoly = PETSC_TRUE;

  PetscOptionsBegin(comm, "", "Projection Test Options", "DMPlex");
  PetscCall(PetscOptionsInt("-qorder", "The quadrature order", "ex8.c", options->qorder, &options->qorder, NULL));
  PetscCall(PetscOptionsInt("-num_comp", "The number of field components", "ex8.c", options->Nc, &options->Nc, NULL));
  PetscCall(PetscOptionsInt("-porder", "The order of polynomials to test", "ex8.c", options->porder, &options->porder, NULL));
  PetscCall(PetscOptionsInt("-K", "The number of coarse modes used in optimization", "ex8.c", options->K, &options->K, NULL));
  PetscCall(PetscOptionsBool("-use_poly", "Use polynomials (or harmonics) to adapt interpolator", "ex8.c", options->usePoly, &options->usePoly, NULL));
  PetscOptionsEnd();
  PetscFunctionReturn(0);
}

static PetscErrorCode CreateMesh(MPI_Comm comm, AppCtx *user, DM *dm)
{
  PetscFunctionBeginUser;
  PetscCall(DMCreate(comm, dm));
  PetscCall(DMSetType(*dm, DMPLEX));
  PetscCall(DMSetFromOptions(*dm));
  PetscCall(DMViewFromOptions(*dm, NULL, "-dm_view"));
  PetscFunctionReturn(0);
}

/* Setup functions to approximate */
static PetscErrorCode SetupFunctions(DM dm, PetscBool usePoly, PetscInt order, PetscInt dir, PetscErrorCode (**exactFuncs)(PetscInt, PetscReal, const PetscReal[], PetscInt, PetscScalar *, void *), PetscErrorCode (**exactFuncDers)(PetscInt, PetscReal, const PetscReal[], const PetscReal[], PetscInt, PetscScalar *, void *), AppCtx *user)
{
  PetscInt dim;

  PetscFunctionBeginUser;
  user->dir = dir;
  if (usePoly) {
    switch (order) {
    case 0:
      exactFuncs[0]    = constant;
      exactFuncDers[0] = constantDer;
      break;
    case 1:
      exactFuncs[0]    = linear;
      exactFuncDers[0] = linearDer;
      break;
    case 2:
      exactFuncs[0]    = quadratic;
      exactFuncDers[0] = quadraticDer;
      break;
    case 3:
      exactFuncs[0]    = cubic;
      exactFuncDers[0] = cubicDer;
      break;
    case 4:
      exactFuncs[0]    = quartic;
      exactFuncDers[0] = quarticDer;
      break;
    default:
      PetscCall(DMGetDimension(dm, &dim));
      SETERRQ(PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_OUTOFRANGE, "Could not determine functions to test for dimension %" PetscInt_FMT " order %" PetscInt_FMT, dim, order);
    }
  } else {
    user->m          = order;
    exactFuncs[0]    = trig;
    exactFuncDers[0] = trigDer;
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode ComputeError(DM dm, PetscErrorCode (**exactFuncs)(PetscInt, PetscReal, const PetscReal[], PetscInt, PetscScalar *, void *), PetscErrorCode (**exactFuncDers)(PetscInt, PetscReal, const PetscReal[], const PetscReal[], PetscInt, PetscScalar *, void *), void **exactCtxs, PetscReal *error, PetscReal *errorDer, AppCtx *user)
{
  Vec       u;
  PetscReal n[3] = {1.0, 1.0, 1.0};

  PetscFunctionBeginUser;
  PetscCall(DMGetGlobalVector(dm, &u));
  /* Project function into FE function space */
  PetscCall(DMProjectFunction(dm, 0.0, exactFuncs, exactCtxs, INSERT_ALL_VALUES, u));
  PetscCall(VecViewFromOptions(u, NULL, "-projection_view"));
  /* Compare approximation to exact in L_2 */
  PetscCall(DMComputeL2Diff(dm, 0.0, exactFuncs, exactCtxs, u, error));
  PetscCall(DMComputeL2GradientDiff(dm, 0.0, exactFuncDers, exactCtxs, u, n, errorDer));
  PetscCall(DMRestoreGlobalVector(dm, &u));
  PetscFunctionReturn(0);
}

static PetscErrorCode CheckFunctions(DM dm, PetscInt order, AppCtx *user)
{
  PetscErrorCode (*exactFuncs[1])(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nf, PetscScalar *u, void *ctx);
  PetscErrorCode (*exactFuncDers[1])(PetscInt dim, PetscReal time, const PetscReal x[], const PetscReal n[], PetscInt Nf, PetscScalar *u, void *ctx);
  void     *exactCtxs[3];
  MPI_Comm  comm;
  PetscReal error, errorDer, tol = PETSC_SMALL;

  PetscFunctionBeginUser;
  exactCtxs[0]       = user;
  exactCtxs[1]       = user;
  exactCtxs[2]       = user;
  user->constants[0] = 1.0;
  user->constants[1] = 2.0;
  user->constants[2] = 3.0;
  PetscCall(PetscObjectGetComm((PetscObject)dm, &comm));
  PetscCall(SetupFunctions(dm, PETSC_TRUE, order, 0, exactFuncs, exactFuncDers, user));
  PetscCall(ComputeError(dm, exactFuncs, exactFuncDers, exactCtxs, &error, &errorDer, user));
  /* Report result */
  if (error > tol) PetscCall(PetscPrintf(comm, "Function tests FAIL for order %" PetscInt_FMT " at tolerance %g error %g\n", order, (double)tol, (double)error));
  else PetscCall(PetscPrintf(comm, "Function tests pass for order %" PetscInt_FMT " at tolerance %g\n", order, (double)tol));
  if (errorDer > tol) PetscCall(PetscPrintf(comm, "Function tests FAIL for order %" PetscInt_FMT " derivatives at tolerance %g error %g\n", order, (double)tol, (double)errorDer));
  else PetscCall(PetscPrintf(comm, "Function tests pass for order %" PetscInt_FMT " derivatives at tolerance %g\n", order, (double)tol));
  PetscFunctionReturn(0);
}

/* Compare approximation to exact in L_2 */
static PetscErrorCode CheckTransferError(DM fdm, PetscBool usePoly, PetscInt order, PetscInt dir, const char *testname, Vec fu, AppCtx *user)
{
  PetscErrorCode (*exactFuncs[1])(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nf, PetscScalar *u, void *ctx);
  PetscErrorCode (*exactFuncDers[1])(PetscInt dim, PetscReal time, const PetscReal x[], const PetscReal n[], PetscInt Nf, PetscScalar *u, void *ctx);
  PetscReal n[3] = {1.0, 1.0, 1.0};
  void     *exactCtxs[3];
  MPI_Comm  comm;
  PetscReal error, errorDer, tol = PETSC_SMALL;

  PetscFunctionBeginUser;
  exactCtxs[0]       = user;
  exactCtxs[1]       = user;
  exactCtxs[2]       = user;
  user->constants[0] = 1.0;
  user->constants[1] = 2.0;
  user->constants[2] = 3.0;
  PetscCall(PetscObjectGetComm((PetscObject)fdm, &comm));
  PetscCall(SetupFunctions(fdm, usePoly, order, dir, exactFuncs, exactFuncDers, user));
  PetscCall(DMComputeL2Diff(fdm, 0.0, exactFuncs, exactCtxs, fu, &error));
  PetscCall(DMComputeL2GradientDiff(fdm, 0.0, exactFuncDers, exactCtxs, fu, n, &errorDer));
  /* Report result */
  if (error > tol) PetscCall(PetscPrintf(comm, "%s tests FAIL for order %" PetscInt_FMT " at tolerance %g error %g\n", testname, order, (double)tol, (double)error));
  else PetscCall(PetscPrintf(comm, "%s tests pass for order %" PetscInt_FMT " at tolerance %g\n", testname, order, (double)tol));
  if (errorDer > tol) PetscCall(PetscPrintf(comm, "%s tests FAIL for order %" PetscInt_FMT " derivatives at tolerance %g error %g\n", testname, order, (double)tol, (double)errorDer));
  else PetscCall(PetscPrintf(comm, "%s tests pass for order %" PetscInt_FMT " derivatives at tolerance %g\n", testname, order, (double)tol));
  PetscFunctionReturn(0);
}

static PetscErrorCode CheckTransfer(DM dm, InterpType inType, PetscInt order, AppCtx *user)
{
  PetscErrorCode (*exactFuncs[1])(PetscInt, PetscReal, const PetscReal x[], PetscInt, PetscScalar *u, void *ctx);
  PetscErrorCode (*exactFuncDers[1])(PetscInt, PetscReal, const PetscReal x[], const PetscReal n[], PetscInt, PetscScalar *u, void *ctx);
  void       *exactCtxs[3];
  DM          rdm = NULL, idm = NULL, fdm = NULL;
  Mat         Interp, InterpAdapt = NULL;
  Vec         iu, fu, scaling = NULL;
  MPI_Comm    comm;
  const char *testname = "Unknown";
  char        checkname[PETSC_MAX_PATH_LEN];

  PetscFunctionBeginUser;
  exactCtxs[0] = exactCtxs[1] = exactCtxs[2] = user;
  PetscCall(PetscObjectGetComm((PetscObject)dm, &comm));
  PetscCall(DMRefine(dm, comm, &rdm));
  PetscCall(DMViewFromOptions(rdm, NULL, "-ref_dm_view"));
  PetscCall(DMSetCoarseDM(rdm, dm));
  PetscCall(DMCopyDisc(dm, rdm));
  switch (inType) {
  case INTERPOLATION:
    testname = "Interpolation";
    idm      = dm;
    fdm      = rdm;
    break;
  case RESTRICTION:
    testname = "Restriction";
    idm      = rdm;
    fdm      = dm;
    break;
  case INJECTION:
    testname = "Injection";
    idm      = rdm;
    fdm      = dm;
    break;
  }
  PetscCall(DMGetGlobalVector(idm, &iu));
  PetscCall(DMGetGlobalVector(fdm, &fu));
  PetscCall(DMSetApplicationContext(dm, user));
  PetscCall(DMSetApplicationContext(rdm, user));
  /* Project function into initial FE function space */
  PetscCall(SetupFunctions(dm, PETSC_TRUE, order, 0, exactFuncs, exactFuncDers, user));
  PetscCall(DMProjectFunction(idm, 0.0, exactFuncs, exactCtxs, INSERT_ALL_VALUES, iu));
  /* Interpolate function into final FE function space */
  switch (inType) {
  case INTERPOLATION:
    PetscCall(DMCreateInterpolation(dm, rdm, &Interp, &scaling));
    PetscCall(MatInterpolate(Interp, iu, fu));
    break;
  case RESTRICTION:
    PetscCall(DMCreateInterpolation(dm, rdm, &Interp, &scaling));
    PetscCall(MatRestrict(Interp, iu, fu));
    PetscCall(VecPointwiseMult(fu, scaling, fu));
    break;
  case INJECTION:
    PetscCall(DMCreateInjection(dm, rdm, &Interp));
    PetscCall(MatRestrict(Interp, iu, fu));
    break;
  }
  PetscCall(CheckTransferError(fdm, PETSC_TRUE, order, 0, testname, fu, user));
  if (user->K && (inType == INTERPOLATION)) {
    KSP      smoother;
    Mat      A, iVM, fVM;
    Vec      iV, fV;
    PetscInt k, dim, d, im, fm;

    PetscCall(PetscPrintf(comm, " Adapting interpolator using %s\n", user->usePoly ? "polynomials" : "harmonics"));
    PetscCall(DMGetDimension(dm, &dim));
    /* Project coarse modes into initial and final FE function space */
    PetscCall(DMGetGlobalVector(idm, &iV));
    PetscCall(DMGetGlobalVector(fdm, &fV));
    PetscCall(VecGetLocalSize(iV, &im));
    PetscCall(VecGetLocalSize(fV, &fm));
    PetscCall(MatCreateDense(PetscObjectComm((PetscObject)dm), im, PETSC_DECIDE, PETSC_DECIDE, user->K * dim, NULL, &iVM));
    PetscCall(MatCreateDense(PetscObjectComm((PetscObject)dm), fm, PETSC_DECIDE, PETSC_DECIDE, user->K * dim, NULL, &fVM));
    PetscCall(DMRestoreGlobalVector(idm, &iV));
    PetscCall(DMRestoreGlobalVector(fdm, &fV));
    for (k = 0; k < user->K; ++k) {
      for (d = 0; d < dim; ++d) {
        PetscCall(MatDenseGetColumnVecWrite(iVM, k * dim + d, &iV));
        PetscCall(MatDenseGetColumnVecWrite(fVM, k * dim + d, &fV));
        PetscCall(SetupFunctions(idm, user->usePoly, user->usePoly ? k : k + 1, d, exactFuncs, exactFuncDers, user));
        PetscCall(DMProjectFunction(idm, 0.0, exactFuncs, exactCtxs, INSERT_ALL_VALUES, iV));
        PetscCall(DMProjectFunction(fdm, 0.0, exactFuncs, exactCtxs, INSERT_ALL_VALUES, fV));
        PetscCall(MatDenseRestoreColumnVecWrite(iVM, k * dim + d, &iV));
        PetscCall(MatDenseRestoreColumnVecWrite(fVM, k * dim + d, &fV));
      }
    }

    /* Adapt interpolator */
    PetscCall(DMCreateMatrix(rdm, &A));
    PetscCall(MatShift(A, 1.0));
    PetscCall(KSPCreate(comm, &smoother));
    PetscCall(KSPSetFromOptions(smoother));
    PetscCall(KSPSetOperators(smoother, A, A));
    PetscCall(DMAdaptInterpolator(dm, rdm, Interp, smoother, fVM, iVM, &InterpAdapt, user));
    /* Interpolate function into final FE function space */
    PetscCall(PetscSNPrintf(checkname, PETSC_MAX_PATH_LEN, "  %s poly", testname));
    PetscCall(MatInterpolate(InterpAdapt, iu, fu));
    PetscCall(CheckTransferError(fdm, PETSC_TRUE, order, 0, checkname, fu, user));
    for (k = 0; k < user->K; ++k) {
      for (d = 0; d < dim; ++d) {
        PetscCall(PetscSNPrintf(checkname, PETSC_MAX_PATH_LEN, "  %s trig (%" PetscInt_FMT ", %" PetscInt_FMT ")", testname, k, d));
        PetscCall(MatDenseGetColumnVecRead(iVM, k * dim + d, &iV));
        PetscCall(MatDenseGetColumnVecWrite(fVM, k * dim + d, &fV));
        PetscCall(MatInterpolate(InterpAdapt, iV, fV));
        PetscCall(CheckTransferError(fdm, PETSC_FALSE, k + 1, d, checkname, fV, user));
        PetscCall(MatDenseRestoreColumnVecRead(iVM, k * dim + d, &iV));
        PetscCall(MatDenseRestoreColumnVecWrite(fVM, k * dim + d, &fV));
      }
    }
    /* Cleanup */
    PetscCall(KSPDestroy(&smoother));
    PetscCall(MatDestroy(&A));
    PetscCall(MatDestroy(&InterpAdapt));
    PetscCall(MatDestroy(&iVM));
    PetscCall(MatDestroy(&fVM));
  }
  PetscCall(DMRestoreGlobalVector(idm, &iu));
  PetscCall(DMRestoreGlobalVector(fdm, &fu));
  PetscCall(MatDestroy(&Interp));
  PetscCall(VecDestroy(&scaling));
  PetscCall(DMDestroy(&rdm));
  PetscFunctionReturn(0);
}

int main(int argc, char **argv)
{
  DM        dm;
  PetscFE   fe;
  AppCtx    user;
  PetscInt  dim;
  PetscBool simplex;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  PetscCall(ProcessOptions(PETSC_COMM_WORLD, &user));
  PetscCall(CreateMesh(PETSC_COMM_WORLD, &user, &dm));

  PetscCall(DMGetDimension(dm, &dim));
  PetscCall(DMPlexIsSimplex(dm, &simplex));
  PetscCall(PetscFECreateDefault(PETSC_COMM_WORLD, dim, user.Nc < 0 ? dim : user.Nc, simplex, NULL, user.qorder, &fe));
  PetscCall(DMSetField(dm, 0, NULL, (PetscObject)fe));
  PetscCall(PetscFEDestroy(&fe));
  PetscCall(DMCreateDS(dm));

  PetscCall(CheckFunctions(dm, user.porder, &user));
  PetscCall(CheckTransfer(dm, INTERPOLATION, user.porder, &user));
  PetscCall(CheckTransfer(dm, INJECTION, user.porder, &user));
  PetscCall(DMDestroy(&dm));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

  # TODO dim 3 will not work until I get composite elements in 3D (see plexrefine.c:34)
  # 2D/3D P_1 on a simplex
  test:
    suffix: p1
    requires: triangle ctetgen
    args: -dm_plex_dim {{2}separate output} -dm_plex_box_faces 2,2,2 -petscspace_degree 1 -num_comp 1 -qorder 1 -porder {{1}separate output}
  test:
    suffix: p1_pragmatic
    requires: triangle ctetgen pragmatic
    args: -dm_plex_dim {{2}separate output} -dm_plex_box_faces 2,2,2 -petscspace_degree 1 -qorder 1 -dm_plex_hash_location -porder {{1 2}separate output}
  test:
    suffix: p1_adapt
    requires: triangle ctetgen
    args: -dm_plex_dim {{2}separate output} -dm_plex_box_faces 2,2,2 -dm_refine 3 -petscspace_degree 1 -qorder 1 -porder {{1 2}separate output}

  # TODO dim 3 will not work until I get composite elements in 3D (see plexrefine.c:34)
  # 2D/3D P_2 on a simplex
  test:
    suffix: p2
    requires: triangle ctetgen
    args: -dm_plex_dim {{2}separate output} -dm_plex_box_faces 2,2,2 -petscspace_degree 2 -qorder 2 -porder {{1 2 3}separate output}
  test:
    suffix: p2_pragmatic
    requires: triangle ctetgen pragmatic
    args: -dm_plex_dim {{2}separate output} -dm_plex_box_faces 2,2,2 -petscspace_degree 2 -qorder 2 -dm_plex_hash_location -porder {{1 2 3}separate output}

  # TODO dim 3 will not work until I get composite elements in 3D (see plexrefine.c:34)
  # TODO This is broken. Check ex3 which worked
  # 2D/3D P_3 on a simplex
  test:
    TODO: gll Lagrange nodes break this
    suffix: p3
    requires: triangle ctetgen !single
    args: -dm_plex_dim {{2}separate output} -dm_plex_box_faces 2,2,2 -petscspace_degree 3 -qorder 3 -porder {{1 2 3 4}separate output}
  test:
    TODO: gll Lagrange nodes break this
    suffix: p3_pragmatic
    requires: triangle ctetgen pragmatic !single
    args: -dm_plex_dim {{2}separate output} -dm_plex_box_faces 2,2,2 -petscspace_degree 3 -qorder 3 -dm_plex_hash_location -porder {{1 2 3 4}separate output}

  # 2D/3D Q_1 on a tensor cell
  test:
    suffix: q1
    args: -dm_plex_dim {{2 3}separate output} -dm_plex_box_faces 2,2,2 -dm_plex_simplex 0 -petscspace_degree 1 -qorder 1 -porder {{1 2}separate output}

  # 2D/3D Q_2 on a tensor cell
  test:
    suffix: q2
    requires: !single
    args: -dm_plex_dim {{2 3}separate output} -dm_plex_box_faces 2,2,2 -dm_plex_simplex 0 -petscspace_degree 2 -qorder 2 -porder {{1 2 3}separate output}

  # 2D/3D Q_3 on a tensor cell
  test:
    TODO: gll Lagrange nodes break this
    suffix: q3
    requires: !single
    args: -dm_plex_dim {{2 3}separate output} -dm_plex_box_faces 2,2,2 -dm_plex_simplex 0 -petscspace_degree 3 -qorder 3 -porder {{1 2 3 4}separate output}

  # 2D/3D P_1disc on a triangle/quadrilateral
  # TODO Missing injection functional for simplices
  test:
    suffix: p1d
    requires: triangle ctetgen
    args: -dm_plex_dim {{2}separate output} -dm_plex_box_faces 2,2,2 -dm_plex_simplex {{0}separate output} -petscspace_degree 1 -petscdualspace_lagrange_continuity 0 -qorder 1 -porder {{1 2}separate output}

TEST*/
