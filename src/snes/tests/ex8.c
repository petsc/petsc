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
  /* Domain and mesh definition */
  PetscInt  dim;               /* The topological mesh dimension */
  PetscBool simplex;           /* Flag for simplex or tensor product mesh */
  PetscBool interpolate;       /* Generate intermediate mesh elements */
  PetscReal refinementLimit;   /* The largest allowable cell volume */
  /* Element definition */
  PetscInt  qorder;            /* Order of the quadrature */
  PetscInt  Nc;                /* Number of field components */
  PetscFE   fe;                /* The finite element */
  /* Testing space */
  PetscInt  porder;            /* Order of polynomials to test */
  PetscReal constants[3];      /* Constant values for each dimension */
  PetscInt  m;                 /* The frequency of sinusoids to use */
  PetscInt  dir;               /* The direction of sinusoids to use */
  /* Adaptation */
  PetscInt  K;                 /* Number of coarse modes used for optimization */
  PetscBool usePoly;           /* Use polynomials, or harmonics, to adapt interpolator */
} AppCtx;

typedef enum {INTERPOLATION, RESTRICTION, INJECTION} InterpType;

/* u = 1 */
PetscErrorCode constant(PetscInt dim, PetscReal time, const PetscReal coords[], PetscInt Nc, PetscScalar *u, void *ctx)
{
  AppCtx  *user = (AppCtx *) ctx;
  PetscInt d = user->dir;

  if (Nc > 1) {
    for (d = 0; d < Nc; ++d) u[d] = user->constants[d];
  } else {
    u[0] = user->constants[d];
  }
  return 0;
}
PetscErrorCode constantDer(PetscInt dim, PetscReal time, const PetscReal coords[], const PetscReal n[], PetscInt Nc, PetscScalar *u, void *ctx)
{
  AppCtx  *user = (AppCtx *) ctx;
  PetscInt d = user->dir;

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
  AppCtx  *user = (AppCtx *) ctx;
  PetscInt d = user->dir;

  if (Nc > 1) {
    for (d = 0; d < Nc; ++d) u[d] = coords[d];
  } else {
    u[0] = coords[d];
  }
  return 0;
}
PetscErrorCode linearDer(PetscInt dim, PetscReal time, const PetscReal coords[], const PetscReal n[], PetscInt Nc, PetscScalar *u, void *ctx)
{
  AppCtx  *user = (AppCtx *) ctx;
  PetscInt d = user->dir;

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
  AppCtx  *user = (AppCtx *) ctx;
  PetscInt d = user->dir;

  if (Nc > 1) {
    if (Nc > 2) {u[0] = coords[0]*coords[1]; u[1] = coords[1]*coords[2]; u[2] = coords[2]*coords[0];}
    else        {u[0] = coords[0]*coords[0]; u[1] = coords[0]*coords[1];}
  } else {
    u[0] = coords[d]*coords[d];
  }
  return 0;
}
PetscErrorCode quadraticDer(PetscInt dim, PetscReal time, const PetscReal coords[], const PetscReal n[], PetscInt Nc, PetscScalar *u, void *ctx)
{
  AppCtx  *user = (AppCtx *) ctx;
  PetscInt d = user->dir;

  if (Nc > 1) {
    if (Nc > 2) {u[0] = coords[1]*n[0] + coords[0]*n[1]; u[1] = coords[2]*n[1] + coords[1]*n[2]; u[2] = coords[2]*n[0] + coords[0]*n[2];}
    else        {u[0] = 2.0*coords[0]*n[0]; u[1] = coords[1]*n[0] + coords[0]*n[1];}
  } else {
    u[0] = 2.0*coords[d]*n[d];
  }
  return 0;
}

/* u = x^3 or u = (x^3, x^2y) or u = (x^2y, y^2z, z^2x) */
PetscErrorCode cubic(PetscInt dim, PetscReal time, const PetscReal coords[], PetscInt Nc, PetscScalar *u, void *ctx)
{
  AppCtx  *user = (AppCtx *) ctx;
  PetscInt d = user->dir;

  if (Nc > 1) {
    if (Nc > 2) {u[0] = coords[0]*coords[0]*coords[1]; u[1] = coords[1]*coords[1]*coords[2]; u[2] = coords[2]*coords[2]*coords[0];}
    else        {u[0] = coords[0]*coords[0]*coords[0]; u[1] = coords[0]*coords[0]*coords[1];}
  } else {
    u[0] = coords[d]*coords[d]*coords[d];
  }
  return 0;
}
PetscErrorCode cubicDer(PetscInt dim, PetscReal time, const PetscReal coords[], const PetscReal n[], PetscInt Nc, PetscScalar *u, void *ctx)
{
  AppCtx  *user = (AppCtx *) ctx;
  PetscInt d = user->dir;

  if (Nc > 1) {
    if (Nc > 2) {u[0] = 2.0*coords[0]*coords[1]*n[0] + coords[0]*coords[0]*n[1]; u[1] = 2.0*coords[1]*coords[2]*n[1] + coords[1]*coords[1]*n[2]; u[2] = 2.0*coords[2]*coords[0]*n[2] + coords[2]*coords[2]*n[0];}
    else        {u[0] = 3.0*coords[0]*coords[0]*n[0]; u[1] = 2.0*coords[0]*coords[1]*n[0] + coords[0]*coords[0]*n[1];}
  } else {
    u[0] = 3.0*coords[d]*coords[d]*n[d];
  }
  return 0;
}

/* u = x^4 or u = (x^4, x^2y^2) or u = (x^2y^2, y^2z^2, z^2x^2) */
PetscErrorCode quartic(PetscInt dim, PetscReal time, const PetscReal coords[], PetscInt Nc, PetscScalar *u, void *ctx)
{
  AppCtx  *user = (AppCtx *) ctx;
  PetscInt d = user->dir;

  if (Nc > 1) {
    if (Nc > 2) {u[0] = coords[0]*coords[0]*coords[1]*coords[1]; u[1] = coords[1]*coords[1]*coords[2]*coords[2]; u[2] = coords[2]*coords[2]*coords[0]*coords[0];}
    else        {u[0] = coords[0]*coords[0]*coords[0]*coords[0]; u[1] = coords[0]*coords[0]*coords[1]*coords[1];}
  } else {
    u[0] = coords[d]*coords[d]*coords[d]*coords[d];
  }
  return 0;
}
PetscErrorCode quarticDer(PetscInt dim, PetscReal time, const PetscReal coords[], const PetscReal n[], PetscInt Nc, PetscScalar *u, void *ctx)
{
  AppCtx  *user = (AppCtx *) ctx;
  PetscInt d = user->dir;

  if (Nc > 1) {
    if (Nc > 2) {u[0] = 2.0*coords[0]*coords[1]*coords[1]*n[0] + 2.0*coords[0]*coords[0]*coords[1]*n[1];
                 u[1] = 2.0*coords[1]*coords[2]*coords[2]*n[1] + 2.0*coords[1]*coords[1]*coords[2]*n[2];
                 u[2] = 2.0*coords[2]*coords[0]*coords[0]*n[2] + 2.0*coords[2]*coords[2]*coords[0]*n[0];}
    else        {u[0] = 4.0*coords[0]*coords[0]*coords[0]*n[0]; u[1] = 2.0*coords[0]*coords[1]*coords[1]*n[0] + 2.0*coords[0]*coords[0]*coords[1]*n[1];}
  } else {
    u[0] = 4.0*coords[d]*coords[d]*coords[d]*n[d];
  }
  return 0;
}

PetscErrorCode mytanh(PetscInt dim, PetscReal time, const PetscReal coords[], PetscInt Nc, PetscScalar *u, void *ctx)
{
  AppCtx  *user = (AppCtx *) ctx;
  PetscInt d = user->dir;

  if (Nc > 1) {
    for (d = 0; d < Nc; ++d) u[d] = PetscTanhReal(coords[d] - 0.5);
  } else {
    u[0] = PetscTanhReal(coords[d] - 0.5);
  }
  return 0;
}
PetscErrorCode mytanhDer(PetscInt dim, PetscReal time, const PetscReal coords[], const PetscReal n[], PetscInt Nc, PetscScalar *u, void *ctx)
{
  AppCtx  *user = (AppCtx *) ctx;
  PetscInt d = user->dir;

  if (Nc > 1) {
    for (d = 0; d < Nc; ++d) u[d] = 1.0/PetscSqr(PetscCoshReal(coords[d] - 0.5)) * n[d];
  } else {
    u[0] = 1.0/PetscSqr(PetscCoshReal(coords[d] - 0.5)) * n[d];
  }
  return 0;
}

PetscErrorCode trig(PetscInt dim, PetscReal time, const PetscReal coords[], PetscInt Nc, PetscScalar *u, void *ctx)
{
  AppCtx  *user = (AppCtx *) ctx;
  PetscInt m = user->m, d = user->dir;

  if (Nc > 1) {
    for (d = 0; d < Nc; ++d) u[d] = PetscSinReal(PETSC_PI*m*coords[d]);
  } else {
    u[0] = PetscSinReal(PETSC_PI*m*coords[d]);
  }
  return 0;
}
PetscErrorCode trigDer(PetscInt dim, PetscReal time, const PetscReal coords[], const PetscReal n[], PetscInt Nc, PetscScalar *u, void *ctx)
{
  AppCtx  *user = (AppCtx *) ctx;
  PetscInt m = user->m, d = user->dir;

  if (Nc > 1) {
    for (d = 0; d < Nc; ++d) u[d] = PETSC_PI*m*PetscCosReal(PETSC_PI*m*coords[d]) * n[d];
  } else {
    u[0] = PETSC_PI*m*PetscCosReal(PETSC_PI*m*coords[d]) * n[d];
  }
  return 0;
}

static PetscErrorCode ProcessOptions(MPI_Comm comm, AppCtx *options)
{
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  options->dim             = 2;
  options->simplex         = PETSC_TRUE;
  options->interpolate     = PETSC_TRUE;
  options->refinementLimit = 0.0;
  options->qorder          = 0;
  options->Nc              = PETSC_DEFAULT;
  options->porder          = 0;
  options->m               = 1;
  options->dir             = 0;
  options->K               = 0;
  options->usePoly         = PETSC_TRUE;

  ierr = PetscOptionsBegin(comm, "", "Projection Test Options", "DMPlex");CHKERRQ(ierr);
  ierr = PetscOptionsInt("-dim", "The topological mesh dimension", "ex8.c", options->dim, &options->dim, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-simplex", "Flag for simplices or hexhedra", "ex8.c", options->simplex, &options->simplex, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-interpolate", "Generate intermediate mesh elements", "ex8.c", options->interpolate, &options->interpolate, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-refinement_limit", "The largest allowable cell volume", "ex8.c", options->refinementLimit, &options->refinementLimit, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-qorder", "The quadrature order", "ex8.c", options->qorder, &options->qorder, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-num_comp", "The number of field components", "ex8.c", options->Nc, &options->Nc, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-porder", "The order of polynomials to test", "ex8.c", options->porder, &options->porder, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-K", "The number of coarse modes used in optimization", "ex8.c", options->K, &options->K, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-use_poly", "Use polynomials (or harmonics) to adapt interpolator", "ex8.c", options->usePoly, &options->usePoly, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);

  options->Nc = options->Nc < 0 ? options->dim : options->Nc;

  PetscFunctionReturn(0);
}

static PetscErrorCode CreateMesh(MPI_Comm comm, AppCtx *user, DM *dm)
{
  DM               pdm      = NULL;
  PetscInt         cells[3] = {2, 2, 2};
  PetscPartitioner part;
  PetscErrorCode   ierr;

  PetscFunctionBeginUser;
  ierr = PetscOptionsGetInt(NULL, NULL, "-da_grid_x", &cells[0], NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL, NULL, "-da_grid_y", &cells[1], NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL, NULL, "-da_grid_z", &cells[2], NULL);CHKERRQ(ierr);
  ierr = DMPlexCreateBoxMesh(comm, user->dim, user->simplex, cells, NULL, NULL, NULL, user->interpolate, dm);CHKERRQ(ierr);
  /* Refine mesh using a volume constraint */
  if (user->simplex) {
    DM rdm = NULL;

    ierr = DMPlexSetRefinementLimit(*dm, user->refinementLimit);CHKERRQ(ierr);
    ierr = DMRefine(*dm, comm, &rdm);CHKERRQ(ierr);
    if (rdm) {
      ierr = DMDestroy(dm);CHKERRQ(ierr);
      *dm  = rdm;
    }
  }
  ierr = DMPlexSetRefinementUniform(*dm, PETSC_TRUE);CHKERRQ(ierr);
  /* Distribute mesh over processes */
  ierr = DMPlexGetPartitioner(*dm, &part);CHKERRQ(ierr);
  ierr = PetscPartitionerSetFromOptions(part);CHKERRQ(ierr);
  ierr = DMPlexDistribute(*dm, 0, NULL, &pdm);CHKERRQ(ierr);
  if (pdm) {
    ierr = DMDestroy(dm);CHKERRQ(ierr);
    *dm  = pdm;
  }
  ierr = PetscObjectSetName((PetscObject) *dm, user->simplex ? "Simplicial Mesh" : "Hexahedral Mesh");CHKERRQ(ierr);
  ierr = DMSetFromOptions(*dm);CHKERRQ(ierr);
  ierr = DMViewFromOptions(*dm, NULL, "-dm_view");CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* Setup functions to approximate */
static PetscErrorCode SetupFunctions(DM dm, PetscBool usePoly, PetscInt order, PetscInt dir, PetscErrorCode (**exactFuncs)(PetscInt, PetscReal, const PetscReal[], PetscInt, PetscScalar *, void *),
                                     PetscErrorCode (**exactFuncDers)(PetscInt, PetscReal, const PetscReal[], const PetscReal[], PetscInt, PetscScalar *, void *), AppCtx *user)
{
  PetscInt       dim;
  PetscErrorCode ierr;

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
      ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);
      SETERRQ2(PetscObjectComm((PetscObject) dm), PETSC_ERR_ARG_OUTOFRANGE, "Could not determine functions to test for dimension %d order %d", dim, order);
    }
  } else {
    user->m          = order;
    exactFuncs[0]    = trig;
    exactFuncDers[0] = trigDer;
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode ComputeError(DM dm, PetscErrorCode (**exactFuncs)(PetscInt, PetscReal, const PetscReal[], PetscInt, PetscScalar *, void *),
                                   PetscErrorCode (**exactFuncDers)(PetscInt, PetscReal, const PetscReal[], const PetscReal[], PetscInt, PetscScalar *, void *),
                                   void **exactCtxs, PetscReal *error, PetscReal *errorDer, AppCtx *user)
{
  Vec            u;
  PetscReal      n[3] = {1.0, 1.0, 1.0};
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = DMGetGlobalVector(dm, &u);CHKERRQ(ierr);
  /* Project function into FE function space */
  ierr = DMProjectFunction(dm, 0.0, exactFuncs, exactCtxs, INSERT_ALL_VALUES, u);CHKERRQ(ierr);
  ierr = VecViewFromOptions(u, NULL, "-projection_view");CHKERRQ(ierr);
  /* Compare approximation to exact in L_2 */
  ierr = DMComputeL2Diff(dm, 0.0, exactFuncs, exactCtxs, u, error);CHKERRQ(ierr);
  ierr = DMComputeL2GradientDiff(dm, 0.0, exactFuncDers, exactCtxs, u, n, errorDer);CHKERRQ(ierr);
  ierr = DMRestoreGlobalVector(dm, &u);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode CheckFunctions(DM dm, PetscInt order, AppCtx *user)
{
  PetscErrorCode (*exactFuncs[1]) (PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nf, PetscScalar *u, void *ctx);
  PetscErrorCode (*exactFuncDers[1]) (PetscInt dim, PetscReal time, const PetscReal x[], const PetscReal n[], PetscInt Nf, PetscScalar *u, void *ctx);
  void            *exactCtxs[3];
  MPI_Comm         comm;
  PetscReal        error, errorDer, tol = PETSC_SMALL;
  PetscErrorCode   ierr;

  PetscFunctionBeginUser;
  exactCtxs[0]       = user;
  exactCtxs[1]       = user;
  exactCtxs[2]       = user;
  user->constants[0] = 1.0;
  user->constants[1] = 2.0;
  user->constants[2] = 3.0;
  ierr = PetscObjectGetComm((PetscObject)dm, &comm);CHKERRQ(ierr);
  ierr = SetupFunctions(dm, PETSC_TRUE, order, 0, exactFuncs, exactFuncDers, user);CHKERRQ(ierr);
  ierr = ComputeError(dm, exactFuncs, exactFuncDers, exactCtxs, &error, &errorDer, user);CHKERRQ(ierr);
  /* Report result */
  if (error > tol)    {ierr = PetscPrintf(comm, "Function tests FAIL for order %D at tolerance %g error %g\n", order, (double)tol,(double) error);CHKERRQ(ierr);}
  else                {ierr = PetscPrintf(comm, "Function tests pass for order %D at tolerance %g\n", order, (double)tol);CHKERRQ(ierr);}
  if (errorDer > tol) {ierr = PetscPrintf(comm, "Function tests FAIL for order %D derivatives at tolerance %g error %g\n", order, (double)tol, (double)errorDer);CHKERRQ(ierr);}
  else                {ierr = PetscPrintf(comm, "Function tests pass for order %D derivatives at tolerance %g\n", order, (double)tol);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

/* Compare approximation to exact in L_2 */
static PetscErrorCode CheckTransferError(DM fdm, PetscBool usePoly, PetscInt order, PetscInt dir, const char *testname, Vec fu, AppCtx *user)
{
  PetscErrorCode (*exactFuncs[1]) (PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nf, PetscScalar *u, void *ctx);
  PetscErrorCode (*exactFuncDers[1]) (PetscInt dim, PetscReal time, const PetscReal x[], const PetscReal n[], PetscInt Nf, PetscScalar *u, void *ctx);
  PetscReal        n[3] = {1.0, 1.0, 1.0};
  void            *exactCtxs[3];
  MPI_Comm         comm;
  PetscReal        error, errorDer, tol = PETSC_SMALL;
  PetscErrorCode   ierr;

  PetscFunctionBeginUser;
  exactCtxs[0]       = user;
  exactCtxs[1]       = user;
  exactCtxs[2]       = user;
  user->constants[0] = 1.0;
  user->constants[1] = 2.0;
  user->constants[2] = 3.0;
  ierr = PetscObjectGetComm((PetscObject) fdm, &comm);CHKERRQ(ierr);
  ierr = SetupFunctions(fdm, usePoly, order, dir, exactFuncs, exactFuncDers, user);CHKERRQ(ierr);
  ierr = DMComputeL2Diff(fdm, 0.0, exactFuncs, exactCtxs, fu, &error);CHKERRQ(ierr);
  ierr = DMComputeL2GradientDiff(fdm, 0.0, exactFuncDers, exactCtxs, fu, n, &errorDer);CHKERRQ(ierr);
  /* Report result */
  if (error > tol)    {ierr = PetscPrintf(comm, "%s tests FAIL for order %D at tolerance %g error %g\n", testname, order, (double)tol, (double)error);CHKERRQ(ierr);}
  else                {ierr = PetscPrintf(comm, "%s tests pass for order %D at tolerance %g\n", testname, order, (double)tol);CHKERRQ(ierr);}
  if (errorDer > tol) {ierr = PetscPrintf(comm, "%s tests FAIL for order %D derivatives at tolerance %g error %g\n", testname, order, (double)tol, (double)errorDer);CHKERRQ(ierr);}
  else                {ierr = PetscPrintf(comm, "%s tests pass for order %D derivatives at tolerance %g\n", testname, order, (double)tol);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

static PetscErrorCode CheckTransfer(DM dm, InterpType inType, PetscInt order, AppCtx *user)
{
  PetscErrorCode (*exactFuncs[1]) (PetscInt, PetscReal, const PetscReal x[], PetscInt, PetscScalar *u, void *ctx);
  PetscErrorCode (*exactFuncDers[1]) (PetscInt, PetscReal, const PetscReal x[], const PetscReal n[], PetscInt, PetscScalar *u, void *ctx);
  void           *exactCtxs[3] = {user, user, user};
  DM              rdm, idm, fdm;
  Mat             Interp, InterpAdapt = NULL;
  Vec             iu, fu, scaling;
  MPI_Comm        comm;
  const char     *testname;
  char            checkname[PETSC_MAX_PATH_LEN];
  PetscErrorCode  ierr;

  PetscFunctionBeginUser;
  ierr = PetscObjectGetComm((PetscObject)dm,&comm);CHKERRQ(ierr);
  ierr = DMRefine(dm, comm, &rdm);CHKERRQ(ierr);
  ierr = DMSetCoarseDM(rdm, dm);CHKERRQ(ierr);
  ierr = DMCopyDisc(dm, rdm);CHKERRQ(ierr);
  switch (inType) {
  case INTERPOLATION:
    testname = "Interpolation";
    idm = dm;
    fdm = rdm;
    break;
  case RESTRICTION:
    testname = "Restriction";
    idm = rdm;
    fdm = dm;
    break;
  case INJECTION:
    testname = "Injection";
    idm = rdm;
    fdm = dm;
    break;
  }
  ierr = DMGetGlobalVector(idm, &iu);CHKERRQ(ierr);
  ierr = DMGetGlobalVector(fdm, &fu);CHKERRQ(ierr);
  ierr = DMSetApplicationContext(dm, user);CHKERRQ(ierr);
  ierr = DMSetApplicationContext(rdm, user);CHKERRQ(ierr);
  /* Project function into initial FE function space */
  ierr = SetupFunctions(dm, PETSC_TRUE, order, 0, exactFuncs, exactFuncDers, user);CHKERRQ(ierr);
  ierr = DMProjectFunction(idm, 0.0, exactFuncs, exactCtxs, INSERT_ALL_VALUES, iu);CHKERRQ(ierr);
  /* Interpolate function into final FE function space */
  switch (inType) {
  case INTERPOLATION:
    ierr = DMCreateInterpolation(dm, rdm, &Interp, &scaling);CHKERRQ(ierr);
    ierr = MatInterpolate(Interp, iu, fu);CHKERRQ(ierr);
    break;
  case RESTRICTION:
    ierr = DMCreateInterpolation(dm, rdm, &Interp, &scaling);CHKERRQ(ierr);
    ierr = MatRestrict(Interp, iu, fu);CHKERRQ(ierr);
    ierr = VecPointwiseMult(fu, scaling, fu);CHKERRQ(ierr);
    break;
  case INJECTION:
    ierr = DMCreateInjection(dm, rdm, &Interp);CHKERRQ(ierr);
    ierr = MatRestrict(Interp, iu, fu);CHKERRQ(ierr);
    break;
  }
  ierr = CheckTransferError(fdm, PETSC_TRUE, order, 0, testname, fu, user);CHKERRQ(ierr);
  if (user->K && (inType == INTERPOLATION)) {
    KSP      smoother;
    Mat      A;
    Vec     *iV, *fV;
    PetscInt k, dim, d;

    ierr = PetscPrintf(comm, " Adapting interpolator using %s\n", user->usePoly ? "polynomials" : "harmonics");CHKERRQ(ierr);
    ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);
    ierr = PetscMalloc2(user->K*dim, &iV, user->K*dim, &fV);CHKERRQ(ierr);
    /* Project coarse modes into initial and final FE function space */
    for (k = 0; k < user->K; ++k) {
      for (d = 0; d < dim; ++d) {
        ierr = DMGetGlobalVector(idm, &iV[k*dim+d]);CHKERRQ(ierr);
        ierr = DMGetGlobalVector(fdm, &fV[k*dim+d]);CHKERRQ(ierr);
        ierr = SetupFunctions(idm, user->usePoly, user->usePoly ? k : k+1, d, exactFuncs, exactFuncDers, user);CHKERRQ(ierr);
        ierr = DMProjectFunction(idm, 0.0, exactFuncs, exactCtxs, INSERT_ALL_VALUES, iV[k*dim+d]);CHKERRQ(ierr);
        ierr = DMProjectFunction(fdm, 0.0, exactFuncs, exactCtxs, INSERT_ALL_VALUES, fV[k*dim+d]);CHKERRQ(ierr);
      }
    }
    /* Adapt interpolator */
    ierr = DMCreateMatrix(rdm, &A);CHKERRQ(ierr);
    ierr = MatShift(A, 1.0);CHKERRQ(ierr);
    ierr = KSPCreate(comm, &smoother);CHKERRQ(ierr);
    ierr = KSPSetFromOptions(smoother);CHKERRQ(ierr);
    ierr = KSPSetOperators(smoother, A, A);CHKERRQ(ierr);
    ierr = DMAdaptInterpolator(dm, rdm, Interp, smoother, user->K*dim, fV, iV, &InterpAdapt, user);CHKERRQ(ierr);
    /* Interpolate function into final FE function space */
    ierr = PetscSNPrintf(checkname, PETSC_MAX_PATH_LEN, "  %s poly", testname);CHKERRQ(ierr);
    ierr = MatInterpolate(InterpAdapt, iu, fu);CHKERRQ(ierr);
    ierr = CheckTransferError(fdm, PETSC_TRUE, order, 0, checkname, fu, user);CHKERRQ(ierr);
    for (k = 0; k < user->K; ++k) {
      for (d = 0; d < dim; ++d) {
        ierr = PetscSNPrintf(checkname, PETSC_MAX_PATH_LEN, "  %s trig (%D, %D)", testname, k, d);CHKERRQ(ierr);
        ierr = MatInterpolate(InterpAdapt, iV[k*dim+d], fV[k*dim+d]);CHKERRQ(ierr);
        ierr = CheckTransferError(fdm, PETSC_FALSE, k+1, d, checkname, fV[k*dim+d], user);CHKERRQ(ierr);
      }
    }
    /* Cleanup */
    ierr = KSPDestroy(&smoother);CHKERRQ(ierr);
    ierr = MatDestroy(&A);CHKERRQ(ierr);
    for (k = 0; k < user->K; ++k) {
      for (d = 0; d < dim; ++d) {
        ierr = DMRestoreGlobalVector(idm, &iV[k*dim+d]);CHKERRQ(ierr);
        ierr = DMRestoreGlobalVector(fdm, &fV[k*dim+d]);CHKERRQ(ierr);
      }
    }
    ierr = PetscFree2(iV, fV);CHKERRQ(ierr);
    ierr = MatDestroy(&InterpAdapt);CHKERRQ(ierr);
  }
  ierr = DMRestoreGlobalVector(idm, &iu);CHKERRQ(ierr);
  ierr = DMRestoreGlobalVector(fdm, &fu);CHKERRQ(ierr);
  ierr = MatDestroy(&Interp);CHKERRQ(ierr);
  ierr = VecDestroy(&scaling);CHKERRQ(ierr);
  ierr = DMDestroy(&rdm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

int main(int argc, char **argv)
{
  DM             dm;
  AppCtx         user;                 /* user-defined work context */
  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc, &argv, NULL, help);if (ierr) return ierr;
  ierr = ProcessOptions(PETSC_COMM_WORLD, &user);CHKERRQ(ierr);
  ierr = CreateMesh(PETSC_COMM_WORLD, &user, &dm);CHKERRQ(ierr);
  ierr = PetscFECreateDefault(PETSC_COMM_WORLD, user.dim, user.Nc, user.simplex, NULL, user.qorder, &user.fe);CHKERRQ(ierr);
  ierr = DMSetField(dm, 0, NULL, (PetscObject) user.fe);CHKERRQ(ierr);
  ierr = DMCreateDS(dm);CHKERRQ(ierr);
  ierr = CheckFunctions(dm, user.porder, &user);CHKERRQ(ierr);
  ierr = CheckTransfer(dm, INTERPOLATION, user.porder, &user);CHKERRQ(ierr);
  ierr = CheckTransfer(dm, INJECTION,  user.porder, &user);CHKERRQ(ierr);
  ierr = PetscFEDestroy(&user.fe);CHKERRQ(ierr);
  ierr = DMDestroy(&dm);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

  # TODO dim 3 will not work until I get composite elements in 3D (see plexrefine.c:34)
  # 2D/3D P_1 on a simplex
  test:
    suffix: p1
    requires: triangle ctetgen
    args: -dim {{2}separate output} -petscspace_degree 1 -num_comp 1 -qorder 1 -porder {{1}separate output}
  test:
    suffix: p1_pragmatic
    requires: triangle ctetgen pragmatic
    args: -dim {{2}separate output} -petscspace_degree 1 -qorder 1 -dm_plex_hash_location -porder {{1 2}separate output}
  test:
    suffix: p1_adapt
    requires: triangle ctetgen
    args: -dim {{2}separate output} -dm_refine 3 -petscspace_degree 1 -qorder 1 -porder {{1 2}separate output}

  # TODO dim 3 will not work until I get composite elements in 3D (see plexrefine.c:34)
  # 2D/3D P_2 on a simplex
  test:
    suffix: p2
    requires: triangle ctetgen
    args: -dim {{2}separate output} -petscspace_degree 2 -qorder 2 -porder {{1 2 3}separate output}
  test:
    suffix: p2_pragmatic
    requires: triangle ctetgen pragmatic
    args: -dim {{2}separate output} -petscspace_degree 2 -qorder 2 -dm_plex_hash_location -porder {{1 2 3}separate output}

  # TODO dim 3 will not work until I get composite elements in 3D (see plexrefine.c:34)
  # TODO This is broken. Check ex3 which worked
  # 2D/3D P_3 on a simplex
  test:
    suffix: p3
    requires: triangle ctetgen !single
    args: -dim {{2}separate output} -petscspace_degree 3 -qorder 3 -porder {{1 2 3 4}separate output}
  test:
    suffix: p3_pragmatic
    requires: triangle ctetgen pragmatic !single
    args: -dim {{2}separate output} -petscspace_degree 3 -qorder 3 -dm_plex_hash_location -porder {{1 2 3 4}separate output}

  # 2D/3D Q_1 on a tensor cell
  test:
    suffix: q1
    requires: mpi_type_get_envelope
    args: -dim {{2 3}separate output} -simplex 0 -petscspace_degree 1 -qorder 1 -porder {{1 2}separate output}

  # 2D/3D Q_2 on a tensor cell
  test:
    suffix: q2
    requires: mpi_type_get_envelope
    args: -dim {{2 3}separate output} -simplex 0 -petscspace_degree 2 -qorder 2 -porder {{1 2 3}separate output}

  # 2D/3D Q_3 on a tensor cell
  test:
    suffix: q3
    requires: mpi_type_get_envelope !single
    args: -dim {{2 3}separate output} -simplex 0 -petscspace_degree 3 -qorder 3 -porder {{1 2 3 4}separate output}

  # 2D/3D P_1disc on a triangle/quadrilateral
  # TODO Missing injection functional for simplices
  test:
    suffix: p1d
    requires: triangle ctetgen
    args: -dim {{2}separate output} -simplex {{0}separate output} -petscspace_degree 1 -petscdualspace_lagrange_continuity 0 -qorder 1 -porder {{1 2}separate output}

TEST*/
