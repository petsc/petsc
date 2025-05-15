#include <petscdt.h> /*I "petscdt.h" I*/

#include <petscvec.h>
#include <petscdraw.h>
#include <petsc/private/petscimpl.h>

const char *const DTProbDensityTypes[] = {"constant", "gaussian", "maxwell_boltzmann", "DTProb Density", "DTPROB_DENSITY_", NULL};

/*@
  PetscPDFMaxwellBoltzmann1D - PDF for the Maxwell-Boltzmann distribution in 1D

  Not Collective

  Input Parameters:
+ x     - Speed in $[0, \infty]$
- dummy - Unused

  Output Parameter:
. p - The probability density at `x`

  Level: beginner

.seealso: `PetscCDFMaxwellBoltzmann1D()`
@*/
PetscErrorCode PetscPDFMaxwellBoltzmann1D(const PetscReal x[], const PetscReal dummy[], PetscReal p[])
{
  p[0] = PetscSqrtReal(2. / PETSC_PI) * PetscExpReal(-0.5 * PetscSqr(x[0]));
  return PETSC_SUCCESS;
}

/*@
  PetscCDFMaxwellBoltzmann1D - CDF for the Maxwell-Boltzmann distribution in 1D

  Not Collective

  Input Parameters:
+ x     - Speed in $[0, \infty]$
- dummy - Unused

  Output Parameter:
. p - The probability density at `x`

  Level: beginner

.seealso: `PetscPDFMaxwellBoltzmann1D()`
@*/
PetscErrorCode PetscCDFMaxwellBoltzmann1D(const PetscReal x[], const PetscReal dummy[], PetscReal p[])
{
  p[0] = PetscErfReal(x[0] / PETSC_SQRT2);
  return PETSC_SUCCESS;
}

/*@
  PetscPDFMaxwellBoltzmann2D - PDF for the Maxwell-Boltzmann distribution in 2D

  Not Collective

  Input Parameters:
+ x     - Speed in $[0, \infty]$
- dummy - Unused

  Output Parameter:
. p - The probability density at `x`

  Level: beginner

.seealso: `PetscCDFMaxwellBoltzmann2D()`
@*/
PetscErrorCode PetscPDFMaxwellBoltzmann2D(const PetscReal x[], const PetscReal dummy[], PetscReal p[])
{
  p[0] = x[0] * PetscExpReal(-0.5 * PetscSqr(x[0]));
  return PETSC_SUCCESS;
}

/*@
  PetscCDFMaxwellBoltzmann2D - CDF for the Maxwell-Boltzmann distribution in 2D

  Not Collective

  Input Parameters:
+ x     - Speed in $[0, \infty]$
- dummy - Unused

  Output Parameter:
. p - The probability density at `x`

  Level: beginner

.seealso: `PetscPDFMaxwellBoltzmann2D()`
@*/
PetscErrorCode PetscCDFMaxwellBoltzmann2D(const PetscReal x[], const PetscReal dummy[], PetscReal p[])
{
  p[0] = 1. - PetscExpReal(-0.5 * PetscSqr(x[0]));
  return PETSC_SUCCESS;
}

/*@
  PetscPDFMaxwellBoltzmann3D - PDF for the Maxwell-Boltzmann distribution in 3D

  Not Collective

  Input Parameters:
+ x     - Speed in $[0, \infty]$
- dummy - Unused

  Output Parameter:
. p - The probability density at `x`

  Level: beginner

.seealso: `PetscCDFMaxwellBoltzmann3D()`
@*/
PetscErrorCode PetscPDFMaxwellBoltzmann3D(const PetscReal x[], const PetscReal dummy[], PetscReal p[])
{
  p[0] = PetscSqrtReal(2. / PETSC_PI) * PetscSqr(x[0]) * PetscExpReal(-0.5 * PetscSqr(x[0]));
  return PETSC_SUCCESS;
}

/*@
  PetscCDFMaxwellBoltzmann3D - CDF for the Maxwell-Boltzmann distribution in 3D

  Not Collective

  Input Parameters:
+ x     - Speed in $[0, \infty]$
- dummy - Unused

  Output Parameter:
. p - The probability density at `x`

  Level: beginner

.seealso: `PetscPDFMaxwellBoltzmann3D()`
@*/
PetscErrorCode PetscCDFMaxwellBoltzmann3D(const PetscReal x[], const PetscReal dummy[], PetscReal p[])
{
  p[0] = PetscErfReal(x[0] / PETSC_SQRT2) - PetscSqrtReal(2. / PETSC_PI) * x[0] * PetscExpReal(-0.5 * PetscSqr(x[0]));
  return PETSC_SUCCESS;
}

/*@
  PetscPDFGaussian1D - PDF for the Gaussian distribution in 1D

  Not Collective

  Input Parameters:
+ x     - Coordinate in $[-\infty, \infty]$
- scale - Scaling value

  Output Parameter:
. p - The probability density at `x`

  Level: beginner

.seealso: `PetscPDFMaxwellBoltzmann3D()`
@*/
PetscErrorCode PetscPDFGaussian1D(const PetscReal x[], const PetscReal scale[], PetscReal p[])
{
  const PetscReal sigma = scale ? scale[0] : 1.;
  p[0]                  = PetscSqrtReal(1. / (2. * PETSC_PI)) * PetscExpReal(-0.5 * PetscSqr(x[0] / sigma)) / sigma;
  return PETSC_SUCCESS;
}

PetscErrorCode PetscCDFGaussian1D(const PetscReal x[], const PetscReal scale[], PetscReal p[])
{
  const PetscReal sigma = scale ? scale[0] : 1.;
  p[0]                  = 0.5 * (1. + PetscErfReal(x[0] / PETSC_SQRT2 / sigma));
  return PETSC_SUCCESS;
}

/*@
  PetscPDFSampleGaussian1D - Sample uniformly from a Gaussian distribution in 1D

  Not Collective

  Input Parameters:
+ p     - A uniform variable on $[0, 1]$
- dummy - ignored

  Output Parameter:
. x - Coordinate in $[-\infty, \infty]$

  Level: beginner

  Note:
  See <http://www.mimirgames.com/articles/programming/approximations-of-the-inverse-error-function> and
  <https://stackoverflow.com/questions/27229371/inverse-error-function-in-c>

.seealso: `PetscPDFGaussian2D()`
@*/
PetscErrorCode PetscPDFSampleGaussian1D(const PetscReal p[], const PetscReal dummy[], PetscReal x[])
{
  const PetscReal q       = 2 * p[0] - 1.;
  const PetscInt  maxIter = 100;
  PetscReal       ck[100], r = 0.;
  PetscInt        k, m;

  PetscFunctionBeginHot;
  /* Transform input to [-1, 1] since the code below computes the inverse error function */
  for (k = 0; k < maxIter; ++k) ck[k] = 0.;
  ck[0] = 1;
  r     = ck[0] * (PetscSqrtReal(PETSC_PI) / 2.) * q;
  for (k = 1; k < maxIter; ++k) {
    const PetscReal temp = 2. * k + 1.;

    for (m = 0; m <= k - 1; ++m) {
      PetscReal denom = (m + 1.) * (2. * m + 1.);

      ck[k] += (ck[m] * ck[k - 1 - m]) / denom;
    }
    r += (ck[k] / temp) * PetscPowReal((PetscSqrtReal(PETSC_PI) / 2.) * q, 2. * k + 1.);
  }
  /* Scale erfinv() by \sqrt{\pi/2} */
  x[0] = PetscSqrtReal(PETSC_PI * 0.5) * r;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscPDFGaussian2D - PDF for the Gaussian distribution in 2D

  Not Collective

  Input Parameters:
+ x     - Coordinate in $[-\infty, \infty]^2$
- dummy - ignored

  Output Parameter:
. p - The probability density at `x`

  Level: beginner

.seealso: `PetscPDFSampleGaussian2D()`, `PetscPDFMaxwellBoltzmann3D()`
@*/
PetscErrorCode PetscPDFGaussian2D(const PetscReal x[], const PetscReal dummy[], PetscReal p[])
{
  p[0] = (1. / PETSC_PI) * PetscExpReal(-0.5 * (PetscSqr(x[0]) + PetscSqr(x[1])));
  return PETSC_SUCCESS;
}

/*@
  PetscPDFSampleGaussian2D - Sample uniformly from a Gaussian distribution in 2D
  <https://en.wikipedia.org/wiki/Box%E2%80%93Muller_transform>

  Not Collective

  Input Parameters:
+ p     - A uniform variable on $[0, 1]^2$
- dummy - ignored

  Output Parameter:
. x - Coordinate in $[-\infty, \infty]^2 $

  Level: beginner

.seealso: `PetscPDFGaussian2D()`, `PetscPDFMaxwellBoltzmann3D()`
@*/
PetscErrorCode PetscPDFSampleGaussian2D(const PetscReal p[], const PetscReal dummy[], PetscReal x[])
{
  const PetscReal mag = PetscSqrtReal(-2.0 * PetscLogReal(p[0]));
  x[0]                = mag * PetscCosReal(2.0 * PETSC_PI * p[1]);
  x[1]                = mag * PetscSinReal(2.0 * PETSC_PI * p[1]);
  return PETSC_SUCCESS;
}

/*@
  PetscPDFGaussian3D - PDF for the Gaussian distribution in 3D

  Not Collective

  Input Parameters:
+ x     - Coordinate in $[-\infty, \infty]^3$
- dummy - ignored

  Output Parameter:
. p - The probability density at `x`

  Level: beginner

.seealso: `PetscPDFSampleGaussian3D()`, `PetscPDFMaxwellBoltzmann3D()`
@*/
PetscErrorCode PetscPDFGaussian3D(const PetscReal x[], const PetscReal dummy[], PetscReal p[])
{
  p[0] = (1. / PETSC_PI * PetscSqrtReal(PETSC_PI)) * PetscExpReal(-0.5 * (PetscSqr(x[0]) + PetscSqr(x[1]) + PetscSqr(x[2])));
  return PETSC_SUCCESS;
}

/*@
  PetscPDFSampleGaussian3D - Sample uniformly from a Gaussian distribution in 3D
  <https://en.wikipedia.org/wiki/Box%E2%80%93Muller_transform>

  Not Collective

  Input Parameters:
+ p     - A uniform variable on $[0, 1]^3$
- dummy - ignored

  Output Parameter:
. x - Coordinate in $[-\infty, \infty]^3$

  Level: beginner

.seealso: `PetscPDFGaussian3D()`, `PetscPDFMaxwellBoltzmann3D()`
@*/
PetscErrorCode PetscPDFSampleGaussian3D(const PetscReal p[], const PetscReal dummy[], PetscReal x[])
{
  PetscCall(PetscPDFSampleGaussian1D(p, dummy, x));
  PetscCall(PetscPDFSampleGaussian2D(&p[1], dummy, &x[1]));
  return PETSC_SUCCESS;
}

/*@
  PetscPDFConstant1D - PDF for the uniform distribution in 1D

  Not Collective

  Input Parameters:
+ x     - Coordinate in $[-1, 1]$
- dummy - Unused

  Output Parameter:
. p - The probability density at `x`

  Level: beginner

.seealso: `PetscCDFConstant1D()`, `PetscPDFSampleConstant1D()`, `PetscPDFConstant2D()`, `PetscPDFConstant3D()`
@*/
PetscErrorCode PetscPDFConstant1D(const PetscReal x[], const PetscReal dummy[], PetscReal p[])
{
  p[0] = x[0] > -1. && x[0] <= 1. ? 0.5 : 0.;
  return PETSC_SUCCESS;
}

/*@
  PetscCDFConstant1D - CDF for the uniform distribution in 1D

  Not Collective

  Input Parameters:
+ x     - Coordinate in $[-1, 1]$
- dummy - Unused

  Output Parameter:
. p - The cumulative probability at `x`

  Level: beginner

.seealso: `PetscPDFConstant1D()`, `PetscPDFSampleConstant1D()`, `PetscCDFConstant2D()`, `PetscCDFConstant3D()`
@*/
PetscErrorCode PetscCDFConstant1D(const PetscReal x[], const PetscReal dummy[], PetscReal p[])
{
  p[0] = x[0] <= -1. ? 0. : (x[0] > 1. ? 1. : 0.5 * (x[0] + 1.));
  return PETSC_SUCCESS;
}

/*@
  PetscPDFSampleConstant1D - Sample uniformly from a uniform distribution on [-1, 1] in 1D

  Not Collective

  Input Parameters:
+ p     - A uniform variable on $[0, 1]$
- dummy - Unused

  Output Parameter:
. x - Coordinate in $[-1, 1]$

  Level: beginner

.seealso: `PetscPDFConstant1D()`, `PetscCDFConstant1D()`, `PetscPDFSampleConstant2D()`, `PetscPDFSampleConstant3D()`
@*/
PetscErrorCode PetscPDFSampleConstant1D(const PetscReal p[], const PetscReal dummy[], PetscReal x[])
{
  x[0] = 2. * p[0] - 1.;
  return PETSC_SUCCESS;
}

/*@
  PetscPDFConstant2D - PDF for the uniform distribution in 2D

  Not Collective

  Input Parameters:
+ x     - Coordinate in $[-1, 1]^2$
- dummy - Unused

  Output Parameter:
. p - The probability density at `x`

  Level: beginner

.seealso: `PetscCDFConstant2D()`, `PetscPDFSampleConstant2D()`, `PetscPDFConstant1D()`, `PetscPDFConstant3D()`
@*/
PetscErrorCode PetscPDFConstant2D(const PetscReal x[], const PetscReal dummy[], PetscReal p[])
{
  p[0] = x[0] > -1. && x[0] <= 1. && x[1] > -1. && x[1] <= 1. ? 0.25 : 0.;
  return PETSC_SUCCESS;
}

/*@
  PetscCDFConstant2D - CDF for the uniform distribution in 2D

  Not Collective

  Input Parameters:
+ x     - Coordinate in $[-1, 1]^2$
- dummy - Unused

  Output Parameter:
. p - The cumulative probability at `x`

  Level: beginner

.seealso: `PetscPDFConstant2D()`, `PetscPDFSampleConstant2D()`, `PetscCDFConstant1D()`, `PetscCDFConstant3D()`
@*/
PetscErrorCode PetscCDFConstant2D(const PetscReal x[], const PetscReal dummy[], PetscReal p[])
{
  p[0] = x[0] <= -1. || x[1] <= -1. ? 0. : (x[0] > 1. ? 1. : 0.5 * (x[0] + 1.)) * (x[1] > 1. ? 1. : 0.5 * (x[1] + 1.));
  return PETSC_SUCCESS;
}

/*@
  PetscPDFSampleConstant2D - Sample uniformly from a uniform distribution on $[-1, 1]^2$ in 2D

  Not Collective

  Input Parameters:
+ p     - Two uniform variables on $[0, 1]$
- dummy - Unused

  Output Parameter:
. x - Coordinate in $[-1, 1]^2$

  Level: beginner

.seealso: `PetscPDFConstant2D()`, `PetscCDFConstant2D()`, `PetscPDFSampleConstant1D()`, `PetscPDFSampleConstant3D()`
@*/
PetscErrorCode PetscPDFSampleConstant2D(const PetscReal p[], const PetscReal dummy[], PetscReal x[])
{
  x[0] = 2. * p[0] - 1.;
  x[1] = 2. * p[1] - 1.;
  return PETSC_SUCCESS;
}

/*@
  PetscPDFConstant3D - PDF for the uniform distribution in 3D

  Not Collective

  Input Parameters:
+ x     - Coordinate in $[-1, 1]^3$
- dummy - Unused

  Output Parameter:
. p - The probability density at `x`

  Level: beginner

.seealso: `PetscCDFConstant3D()`, `PetscPDFSampleConstant3D()`, `PetscPDFSampleConstant1D()`, `PetscPDFSampleConstant2D()`
@*/
PetscErrorCode PetscPDFConstant3D(const PetscReal x[], const PetscReal dummy[], PetscReal p[])
{
  p[0] = x[0] > -1. && x[0] <= 1. && x[1] > -1. && x[1] <= 1. && x[2] > -1. && x[2] <= 1. ? 0.125 : 0.;
  return PETSC_SUCCESS;
}

/*@
  PetscCDFConstant3D - CDF for the uniform distribution in 3D

  Not Collective

  Input Parameters:
+ x     - Coordinate in $[-1, 1]^3$
- dummy - Unused

  Output Parameter:
. p - The cumulative probability at `x`

  Level: beginner

.seealso: `PetscPDFConstant3D()`, `PetscPDFSampleConstant3D()`, `PetscCDFConstant1D()`, `PetscCDFConstant2D()`
@*/
PetscErrorCode PetscCDFConstant3D(const PetscReal x[], const PetscReal dummy[], PetscReal p[])
{
  p[0] = x[0] <= -1. || x[1] <= -1. || x[2] <= -1. ? 0. : (x[0] > 1. ? 1. : 0.5 * (x[0] + 1.)) * (x[1] > 1. ? 1. : 0.5 * (x[1] + 1.)) * (x[2] > 1. ? 1. : 0.5 * (x[2] + 1.));
  return PETSC_SUCCESS;
}

/*@
  PetscPDFSampleConstant3D - Sample uniformly from a uniform distribution on $[-1, 1]^3$ in 3D

  Not Collective

  Input Parameters:
+ p     - Three uniform variables on $[0, 1]$
- dummy - Unused

  Output Parameter:
. x - Coordinate in $[-1, 1]^3$

  Level: beginner

.seealso: `PetscPDFConstant3D()`, `PetscCDFConstant3D()`, `PetscPDFSampleConstant1D()`, `PetscPDFSampleConstant2D()`
@*/
PetscErrorCode PetscPDFSampleConstant3D(const PetscReal p[], const PetscReal dummy[], PetscReal x[])
{
  x[0] = 2. * p[0] - 1.;
  x[1] = 2. * p[1] - 1.;
  x[2] = 2. * p[2] - 1.;
  return PETSC_SUCCESS;
}

/*@C
  PetscProbCreateFromOptions - Return the probability distribution specified by the arguments and options

  Not Collective

  Input Parameters:
+ dim    - The dimension of sample points
. prefix - The options prefix, or `NULL`
- name   - The options database name for the probability distribution type

  Output Parameters:
+ pdf     - The PDF of this type, or `NULL`
. cdf     - The CDF of this type, or `NULL`
- sampler - The PDF sampler of this type, or `NULL`

  Level: intermediate

.seealso: `PetscProbFn`, `PetscPDFMaxwellBoltzmann1D()`, `PetscPDFGaussian1D()`, `PetscPDFConstant1D()`
@*/
PetscErrorCode PetscProbCreateFromOptions(PetscInt dim, const char prefix[], const char name[], PetscProbFn **pdf, PetscProbFn **cdf, PetscProbFn **sampler)
{
  DTProbDensityType den = DTPROB_DENSITY_GAUSSIAN;

  PetscFunctionBegin;
  PetscOptionsBegin(PETSC_COMM_SELF, prefix, "PetscProb Options", "DT");
  PetscCall(PetscOptionsEnum(name, "Method to compute PDF <constant, gaussian>", "", DTProbDensityTypes, (PetscEnum)den, (PetscEnum *)&den, NULL));
  PetscOptionsEnd();

  if (pdf) {
    PetscAssertPointer(pdf, 4);
    *pdf = NULL;
  }
  if (cdf) {
    PetscAssertPointer(cdf, 5);
    *cdf = NULL;
  }
  if (sampler) {
    PetscAssertPointer(sampler, 6);
    *sampler = NULL;
  }
  switch (den) {
  case DTPROB_DENSITY_CONSTANT:
    switch (dim) {
    case 1:
      if (pdf) *pdf = PetscPDFConstant1D;
      if (cdf) *cdf = PetscCDFConstant1D;
      if (sampler) *sampler = PetscPDFSampleConstant1D;
      break;
    case 2:
      if (pdf) *pdf = PetscPDFConstant2D;
      if (cdf) *cdf = PetscCDFConstant2D;
      if (sampler) *sampler = PetscPDFSampleConstant2D;
      break;
    case 3:
      if (pdf) *pdf = PetscPDFConstant3D;
      if (cdf) *cdf = PetscCDFConstant3D;
      if (sampler) *sampler = PetscPDFSampleConstant3D;
      break;
    default:
      SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Dimension %" PetscInt_FMT " not supported for density type %s", dim, DTProbDensityTypes[den]);
    }
    break;
  case DTPROB_DENSITY_GAUSSIAN:
    switch (dim) {
    case 1:
      if (pdf) *pdf = PetscPDFGaussian1D;
      if (cdf) *cdf = PetscCDFGaussian1D;
      if (sampler) *sampler = PetscPDFSampleGaussian1D;
      break;
    case 2:
      if (pdf) *pdf = PetscPDFGaussian2D;
      if (sampler) *sampler = PetscPDFSampleGaussian2D;
      break;
    case 3:
      if (pdf) *pdf = PetscPDFGaussian3D;
      if (sampler) *sampler = PetscPDFSampleGaussian3D;
      break;
    default:
      SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Dimension %" PetscInt_FMT " not supported for density type %s", dim, DTProbDensityTypes[den]);
    }
    break;
  case DTPROB_DENSITY_MAXWELL_BOLTZMANN:
    switch (dim) {
    case 1:
      if (pdf) *pdf = PetscPDFMaxwellBoltzmann1D;
      if (cdf) *cdf = PetscCDFMaxwellBoltzmann1D;
      break;
    case 2:
      if (pdf) *pdf = PetscPDFMaxwellBoltzmann2D;
      if (cdf) *cdf = PetscCDFMaxwellBoltzmann2D;
      break;
    case 3:
      if (pdf) *pdf = PetscPDFMaxwellBoltzmann3D;
      if (cdf) *cdf = PetscCDFMaxwellBoltzmann3D;
      break;
    default:
      SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Dimension %" PetscInt_FMT " not supported for density type %s", dim, DTProbDensityTypes[den]);
    }
    break;
  default:
    SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Density type %s is not supported", DTProbDensityTypes[PetscMax(0, PetscMin(den, DTPROB_NUM_DENSITY))]);
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

#ifdef PETSC_HAVE_KS
EXTERN_C_BEGIN
  #include <KolmogorovSmirnovDist.h>
EXTERN_C_END
#endif

typedef enum {
  NONE,
  ASCII,
  DRAW
} OutputType;

static PetscErrorCode KSViewerCreate(PetscObject obj, OutputType *outputType, PetscViewer *viewer)
{
  PetscViewerFormat format;
  PetscOptions      options;
  const char       *prefix;
  PetscBool         flg;
  MPI_Comm          comm;

  PetscFunctionBegin;
  *outputType = NONE;
  PetscCall(PetscObjectGetComm(obj, &comm));
  PetscCall(PetscObjectGetOptionsPrefix(obj, &prefix));
  PetscCall(PetscObjectGetOptions(obj, &options));
  PetscCall(PetscOptionsCreateViewer(comm, options, prefix, "-ks_monitor", viewer, &format, &flg));
  if (flg) {
    PetscCall(PetscObjectTypeCompare((PetscObject)*viewer, PETSCVIEWERASCII, &flg));
    if (flg) *outputType = ASCII;
    PetscCall(PetscObjectTypeCompare((PetscObject)*viewer, PETSCVIEWERDRAW, &flg));
    if (flg) *outputType = DRAW;
    PetscCall(PetscViewerPushFormat(*viewer, format));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode KSViewerDestroy(PetscViewer *viewer)
{
  PetscFunctionBegin;
  if (*viewer) {
    PetscCall(PetscViewerFlush(*viewer));
    PetscCall(PetscViewerPopFormat(*viewer));
    PetscCall(PetscViewerDestroy(viewer));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscProbComputeKSStatistic_Internal(MPI_Comm comm, PetscInt n, PetscReal val[], PetscReal wgt[], PetscProbFn *cdf, PetscReal *alpha, OutputType outputType, PetscViewer viewer)
{
#if !defined(PETSC_HAVE_KS)
  SETERRQ(comm, PETSC_ERR_SUP, "No support for Kolmogorov-Smirnov test.\nReconfigure using --download-ks");
#else
  PetscDraw     draw;
  PetscDrawLG   lg;
  PetscDrawAxis axis;
  const char   *names[2] = {"Analytic", "Empirical"};
  char          title[PETSC_MAX_PATH_LEN];
  PetscReal     Fn = 0., Dn = PETSC_MIN_REAL;
  PetscMPIInt   size;

  PetscFunctionBegin;
  PetscCallMPI(MPI_Comm_size(comm, &size));
  PetscCheck(size == 1, comm, PETSC_ERR_SUP, "Parallel K-S test not yet supported");
  if (outputType == DRAW) {
    PetscCall(PetscViewerDrawGetDraw(viewer, 0, &draw));
    PetscCall(PetscDrawLGCreate(draw, 2, &lg));
    PetscCall(PetscDrawLGSetLegend(lg, names));
  }
  if (wgt) {
    PetscReal *tmpv, *tmpw;
    PetscInt  *perm;

    PetscCall(PetscMalloc3(n, &perm, n, &tmpv, n, &tmpw));
    for (PetscInt i = 0; i < n; ++i) perm[i] = i;
    PetscCall(PetscSortRealWithPermutation(n, val, perm));
    for (PetscInt i = 0; i < n; ++i) {
      tmpv[i] = val[perm[i]];
      tmpw[i] = wgt[perm[i]];
    }
    for (PetscInt i = 0; i < n; ++i) {
      val[i] = tmpv[i];
      wgt[i] = tmpw[i];
    }
    PetscCall(PetscFree3(perm, tmpv, tmpw));
  } else PetscCall(PetscSortReal(n, val));
  // Compute empirical cumulative distribution F_n and discrepancy D_n
  for (PetscInt p = 0; p < n; ++p) {
    const PetscReal x = val[p];
    const PetscReal w = wgt ? wgt[p] : 1. / n;
    PetscReal       F, vals[2];

    Fn += w;
    PetscCall(cdf(&x, NULL, &F));
    Dn = PetscMax(PetscAbsReal(Fn - F), Dn);
    switch (outputType) {
    case ASCII:
      PetscCall(PetscViewerASCIIPrintf(viewer, "x: %g F: %g Fn: %g Dn: %.2g\n", x, F, Fn, Dn));
      break;
    case DRAW:
      vals[0] = F;
      vals[1] = Fn;
      PetscCall(PetscDrawLGAddCommonPoint(lg, x, vals));
      break;
    case NONE:
      break;
    }
  }
  if (outputType == DRAW) {
    PetscCall(PetscDrawLGGetAxis(lg, &axis));
    PetscCall(PetscSNPrintf(title, PETSC_MAX_PATH_LEN, "Kolmogorov-Smirnov Test (Dn %.2g)", Dn));
    PetscCall(PetscDrawAxisSetLabels(axis, title, "x", "CDF(x)"));
    PetscCall(PetscDrawLGDraw(lg));
    PetscCall(PetscDrawLGDestroy(&lg));
  }
  *alpha = KSfbar((int)n, (double)Dn);
  if (outputType == ASCII) PetscCall(PetscViewerASCIIPrintf(viewer, "KSfbar(%" PetscInt_FMT ", %.2g): %g\n", n, Dn, *alpha));
  PetscFunctionReturn(PETSC_SUCCESS);
#endif
}

/*@C
  PetscProbComputeKSStatistic - Compute the Kolmogorov-Smirnov statistic for the empirical distribution for an input vector, compared to an analytic CDF.

  Collective

  Input Parameters:
+ v   - The data vector, blocksize is the sample dimension
- cdf - The analytic CDF

  Output Parameter:
. alpha - The KS statistic

  Level: advanced

  Notes:
  The Kolmogorov-Smirnov statistic for a given cumulative distribution function $F(x)$ is

  $$
  D_n = \sup_x \left| F_n(x) - F(x) \right|
  $$

  where $\sup_x$ is the supremum of the set of distances, and the empirical distribution function $F_n(x)$ is discrete, and given by

  $$
  F_n =  # of samples <= x / n
  $$

  The empirical distribution function $F_n(x)$ is discrete, and thus had a ``stairstep''
  cumulative distribution, making $n$ the number of stairs. Intuitively, the statistic takes
  the largest absolute difference between the two distribution functions across all $x$ values.

  The goodness-of-fit test, or Kolmogorov-Smirnov test, is constructed using the Kolmogorov
  distribution. It rejects the null hypothesis at level $\alpha$ if

  $$
  \sqrt{n} D_{n} > K_{\alpha},
  $$

  where $K_\alpha$ is found from

  $$
  \operatorname{Pr}(K \leq K_{\alpha}) = 1 - \alpha.
  $$

  This means that getting a small alpha says that we have high confidence that the data did not come
  from the input distribution, so we say that it rejects the null hypothesis.

.seealso: `PetscProbComputeKSStatisticWeighted()`, `PetscProbComputeKSStatisticMagnitude()`, `PetscProbFn`
@*/
PetscErrorCode PetscProbComputeKSStatistic(Vec v, PetscProbFn *cdf, PetscReal *alpha)
{
  PetscViewer        viewer     = NULL;
  OutputType         outputType = NONE;
  const PetscScalar *val;
  PetscInt           n;

  PetscFunctionBegin;
  PetscCall(KSViewerCreate((PetscObject)v, &outputType, &viewer));
  PetscCall(VecGetLocalSize(v, &n));
  PetscCall(VecGetArrayRead(v, &val));
  PetscCheck(!PetscDefined(USE_COMPLEX), PetscObjectComm((PetscObject)v), PETSC_ERR_SUP, "K-S test does not support complex");
  PetscCall(PetscProbComputeKSStatistic_Internal(PetscObjectComm((PetscObject)v), n, (PetscReal *)val, NULL, cdf, alpha, outputType, viewer));
  PetscCall(VecRestoreArrayRead(v, &val));
  PetscCall(KSViewerDestroy(&viewer));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PetscProbComputeKSStatisticWeighted - Compute the Kolmogorov-Smirnov statistic for the weighted empirical distribution for an input vector, compared to an analytic CDF.

  Collective

  Input Parameters:
+ v   - The data vector, blocksize is the sample dimension
. w   - The vector of weights for each sample, instead of the default 1/n
- cdf - The analytic CDF

  Output Parameter:
. alpha - The KS statistic

  Level: advanced

  Notes:
  The Kolmogorov-Smirnov statistic for a given cumulative distribution function $F(x)$ is

  $$
  D_n = \sup_x \left| F_n(x) - F(x) \right|
  $$

  where $\sup_x$ is the supremum of the set of distances, and the empirical distribution function $F_n(x)$ is discrete, and given by

  $$
  F_n =  # of samples <= x / n
  $$

  The empirical distribution function $F_n(x)$ is discrete, and thus had a ``stairstep''
  cumulative distribution, making $n$ the number of stairs. Intuitively, the statistic takes
  the largest absolute difference between the two distribution functions across all $x$ values.

  The goodness-of-fit test, or Kolmogorov-Smirnov test, is constructed using the Kolmogorov
  distribution. It rejects the null hypothesis at level $\alpha$ if

  $$
  \sqrt{n} D_{n} > K_{\alpha},
  $$

  where $K_\alpha$ is found from

  $$
  \operatorname{Pr}(K \leq K_{\alpha}) = 1 - \alpha.
  $$

  This means that getting a small alpha says that we have high confidence that the data did not come
  from the input distribution, so we say that it rejects the null hypothesis.

.seealso: `PetscProbComputeKSStatistic()`, `PetscProbComputeKSStatisticMagnitude()`, `PetscProbFn`
@*/
PetscErrorCode PetscProbComputeKSStatisticWeighted(Vec v, Vec w, PetscProbFn *cdf, PetscReal *alpha)
{
  PetscViewer        viewer     = NULL;
  OutputType         outputType = NONE;
  const PetscScalar *val, *wgt;
  PetscInt           n;

  PetscFunctionBegin;
  PetscCall(KSViewerCreate((PetscObject)v, &outputType, &viewer));
  PetscCall(VecGetLocalSize(v, &n));
  PetscCall(VecGetArrayRead(v, &val));
  PetscCall(VecGetArrayRead(w, &wgt));
  PetscCheck(!PetscDefined(USE_COMPLEX), PetscObjectComm((PetscObject)v), PETSC_ERR_SUP, "K-S test does not support complex");
  PetscCall(PetscProbComputeKSStatistic_Internal(PetscObjectComm((PetscObject)v), n, (PetscReal *)val, (PetscReal *)wgt, cdf, alpha, outputType, viewer));
  PetscCall(VecRestoreArrayRead(v, &val));
  PetscCall(VecRestoreArrayRead(w, &wgt));
  PetscCall(KSViewerDestroy(&viewer));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PetscProbComputeKSStatisticMagnitude - Compute the Kolmogorov-Smirnov statistic for the empirical distribution for the magnitude over each block of an input vector, compared to an analytic CDF.

  Collective

  Input Parameters:
+ v   - The data vector, blocksize is the sample dimension
- cdf - The analytic CDF

  Output Parameter:
. alpha - The KS statistic

  Level: advanced

  Notes:
  The Kolmogorov-Smirnov statistic for a given cumulative distribution function $F(x)$ is

  $$
  D_n = \sup_x \left| F_n(x) - F(x) \right|
  $$

  where $\sup_x$ is the supremum of the set of distances, and the empirical distribution function $F_n(x)$ is discrete, and given by

  $$
  F_n =  # of samples <= x / n
  $$

  The empirical distribution function $F_n(x)$ is discrete, and thus had a ``stairstep''
  cumulative distribution, making $n$ the number of stairs. Intuitively, the statistic takes
  the largest absolute difference between the two distribution functions across all $x$ values.

  The goodness-of-fit test, or Kolmogorov-Smirnov test, is constructed using the Kolmogorov
  distribution. It rejects the null hypothesis at level $\alpha$ if

  $$
  \sqrt{n} D_{n} > K_{\alpha},
  $$

  where $K_\alpha$ is found from

  $$
  \operatorname{Pr}(K \leq K_{\alpha}) = 1 - \alpha.
  $$

  This means that getting a small alpha says that we have high confidence that the data did not come
  from the input distribution, so we say that it rejects the null hypothesis.

.seealso: `PetscProbComputeKSStatistic()`, `PetscProbComputeKSStatisticWeighted()`, `PetscProbFn`
@*/
PetscErrorCode PetscProbComputeKSStatisticMagnitude(Vec v, PetscProbFn *cdf, PetscReal *alpha)
{
  PetscViewer        viewer     = NULL;
  OutputType         outputType = NONE;
  const PetscScalar *a;
  PetscReal         *speed;
  PetscInt           n, dim;

  PetscFunctionBegin;
  PetscCall(KSViewerCreate((PetscObject)v, &outputType, &viewer));
  // Convert to a scalar value
  PetscCall(VecGetLocalSize(v, &n));
  PetscCall(VecGetBlockSize(v, &dim));
  n /= dim;
  PetscCall(PetscMalloc1(n, &speed));
  PetscCall(VecGetArrayRead(v, &a));
  for (PetscInt p = 0; p < n; ++p) {
    PetscReal mag = 0.;

    for (PetscInt d = 0; d < dim; ++d) mag += PetscSqr(PetscRealPart(a[p * dim + d]));
    speed[p] = PetscSqrtReal(mag);
  }
  PetscCall(VecRestoreArrayRead(v, &a));
  // Compute statistic
  PetscCall(PetscProbComputeKSStatistic_Internal(PetscObjectComm((PetscObject)v), n, speed, NULL, cdf, alpha, outputType, viewer));
  PetscCall(PetscFree(speed));
  PetscCall(KSViewerDestroy(&viewer));
  PetscFunctionReturn(PETSC_SUCCESS);
}
