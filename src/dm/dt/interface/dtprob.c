#include <petscdt.h> /*I "petscdt.h" I*/

#include <petscvec.h>
#include <petscdraw.h>
#include <petsc/private/petscimpl.h>

const char * const DTProbDensityTypes[] = {"constant", "gaussian", "maxwell_boltzmann", "DTProb Density", "DTPROB_DENSITY_", NULL};

/*@
  PetscPDFMaxwellBoltzmann1D - PDF for the Maxwell-Boltzmann distribution in 1D

  Not collective

  Input Parameter:
. x - Speed in [0, \infty]

  Output Parameter:
. p - The probability density at x

  Level: beginner

.seealso: PetscCDFMaxwellBoltzmann1D
@*/
PetscErrorCode PetscPDFMaxwellBoltzmann1D(const PetscReal x[], const PetscReal dummy[], PetscReal p[])
{
  p[0] = PetscSqrtReal(2. / PETSC_PI) * PetscExpReal(-0.5 * PetscSqr(x[0]));
  return 0;
}

/*@
  PetscCDFMaxwellBoltzmann1D - CDF for the Maxwell-Boltzmann distribution in 1D

  Not collective

  Input Parameter:
. x - Speed in [0, \infty]

  Output Parameter:
. p - The probability density at x

  Level: beginner

.seealso: PetscPDFMaxwellBoltzmann1D
@*/
PetscErrorCode PetscCDFMaxwellBoltzmann1D(const PetscReal x[], const PetscReal dummy[], PetscReal p[])
{
  p[0] = PetscErfReal(x[0] / PETSC_SQRT2);
  return 0;
}

/*@
  PetscPDFMaxwellBoltzmann2D - PDF for the Maxwell-Boltzmann distribution in 2D

  Not collective

  Input Parameter:
. x - Speed in [0, \infty]

  Output Parameter:
. p - The probability density at x

  Level: beginner

.seealso: PetscCDFMaxwellBoltzmann2D
@*/
PetscErrorCode PetscPDFMaxwellBoltzmann2D(const PetscReal x[], const PetscReal dummy[], PetscReal p[])
{
  p[0] = x[0] * PetscExpReal(-0.5 * PetscSqr(x[0]));
  return 0;
}

/*@
  PetscCDFMaxwellBoltzmann2D - CDF for the Maxwell-Boltzmann distribution in 2D

  Not collective

  Input Parameter:
. x - Speed in [0, \infty]

  Output Parameter:
. p - The probability density at x

  Level: beginner

.seealso: PetscPDFMaxwellBoltzmann2D
@*/
PetscErrorCode PetscCDFMaxwellBoltzmann2D(const PetscReal x[], const PetscReal dummy[], PetscReal p[])
{
  p[0] = 1. - PetscExpReal(-0.5 * PetscSqr(x[0]));
  return 0;
}

/*@
  PetscPDFMaxwellBoltzmann3D - PDF for the Maxwell-Boltzmann distribution in 3D

  Not collective

  Input Parameter:
. x - Speed in [0, \infty]

  Output Parameter:
. p - The probability density at x

  Level: beginner

.seealso: PetscCDFMaxwellBoltzmann3D
@*/
PetscErrorCode PetscPDFMaxwellBoltzmann3D(const PetscReal x[], const PetscReal dummy[], PetscReal p[])
{
  p[0] = PetscSqrtReal(2. / PETSC_PI) * PetscSqr(x[0]) * PetscExpReal(-0.5 * PetscSqr(x[0]));
  return 0;
}

/*@
  PetscCDFMaxwellBoltzmann3D - CDF for the Maxwell-Boltzmann distribution in 3D

  Not collective

  Input Parameter:
. x - Speed in [0, \infty]

  Output Parameter:
. p - The probability density at x

  Level: beginner

.seealso: PetscPDFMaxwellBoltzmann3D
@*/
PetscErrorCode PetscCDFMaxwellBoltzmann3D(const PetscReal x[], const PetscReal dummy[], PetscReal p[])
{
  p[0] = PetscErfReal(x[0] / PETSC_SQRT2) - PetscSqrtReal(2. / PETSC_PI) * x[0] * PetscExpReal(-0.5 * PetscSqr(x[0]));
  return 0;
}

/*@
  PetscPDFGaussian1D - PDF for the Gaussian distribution in 1D

  Not collective

  Input Parameter:
. x - Coordinate in [-\infty, \infty]

  Output Parameter:
. p - The probability density at x

  Level: beginner

.seealso: PetscPDFMaxwellBoltzmann3D
@*/
PetscErrorCode PetscPDFGaussian1D(const PetscReal x[], const PetscReal scale[], PetscReal p[])
{
  const PetscReal sigma = scale ? scale[0] : 1.;
  p[0] = PetscSqrtReal(1. / (2.*PETSC_PI)) * PetscExpReal(-0.5 * PetscSqr(x[0] / sigma)) / sigma;
  return 0;
}

PetscErrorCode PetscCDFGaussian1D(const PetscReal x[], const PetscReal scale[], PetscReal p[])
{
  const PetscReal sigma = scale ? scale[0] : 1.;
  p[0] = 0.5 * (1. + PetscErfReal(x[0] / PETSC_SQRT2 / sigma));
  return 0;
}

/*@
  PetscPDFSampleGaussian1D - Sample uniformly from a Gaussian distribution in 1D

  Not collective

  Input Parameter:
. p - A uniform variable on [0, 1]

  Output Parameter:
. x - Coordinate in [-\infty, \infty]

  Level: beginner

  Note: http://www.mimirgames.com/articles/programming/approximations-of-the-inverse-error-function/
  https://stackoverflow.com/questions/27229371/inverse-error-function-in-c
@*/
PetscErrorCode PetscPDFSampleGaussian1D(const PetscReal p[], const PetscReal dummy[], PetscReal x[])
{
  const PetscReal q = 2*p[0] - 1.;
  const PetscInt  maxIter = 100;
  PetscReal       ck[100], r = 0.;
  PetscInt        k, m;

  PetscFunctionBeginHot;
  /* Transform input to [-1, 1] since the code below computes the inverse error function */
  for (k = 0; k < maxIter; ++k) ck[k] = 0.;
  ck[0] = 1;
  r = ck[0]* (PetscSqrtReal(PETSC_PI) / 2.) * q;
  for (k = 1; k < maxIter; ++k) {
    const PetscReal temp = 2.*k + 1.;

    for (m = 0; m <= k-1; ++m) {
      PetscReal denom = (m + 1.)*(2.*m + 1.);

      ck[k] += (ck[m]*ck[k-1-m])/denom;
    }
    r += (ck[k]/temp) * PetscPowReal((PetscSqrtReal(PETSC_PI)/2.) * q, 2.*k+1.);
  }
  /* Scale erfinv() by \sqrt{\pi/2} */
  x[0] = PetscSqrtReal(PETSC_PI * 0.5) * r;
  PetscFunctionReturn(0);
}

/*@
  PetscPDFGaussian2D - PDF for the Gaussian distribution in 2D

  Not collective

  Input Parameter:
. x - Coordinate in [-\infty, \infty]^2

  Output Parameter:
. p - The probability density at x

  Level: beginner

.seealso: PetscPDFSampleGaussian2D(), PetscPDFMaxwellBoltzmann3D()
@*/
PetscErrorCode PetscPDFGaussian2D(const PetscReal x[], const PetscReal dummy[], PetscReal p[])
{
  p[0] = (1. / PETSC_PI) * PetscExpReal(-0.5 * (PetscSqr(x[0]) + PetscSqr(x[1])));
  return 0;
}

/*@
  PetscPDFSampleGaussian2D - Sample uniformly from a Gaussian distribution in 2D

  Not collective

  Input Parameter:
. p - A uniform variable on [0, 1]^2

  Output Parameter:
. x - Coordinate in [-\infty, \infty]^2

  Level: beginner

  Note: https://en.wikipedia.org/wiki/Box%E2%80%93Muller_transform

.seealso: PetscPDFGaussian2D(), PetscPDFMaxwellBoltzmann3D()
@*/
PetscErrorCode PetscPDFSampleGaussian2D(const PetscReal p[], const PetscReal dummy[], PetscReal x[])
{
  const PetscReal mag = PetscSqrtReal(-2.0 * PetscLogReal(p[0]));
  x[0] = mag * PetscCosReal(2.0 * PETSC_PI * p[1]);
  x[1] = mag * PetscSinReal(2.0 * PETSC_PI * p[1]);
  return 0;
}

/*@
  PetscPDFConstant1D - PDF for the uniform distribution in 1D

  Not collective

  Input Parameter:
. x - Coordinate in [-1, 1]

  Output Parameter:
. p - The probability density at x

  Level: beginner

.seealso: PetscCDFConstant1D()
@*/
PetscErrorCode PetscPDFConstant1D(const PetscReal x[], const PetscReal dummy[], PetscReal p[])
{
  p[0] = x[0] > -1. && x[0] <= 1. ? 0.5 : 0.;
  return 0;
}

PetscErrorCode PetscCDFConstant1D(const PetscReal x[], const PetscReal dummy[], PetscReal p[])
{
  p[0] = x[0] <= -1. ? 0. : (x[0] > 1. ? 1. : 0.5*(x[0] + 1.));
  return 0;
}

/*@
  PetscPDFSampleConstant1D - Sample uniformly from a uniform distribution on [-1, 1] in 1D

  Not collective

  Input Parameter:
. p - A uniform variable on [0, 1]

  Output Parameter:
. x - Coordinate in [-1, 1]

  Level: beginner

.seealso: PetscPDFConstant1D(), PetscCDFConstant1D()
@*/
PetscErrorCode PetscPDFSampleConstant1D(const PetscReal p[], const PetscReal dummy[], PetscReal x[])
{
  x[0] = 2.*p[1] - 1.;
  return 0;
}

/*@C
  PetscProbCreateFromOptions - Return the probability distribution specified by the argumetns and options

  Not collective

  Input Parameters:
+ dim    - The dimension of sample points
. prefix - The options prefix, or NULL
- name   - The option name for the probility distribution type

  Output Parameters:
+ pdf - The PDF of this type
. cdf - The CDF of this type
- sampler - The PDF sampler of this type

  Level: intermediate

.seealso: PetscPDFMaxwellBoltzmann1D(), PetscPDFGaussian1D(), PetscPDFConstant1D()
@*/
PetscErrorCode PetscProbCreateFromOptions(PetscInt dim, const char prefix[], const char name[], PetscProbFunc *pdf, PetscProbFunc *cdf, PetscProbFunc *sampler)
{
  DTProbDensityType den = DTPROB_DENSITY_GAUSSIAN;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ierr = PetscOptionsBegin(PETSC_COMM_SELF, prefix, "PetscProb Options", "DT");CHKERRQ(ierr);
  ierr = PetscOptionsEnum(name, "Method to compute PDF <constant, gaussian>", "", DTProbDensityTypes, (PetscEnum) den, (PetscEnum *) &den, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);

  if (pdf) {PetscValidPointer(pdf, 4); *pdf = NULL;}
  if (cdf) {PetscValidPointer(cdf, 5); *cdf = NULL;}
  if (sampler) {PetscValidPointer(sampler, 6); *sampler = NULL;}
  switch (den) {
    case DTPROB_DENSITY_CONSTANT:
      switch (dim) {
        case 1:
          if (pdf) *pdf = PetscPDFConstant1D;
          if (cdf) *cdf = PetscCDFConstant1D;
          if (sampler) *sampler = PetscPDFSampleConstant1D;
          break;
        default: SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Dimension %" PetscInt_FMT " not supported for density type %s", dim, DTProbDensityTypes[den]);
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
        default: SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Dimension %" PetscInt_FMT " not supported for density type %s", dim, DTProbDensityTypes[den]);
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
        default: SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Dimension %" PetscInt_FMT " not supported for density type %s", dim, DTProbDensityTypes[den]);
      }
      break;
    default: SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Density type %s is not supported", DTProbDensityTypes[PetscMax(0, PetscMin(den, DTPROB_NUM_DENSITY))]);
  }
  PetscFunctionReturn(0);
}

#ifdef PETSC_HAVE_KS
EXTERN_C_BEGIN
#include <KolmogorovSmirnovDist.h>
EXTERN_C_END
#endif

/*@C
  PetscProbComputeKSStatistic - Compute the Kolmogorov-Smirnov statistic for the empirical distribution for an input vector, compared to an analytic CDF.

  Collective on v

  Input Parameters:
+ v   - The data vector, blocksize is the sample dimension
- cdf - The analytic CDF

  Output Parameter:
. alpha - The KS statisic

  Level: advanced

  Note: The Kolmogorov-Smirnov statistic for a given cumulative distribution function $F(x)$ is
$
$    D_n = \sup_x \left| F_n(x) - F(x) \right|
$
where $\sup_x$ is the supremum of the set of distances, and the empirical distribution
function $F_n(x)$ is discrete, and given by
$
$    F_n =  # of samples <= x / n
$
The empirical distribution function $F_n(x)$ is discrete, and thus had a ``stairstep''
cumulative distribution, making $n$ the number of stairs. Intuitively, the statistic takes
the largest absolute difference between the two distribution functions across all $x$ values.

.seealso: PetscProbFunc
@*/
PetscErrorCode PetscProbComputeKSStatistic(Vec v, PetscProbFunc cdf, PetscReal *alpha)
{
#if !defined(PETSC_HAVE_KS)
  SETERRQ(PetscObjectComm((PetscObject) v), PETSC_ERR_SUP, "No support for Kolmogorov-Smirnov test.\nReconfigure using --download-ks");
#else
  PetscViewer        viewer = NULL;
  PetscViewerFormat  format;
  PetscDraw          draw;
  PetscDrawLG        lg;
  PetscDrawAxis      axis;
  const PetscScalar *a;
  PetscReal         *speed, Dn = PETSC_MIN_REAL;
  PetscBool          isascii = PETSC_FALSE, isdraw = PETSC_FALSE, flg;
  PetscInt           n, p, dim, d;
  PetscMPIInt        size;
  const char        *names[2] = {"Analytic", "Empirical"};
  char               title[PETSC_MAX_PATH_LEN];
  PetscOptions       options;
  const char        *prefix;
  MPI_Comm           comm;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject) v, &comm);CHKERRQ(ierr);
  ierr = PetscObjectGetOptionsPrefix((PetscObject) v, &prefix);CHKERRQ(ierr);
  ierr = PetscObjectGetOptions((PetscObject) v, &options);CHKERRQ(ierr);
  ierr = PetscOptionsGetViewer(comm, options, prefix, "-ks_monitor", &viewer, &format, &flg);CHKERRQ(ierr);
  if (flg) {
    ierr = PetscObjectTypeCompare((PetscObject) viewer, PETSCVIEWERASCII, &isascii);CHKERRQ(ierr);
    ierr = PetscObjectTypeCompare((PetscObject) viewer, PETSCVIEWERDRAW,  &isdraw);CHKERRQ(ierr);
  }
  if (isascii) {
    ierr = PetscViewerPushFormat(viewer, format);CHKERRQ(ierr);
  } else if (isdraw) {
    ierr = PetscViewerPushFormat(viewer, format);CHKERRQ(ierr);
    ierr = PetscViewerDrawGetDraw(viewer, 0, &draw);CHKERRQ(ierr);
    ierr = PetscDrawLGCreate(draw, 2, &lg);CHKERRQ(ierr);
    ierr = PetscDrawLGSetLegend(lg, names);CHKERRQ(ierr);
  }

  ierr = MPI_Comm_size(PetscObjectComm((PetscObject) v), &size);CHKERRMPI(ierr);
  ierr = VecGetLocalSize(v, &n);CHKERRQ(ierr);
  ierr = VecGetBlockSize(v, &dim);CHKERRQ(ierr);
  n   /= dim;
  /* TODO Parallel algorithm is harder */
  if (size == 1) {
    ierr = PetscMalloc1(n, &speed);CHKERRQ(ierr);
    ierr = VecGetArrayRead(v, &a);CHKERRQ(ierr);
    for (p = 0; p < n; ++p) {
      PetscReal mag = 0.;

      for (d = 0; d < dim; ++d) mag += PetscSqr(PetscRealPart(a[p*dim+d]));
      speed[p] = PetscSqrtReal(mag);
    }
    ierr = PetscSortReal(n, speed);CHKERRQ(ierr);
    ierr = VecRestoreArrayRead(v, &a);CHKERRQ(ierr);
    for (p = 0; p < n; ++p) {
      const PetscReal x = speed[p], Fn = ((PetscReal) p) / n;
      PetscReal       F, vals[2];

      ierr = cdf(&x, NULL, &F);CHKERRQ(ierr);
      Dn = PetscMax(PetscAbsReal(Fn - F), Dn);
      if (isascii) {ierr = PetscViewerASCIIPrintf(viewer, "x: %g F: %g Fn: %g Dn: %.2g\n", x, F, Fn, Dn);CHKERRQ(ierr);}
      if (isdraw)  {vals[0] = F; vals[1] = Fn; ierr = PetscDrawLGAddCommonPoint(lg, x, vals);CHKERRQ(ierr);}
    }
    if (speed[n-1] < 6.) {
      const PetscReal k = (PetscInt) (6. - speed[n-1]) + 1, dx = (6. - speed[n-1])/k;
      for (p = 0; p <= k; ++p) {
        const PetscReal x = speed[n-1] + p*dx, Fn = 1.0;
        PetscReal       F, vals[2];

        ierr = cdf(&x, NULL, &F);CHKERRQ(ierr);
        Dn = PetscMax(PetscAbsReal(Fn - F), Dn);
        if (isascii) {ierr = PetscViewerASCIIPrintf(viewer, "x: %g F: %g Fn: %g Dn: %.2g\n", x, F, Fn, Dn);CHKERRQ(ierr);}
        if (isdraw)  {vals[0] = F; vals[1] = Fn; ierr = PetscDrawLGAddCommonPoint(lg, x, vals);CHKERRQ(ierr);}
      }
    }
    ierr = PetscFree(speed);CHKERRQ(ierr);
  }
  if (isdraw) {
    ierr = PetscDrawLGGetAxis(lg, &axis);CHKERRQ(ierr);
    ierr = PetscSNPrintf(title, PETSC_MAX_PATH_LEN, "Kolmogorov-Smirnov Test (Dn %.2g)", Dn);CHKERRQ(ierr);
    ierr = PetscDrawAxisSetLabels(axis, title, "x", "CDF(x)");CHKERRQ(ierr);
    ierr = PetscDrawLGDraw(lg);CHKERRQ(ierr);
    ierr = PetscDrawLGDestroy(&lg);CHKERRQ(ierr);
  }
  if (viewer) {
    ierr = PetscViewerFlush(viewer);CHKERRQ(ierr);
    ierr = PetscViewerPopFormat(viewer);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
  }
  *alpha = KSfbar((int) n, (double) Dn);
  PetscFunctionReturn(0);
#endif
}
