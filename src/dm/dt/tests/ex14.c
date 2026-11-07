const char help[] = "Tests properties of probability distributions";

#include <petscdt.h>

/* Checks that
   - the PDF integrates to 1
   - the incomplete integral of the PDF is the CDF at many points
*/
static PetscErrorCode VerifyDistribution(const char name[], PetscBool pos, PetscProbFn *pdf, PetscProbFn *cdf)
{
  const PetscInt digits = 14;
  PetscReal      lower = pos ? 0. : -10., upper = 10.;
  PetscReal      integral, integral2;
  PetscInt       i;

  PetscFunctionBeginUser;
  PetscCall(PetscDTTanhSinhIntegrate((void (*)(const PetscReal[], void *, PetscReal *))pdf, lower, upper, digits, NULL, &integral));
  PetscCheck(PetscAbsReal(integral - 1) < 100 * PETSC_MACHINE_EPSILON, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "PDF %s must integrate to 1, not %g", name, (double)integral);
  for (i = 0; i <= 10; ++i) {
    const PetscReal x = i;

    PetscCall(PetscDTTanhSinhIntegrate((void (*)(const PetscReal[], void *, PetscReal *))pdf, lower, x, digits, NULL, &integral));
    PetscCall(cdf(&x, NULL, &integral2));
    PetscCheck(PetscAbsReal(integral - integral2) < PETSC_SQRT_MACHINE_EPSILON, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Integral of PDF %s %g != %g CDF at x = %g", name, (double)integral, (double)integral2, (double)x);
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* The 2D and 3D Gaussians depend only on |x|, so their integral over R^d is the integral over r of p(x) with |x| = r
   times the measure of the sphere |x| = r, which is 2 pi r in 2D and 4 pi r^2 in 3D. We take x on the diagonal so every
   coordinate is used. */
static void Gaussian2DRadial(const PetscReal r[], PetscCtx ctx, PetscReal *f)
{
  const PetscReal x[2] = {r[0] / PetscSqrtReal(2.), r[0] / PetscSqrtReal(2.)};

  PetscCallAbort(PETSC_COMM_SELF, PetscPDFGaussian2D(x, NULL, f));
  *f *= 2. * PETSC_PI * r[0];
}

static void Gaussian3DRadial(const PetscReal r[], PetscCtx ctx, PetscReal *f)
{
  const PetscReal x[3] = {r[0] / PetscSqrtReal(3.), r[0] / PetscSqrtReal(3.), r[0] / PetscSqrtReal(3.)};

  PetscCallAbort(PETSC_COMM_SELF, PetscPDFGaussian3D(x, NULL, f));
  *f *= 4. * PETSC_PI * PetscSqr(r[0]);
}

static PetscErrorCode TestDistributions()
{
  PetscProbFn *pdf[]  = {PetscPDFMaxwellBoltzmann1D, PetscPDFMaxwellBoltzmann2D, PetscPDFMaxwellBoltzmann3D, PetscPDFGaussian1D};
  PetscProbFn *cdf[]  = {PetscCDFMaxwellBoltzmann1D, PetscCDFMaxwellBoltzmann2D, PetscCDFMaxwellBoltzmann3D, PetscCDFGaussian1D};
  PetscBool    pos[]  = {PETSC_TRUE, PETSC_TRUE, PETSC_TRUE, PETSC_FALSE};
  const char  *name[] = {"Maxwell-Boltzmann 1D", "Maxwell-Boltzmann 2D", "Maxwell-Boltzmann 3D", "Gaussian"};

  PetscFunctionBeginUser;
  for (PetscInt i = 0; i < (PetscInt)PETSC_STATIC_ARRAY_LENGTH(pdf); ++i) PetscCall(VerifyDistribution(name[i], pos[i], pdf[i], cdf[i]));
  for (PetscInt d = 2; d <= 3; ++d) {
    PetscReal integral;

    PetscCall(PetscDTTanhSinhIntegrate(d == 2 ? Gaussian2DRadial : Gaussian3DRadial, 0., 10., 14, NULL, &integral));
    PetscCheck(PetscAbsReal(integral - 1) < 100 * PETSC_MACHINE_EPSILON, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "PDF Gaussian %" PetscInt_FMT "D must integrate to 1, not %g", d, (double)integral);
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TestSampling()
{
  PetscProbFn *cdf[2]     = {PetscCDFMaxwellBoltzmann1D, PetscCDFMaxwellBoltzmann2D};
  PetscProbFn *sampler[2] = {PetscPDFSampleGaussian1D, PetscPDFSampleGaussian2D};
  PetscInt     dim[2]     = {1, 2};
  PetscRandom  rnd;
  Vec          v;
  PetscScalar *a;
  PetscReal    alpha, confidenceLevel = 0.05;
  PetscInt     n = 1000, s, i, d;

  PetscFunctionBeginUser;
  PetscCall(PetscRandomCreate(PETSC_COMM_SELF, &rnd));
  PetscCall(PetscRandomSetInterval(rnd, 0, 1.));
  PetscCall(PetscRandomSetFromOptions(rnd));

  for (s = 0; s < 2; ++s) {
    PetscCall(VecCreateSeq(PETSC_COMM_SELF, n * dim[s], &v));
    PetscCall(VecSetBlockSize(v, dim[s]));
    PetscCall(VecGetArray(v, &a));
    for (i = 0; i < n; ++i) {
      PetscReal r[3], o[3];

      for (d = 0; d < dim[s]; ++d) PetscCall(PetscRandomGetValueReal(rnd, &r[d]));
      PetscCall(sampler[s](r, NULL, o));
      for (d = 0; d < dim[s]; ++d) a[i * dim[s] + d] = o[d];
    }
    PetscCall(VecRestoreArray(v, &a));
    PetscCall(PetscProbComputeKSStatistic(v, cdf[s], &alpha));
    PetscCheck(alpha < confidenceLevel, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "KS finds sampling does not match the distribution at confidence level %.2g", (double)confidenceLevel);
    PetscCall(VecDestroy(&v));
  }
  PetscCall(PetscRandomDestroy(&rnd));
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv)
{
  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  PetscCall(TestDistributions());
  PetscCall(TestSampling());
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

  test:
    suffix: 0
    requires: ks
    args:
    output_file: output/empty.out

TEST*/
