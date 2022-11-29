const char help[] = "Tests properties of probability distributions";

#include <petscdt.h>

/* Checks that
   - the PDF integrates to 1
   - the incomplete integral of the PDF is the CDF at many points
*/
static PetscErrorCode VerifyDistribution(const char name[], PetscBool pos, PetscProbFunc pdf, PetscProbFunc cdf)
{
  const PetscInt digits = 14;
  PetscReal      lower = pos ? 0. : -10., upper = 10.;
  PetscReal      integral, integral2;
  PetscInt       i;

  PetscFunctionBeginUser;
  PetscCall(PetscDTTanhSinhIntegrate((void (*)(const PetscReal[], void *, PetscReal *))pdf, lower, upper, digits, NULL, &integral));
  PetscCheck(PetscAbsReal(integral - 1.0) < 100 * PETSC_MACHINE_EPSILON, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "PDF %s must integrate to 1, not %g", name, (double)integral);
  for (i = 0; i <= 10; ++i) {
    const PetscReal x = i;

    PetscCall(PetscDTTanhSinhIntegrate((void (*)(const PetscReal[], void *, PetscReal *))pdf, lower, x, digits, NULL, &integral));
    PetscCall(cdf(&x, NULL, &integral2));
    PetscCheck(PetscAbsReal(integral - integral2) < PETSC_SQRT_MACHINE_EPSILON, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Integral of PDF %s %g != %g CDF at x = %g", name, (double)integral, (double)integral2, (double)x);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode TestDistributions()
{
  PetscProbFunc pdf[]  = {PetscPDFMaxwellBoltzmann1D, PetscPDFMaxwellBoltzmann2D, PetscPDFMaxwellBoltzmann3D, PetscPDFGaussian1D};
  PetscProbFunc cdf[]  = {PetscCDFMaxwellBoltzmann1D, PetscCDFMaxwellBoltzmann2D, PetscCDFMaxwellBoltzmann3D, PetscCDFGaussian1D};
  PetscBool     pos[]  = {PETSC_TRUE, PETSC_TRUE, PETSC_TRUE, PETSC_FALSE};
  const char   *name[] = {"Maxwell-Boltzmann 1D", "Maxwell-Boltzmann 2D", "Maxwell-Boltzmann 3D", "Gaussian"};

  PetscFunctionBeginUser;
  for (PetscInt i = 0; i < (PetscInt)(sizeof(pdf) / sizeof(PetscProbFunc)); ++i) PetscCall(VerifyDistribution(name[i], pos[i], pdf[i], cdf[i]));
  PetscFunctionReturn(0);
}

static PetscErrorCode TestSampling()
{
  PetscProbFunc cdf[2]     = {PetscCDFMaxwellBoltzmann1D, PetscCDFMaxwellBoltzmann2D};
  PetscProbFunc sampler[2] = {PetscPDFSampleGaussian1D, PetscPDFSampleGaussian2D};
  PetscInt      dim[2]     = {1, 2};
  PetscRandom   rnd;
  Vec           v;
  PetscScalar  *a;
  PetscReal     alpha, confidenceLevel = 0.05;
  PetscInt      n = 1000, s, i, d;

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
  PetscFunctionReturn(0);
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

TEST*/
