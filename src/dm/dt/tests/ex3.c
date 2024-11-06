static char help[] = "Tests quadrature.\n\n";

#include <petscdt.h>

static void func1(const PetscReal a[], void *dummy, PetscReal *val)
{
  const PetscReal x = a[0];
  *val              = x * PetscLogReal(1 + x);
}

static void func2(const PetscReal a[], void *dummy, PetscReal *val)
{
  const PetscReal x = a[0];
  *val              = x * x * PetscAtanReal(x);
}

static void func3(const PetscReal a[], void *dummy, PetscReal *val)
{
  const PetscReal x = a[0];
  *val              = PetscExpReal(x) * PetscCosReal(x);
}

static void func4(const PetscReal a[], void *dummy, PetscReal *val)
{
  const PetscReal x = a[0];
  const PetscReal u = PetscSqrtReal(2.0 + x * x);
  *val              = PetscAtanReal(u) / ((1.0 + x * x) * u);
}

static void func5(const PetscReal a[], void *dummy, PetscReal *val)
{
  const PetscReal x = a[0];
  if (x == 0.0) *val = 0.0;
  else *val = PetscSqrtReal(x) * PetscLogReal(x);
}

static void func6(const PetscReal a[], void *dummy, PetscReal *val)
{
  const PetscReal x = a[0];
  *val              = PetscSqrtReal(1 - x * x);
}

static void func7(const PetscReal a[], void *dummy, PetscReal *val)
{
  const PetscReal x = a[0];
  if (x == 1.0) *val = PETSC_INFINITY;
  else *val = PetscSqrtReal(x) / PetscSqrtReal(1 - x * x);
}

static void func8(const PetscReal a[], void *dummy, PetscReal *val)
{
  const PetscReal x = a[0];
  if (x == 0.0) *val = PETSC_INFINITY;
  else *val = PetscLogReal(x) * PetscLogReal(x);
}

static void func9(const PetscReal x[], void *dummy, PetscReal *val)
{
  *val = PetscLogReal(PetscCosReal(x[0]));
}

static void func10(const PetscReal a[], void *dummy, PetscReal *val)
{
  const PetscReal x = a[0];
  if (x == 0.0) *val = 0.0;
  else if (x == 1.0) *val = PETSC_INFINITY;
  *val = PetscSqrtReal(PetscTanReal(x));
}

static void func11(const PetscReal a[], void *dummy, PetscReal *val)
{
  const PetscReal x = a[0];
  *val              = 1 / (1 - 2 * x + 2 * x * x);
}

static void func12(const PetscReal a[], void *dummy, PetscReal *val)
{
  const PetscReal x = a[0];
  if (x == 0.0) *val = 0.0;
  else if (x == 1.0) *val = PETSC_INFINITY;
  else *val = PetscExpReal(1 - 1 / x) / PetscSqrtReal(x * x * x - x * x * x * x);
}

static void func13(const PetscReal a[], void *dummy, PetscReal *val)
{
  const PetscReal x = a[0];
  if (x == 0.0) *val = 0.0;
  else if (x == 1.0) *val = 1.0;
  else *val = PetscExpReal(-(1 / x - 1) * (1 / x - 1) / 2) / (x * x);
}

static void func14(const PetscReal a[], void *dummy, PetscReal *val)
{
  const PetscReal x = a[0];
  if (x == 0.0) *val = 0.0;
  else if (x == 1.0) *val = 1.0;
  else *val = PetscExpReal(1 - 1 / x) * PetscCosReal(1 / x - 1) / (x * x);
}

int main(int argc, char **argv)
{
#if defined(PETSC_USE_REAL_SINGLE)
  PetscInt digits = 7;
#else
  PetscInt digits = 14;
#endif
  /* for some reason in __float128 precision it cannot get more accuracy for some of the integrals */
#if defined(PETSC_USE_REAL___FLOAT128)
  const PetscReal epsilon = 2.2204460492503131e-16;
#else
  const PetscReal epsilon = 2500. * PETSC_MACHINE_EPSILON;
#endif
  const PetscReal bounds[28] = {0.0, 1.0, 0.0, 1.0, 0.0, PETSC_PI / 2., 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, PETSC_PI / 2., 0.0, PETSC_PI / 2., 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0};
  const PetscReal analytic[14] = {0.250000000000000, 0.210657251225806988108092302182988001695680805674, 1.905238690482675827736517833351916563195085437332, 0.514041895890070761397629739576882871630921844127, -.444444444444444444444444444444444444444444444444, 0.785398163397448309615660845819875721049292349843, 1.198140234735592207439922492280323878227212663216, 2.000000000000000000000000000000000000000000000000, -1.08879304515180106525034444911880697366929185018, 2.221441469079183123507940495030346849307310844687, 1.570796326794896619231321691639751442098584699687, 1.772453850905516027298167483341145182797549456122, 1.253314137315500251207882642405522626503493370304, 0.500000000000000000000000000000000000000000000000};
  void (*funcs[14])(const PetscReal[], void *, PetscReal *) = {func1, func2, func3, func4, func5, func6, func7, func8, func9, func10, func11, func12, func13, func14};
  PetscInt f;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  PetscOptionsBegin(PETSC_COMM_WORLD, "", "Test Options", "none");
  PetscCall(PetscOptionsBoundedInt("-digits", "The number of significant digits for the integral", "ex3.c", digits, &digits, NULL, 1));
  PetscOptionsEnd();

  /* Integrate each function */
  for (f = 0; f < 14; ++f) {
    PetscReal integral;

    /* These can only be integrated accuractely using MPFR */
    if ((f == 6) || (f == 7) || (f == 9) || (f == 11)) continue;
#if defined(PETSC_USE_REAL_SINGLE)
    if (f == 8) continue;
#endif
    PetscCall(PetscDTTanhSinhIntegrate(funcs[f], bounds[f * 2 + 0], bounds[f * 2 + 1], digits, NULL, &integral));
    if (PetscAbsReal(integral - analytic[f]) > PetscMax(epsilon, PetscPowRealInt(10.0, -digits)) || PetscIsInfOrNanScalar(integral - analytic[f])) {
      PetscCall(PetscPrintf(PETSC_COMM_SELF, "The integral of func%2" PetscInt_FMT " is wrong: %g (%g)\n", f + 1, (double)integral, (double)PetscAbsReal(integral - analytic[f])));
    }
  }
#if defined(PETSC_HAVE_MPFR)
  for (f = 0; f < 14; ++f) {
    PetscReal integral;

    PetscCall(PetscDTTanhSinhIntegrateMPFR(funcs[f], bounds[f * 2 + 0], bounds[f * 2 + 1], digits, NULL, &integral));
    if (PetscAbsReal(integral - analytic[f]) > PetscPowRealInt(10.0, -digits)) PetscCall(PetscPrintf(PETSC_COMM_SELF, "The integral of func%2d is wrong: %g (%g)\n", f + 1, (double)integral, (double)PetscAbsReal(integral - analytic[f])));
  }
#endif
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST
  build:
    requires: !complex

  test:
    suffix: 0
TEST*/
