static char help[] = "Tests quadrature.\n\n";

#include <petscdt.h>

#undef __FUNCT__
#define __FUNCT__ "func1"
static void func1(PetscReal x, PetscReal *val)
{
  *val = x*PetscLogReal(1+x);
}

#undef __FUNCT__
#define __FUNCT__ "func2"
static void func2(PetscReal x, PetscReal *val)
{
  *val = x*x*PetscAtanReal(x);
}

#undef __FUNCT__
#define __FUNCT__ "func3"
static void func3(PetscReal x, PetscReal *val)
{
  *val = PetscExpReal(x)*PetscCosReal(x);
}

#undef __FUNCT__
#define __FUNCT__ "func4"
static void func4(PetscReal x, PetscReal *val)
{
  const PetscReal u = PetscSqrtReal(2.0 + x*x);
  *val = PetscAtanReal(u)/((1.0 + x*x)*u);
}

#undef __FUNCT__
#define __FUNCT__ "func5"
static void func5(PetscReal x, PetscReal *val)
{
  if (x == 0.0) *val = 0.0;
  else *val = PetscSqrtReal(x)*PetscLogReal(x);
}

#undef __FUNCT__
#define __FUNCT__ "func6"
static void func6(PetscReal x, PetscReal *val)
{
  *val = PetscSqrtReal(1-x*x);
}

#undef __FUNCT__
#define __FUNCT__ "func7"
static void func7(PetscReal x, PetscReal *val)
{
  if (x == 1.0) *val = PETSC_INFINITY;
  else *val = PetscSqrtReal(x)/PetscSqrtReal(1-x*x);
}

#undef __FUNCT__
#define __FUNCT__ "func8"
static void func8(PetscReal x, PetscReal *val)
{
  if (x == 0.0) *val = PETSC_INFINITY;
  else *val = PetscLogReal(x)*PetscLogReal(x);
}

#undef __FUNCT__
#define __FUNCT__ "func9"
static void func9(PetscReal x, PetscReal *val)
{
  *val = PetscLogReal(PetscCosReal(x));
}

#undef __FUNCT__
#define __FUNCT__ "func10"
static void func10(PetscReal x, PetscReal *val)
{
  if (x == 0.0) *val = 0.0;
  else if (x == 1.0) *val = PETSC_INFINITY;
   *val = PetscSqrtReal(PetscTanReal(x));
}
#undef __FUNCT__
#define __FUNCT__ "func11"
static void func11(PetscReal x, PetscReal *val)
{
  *val = 1/(1-2*x+2*x*x);
}
#undef __FUNCT__
#define __FUNCT__ "func12"
static void func12(PetscReal x, PetscReal *val)
{
  if (x == 0.0) *val = 0.0;
  else if (x == 1.0) *val = PETSC_INFINITY;
  else *val = PetscExpReal(1-1/x)/PetscSqrtReal(x*x*x-x*x*x*x);
}
#undef __FUNCT__
#define __FUNCT__ "func13"
static void func13(PetscReal x, PetscReal *val)
{
  if (x == 0.0) *val = 0.0;
  else if (x == 1.0) *val = 1.0;
  else *val = PetscExpReal(-(1/x-1)*(1/x-1)/2)/(x*x);
}
#undef __FUNCT__
#define __FUNCT__ "func14"
static void func14(PetscReal x, PetscReal *val)
{
  if (x == 0.0) *val = 0.0;
  else if (x == 1.0) *val = 1.0;
  else *val = PetscExpReal(1-1/x)*PetscCosReal(1/x-1)/(x*x);
}
#undef __FUNCT__


#define __FUNCT__ "main"
int main(int argc, char **argv)
{
#if PETSC_SCALAR_SIZE == 32
  PetscInt  digits       = 7;
#elif PETSC_SCALAR_SIZE == 64
  PetscInt  digits       = 14;
#else
  PetscInt  digits       = 14;
#endif
  /* for some reason in __float128 precision it cannot get more accuracy for some of the integrals */
#if defined(PETSC_USE_REAL___FLOAT128)
  const PetscReal epsilon      = 2.2204460492503131e-16;
#else
  const PetscReal epsilon      = 2500.*PETSC_MACHINE_EPSILON;
#endif
  const PetscReal bounds[28]   =
    {
      0.0, 1.0,
      0.0, 1.0,
      0.0, PETSC_PI/2.,
      0.0, 1.0,
      0.0, 1.0,
      0.0, 1.0,
      0.0, 1.0,
      0.0, 1.0,
      0.0, PETSC_PI/2.,
      0.0, PETSC_PI/2.,
      0.0, 1.0,
      0.0, 1.0,
      0.0, 1.0,
      0.0, 1.0
    };
  const PetscReal analytic[14] =
    {
      0.250000000000000,
      0.210657251225806988108092302182988001695680805674,
      1.905238690482675827736517833351916563195085437332,
      0.514041895890070761397629739576882871630921844127,
      -.444444444444444444444444444444444444444444444444,
      0.785398163397448309615660845819875721049292349843,
      1.198140234735592207439922492280323878227212663216,
      2.000000000000000000000000000000000000000000000000,
      -1.08879304515180106525034444911880697366929185018,
      2.221441469079183123507940495030346849307310844687,
      1.570796326794896619231321691639751442098584699687,
      1.772453850905516027298167483341145182797549456122,
      1.253314137315500251207882642405522626503493370304,
      0.500000000000000000000000000000000000000000000000
    };
  void          (*funcs[14])(PetscReal, PetscReal *) = {func1, func2, func3, func4, func5, func6, func7, func8, func9, func10, func11, func12, func13, func14};
  PetscInt        f;
  PetscErrorCode  ierr;

  ierr = PetscInitialize(&argc, &argv, PETSC_NULL, help);CHKERRQ(ierr);

  ierr = PetscOptionsBegin(PETSC_COMM_WORLD,"","Test Options","none");CHKERRQ(ierr);
  ierr = PetscOptionsInt("-digits", "The number of significant digits for the integral","ex3.c",digits,&digits,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();

  /* Integrate each function */
  for (f = 0; f < 14; ++f) {
    PetscReal integral;

    /* These can only be integrated accuractely using MPFR */
    if ((f == 6) || (f == 7) || (f == 9) || (f == 11)) continue;
#ifdef PETSC_USE_REAL_SINGLE
    if (f == 8) continue;
#endif
    ierr = PetscDTTanhSinhIntegrate(funcs[f], bounds[f*2+0], bounds[f*2+1], digits, &integral);CHKERRQ(ierr);
    if (PetscAbsReal(integral - analytic[f]) > PetscMax(epsilon, PetscPowRealInt(10.0, -digits)) || PetscIsInfOrNanScalar(integral - analytic[f])) {
      ierr = PetscPrintf(PETSC_COMM_SELF, "The integral of func%2d is wrong: %g (%g)\n", f+1, (double)integral, (double) PetscAbsReal(integral - analytic[f]));CHKERRQ(ierr);
    }
  }
#ifdef PETSC_HAVE_MPFR
  for (f = 0; f < 14; ++f) {
    PetscReal integral;

    ierr = PetscDTTanhSinhIntegrateMPFR(funcs[f], bounds[f*2+0], bounds[f*2+1], digits, &integral);CHKERRQ(ierr);
    if (PetscAbsReal(integral - analytic[f]) > PetscPowRealInt(10.0, -digits)) {
      ierr = PetscPrintf(PETSC_COMM_SELF, "The integral of func%2d is wrong: %g (%g)\n", f+1, (double)integral, (double)PetscAbsReal(integral - analytic[f]));CHKERRQ(ierr);
    }
  }
#endif
  PetscFinalize();
  return 0;
}
