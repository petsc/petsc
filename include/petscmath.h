/*
    PETSc mathematics include file. Defines certain basic mathematical
    constants and functions for working with single, double, and quad precision
    floating point numbers as well as complex single and double.

    This file is included by petscsys.h and should not be used directly.
*/
#pragma once

#include <math.h>
#include <petscmacros.h>
#include <petscsystypes.h>

/* SUBMANSEC = Sys */

/*
   Defines operations that are different for complex and real numbers.
   All PETSc objects in one program are built around the object
   PetscScalar which is either always a real or a complex.
*/

/*
    Real number definitions
 */
#if PetscDefined(USE_REAL_SINGLE)
  /*MC
    PetscSqrtReal - Returns the square root of a `PetscReal` value

    Synopsis:
    #include <petscmath.h>
    PetscReal PetscSqrtReal(PetscReal a)

    Not Collective; No Fortran Support

    Input Parameter:
  . a - the value

    Level: beginner

.seealso: `PetscReal`, `PetscScalar`, `PetscComplex`, `PetscSqrtComplex()`, `PetscSqrtScalar()`
  M*/
  #define PetscSqrtReal(a) sqrtf(a)
  /*MC
    PetscCbrtReal - Returns the cube root of a `PetscReal` value

    Synopsis:
    #include <petscmath.h>
    PetscReal PetscCbrtReal(PetscReal a)

    Not Collective; No Fortran Support

    Input Parameter:
  . a - the value

    Level: beginner

.seealso: `PetscReal`, `PetscScalar`, `PetscComplex`
  M*/
  #define PetscCbrtReal(a) cbrtf(a)
  /*MC
    PetscHypotReal - Returns the Euclidean distance `sqrt(a*a + b*b)` of two `PetscReal` values, without intermediate overflow

    Synopsis:
    #include <petscmath.h>
    PetscReal PetscHypotReal(PetscReal a, PetscReal b)

    Not Collective; No Fortran Support

    Input Parameters:
  + a - the value
  - b - the value

    Level: beginner

.seealso: `PetscReal`, `PetscScalar`, `PetscComplex`
  M*/
  #define PetscHypotReal(a, b) hypotf(a, b)
  /*MC
    PetscAtan2Real - Returns the arc tangent of `a/b`, in the range $[-\pi, \pi]$, using the signs of `a` and `b` to select the quadrant

    Synopsis:
    #include <petscmath.h>
    PetscReal PetscAtan2Real(PetscReal a, PetscReal b)

    Not Collective; No Fortran Support

    Input Parameters:
  + a - the value
  - b - the value

    Level: beginner

.seealso: `PetscReal`, `PetscScalar`, `PetscComplex`
  M*/
  #define PetscAtan2Real(a, b) atan2f(a, b)
  /*MC
    PetscPowReal - Returns the value of `a` raised to the power `b`, both `PetscReal`

    Synopsis:
    #include <petscmath.h>
    PetscReal PetscPowReal(PetscReal a, PetscReal b)

    Not Collective; No Fortran Support

    Input Parameters:
  + a - the value
  - b - the value

    Level: beginner

.seealso: `PetscReal`, `PetscScalar`, `PetscComplex`, `PetscPowComplex()`, `PetscPowScalar()`
  M*/
  #define PetscPowReal(a, b) powf(a, b)
  /*MC
    PetscExpReal - Returns the natural exponential $e^a$ of a `PetscReal` value

    Synopsis:
    #include <petscmath.h>
    PetscReal PetscExpReal(PetscReal a)

    Not Collective; No Fortran Support

    Input Parameter:
  . a - the value

    Level: beginner

.seealso: `PetscReal`, `PetscScalar`, `PetscComplex`, `PetscExpComplex()`, `PetscExpScalar()`
  M*/
  #define PetscExpReal(a) expf(a)
  /*MC
    PetscLogReal - Returns the natural logarithm of a `PetscReal` value

    Synopsis:
    #include <petscmath.h>
    PetscReal PetscLogReal(PetscReal a)

    Not Collective; No Fortran Support

    Input Parameter:
  . a - the value

    Level: beginner

.seealso: `PetscReal`, `PetscScalar`, `PetscComplex`, `PetscLogComplex()`, `PetscLogScalar()`
  M*/
  #define PetscLogReal(a) logf(a)
  /*MC
    PetscLog10Real - Returns the base-10 logarithm of a `PetscReal` value

    Synopsis:
    #include <petscmath.h>
    PetscReal PetscLog10Real(PetscReal a)

    Not Collective; No Fortran Support

    Input Parameter:
  . a - the value

    Level: beginner

.seealso: `PetscReal`, `PetscScalar`, `PetscComplex`
  M*/
  #define PetscLog10Real(a) log10f(a)
  /*MC
    PetscLog2Real - Returns the base-2 logarithm of a `PetscReal` value

    Synopsis:
    #include <petscmath.h>
    PetscReal PetscLog2Real(PetscReal a)

    Not Collective; No Fortran Support

    Input Parameter:
  . a - the value

    Level: beginner

.seealso: `PetscReal`, `PetscScalar`, `PetscComplex`
  M*/
  #define PetscLog2Real(a) log2f(a)
  /*MC
    PetscSinReal - Returns the sine of a `PetscReal` value, with argument in radians

    Synopsis:
    #include <petscmath.h>
    PetscReal PetscSinReal(PetscReal a)

    Not Collective; No Fortran Support

    Input Parameter:
  . a - the value

    Level: beginner

.seealso: `PetscReal`, `PetscScalar`, `PetscComplex`, `PetscSinComplex()`, `PetscSinScalar()`
  M*/
  #define PetscSinReal(a) sinf(a)
  /*MC
    PetscCosReal - Returns the cosine of a `PetscReal` value, with argument in radians

    Synopsis:
    #include <petscmath.h>
    PetscReal PetscCosReal(PetscReal a)

    Not Collective; No Fortran Support

    Input Parameter:
  . a - the value

    Level: beginner

.seealso: `PetscReal`, `PetscScalar`, `PetscComplex`, `PetscCosComplex()`, `PetscCosScalar()`
  M*/
  #define PetscCosReal(a) cosf(a)
  /*MC
    PetscTanReal - Returns the tangent of a `PetscReal` value, with argument in radians

    Synopsis:
    #include <petscmath.h>
    PetscReal PetscTanReal(PetscReal a)

    Not Collective; No Fortran Support

    Input Parameter:
  . a - the value

    Level: beginner

.seealso: `PetscReal`, `PetscScalar`, `PetscComplex`, `PetscTanComplex()`, `PetscTanScalar()`
  M*/
  #define PetscTanReal(a) tanf(a)
  /*MC
    PetscAsinReal - Returns the arc sine of a `PetscReal` value, returned in radians in the range $[-\pi/2, \pi/2]$

    Synopsis:
    #include <petscmath.h>
    PetscReal PetscAsinReal(PetscReal a)

    Not Collective; No Fortran Support

    Input Parameter:
  . a - the value

    Level: beginner

.seealso: `PetscReal`, `PetscScalar`, `PetscComplex`, `PetscAsinComplex()`, `PetscAsinScalar()`
  M*/
  #define PetscAsinReal(a) asinf(a)
  /*MC
    PetscAcosReal - Returns the arc cosine of a `PetscReal` value, returned in radians in the range $[0, \pi]$

    Synopsis:
    #include <petscmath.h>
    PetscReal PetscAcosReal(PetscReal a)

    Not Collective; No Fortran Support

    Input Parameter:
  . a - the value

    Level: beginner

.seealso: `PetscReal`, `PetscScalar`, `PetscComplex`, `PetscAcosComplex()`, `PetscAcosScalar()`
  M*/
  #define PetscAcosReal(a) acosf(a)
  /*MC
    PetscAtanReal - Returns the arc tangent of a `PetscReal` value, returned in radians in the range $[-\pi/2, \pi/2]$

    Synopsis:
    #include <petscmath.h>
    PetscReal PetscAtanReal(PetscReal a)

    Not Collective; No Fortran Support

    Input Parameter:
  . a - the value

    Level: beginner

.seealso: `PetscReal`, `PetscScalar`, `PetscComplex`, `PetscAtanComplex()`, `PetscAtanScalar()`
  M*/
  #define PetscAtanReal(a) atanf(a)
  /*MC
    PetscSinhReal - Returns the hyperbolic sine of a `PetscReal` value

    Synopsis:
    #include <petscmath.h>
    PetscReal PetscSinhReal(PetscReal a)

    Not Collective; No Fortran Support

    Input Parameter:
  . a - the value

    Level: beginner

.seealso: `PetscReal`, `PetscScalar`, `PetscComplex`, `PetscSinhComplex()`, `PetscSinhScalar()`
  M*/
  #define PetscSinhReal(a) sinhf(a)
  /*MC
    PetscCoshReal - Returns the hyperbolic cosine of a `PetscReal` value

    Synopsis:
    #include <petscmath.h>
    PetscReal PetscCoshReal(PetscReal a)

    Not Collective; No Fortran Support

    Input Parameter:
  . a - the value

    Level: beginner

.seealso: `PetscReal`, `PetscScalar`, `PetscComplex`, `PetscCoshComplex()`, `PetscCoshScalar()`
  M*/
  #define PetscCoshReal(a) coshf(a)
  /*MC
    PetscTanhReal - Returns the hyperbolic tangent of a `PetscReal` value

    Synopsis:
    #include <petscmath.h>
    PetscReal PetscTanhReal(PetscReal a)

    Not Collective; No Fortran Support

    Input Parameter:
  . a - the value

    Level: beginner

.seealso: `PetscReal`, `PetscScalar`, `PetscComplex`, `PetscTanhComplex()`, `PetscTanhScalar()`
  M*/
  #define PetscTanhReal(a) tanhf(a)
  /*MC
    PetscAsinhReal - Returns the inverse hyperbolic sine of a `PetscReal` value

    Synopsis:
    #include <petscmath.h>
    PetscReal PetscAsinhReal(PetscReal a)

    Not Collective; No Fortran Support

    Input Parameter:
  . a - the value

    Level: beginner

.seealso: `PetscReal`, `PetscScalar`, `PetscComplex`, `PetscAsinhComplex()`, `PetscAsinhScalar()`
  M*/
  #define PetscAsinhReal(a) asinhf(a)
  /*MC
    PetscAcoshReal - Returns the inverse hyperbolic cosine of a `PetscReal` value

    Synopsis:
    #include <petscmath.h>
    PetscReal PetscAcoshReal(PetscReal a)

    Not Collective; No Fortran Support

    Input Parameter:
  . a - the value

    Level: beginner

.seealso: `PetscReal`, `PetscScalar`, `PetscComplex`, `PetscAcoshComplex()`, `PetscAcoshScalar()`
  M*/
  #define PetscAcoshReal(a) acoshf(a)
  /*MC
    PetscAtanhReal - Returns the inverse hyperbolic tangent of a `PetscReal` value

    Synopsis:
    #include <petscmath.h>
    PetscReal PetscAtanhReal(PetscReal a)

    Not Collective; No Fortran Support

    Input Parameter:
  . a - the value

    Level: beginner

.seealso: `PetscReal`, `PetscScalar`, `PetscComplex`, `PetscAtanhComplex()`, `PetscAtanhScalar()`
  M*/
  #define PetscAtanhReal(a) atanhf(a)
  /*MC
    PetscErfReal - Returns the error function of a `PetscReal` value

    Synopsis:
    #include <petscmath.h>
    PetscReal PetscErfReal(PetscReal a)

    Not Collective; No Fortran Support

    Input Parameter:
  . a - the value

    Level: beginner

.seealso: `PetscReal`, `PetscScalar`, `PetscComplex`
  M*/
  #define PetscErfReal(a) erff(a)
  /*MC
    PetscCeilReal - Returns the smallest integer value, returned as a `PetscReal`, that is not less than `a`

    Synopsis:
    #include <petscmath.h>
    PetscReal PetscCeilReal(PetscReal a)

    Not Collective; No Fortran Support

    Input Parameter:
  . a - the value

    Level: beginner

.seealso: `PetscReal`, `PetscScalar`, `PetscComplex`
  M*/
  #define PetscCeilReal(a) ceilf(a)
  /*MC
    PetscFloorReal - Returns the largest integer value, returned as a `PetscReal`, that is not greater than `a`

    Synopsis:
    #include <petscmath.h>
    PetscReal PetscFloorReal(PetscReal a)

    Not Collective; No Fortran Support

    Input Parameter:
  . a - the value

    Level: beginner

.seealso: `PetscReal`, `PetscScalar`, `PetscComplex`
  M*/
  #define PetscFloorReal(a) floorf(a)
  /*MC
    PetscRintReal - Returns the value of `a` rounded to the nearest integer, returned as a `PetscReal` in the configured precision

    Synopsis:
    #include <petscmath.h>
    PetscReal PetscRintReal(PetscReal a)

    Not Collective; No Fortran Support

    Input Parameter:
  . a - the value

    Level: beginner

.seealso: `PetscReal`, `PetscScalar`, `PetscComplex`
  M*/
  #define PetscRintReal(a) rintf(a)
  /*MC
    PetscFmodReal - Returns the floating-point remainder of `a/b`, with the same sign as `a`, returned as a `PetscReal`

    Synopsis:
    #include <petscmath.h>
    PetscReal PetscFmodReal(PetscReal a, PetscReal b)

    Not Collective; No Fortran Support

    Input Parameters:
  + a - the value
  - b - the value

    Level: beginner

.seealso: `PetscReal`, `PetscScalar`, `PetscComplex`
  M*/
  #define PetscFmodReal(a, b) fmodf(a, b)
  /*MC
    PetscCopysignReal - Returns the value with the magnitude of `a` and the sign of `b`, as a `PetscReal`

    Synopsis:
    #include <petscmath.h>
    PetscReal PetscCopysignReal(PetscReal a, PetscReal b)

    Not Collective; No Fortran Support

    Input Parameters:
  + a - the value
  - b - the value

    Level: beginner

.seealso: `PetscReal`, `PetscScalar`, `PetscComplex`
  M*/
  #define PetscCopysignReal(a, b) copysignf(a, b)
  /*MC
    PetscTGamma - Returns the value of the gamma function $\Gamma(a)$ in the configured `PetscReal` precision

    Synopsis:
    #include <petscmath.h>
    PetscReal PetscTGamma(PetscReal a)

    Not Collective; No Fortran Support

    Input Parameter:
  . a - the value

    Level: beginner

.seealso: `PetscReal`, `PetscScalar`, `PetscComplex`
  M*/
  #define PetscTGamma(a) tgammaf(a)
  #if PetscDefined(HAVE_LGAMMA_IS_GAMMA)
    /*MC
      PetscLGamma - Returns the natural logarithm of the absolute value of the gamma function in the configured `PetscReal` precision

      Synopsis:
      #include <petscmath.h>
      PetscReal PetscLGamma(PetscReal a)

      Not Collective; No Fortran Support

      Input Parameter:
    . a - the value

      Level: beginner

.seealso: `PetscReal`, `PetscScalar`, `PetscComplex`
    M*/
    #define PetscLGamma(a) gammaf(a)
  #else
    #define PetscLGamma(a) lgammaf(a)
  #endif

#elif PetscDefined(USE_REAL_DOUBLE)
  #define PetscSqrtReal(a)        sqrt(a)
  #define PetscCbrtReal(a)        cbrt(a)
  #define PetscHypotReal(a, b)    hypot(a, b)
  #define PetscAtan2Real(a, b)    atan2(a, b)
  #define PetscPowReal(a, b)      pow(a, b)
  #define PetscExpReal(a)         exp(a)
  #define PetscLogReal(a)         log(a)
  #define PetscLog10Real(a)       log10(a)
  #define PetscLog2Real(a)        log2(a)
  #define PetscSinReal(a)         sin(a)
  #define PetscCosReal(a)         cos(a)
  #define PetscTanReal(a)         tan(a)
  #define PetscAsinReal(a)        asin(a)
  #define PetscAcosReal(a)        acos(a)
  #define PetscAtanReal(a)        atan(a)
  #define PetscSinhReal(a)        sinh(a)
  #define PetscCoshReal(a)        cosh(a)
  #define PetscTanhReal(a)        tanh(a)
  #define PetscAsinhReal(a)       asinh(a)
  #define PetscAcoshReal(a)       acosh(a)
  #define PetscAtanhReal(a)       atanh(a)
  #define PetscErfReal(a)         erf(a)
  #define PetscCeilReal(a)        ceil(a)
  #define PetscFloorReal(a)       floor(a)
  #define PetscRintReal(a)        rint(a)
  #define PetscFmodReal(a, b)     fmod(a, b)
  #define PetscCopysignReal(a, b) copysign(a, b)
  #define PetscTGamma(a)          tgamma(a)
  #if PetscDefined(HAVE_LGAMMA_IS_GAMMA)
    #define PetscLGamma(a) gamma(a)
  #else
    #define PetscLGamma(a) lgamma(a)
  #endif

#elif PetscDefined(USE_REAL___FLOAT128)
  #define PetscSqrtReal(a)        sqrtq(a)
  #define PetscCbrtReal(a)        cbrtq(a)
  #define PetscHypotReal(a, b)    hypotq(a, b)
  #define PetscAtan2Real(a, b)    atan2q(a, b)
  #define PetscPowReal(a, b)      powq(a, b)
  #define PetscExpReal(a)         expq(a)
  #define PetscLogReal(a)         logq(a)
  #define PetscLog10Real(a)       log10q(a)
  #define PetscLog2Real(a)        log2q(a)
  #define PetscSinReal(a)         sinq(a)
  #define PetscCosReal(a)         cosq(a)
  #define PetscTanReal(a)         tanq(a)
  #define PetscAsinReal(a)        asinq(a)
  #define PetscAcosReal(a)        acosq(a)
  #define PetscAtanReal(a)        atanq(a)
  #define PetscSinhReal(a)        sinhq(a)
  #define PetscCoshReal(a)        coshq(a)
  #define PetscTanhReal(a)        tanhq(a)
  #define PetscAsinhReal(a)       asinhq(a)
  #define PetscAcoshReal(a)       acoshq(a)
  #define PetscAtanhReal(a)       atanhq(a)
  #define PetscErfReal(a)         erfq(a)
  #define PetscCeilReal(a)        ceilq(a)
  #define PetscFloorReal(a)       floorq(a)
  #define PetscRintReal(a)        rintq(a)
  #define PetscFmodReal(a, b)     fmodq(a, b)
  #define PetscCopysignReal(a, b) copysignq(a, b)
  #define PetscTGamma(a)          tgammaq(a)
  #if PetscDefined(HAVE_LGAMMA_IS_GAMMA)
    #define PetscLGamma(a) gammaq(a)
  #else
    #define PetscLGamma(a) lgammaq(a)
  #endif

#elif PetscDefined(USE_REAL___FP16)
  #define PetscSqrtReal(a)        sqrtf(a)
  #define PetscCbrtReal(a)        cbrtf(a)
  #define PetscHypotReal(a, b)    hypotf(a, b)
  #define PetscAtan2Real(a, b)    atan2f(a, b)
  #define PetscPowReal(a, b)      powf(a, b)
  #define PetscExpReal(a)         expf(a)
  #define PetscLogReal(a)         logf(a)
  #define PetscLog10Real(a)       log10f(a)
  #define PetscLog2Real(a)        log2f(a)
  #define PetscSinReal(a)         sinf(a)
  #define PetscCosReal(a)         cosf(a)
  #define PetscTanReal(a)         tanf(a)
  #define PetscAsinReal(a)        asinf(a)
  #define PetscAcosReal(a)        acosf(a)
  #define PetscAtanReal(a)        atanf(a)
  #define PetscSinhReal(a)        sinhf(a)
  #define PetscCoshReal(a)        coshf(a)
  #define PetscTanhReal(a)        tanhf(a)
  #define PetscAsinhReal(a)       asinhf(a)
  #define PetscAcoshReal(a)       acoshf(a)
  #define PetscAtanhReal(a)       atanhf(a)
  #define PetscErfReal(a)         erff(a)
  #define PetscCeilReal(a)        ceilf(a)
  #define PetscFloorReal(a)       floorf(a)
  #define PetscRintReal(a)        rintf(a)
  #define PetscFmodReal(a, b)     fmodf(a, b)
  #define PetscCopysignReal(a, b) copysignf(a, b)
  #define PetscTGamma(a)          tgammaf(a)
  #if PetscDefined(HAVE_LGAMMA_IS_GAMMA)
    #define PetscLGamma(a) gammaf(a)
  #else
    #define PetscLGamma(a) lgammaf(a)
  #endif

#endif /* PETSC_USE_REAL_* */

static inline PetscReal PetscSignReal(PetscReal a)
{
  return (PetscReal)((a < (PetscReal)0) ? -1 : ((a > (PetscReal)0) ? 1 : 0));
}

#if !PetscDefined(HAVE_LOG2)
  #undef PetscLog2Real
static inline PetscReal PetscLog2Real(PetscReal a)
{
  return PetscLogReal(a) / PetscLogReal((PetscReal)2);
}
#endif

#if PetscDefined(HAVE_REAL___FLOAT128) && !PetscDefined(SKIP_REAL___FLOAT128)
PETSC_EXTERN MPI_Datatype MPIU___FLOAT128 PETSC_ATTRIBUTE_MPI_TYPE_TAG(__float128);
#endif
#if PetscDefined(HAVE_REAL___FP16) && !PetscDefined(SKIP_REAL___FP16)
PETSC_EXTERN MPI_Datatype MPIU___FP16 PETSC_ATTRIBUTE_MPI_TYPE_TAG(__fp16);
#endif

/*MC
   MPIU_REAL - Portable MPI datatype corresponding to `PetscReal` independent of what precision `PetscReal` is in

   Level: beginner

   Note:
   In MPI calls that require an MPI datatype that matches a `PetscReal` or array of `PetscReal` values, pass this value.

.seealso: `PetscReal`, `PetscScalar`, `PetscComplex`, `PetscInt`, `MPIU_SCALAR`, `MPIU_COMPLEX`, `MPIU_INT`
M*/
#if PetscDefined(USE_REAL_SINGLE)
  #define MPIU_REAL MPI_FLOAT
#elif PetscDefined(USE_REAL_DOUBLE)
  #define MPIU_REAL MPI_DOUBLE
#elif PetscDefined(USE_REAL___FLOAT128)
  #define MPIU_REAL MPIU___FLOAT128
#elif PetscDefined(USE_REAL___FP16)
  #define MPIU_REAL MPIU___FP16
#endif /* PETSC_USE_REAL_* */

/*
    Complex number definitions
 */
#if PetscDefined(HAVE_COMPLEX)
  #if defined(__cplusplus) && !PetscDefined(USE_REAL___FLOAT128)
  /* C++ support of complex number */

    /*MC
      PetscRealPartComplex - Returns the real part of a `PetscComplex` value, as a `PetscReal`

      Synopsis:
      #include <petscmath.h>
      PetscReal PetscRealPartComplex(PetscComplex a)

      Not Collective; No Fortran Support

      Input Parameter:
    . a - the value

      Level: beginner

.seealso: `PetscReal`, `PetscScalar`, `PetscComplex`
    M*/
    #define PetscRealPartComplex(a) (static_cast<PetscComplex>(a)).real()
    /*MC
      PetscImaginaryPartComplex - Returns the imaginary part of a `PetscComplex` value, as a `PetscReal`

      Synopsis:
      #include <petscmath.h>
      PetscReal PetscImaginaryPartComplex(PetscComplex a)

      Not Collective; No Fortran Support

      Input Parameter:
    . a - the value

      Level: beginner

.seealso: `PetscReal`, `PetscScalar`, `PetscComplex`
    M*/
    #define PetscImaginaryPartComplex(a) (static_cast<PetscComplex>(a)).imag()
    /*MC
      PetscAbsComplex - Returns the absolute value (magnitude) of a `PetscComplex` value, as a `PetscReal`

      Synopsis:
      #include <petscmath.h>
      PetscReal PetscAbsComplex(PetscComplex a)

      Not Collective; No Fortran Support

      Input Parameter:
    . a - the value

      Level: beginner

.seealso: `PetscReal`, `PetscScalar`, `PetscComplex`
    M*/
    #define PetscAbsComplex(a) petsccomplexlib::abs(static_cast<PetscComplex>(a))
    /*MC
      PetscArgComplex - Returns the argument (phase angle) of a `PetscComplex` value, as a `PetscReal`

      Synopsis:
      #include <petscmath.h>
      PetscReal PetscArgComplex(PetscComplex a)

      Not Collective; No Fortran Support

      Input Parameter:
    . a - the value

      Level: beginner

.seealso: `PetscReal`, `PetscScalar`, `PetscComplex`
    M*/
    #define PetscArgComplex(a) petsccomplexlib::arg(static_cast<PetscComplex>(a))
    /*MC
      PetscConjComplex - Returns the complex conjugate of a `PetscComplex` value

      Synopsis:
      #include <petscmath.h>
      PetscComplex PetscConjComplex(PetscComplex a)

      Not Collective; No Fortran Support

      Input Parameter:
    . a - the value

      Level: beginner

.seealso: `PetscReal`, `PetscScalar`, `PetscComplex`, `PetscConj()`
    M*/
    #define PetscConjComplex(a) petsccomplexlib::conj(static_cast<PetscComplex>(a))
    /*MC
      PetscSqrtComplex - Returns the principal value of the square root of a `PetscComplex` value

      Synopsis:
      #include <petscmath.h>
      PetscComplex PetscSqrtComplex(PetscComplex a)

      Not Collective; No Fortran Support

      Input Parameter:
    . a - the value

      Level: beginner

.seealso: `PetscReal`, `PetscScalar`, `PetscComplex`, `PetscSqrtReal()`, `PetscSqrtScalar()`
    M*/
    #define PetscSqrtComplex(a) petsccomplexlib::sqrt(static_cast<PetscComplex>(a))
    /*MC
      PetscPowComplex - Returns the complex value `a` raised to the complex power `b`

      Synopsis:
      #include <petscmath.h>
      PetscComplex PetscPowComplex(PetscComplex a, PetscComplex b)

      Not Collective; No Fortran Support

      Input Parameters:
    + a - the value
    - b - the value

      Level: beginner

.seealso: `PetscReal`, `PetscScalar`, `PetscComplex`, `PetscPowReal()`, `PetscPowScalar()`
    M*/
    #define PetscPowComplex(a, b) petsccomplexlib::pow(static_cast<PetscComplex>(a), static_cast<PetscComplex>(b))
    /*MC
      PetscExpComplex - Returns the complex exponential $e^a$ of a `PetscComplex` value

      Synopsis:
      #include <petscmath.h>
      PetscComplex PetscExpComplex(PetscComplex a)

      Not Collective; No Fortran Support

      Input Parameter:
    . a - the value

      Level: beginner

.seealso: `PetscReal`, `PetscScalar`, `PetscComplex`, `PetscExpReal()`, `PetscExpScalar()`
    M*/
    #define PetscExpComplex(a) petsccomplexlib::exp(static_cast<PetscComplex>(a))
    /*MC
      PetscLogComplex - Returns the principal value of the complex natural logarithm of a `PetscComplex` value

      Synopsis:
      #include <petscmath.h>
      PetscComplex PetscLogComplex(PetscComplex a)

      Not Collective; No Fortran Support

      Input Parameter:
    . a - the value

      Level: beginner

.seealso: `PetscReal`, `PetscScalar`, `PetscComplex`, `PetscLogReal()`, `PetscLogScalar()`
    M*/
    #define PetscLogComplex(a) petsccomplexlib::log(static_cast<PetscComplex>(a))
    /*MC
      PetscSinComplex - Returns the complex sine of a `PetscComplex` value

      Synopsis:
      #include <petscmath.h>
      PetscComplex PetscSinComplex(PetscComplex a)

      Not Collective; No Fortran Support

      Input Parameter:
    . a - the value

      Level: beginner

.seealso: `PetscReal`, `PetscScalar`, `PetscComplex`, `PetscSinReal()`, `PetscSinScalar()`
    M*/
    #define PetscSinComplex(a) petsccomplexlib::sin(static_cast<PetscComplex>(a))
    /*MC
      PetscCosComplex - Returns the complex cosine of a `PetscComplex` value

      Synopsis:
      #include <petscmath.h>
      PetscComplex PetscCosComplex(PetscComplex a)

      Not Collective; No Fortran Support

      Input Parameter:
    . a - the value

      Level: beginner

.seealso: `PetscReal`, `PetscScalar`, `PetscComplex`, `PetscCosReal()`, `PetscCosScalar()`
    M*/
    #define PetscCosComplex(a) petsccomplexlib::cos(static_cast<PetscComplex>(a))
    /*MC
      PetscTanComplex - Returns the complex tangent of a `PetscComplex` value

      Synopsis:
      #include <petscmath.h>
      PetscComplex PetscTanComplex(PetscComplex a)

      Not Collective; No Fortran Support

      Input Parameter:
    . a - the value

      Level: beginner

.seealso: `PetscReal`, `PetscScalar`, `PetscComplex`, `PetscTanReal()`, `PetscTanScalar()`
    M*/
    #define PetscTanComplex(a) petsccomplexlib::tan(static_cast<PetscComplex>(a))
    /*MC
      PetscAsinComplex - Returns the complex arc sine of a `PetscComplex` value

      Synopsis:
      #include <petscmath.h>
      PetscComplex PetscAsinComplex(PetscComplex a)

      Not Collective; No Fortran Support

      Input Parameter:
    . a - the value

      Level: beginner

.seealso: `PetscReal`, `PetscScalar`, `PetscComplex`, `PetscAsinReal()`, `PetscAsinScalar()`
    M*/
    #define PetscAsinComplex(a) petsccomplexlib::asin(static_cast<PetscComplex>(a))
    /*MC
      PetscAcosComplex - Returns the complex arc cosine of a `PetscComplex` value

      Synopsis:
      #include <petscmath.h>
      PetscComplex PetscAcosComplex(PetscComplex a)

      Not Collective; No Fortran Support

      Input Parameter:
    . a - the value

      Level: beginner

.seealso: `PetscReal`, `PetscScalar`, `PetscComplex`, `PetscAcosReal()`, `PetscAcosScalar()`
    M*/
    #define PetscAcosComplex(a) petsccomplexlib::acos(static_cast<PetscComplex>(a))
    /*MC
      PetscAtanComplex - Returns the complex arc tangent of a `PetscComplex` value

      Synopsis:
      #include <petscmath.h>
      PetscComplex PetscAtanComplex(PetscComplex a)

      Not Collective; No Fortran Support

      Input Parameter:
    . a - the value

      Level: beginner

.seealso: `PetscReal`, `PetscScalar`, `PetscComplex`, `PetscAtanReal()`, `PetscAtanScalar()`
    M*/
    #define PetscAtanComplex(a) petsccomplexlib::atan(static_cast<PetscComplex>(a))
    /*MC
      PetscSinhComplex - Returns the complex hyperbolic sine of a `PetscComplex` value

      Synopsis:
      #include <petscmath.h>
      PetscComplex PetscSinhComplex(PetscComplex a)

      Not Collective; No Fortran Support

      Input Parameter:
    . a - the value

      Level: beginner

.seealso: `PetscReal`, `PetscScalar`, `PetscComplex`, `PetscSinhReal()`, `PetscSinhScalar()`
    M*/
    #define PetscSinhComplex(a) petsccomplexlib::sinh(static_cast<PetscComplex>(a))
    /*MC
      PetscCoshComplex - Returns the complex hyperbolic cosine of a `PetscComplex` value

      Synopsis:
      #include <petscmath.h>
      PetscComplex PetscCoshComplex(PetscComplex a)

      Not Collective; No Fortran Support

      Input Parameter:
    . a - the value

      Level: beginner

.seealso: `PetscReal`, `PetscScalar`, `PetscComplex`, `PetscCoshReal()`, `PetscCoshScalar()`
    M*/
    #define PetscCoshComplex(a) petsccomplexlib::cosh(static_cast<PetscComplex>(a))
    /*MC
      PetscTanhComplex - Returns the complex hyperbolic tangent of a `PetscComplex` value

      Synopsis:
      #include <petscmath.h>
      PetscComplex PetscTanhComplex(PetscComplex a)

      Not Collective; No Fortran Support

      Input Parameter:
    . a - the value

      Level: beginner

.seealso: `PetscReal`, `PetscScalar`, `PetscComplex`, `PetscTanhReal()`, `PetscTanhScalar()`
    M*/
    #define PetscTanhComplex(a) petsccomplexlib::tanh(static_cast<PetscComplex>(a))
    /*MC
      PetscAsinhComplex - Returns the complex inverse hyperbolic sine of a `PetscComplex` value

      Synopsis:
      #include <petscmath.h>
      PetscComplex PetscAsinhComplex(PetscComplex a)

      Not Collective; No Fortran Support

      Input Parameter:
    . a - the value

      Level: beginner

.seealso: `PetscReal`, `PetscScalar`, `PetscComplex`, `PetscAsinhReal()`, `PetscAsinhScalar()`
    M*/
    #define PetscAsinhComplex(a) petsccomplexlib::asinh(static_cast<PetscComplex>(a))
    /*MC
      PetscAcoshComplex - Returns the complex inverse hyperbolic cosine of a `PetscComplex` value

      Synopsis:
      #include <petscmath.h>
      PetscComplex PetscAcoshComplex(PetscComplex a)

      Not Collective; No Fortran Support

      Input Parameter:
    . a - the value

      Level: beginner

.seealso: `PetscReal`, `PetscScalar`, `PetscComplex`, `PetscAcoshReal()`, `PetscAcoshScalar()`
    M*/
    #define PetscAcoshComplex(a) petsccomplexlib::acosh(static_cast<PetscComplex>(a))
    /*MC
      PetscAtanhComplex - Returns the complex inverse hyperbolic tangent of a `PetscComplex` value

      Synopsis:
      #include <petscmath.h>
      PetscComplex PetscAtanhComplex(PetscComplex a)

      Not Collective; No Fortran Support

      Input Parameter:
    . a - the value

      Level: beginner

.seealso: `PetscReal`, `PetscScalar`, `PetscComplex`, `PetscAtanhReal()`, `PetscAtanhScalar()`
    M*/
    #define PetscAtanhComplex(a) petsccomplexlib::atanh(static_cast<PetscComplex>(a))

  /* TODO: Add configure tests

#if !PetscDefined(HAVE_CXX_TAN_COMPLEX)
#undef PetscTanComplex
static inline PetscComplex PetscTanComplex(PetscComplex z)
{
  return PetscSinComplex(z)/PetscCosComplex(z);
}
#endif

#if !PetscDefined(HAVE_CXX_TANH_COMPLEX)
#undef PetscTanhComplex
static inline PetscComplex PetscTanhComplex(PetscComplex z)
{
  return PetscSinhComplex(z)/PetscCoshComplex(z);
}
#endif

#if !PetscDefined(HAVE_CXX_ASIN_COMPLEX)
#undef PetscAsinComplex
static inline PetscComplex PetscAsinComplex(PetscComplex z)
{
  const PetscComplex j(0,1);
  return -j*PetscLogComplex(j*z+PetscSqrtComplex(1.0f-z*z));
}
#endif

#if !PetscDefined(HAVE_CXX_ACOS_COMPLEX)
#undef PetscAcosComplex
static inline PetscComplex PetscAcosComplex(PetscComplex z)
{
  const PetscComplex j(0,1);
  return j*PetscLogComplex(z-j*PetscSqrtComplex(1.0f-z*z));
}
#endif

#if !PetscDefined(HAVE_CXX_ATAN_COMPLEX)
#undef PetscAtanComplex
static inline PetscComplex PetscAtanComplex(PetscComplex z)
{
  const PetscComplex j(0,1);
  return 0.5f*j*PetscLogComplex((1.0f-j*z)/(1.0f+j*z));
}
#endif

#if !PetscDefined(HAVE_CXX_ASINH_COMPLEX)
#undef PetscAsinhComplex
static inline PetscComplex PetscAsinhComplex(PetscComplex z)
{
  return PetscLogComplex(z+PetscSqrtComplex(z*z+1.0f));
}
#endif

#if !PetscDefined(HAVE_CXX_ACOSH_COMPLEX)
#undef PetscAcoshComplex
static inline PetscComplex PetscAcoshComplex(PetscComplex z)
{
  return PetscLogComplex(z+PetscSqrtComplex(z*z-1.0f));
}
#endif

#if !PetscDefined(HAVE_CXX_ATANH_COMPLEX)
#undef PetscAtanhComplex
static inline PetscComplex PetscAtanhComplex(PetscComplex z)
{
  return 0.5f*PetscLogComplex((1.0f+z)/(1.0f-z));
}
#endif

*/

  #else /* C99 support of complex number */

    #if PetscDefined(USE_REAL_SINGLE)
      #define PetscRealPartComplex(a)      crealf(a)
      #define PetscImaginaryPartComplex(a) cimagf(a)
      #define PetscAbsComplex(a)           cabsf(a)
      #define PetscArgComplex(a)           cargf(a)
      #define PetscConjComplex(a)          conjf(a)
      #define PetscSqrtComplex(a)          csqrtf(a)
      #define PetscPowComplex(a, b)        cpowf(a, b)
      #define PetscExpComplex(a)           cexpf(a)
      #define PetscLogComplex(a)           clogf(a)
      #define PetscSinComplex(a)           csinf(a)
      #define PetscCosComplex(a)           ccosf(a)
      #define PetscTanComplex(a)           ctanf(a)
      #define PetscAsinComplex(a)          casinf(a)
      #define PetscAcosComplex(a)          cacosf(a)
      #define PetscAtanComplex(a)          catanf(a)
      #define PetscSinhComplex(a)          csinhf(a)
      #define PetscCoshComplex(a)          ccoshf(a)
      #define PetscTanhComplex(a)          ctanhf(a)
      #define PetscAsinhComplex(a)         casinhf(a)
      #define PetscAcoshComplex(a)         cacoshf(a)
      #define PetscAtanhComplex(a)         catanhf(a)

    #elif PetscDefined(USE_REAL_DOUBLE)
      #define PetscRealPartComplex(a)      creal(a)
      #define PetscImaginaryPartComplex(a) cimag(a)
      #define PetscAbsComplex(a)           cabs(a)
      #define PetscArgComplex(a)           carg(a)
      #define PetscConjComplex(a)          conj(a)
      #define PetscSqrtComplex(a)          csqrt(a)
      #define PetscPowComplex(a, b)        cpow(a, b)
      #define PetscExpComplex(a)           cexp(a)
      #define PetscLogComplex(a)           clog(a)
      #define PetscSinComplex(a)           csin(a)
      #define PetscCosComplex(a)           ccos(a)
      #define PetscTanComplex(a)           ctan(a)
      #define PetscAsinComplex(a)          casin(a)
      #define PetscAcosComplex(a)          cacos(a)
      #define PetscAtanComplex(a)          catan(a)
      #define PetscSinhComplex(a)          csinh(a)
      #define PetscCoshComplex(a)          ccosh(a)
      #define PetscTanhComplex(a)          ctanh(a)
      #define PetscAsinhComplex(a)         casinh(a)
      #define PetscAcoshComplex(a)         cacosh(a)
      #define PetscAtanhComplex(a)         catanh(a)

    #elif PetscDefined(USE_REAL___FLOAT128)
      #define PetscRealPartComplex(a)      crealq(a)
      #define PetscImaginaryPartComplex(a) cimagq(a)
      #define PetscAbsComplex(a)           cabsq(a)
      #define PetscArgComplex(a)           cargq(a)
      #define PetscConjComplex(a)          conjq(a)
      #define PetscSqrtComplex(a)          csqrtq(a)
      #define PetscPowComplex(a, b)        cpowq(a, b)
      #define PetscExpComplex(a)           cexpq(a)
      #define PetscLogComplex(a)           clogq(a)
      #define PetscSinComplex(a)           csinq(a)
      #define PetscCosComplex(a)           ccosq(a)
      #define PetscTanComplex(a)           ctanq(a)
      #define PetscAsinComplex(a)          casinq(a)
      #define PetscAcosComplex(a)          cacosq(a)
      #define PetscAtanComplex(a)          catanq(a)
      #define PetscSinhComplex(a)          csinhq(a)
      #define PetscCoshComplex(a)          ccoshq(a)
      #define PetscTanhComplex(a)          ctanhq(a)
      #define PetscAsinhComplex(a)         casinhq(a)
      #define PetscAcoshComplex(a)         cacoshq(a)
      #define PetscAtanhComplex(a)         catanhq(a)

    #endif /* PETSC_USE_REAL_* */
  #endif   /* (__cplusplus) */

/*MC
   PETSC_i - the pure imaginary complex number i

   Level: intermediate

.seealso: `PetscComplex`, `PetscScalar`
M*/
PETSC_EXTERN PetscComplex PETSC_i;

/*
   Try to do the right thing for complex number construction: see
   http://www.open-std.org/jtc1/sc22/wg14/www/docs/n1464.htm
   for details
*/
static inline PetscComplex PetscCMPLX(PetscReal x, PetscReal y)
{
  #if defined(__cplusplus) && !PetscDefined(USE_REAL___FLOAT128)
  return PetscComplex(x, y);
  #elif defined(_Imaginary_I)
  return x + y * _Imaginary_I;
  #else
  { /* In both C99 and C11 (ISO/IEC 9899, Section 6.2.5),

       "For each floating type there is a corresponding real type, which is always a real floating
       type. For real floating types, it is the same type. For complex types, it is the type given
       by deleting the keyword _Complex from the type name."

       So type punning should be portable. */
    union
    {
      PetscComplex z;
      PetscReal    f[2];
    } uz;

    uz.f[0] = x;
    uz.f[1] = y;
    return uz.z;
  }
  #endif
}

  #define MPIU_C_COMPLEX        MPI_C_COMPLEX PETSC_DEPRECATED_MACRO(3, 15, 0, "MPI_C_COMPLEX", )
  #define MPIU_C_DOUBLE_COMPLEX MPI_C_DOUBLE_COMPLEX PETSC_DEPRECATED_MACRO(3, 15, 0, "MPI_C_DOUBLE_COMPLEX", )

  #if PetscDefined(HAVE_REAL___FLOAT128) && !PetscDefined(SKIP_REAL___FLOAT128)
    // if complex is not used, then quadmath.h won't be included by petscsystypes.h
    #if PetscDefined(USE_COMPLEX)
      #define MPIU___COMPLEX128_ATTR_TAG PETSC_ATTRIBUTE_MPI_TYPE_TAG(__complex128)
    #else
      #define MPIU___COMPLEX128_ATTR_TAG
    #endif

PETSC_EXTERN MPI_Datatype MPIU___COMPLEX128 MPIU___COMPLEX128_ATTR_TAG;

    #undef MPIU___COMPLEX128_ATTR_TAG
  #endif /* PETSC_HAVE_REAL___FLOAT128 */

  /*MC
   MPIU_COMPLEX - Portable MPI datatype corresponding to `PetscComplex` independent of the precision of `PetscComplex`

   Level: beginner

   Note:
   In MPI calls that require an MPI datatype that matches a `PetscComplex` or array of `PetscComplex` values, pass this value.

.seealso: `PetscReal`, `PetscScalar`, `PetscComplex`, `PetscInt`, `MPIU_REAL`, `MPIU_SCALAR`, `MPIU_COMPLEX`, `MPIU_INT`, `PETSC_i`
M*/
  #if PetscDefined(USE_REAL_SINGLE)
    #define MPIU_COMPLEX MPI_C_COMPLEX
  #elif PetscDefined(USE_REAL_DOUBLE)
    #define MPIU_COMPLEX MPI_C_DOUBLE_COMPLEX
  #elif PetscDefined(USE_REAL___FLOAT128)
    #define MPIU_COMPLEX MPIU___COMPLEX128
  #elif PetscDefined(USE_REAL___FP16)
    #define MPIU_COMPLEX MPI_C_COMPLEX
  #endif /* PETSC_USE_REAL_* */

#endif /* PETSC_HAVE_COMPLEX */

/*
    Scalar number definitions
 */
#if PetscDefined(USE_COMPLEX) && PetscDefined(HAVE_COMPLEX)
  /*MC
   MPIU_SCALAR - Portable MPI datatype corresponding to `PetscScalar` independent of the precision of `PetscScalar`

   Level: beginner

   Note:
   In MPI calls that require an MPI datatype that matches a `PetscScalar` or array of `PetscScalar` values, pass this value.

.seealso: `PetscReal`, `PetscScalar`, `PetscComplex`, `PetscInt`, `MPIU_REAL`, `MPIU_COMPLEX`, `MPIU_INT`
M*/
  #define MPIU_SCALAR MPIU_COMPLEX

  /*MC
   PetscRealPart - Returns the real part of a `PetscScalar`

   Synopsis:
   #include <petscmath.h>
   PetscReal PetscRealPart(PetscScalar v)

   Not Collective

   Input Parameter:
.  v - value to find the real part of

   Level: beginner

.seealso: `PetscScalar`, `PetscImaginaryPart()`, `PetscMax()`, `PetscClipInterval()`, `PetscAbsInt()`, `PetscAbsReal()`, `PetscSqr()`
M*/
  #define PetscRealPart(a) PetscRealPartComplex(a)

  /*MC
   PetscImaginaryPart - Returns the imaginary part of a `PetscScalar`

   Synopsis:
   #include <petscmath.h>
   PetscReal PetscImaginaryPart(PetscScalar v)

   Not Collective

   Input Parameter:
.  v - value to find the imaginary part of

   Level: beginner

   Note:
   If PETSc was configured for real numbers then this always returns the value 0

.seealso: `PetscScalar`, `PetscRealPart()`, `PetscMax()`, `PetscClipInterval()`, `PetscAbsInt()`, `PetscAbsReal()`, `PetscSqr()`
M*/
  #define PetscImaginaryPart(a) PetscImaginaryPartComplex(a)

  /*MC
    PetscAbsScalar - Returns the absolute value (magnitude) of a `PetscScalar` value, as a `PetscReal`

    Synopsis:
    #include <petscmath.h>
    PetscReal PetscAbsScalar(PetscScalar a)

    Not Collective; No Fortran Support

    Input Parameter:
  . a - the value

    Level: beginner

.seealso: `PetscReal`, `PetscScalar`, `PetscComplex`
  M*/
  #define PetscAbsScalar(a) PetscAbsComplex(a)
  /*MC
    PetscArgScalar - Returns the argument (phase angle) of a `PetscScalar` value, as a `PetscReal`; for real `PetscScalar` this is `0` for nonnegative values and `PETSC_PI` otherwise

    Synopsis:
    #include <petscmath.h>
    PetscReal PetscArgScalar(PetscScalar a)

    Not Collective; No Fortran Support

    Input Parameter:
  . a - the value

    Level: beginner

.seealso: `PetscReal`, `PetscScalar`, `PetscComplex`
  M*/
  #define PetscArgScalar(a) PetscArgComplex(a)
  /*MC
    PetscConj - Returns the complex conjugate of a `PetscScalar` value; the identity when `PetscScalar` is real

    Synopsis:
    #include <petscmath.h>
    PetscScalar PetscConj(PetscScalar a)

    Not Collective; No Fortran Support

    Input Parameter:
  . a - the value

    Level: beginner

.seealso: `PetscReal`, `PetscScalar`, `PetscComplex`, `PetscConjComplex()`
  M*/
  #define PetscConj(a) PetscConjComplex(a)
  /*MC
    PetscSqrtScalar - Returns the square root of a `PetscScalar` value

    Synopsis:
    #include <petscmath.h>
    PetscScalar PetscSqrtScalar(PetscScalar a)

    Not Collective; No Fortran Support

    Input Parameter:
  . a - the value

    Level: beginner

.seealso: `PetscReal`, `PetscScalar`, `PetscComplex`, `PetscSqrtReal()`, `PetscSqrtComplex()`
  M*/
  #define PetscSqrtScalar(a) PetscSqrtComplex(a)
  /*MC
    PetscPowScalar - Returns the value of `a` raised to the power `b`, both `PetscScalar`

    Synopsis:
    #include <petscmath.h>
    PetscScalar PetscPowScalar(PetscScalar a, PetscScalar b)

    Not Collective; No Fortran Support

    Input Parameters:
  + a - the value
  - b - the value

    Level: beginner

.seealso: `PetscReal`, `PetscScalar`, `PetscComplex`, `PetscPowReal()`, `PetscPowComplex()`
  M*/
  #define PetscPowScalar(a, b) PetscPowComplex(a, b)
  /*MC
    PetscExpScalar - Returns the exponential $e^a$ of a `PetscScalar` value

    Synopsis:
    #include <petscmath.h>
    PetscScalar PetscExpScalar(PetscScalar a)

    Not Collective; No Fortran Support

    Input Parameter:
  . a - the value

    Level: beginner

.seealso: `PetscReal`, `PetscScalar`, `PetscComplex`, `PetscExpReal()`, `PetscExpComplex()`
  M*/
  #define PetscExpScalar(a) PetscExpComplex(a)
  /*MC
    PetscLogScalar - Returns the natural logarithm of a `PetscScalar` value

    Synopsis:
    #include <petscmath.h>
    PetscScalar PetscLogScalar(PetscScalar a)

    Not Collective; No Fortran Support

    Input Parameter:
  . a - the value

    Level: beginner

.seealso: `PetscReal`, `PetscScalar`, `PetscComplex`, `PetscLogReal()`, `PetscLogComplex()`
  M*/
  #define PetscLogScalar(a) PetscLogComplex(a)
  /*MC
    PetscSinScalar - Returns the sine of a `PetscScalar` value

    Synopsis:
    #include <petscmath.h>
    PetscScalar PetscSinScalar(PetscScalar a)

    Not Collective; No Fortran Support

    Input Parameter:
  . a - the value

    Level: beginner

.seealso: `PetscReal`, `PetscScalar`, `PetscComplex`, `PetscSinReal()`, `PetscSinComplex()`
  M*/
  #define PetscSinScalar(a) PetscSinComplex(a)
  /*MC
    PetscCosScalar - Returns the cosine of a `PetscScalar` value

    Synopsis:
    #include <petscmath.h>
    PetscScalar PetscCosScalar(PetscScalar a)

    Not Collective; No Fortran Support

    Input Parameter:
  . a - the value

    Level: beginner

.seealso: `PetscReal`, `PetscScalar`, `PetscComplex`, `PetscCosReal()`, `PetscCosComplex()`
  M*/
  #define PetscCosScalar(a) PetscCosComplex(a)
  /*MC
    PetscTanScalar - Returns the tangent of a `PetscScalar` value

    Synopsis:
    #include <petscmath.h>
    PetscScalar PetscTanScalar(PetscScalar a)

    Not Collective; No Fortran Support

    Input Parameter:
  . a - the value

    Level: beginner

.seealso: `PetscReal`, `PetscScalar`, `PetscComplex`, `PetscTanReal()`, `PetscTanComplex()`
  M*/
  #define PetscTanScalar(a) PetscTanComplex(a)
  /*MC
    PetscAsinScalar - Returns the arc sine of a `PetscScalar` value

    Synopsis:
    #include <petscmath.h>
    PetscScalar PetscAsinScalar(PetscScalar a)

    Not Collective; No Fortran Support

    Input Parameter:
  . a - the value

    Level: beginner

.seealso: `PetscReal`, `PetscScalar`, `PetscComplex`, `PetscAsinReal()`, `PetscAsinComplex()`
  M*/
  #define PetscAsinScalar(a) PetscAsinComplex(a)
  /*MC
    PetscAcosScalar - Returns the arc cosine of a `PetscScalar` value

    Synopsis:
    #include <petscmath.h>
    PetscScalar PetscAcosScalar(PetscScalar a)

    Not Collective; No Fortran Support

    Input Parameter:
  . a - the value

    Level: beginner

.seealso: `PetscReal`, `PetscScalar`, `PetscComplex`, `PetscAcosReal()`, `PetscAcosComplex()`
  M*/
  #define PetscAcosScalar(a) PetscAcosComplex(a)
  /*MC
    PetscAtanScalar - Returns the arc tangent of a `PetscScalar` value

    Synopsis:
    #include <petscmath.h>
    PetscScalar PetscAtanScalar(PetscScalar a)

    Not Collective; No Fortran Support

    Input Parameter:
  . a - the value

    Level: beginner

.seealso: `PetscReal`, `PetscScalar`, `PetscComplex`, `PetscAtanReal()`, `PetscAtanComplex()`
  M*/
  #define PetscAtanScalar(a) PetscAtanComplex(a)
  /*MC
    PetscSinhScalar - Returns the hyperbolic sine of a `PetscScalar` value

    Synopsis:
    #include <petscmath.h>
    PetscScalar PetscSinhScalar(PetscScalar a)

    Not Collective; No Fortran Support

    Input Parameter:
  . a - the value

    Level: beginner

.seealso: `PetscReal`, `PetscScalar`, `PetscComplex`, `PetscSinhReal()`, `PetscSinhComplex()`
  M*/
  #define PetscSinhScalar(a) PetscSinhComplex(a)
  /*MC
    PetscCoshScalar - Returns the hyperbolic cosine of a `PetscScalar` value

    Synopsis:
    #include <petscmath.h>
    PetscScalar PetscCoshScalar(PetscScalar a)

    Not Collective; No Fortran Support

    Input Parameter:
  . a - the value

    Level: beginner

.seealso: `PetscReal`, `PetscScalar`, `PetscComplex`, `PetscCoshReal()`, `PetscCoshComplex()`
  M*/
  #define PetscCoshScalar(a) PetscCoshComplex(a)
  /*MC
    PetscTanhScalar - Returns the hyperbolic tangent of a `PetscScalar` value

    Synopsis:
    #include <petscmath.h>
    PetscScalar PetscTanhScalar(PetscScalar a)

    Not Collective; No Fortran Support

    Input Parameter:
  . a - the value

    Level: beginner

.seealso: `PetscReal`, `PetscScalar`, `PetscComplex`, `PetscTanhReal()`, `PetscTanhComplex()`
  M*/
  #define PetscTanhScalar(a) PetscTanhComplex(a)
  /*MC
    PetscAsinhScalar - Returns the inverse hyperbolic sine of a `PetscScalar` value

    Synopsis:
    #include <petscmath.h>
    PetscScalar PetscAsinhScalar(PetscScalar a)

    Not Collective; No Fortran Support

    Input Parameter:
  . a - the value

    Level: beginner

.seealso: `PetscReal`, `PetscScalar`, `PetscComplex`, `PetscAsinhReal()`, `PetscAsinhComplex()`
  M*/
  #define PetscAsinhScalar(a) PetscAsinhComplex(a)
  /*MC
    PetscAcoshScalar - Returns the inverse hyperbolic cosine of a `PetscScalar` value

    Synopsis:
    #include <petscmath.h>
    PetscScalar PetscAcoshScalar(PetscScalar a)

    Not Collective; No Fortran Support

    Input Parameter:
  . a - the value

    Level: beginner

.seealso: `PetscReal`, `PetscScalar`, `PetscComplex`, `PetscAcoshReal()`, `PetscAcoshComplex()`
  M*/
  #define PetscAcoshScalar(a) PetscAcoshComplex(a)
  /*MC
    PetscAtanhScalar - Returns the inverse hyperbolic tangent of a `PetscScalar` value

    Synopsis:
    #include <petscmath.h>
    PetscScalar PetscAtanhScalar(PetscScalar a)

    Not Collective; No Fortran Support

    Input Parameter:
  . a - the value

    Level: beginner

.seealso: `PetscReal`, `PetscScalar`, `PetscComplex`, `PetscAtanhReal()`, `PetscAtanhComplex()`
  M*/
  #define PetscAtanhScalar(a) PetscAtanhComplex(a)

#else /* PETSC_USE_COMPLEX */
  #define MPIU_SCALAR           MPIU_REAL
  #define PetscRealPart(a)      (a)
  #define PetscImaginaryPart(a) ((PetscReal)0)
  #define PetscAbsScalar(a)     PetscAbsReal(a)
  #define PetscArgScalar(a)     (((a) < (PetscReal)0) ? PETSC_PI : (PetscReal)0)
  #define PetscConj(a)          (a)
  #define PetscSqrtScalar(a)    PetscSqrtReal(a)
  #define PetscPowScalar(a, b)  PetscPowReal(a, b)
  #define PetscExpScalar(a)     PetscExpReal(a)
  #define PetscLogScalar(a)     PetscLogReal(a)
  #define PetscSinScalar(a)     PetscSinReal(a)
  #define PetscCosScalar(a)     PetscCosReal(a)
  #define PetscTanScalar(a)     PetscTanReal(a)
  #define PetscAsinScalar(a)    PetscAsinReal(a)
  #define PetscAcosScalar(a)    PetscAcosReal(a)
  #define PetscAtanScalar(a)    PetscAtanReal(a)
  #define PetscSinhScalar(a)    PetscSinhReal(a)
  #define PetscCoshScalar(a)    PetscCoshReal(a)
  #define PetscTanhScalar(a)    PetscTanhReal(a)
  #define PetscAsinhScalar(a)   PetscAsinhReal(a)
  #define PetscAcoshScalar(a)   PetscAcoshReal(a)
  #define PetscAtanhScalar(a)   PetscAtanhReal(a)

#endif /* PETSC_USE_COMPLEX */

/*E
   PetscScalarPrecision - Names of the floating-point precisions that a `PetscScalar` could in principle be built with

   Values:
+   `PETSC_SCALAR_DOUBLE`      - IEEE double precision
.   `PETSC_SCALAR_SINGLE`      - IEEE single precision
.   `PETSC_SCALAR_LONG_DOUBLE` - C `long double` precision
-   `PETSC_SCALAR_HALF`        - IEEE half precision

   Level: developer

   Note:
   This enum is reserved for situations where an object might be created in a precision different from `PetscScalar`; the current PETSc code does not use it.

.seealso: `PetscScalar`, `PetscReal`, `PetscComplex`
E*/
typedef enum {
  PETSC_SCALAR_DOUBLE,
  PETSC_SCALAR_SINGLE,
  PETSC_SCALAR_LONG_DOUBLE,
  PETSC_SCALAR_HALF
} PetscScalarPrecision;

/*MC
   PetscAbs - Returns the absolute value of a number

   Synopsis:
   #include <petscmath.h>
   type PetscAbs(type v)

   Not Collective

   Input Parameter:
.  v - the number

   Level: beginner

   Note:
   The type can be integer or real floating point value, but cannot be complex

.seealso: `PetscAbsInt()`, `PetscAbsReal()`, `PetscAbsScalar()`, `PetscSign()`
M*/
#define PetscAbs(a) (((a) >= 0) ? (a) : (-(a)))

/*MC
   PetscSign - Returns the sign of a number as an integer of value -1, 0, or 1

   Synopsis:
   #include <petscmath.h>
   int PetscSign(type v)

   Not Collective

   Input Parameter:
.  v - the number

   Level: beginner

   Note:
   The type can be integer or real floating point value

.seealso: `PetscAbsInt()`, `PetscAbsReal()`, `PetscAbsScalar()`
M*/
#define PetscSign(a) (((a) >= 0) ? ((a) == 0 ? 0 : 1) : -1)

/*MC
   PetscMin - Returns minimum of two numbers

   Synopsis:
   #include <petscmath.h>
   type PetscMin(type v1,type v2)

   Not Collective

   Input Parameters:
+  v1 - first value to find minimum of
-  v2 - second value to find minimum of

   Level: beginner

   Note:
   The type can be integer or floating point value, but cannot be complex

.seealso: `PetscMax()`, `PetscClipInterval()`, `PetscAbsInt()`, `PetscAbsReal()`, `PetscSqr()`
M*/
#define PetscMin(a, b) (((a) < (b)) ? (a) : (b))

/*MC
   PetscMax - Returns maximum of two numbers

   Synopsis:
   #include <petscmath.h>
   type max PetscMax(type v1,type v2)

   Not Collective

   Input Parameters:
+  v1 - first value to find maximum of
-  v2 - second value to find maximum of

   Level: beginner

   Note:
   The type can be integer or floating point value

.seealso: `PetscMin()`, `PetscClipInterval()`, `PetscAbsInt()`, `PetscAbsReal()`, `PetscSqr()`
M*/
#define PetscMax(a, b) (((a) < (b)) ? (b) : (a))

/*MC
   PetscClipInterval - Returns a number clipped to be within an interval

   Synopsis:
   #include <petscmath.h>
   type clip PetscClipInterval(type x,type a,type b)

   Not Collective

   Input Parameters:
+  x - value to use if within interval [a,b]
.  a - lower end of interval
-  b - upper end of interval

   Level: beginner

   Note:
   The type can be integer or floating point value

   Example\:
.vb
  PetscInt c = PetscClipInterval(5, 2, 3); // the value of c is 3
  PetscInt c = PetscClipInterval(5, 2, 6); // the value of c is 5
.ve

.seealso: `PetscMin()`, `PetscMax()`, `PetscAbsInt()`, `PetscAbsReal()`, `PetscSqr()`
M*/
#define PetscClipInterval(x, a, b) (PetscMax((a), PetscMin((x), (b))))

/*MC
   PetscAbsInt - Returns the absolute value of an integer

   Synopsis:
   #include <petscmath.h>
   int abs PetscAbsInt(int v1)

   Input Parameter:
.   v1 - the integer

   Level: beginner

.seealso: `PetscMax()`, `PetscMin()`, `PetscAbsReal()`, `PetscSqr()`
M*/
#define PetscAbsInt(a) (((a) < 0) ? (-(a)) : (a))

/*MC
   PetscAbsReal - Returns the absolute value of a real number

   Synopsis:
   #include <petscmath.h>
   Real abs PetscAbsReal(PetscReal v1)

   Input Parameter:
.   v1 - the `PetscReal` value

   Level: beginner

.seealso: `PetscReal`, `PetscMax()`, `PetscMin()`, `PetscAbsInt()`, `PetscSqr()`
M*/
#if PetscDefined(USE_REAL_SINGLE)
  #define PetscAbsReal(a) fabsf(a)
#elif PetscDefined(USE_REAL_DOUBLE)
  #define PetscAbsReal(a) fabs(a)
#elif PetscDefined(USE_REAL___FLOAT128)
  #define PetscAbsReal(a) fabsq(a)
#elif PetscDefined(USE_REAL___FP16)
  #define PetscAbsReal(a) fabsf(a)
#endif

/*MC
   PetscSqr - Returns the square of a number

   Synopsis:
   #include <petscmath.h>
   type sqr PetscSqr(type v1)

   Not Collective

   Input Parameter:
.   v1 - the value

   Level: beginner

   Note:
   The type can be integer, floating point, or complex floating point

.seealso: `PetscMax()`, `PetscMin()`, `PetscAbsInt()`, `PetscAbsReal()`
M*/
#define PetscSqr(a) ((a) * (a))

/*MC
   PetscRealConstant - a compile time macro that ensures a given constant real number is properly represented in the configured
   precision of `PetscReal` be it half, single, double or 128-bit representation

   Synopsis:
   #include <petscmath.h>
   PetscReal PetscRealConstant(real_number)

   Not Collective

   Input Parameter:
.   v1 - the real number, for example 1.5

   Level: beginner

   Note:
   For example, if PETSc is configured with `--with-precision=__float128` and one writes
.vb
   PetscReal d = 1.5;
.ve
   the result is 1.5 in double precision extended to 128 bit representation, meaning it is very far from the correct value. Hence, one should write
.vb
   PetscReal d = PetscRealConstant(1.5);
.ve

.seealso: `PetscReal`
M*/
#if PetscDefined(USE_REAL_SINGLE)
  #define PetscRealConstant(constant) constant##F
#elif PetscDefined(USE_REAL_DOUBLE)
  #define PetscRealConstant(constant) constant
#elif PetscDefined(USE_REAL___FLOAT128)
  #define PetscRealConstant(constant) constant##Q
#elif PetscDefined(USE_REAL___FP16)
  #define PetscRealConstant(constant) constant##F
#endif

/*
     Basic constants
*/
/*MC
  PETSC_PI - the value of $ \pi$ to the correct precision of `PetscReal`.

  Level: beginner

.seealso: `PetscReal`, `PETSC_PHI`, `PETSC_SQRT2`
M*/

/*MC
  PETSC_PHI - the value of $ \phi$, the Golden Ratio, to the correct precision of `PetscReal`.

  Level: beginner

.seealso: `PetscReal`, `PETSC_PI`, `PETSC_SQRT2`
M*/

/*MC
  PETSC_SQRT2 - the value of $ \sqrt{2} $ to the correct precision of `PetscReal`.

  Level: beginner

.seealso: `PetscReal`, `PETSC_PI`, `PETSC_PHI`
M*/

/*MC
  PETSC_E - the value of Euler's constant $ e $ to the correct precision of `PetscReal`.

  Level: beginner

.seealso: `PetscReal`, `PETSC_PI`, `PETSC_PHI`
M*/

#define PETSC_PI    PetscRealConstant(3.1415926535897932384626433832795029)
#define PETSC_PHI   PetscRealConstant(1.6180339887498948482045868343656381)
#define PETSC_SQRT2 PetscRealConstant(1.4142135623730950488016887242096981)
#define PETSC_E     PetscRealConstant(2.7182818284590452353602874713526625)

/*MC
  PETSC_MAX_REAL - the largest real value that can be stored in a `PetscReal`

  Level: beginner

.seealso: `PETSC_MIN_REAL`, `PETSC_REAL_MIN`, `PETSC_MACHINE_EPSILON`, `PETSC_SQRT_MACHINE_EPSILON`, `PETSC_SMALL`
M*/

/*MC
  PETSC_MIN_REAL - the smallest real value that can be stored in a `PetscReal`, generally this is - `PETSC_MAX_REAL`

  Level: beginner

.seealso `PETSC_MAX_REAL`, `PETSC_REAL_MIN`, `PETSC_MACHINE_EPSILON`, `PETSC_SQRT_MACHINE_EPSILON`, `PETSC_SMALL`
M*/

/*MC
  PETSC_REAL_MIN - the smallest positive normalized real value that can be stored in a `PetscReal`.

  Level: beginner

  Note:
  See <https://en.wikipedia.org/wiki/Subnormal_number> for a discussion of normalized and subnormal floating point numbers

  Developer Note:
  The naming is confusing as there is both a `PETSC_REAL_MIN` and `PETSC_MIN_REAL` with different meanings.

.seealso `PETSC_MAX_REAL`, `PETSC_MIN_REAL`, `PETSC_MACHINE_EPSILON`, `PETSC_SQRT_MACHINE_EPSILON`, `PETSC_SMALL`
M*/

/*MC
  PETSC_MACHINE_EPSILON - the machine epsilon for the precision of `PetscReal`

  Level: beginner

  Note:
  See <https://en.wikipedia.org/wiki/Machine_epsilon>

.seealso `PETSC_MAX_REAL`, `PETSC_MIN_REAL`, `PETSC_REAL_MIN`, `PETSC_SQRT_MACHINE_EPSILON`, `PETSC_SMALL`
M*/

/*MC
  PETSC_SQRT_MACHINE_EPSILON - the square root of the machine epsilon for the precision of `PetscReal`

  Level: beginner

  Note:
  See `PETSC_MACHINE_EPSILON`

.seealso `PETSC_MAX_REAL`, `PETSC_MIN_REAL`, `PETSC_REAL_MIN`, `PETSC_MACHINE_EPSILON`, `PETSC_SMALL`
M*/

/*MC
  PETSC_SMALL - an arbitrary "small" number which depends on the precision of `PetscReal` used in some PETSc examples
  and in `PetscApproximateLTE()` and `PetscApproximateGTE()` to determine if a computation was successful.

  Level: beginner

  Note:
  See `PETSC_MACHINE_EPSILON`

.seealso `PetscApproximateLTE()`, `PetscApproximateGTE()`, `PETSC_MAX_REAL`, `PETSC_MIN_REAL`, `PETSC_REAL_MIN`, `PETSC_MACHINE_EPSILON`,
         `PETSC_SQRT_MACHINE_EPSILON`
M*/

#if PetscDefined(USE_REAL_SINGLE)
  #define PETSC_MAX_REAL             3.40282346638528860e+38F
  #define PETSC_MIN_REAL             (-PETSC_MAX_REAL)
  #define PETSC_REAL_MIN             1.1754944e-38F
  #define PETSC_MACHINE_EPSILON      1.19209290e-07F
  #define PETSC_SQRT_MACHINE_EPSILON 3.45266983e-04F
  #define PETSC_SMALL                1.e-5F
#elif PetscDefined(USE_REAL_DOUBLE)
  #define PETSC_MAX_REAL             1.7976931348623157e+308
  #define PETSC_MIN_REAL             (-PETSC_MAX_REAL)
  #define PETSC_REAL_MIN             2.225073858507201e-308
  #define PETSC_MACHINE_EPSILON      2.2204460492503131e-16
  #define PETSC_SQRT_MACHINE_EPSILON 1.490116119384766e-08
  #define PETSC_SMALL                1.e-10
#elif PetscDefined(USE_REAL___FLOAT128)
  #define PETSC_MAX_REAL             FLT128_MAX
  #define PETSC_MIN_REAL             (-FLT128_MAX)
  #define PETSC_REAL_MIN             FLT128_MIN
  #define PETSC_MACHINE_EPSILON      FLT128_EPSILON
  #define PETSC_SQRT_MACHINE_EPSILON 1.38777878078144567552953958511352539e-17Q
  #define PETSC_SMALL                1.e-20Q
#elif PetscDefined(USE_REAL___FP16)
  #define PETSC_MAX_REAL             65504.0F
  #define PETSC_MIN_REAL             (-PETSC_MAX_REAL)
  #define PETSC_REAL_MIN             .00006103515625F
  #define PETSC_MACHINE_EPSILON      .0009765625F
  #define PETSC_SQRT_MACHINE_EPSILON .03125F
  #define PETSC_SMALL                5.e-3F
#endif

/*MC
  PETSC_INFINITY - a finite number that represents infinity for setting certain bounds in `Tao`

  Level: intermediate

  Note:
  This is not the IEEE infinity value

.seealso: `PETSC_NINFINITY`, `SNESVIGetVariableBounds()`, `SNESVISetComputeVariableBounds()`, `SNESVISetVariableBounds()`
M*/
#define PETSC_INFINITY (PETSC_MAX_REAL / 4)

/*MC
  PETSC_NINFINITY - a finite number that represents negative infinity for setting certain bounds in `Tao`

  Level: intermediate

  Note:
  This is not the negative IEEE infinity value

.seealso: `PETSC_INFINITY`, `SNESVIGetVariableBounds()`, `SNESVISetComputeVariableBounds()`, `SNESVISetVariableBounds()`
M*/
#define PETSC_NINFINITY (-PETSC_INFINITY)

PETSC_EXTERN PetscBool  PetscIsInfReal(PetscReal);
PETSC_EXTERN PetscBool  PetscIsNanReal(PetscReal);
PETSC_EXTERN PetscBool  PetscIsNormalReal(PetscReal);
static inline PetscBool PetscIsInfOrNanReal(PetscReal v)
{
  return PetscIsInfReal(v) || PetscIsNanReal(v) ? PETSC_TRUE : PETSC_FALSE;
}
static inline PetscBool PetscIsInfScalar(PetscScalar v)
{
  return PetscIsInfReal(PetscAbsScalar(v));
}
static inline PetscBool PetscIsNanScalar(PetscScalar v)
{
  return PetscIsNanReal(PetscAbsScalar(v));
}
static inline PetscBool PetscIsInfOrNanScalar(PetscScalar v)
{
  return PetscIsInfOrNanReal(PetscAbsScalar(v));
}
static inline PetscBool PetscIsNormalScalar(PetscScalar v)
{
  return PetscIsNormalReal(PetscAbsScalar(v));
}

PETSC_EXTERN PetscBool PetscIsCloseAtTol(PetscReal, PetscReal, PetscReal, PetscReal);
PETSC_EXTERN PetscBool PetscEqualReal(PetscReal, PetscReal);
PETSC_EXTERN PetscBool PetscEqualScalar(PetscScalar, PetscScalar);

/*@C
  PetscIsCloseAtTolScalar - Like `PetscIsCloseAtTol()` but for `PetscScalar`

  Input Parameters:
+ lhs  - The first number
. rhs  - The second number
. rtol - The relative tolerance
- atol - The absolute tolerance

  Level: beginner

  Note:
  This routine is equivalent to `PetscIsCloseAtTol()` when PETSc is configured without complex
  numbers.

.seealso: `PetscIsCloseAtTol()`
@*/
static inline PetscBool PetscIsCloseAtTolScalar(PetscScalar lhs, PetscScalar rhs, PetscReal rtol, PetscReal atol)
{
  PetscBool close = PetscIsCloseAtTol(PetscRealPart(lhs), PetscRealPart(rhs), rtol, atol);

  if (PetscDefined(USE_COMPLEX)) close = (PetscBool)(close && PetscIsCloseAtTol(PetscImaginaryPart(lhs), PetscImaginaryPart(rhs), rtol, atol));
  return close;
}

/*
    These macros are currently hardwired to match the regular data types, so there is no support for a different
    MatScalar from PetscScalar. We left the MatScalar in the source just in case we use it again.
 */
#define MPIU_MATSCALAR MPIU_SCALAR

/*MC
   MatScalar - Datatype used inside matrix implementations to store matrix entries

   Level: developer

   Note:
   `MatScalar` is currently a synonym for `PetscScalar`. It exists so that matrix internals can, in principle, store entries in a different precision from the rest of PETSc. No PETSc matrix implementation currently exploits this.

.seealso: `PetscScalar`, `MatReal`, `Mat`, `PETSC_USE_COMPLEX`
M*/
typedef PetscScalar MatScalar;

/*MC
   MatReal - Datatype used inside matrix implementations to store real-valued matrix data

   Level: developer

   Note:
   `MatReal` is currently a synonym for `PetscReal`. It exists so that matrix internals can, in principle, store data in a different precision from the rest of PETSc. No PETSc matrix implementation currently exploits this.

.seealso: `PetscReal`, `MatScalar`, `Mat`
M*/
typedef PetscReal MatReal;

struct petsc_mpiu_2scalar {
  PetscScalar a, b;
};
PETSC_EXTERN MPI_Datatype MPIU_2SCALAR PETSC_ATTRIBUTE_MPI_TYPE_TAG_LAYOUT_COMPATIBLE(struct petsc_mpiu_2scalar);

/* MPI Datatypes for composite reductions */
struct petsc_mpiu_real_int {
  PetscReal v;
  PetscInt  i;
};

struct petsc_mpiu_scalar_int {
  PetscScalar v;
  PetscInt    i;
};

PETSC_EXTERN MPI_Datatype MPIU_REAL_INT PETSC_ATTRIBUTE_MPI_TYPE_TAG_LAYOUT_COMPATIBLE(struct petsc_mpiu_real_int);
PETSC_EXTERN MPI_Datatype MPIU_SCALAR_INT PETSC_ATTRIBUTE_MPI_TYPE_TAG_LAYOUT_COMPATIBLE(struct petsc_mpiu_scalar_int);

#if PetscDefined(USE_64BIT_INDICES)
struct /* __attribute__((packed, aligned(alignof(PetscInt *)))) */ petsc_mpiu_2int {
  PetscInt a;
  PetscInt b;
};
struct __attribute__((packed)) petsc_mpiu_int_mpiint {
  PetscInt    a;
  PetscMPIInt b;
};
/*
 static_assert(sizeof(struct petsc_mpiu_2int) == 2 * sizeof(PetscInt), "");
 static_assert(alignof(struct petsc_mpiu_2int) == alignof(PetscInt *), "");
 static_assert(alignof(struct petsc_mpiu_2int) == alignof(PetscInt[2]), "");

 clang generates warnings that petsc_mpiu_2int is not layout compatible with PetscInt[2] or
 PetscInt *, even though (with everything else uncommented) both of the static_asserts above
 pass! So we just comment it out...
*/
PETSC_EXTERN MPI_Datatype MPIU_2INT /* PETSC_ATTRIBUTE_MPI_TYPE_TAG_LAYOUT_COMPATIBLE(struct petsc_mpiu_2int) */;
PETSC_EXTERN MPI_Datatype MPIU_INT_MPIINT /* PETSC_ATTRIBUTE_MPI_TYPE_TAG_LAYOUT_COMPATIBLE(struct petsc_mpiu_int_mpiint) */;
#else
  #define MPIU_2INT       MPI_2INT
  #define MPIU_INT_MPIINT MPI_2INT
#endif
PETSC_EXTERN MPI_Datatype MPI_4INT;
PETSC_EXTERN MPI_Datatype MPIU_4INT;

static inline PetscInt PetscPowInt(PetscInt base, PetscInt power)
{
  PetscInt result = 1;
  while (power) {
    if (power & 1) result *= base;
    power >>= 1;
    if (power) base *= base;
  }
  return result;
}

static inline PetscInt64 PetscPowInt64(PetscInt base, PetscInt power)
{
  PetscInt64 result = 1;
  while (power) {
    if (power & 1) result *= base;
    power >>= 1;
    if (power) base *= base;
  }
  return result;
}

static inline PetscReal PetscPowRealInt(PetscReal base, PetscInt power)
{
  PetscReal result = 1;
  if (power < 0) {
    power = -power;
    base  = ((PetscReal)1) / base;
  }
  while (power) {
    if (power & 1) result *= base;
    power >>= 1;
    if (power) base *= base;
  }
  return result;
}

static inline PetscScalar PetscPowScalarInt(PetscScalar base, PetscInt power)
{
  PetscScalar result = (PetscReal)1;
  if (power < 0) {
    power = -power;
    base  = ((PetscReal)1) / base;
  }
  while (power) {
    if (power & 1) result *= base;
    power >>= 1;
    if (power) base *= base;
  }
  return result;
}

static inline PetscScalar PetscPowScalarReal(PetscScalar base, PetscReal power)
{
  PetscScalar cpower = power;
  return PetscPowScalar(base, cpower);
}

/*MC
   PetscApproximateLTE - Performs a less than or equal to on a given constant with a fudge for floating point numbers

   Synopsis:
   #include <petscmath.h>
   bool PetscApproximateLTE(PetscReal x,constant float)

   Not Collective

   Input Parameters:
+   x - the variable
-   b - the constant float it is checking if `x` is less than or equal to

   Level: advanced

   Notes:
   The fudge factor is the value `PETSC_SMALL`

   The constant numerical value is automatically set to the appropriate precision of PETSc so can just be provided as, for example, 3.2

   This is used in several examples for setting initial conditions based on coordinate values that are computed with i*h that produces inexact
   floating point results.

   Example\:
.vb
  PetscReal x;
  if (PetscApproximateLTE(x, 3.2)) { // replaces if (x <= 3.2) {
.ve

.seealso: `PetscMax()`, `PetscMin()`, `PetscAbsInt()`, `PetscAbsReal()`, `PetscApproximateGTE()`
M*/
#define PetscApproximateLTE(x, b) ((x) <= (PetscRealConstant(b) + PETSC_SMALL))

/*MC
   PetscApproximateGTE - Performs a greater than or equal to on a given constant with a fudge for floating point numbers

   Synopsis:
   #include <petscmath.h>
   bool PetscApproximateGTE(PetscReal x,constant float)

   Not Collective

   Input Parameters:
+   x - the variable
-   b - the constant float it is checking if `x` is greater than or equal to

   Level: advanced

   Notes:
   The fudge factor is the value `PETSC_SMALL`

   The constant numerical value is automatically set to the appropriate precision of PETSc so can just be provided as, for example, 3.2

   This is used in several examples for setting initial conditions based on coordinate values that are computed with i*h that produces inexact
   floating point results.

   Example\:
.vb
  PetscReal x;
  if (PetscApproximateGTE(x, 3.2)) {  // replaces if (x >= 3.2) {
.ve

.seealso: `PetscMax()`, `PetscMin()`, `PetscAbsInt()`, `PetscAbsReal()`, `PetscApproximateLTE()`
M*/
#define PetscApproximateGTE(x, b) ((x) >= (PetscRealConstant(b) - PETSC_SMALL))

/*@C
   PetscCeilInt - Returns the ceiling of the quotation of two positive integers

   Not Collective

   Input Parameters:
+   x - the numerator
-   y - the denominator

   Level: advanced

  Example\:
.vb
  PetscInt n = PetscCeilInt(10, 3); // n has the value of 4
.ve

.seealso: `PetscCeilInt64()`, `PetscMax()`, `PetscMin()`, `PetscAbsInt()`, `PetscAbsReal()`, `PetscApproximateLTE()`
@*/
static inline PetscInt PetscCeilInt(PetscInt x, PetscInt y)
{
  return x / y + (x % y ? 1 : 0);
}

/*@C
   PetscCeilInt64 - Returns the ceiling of the quotation of two positive integers

   Not Collective

   Input Parameters:
+   x - the numerator
-   y - the denominator

   Level: advanced

  Example\:
.vb
  PetscInt64 n = PetscCeilInt64(10, 3); // n has the value of 4
.ve

.seealso: `PetscCeilInt()`, `PetscMax()`, `PetscMin()`, `PetscAbsInt()`, `PetscAbsReal()`, `PetscApproximateLTE()`
@*/
static inline PetscInt64 PetscCeilInt64(PetscInt64 x, PetscInt64 y)
{
  return x / y + (x % y ? 1 : 0);
}

/*@C
   PetscGCD - Returns the greatest common divisor of two integers

   Not Collective

   Input Parameters:
+   a - first number
-   b - second number

   Level: advanced

.seealso: `PetscLCM()`
@*/
static inline PetscInt PetscGCD(PetscInt a, PetscInt b)
{
  a = PetscAbsInt(a);
  b = PetscAbsInt(b);
  while (b != 0) {
    PetscInt tmp = b;

    b = a % b;
    a = tmp;
  }
  return a;
}

/*@C
   PetscLCM - Returns the least common multiple of two integers

   Not Collective

   Input Parameters:
+   a - first number
-   b - second number

   Level: advanced

.seealso: `PetscGCD()`
@*/
static inline PetscInt PetscLCM(PetscInt a, PetscInt b)
{
  PetscInt gcd;

  a = PetscAbsInt(a);
  b = PetscAbsInt(b);

  gcd = PetscGCD(a, b);
  return gcd ? a * (b / gcd) : 0;
}

PETSC_EXTERN PetscErrorCode PetscLinearRegression(PetscInt, const PetscReal[], const PetscReal[], PetscReal *, PetscReal *);
