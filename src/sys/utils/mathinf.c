#define PETSC_SKIP_COMPLEX
#include <petscsys.h>
/*@C
      PetscIsNormalReal - Returns PETSC_TRUE if the input value satisfies isnormal()

    Input Parameter:
.     a - the PetscReal Value

     Notes: uses the C99 standard isnormal() on systems where they exist.
      Uses isnormalq() with __float128
      Otherwises always returns true

     Level: beginner
@*/
#if defined(PETSC_USE_REAL___FLOAT128) || defined(PETSC_USE_REAL___FP16)
PetscBool PetscIsNormalReal(PetscReal a)
{
  return PETSC_TRUE;
}
#elif defined(PETSC_HAVE_ISNORMAL)
PetscBool PetscIsNormalReal(PetscReal a)
{
  return isnormal(a) ? PETSC_TRUE : PETSC_FALSE;
}
#else
PetscBool PetscIsNormalReal(PetscReal a)
{
  return PETSC_TRUE;
}
#endif

/*@C
      PetscIsInfOrNanReal - Returns whether the input is an infinity or Not-a-Number (NaN) value.

    Input Parameter:
.     a - the floating point number

     Notes: uses the C99 standard isinf() and isnan() on systems where they exist.
      Otherwises uses ((a - a) != 0.0), note that some optimizing compiles compile
      out this form, thus removing the check.

     Level: beginner
@*/
#if defined(PETSC_USE_REAL___FLOAT128)
PetscBool PetscIsInfOrNanReal(PetscReal a)
{
  return isinfq(a) || isnanq(a) ? PETSC_TRUE : PETSC_FALSE;
}
#elif defined(PETSC_HAVE_ISINF) && defined(PETSC_HAVE_ISNAN)
PetscBool PetscIsInfOrNanReal(PetscReal a)
{
  return isinf(a) || isnan(a) ? PETSC_TRUE : PETSC_FALSE;
}
#elif defined(PETSC_HAVE__FINITE) && defined(PETSC_HAVE__ISNAN)
#if defined(PETSC_HAVE_FLOAT_H)
#include <float.h>  /* Microsoft Windows defines _finite() in float.h */
#endif
#if defined(PETSC_HAVE_IEEEFP_H)
#include <ieeefp.h>  /* Solaris prototypes these here */
#endif
PetscBool PetscIsInfOrNanReal(PetscReal a)
{
  return !_finite(a) || _isnan(a) ? PETSC_TRUE : PETSC_FALSE;
}
#else
PetscBool PetscIsInfOrNanReal(PetscReal a)
{
  return ((a - a) != 0) ? PETSC_TRUE : PETSC_FALSE;
}
#endif

/*@C
      PetscIsNanReal - Returns whether the input is a Not-a-Number (NaN) value.

    Input Parameter:
.     a - the floating point number

     Notes: uses the C99 standard isnan() on systems where it exists.
      Otherwises uses (a != a), note that some optimizing compiles compile
      out this form, thus removing the check.

     Level: beginner
@*/
#if defined(PETSC_USE_REAL___FLOAT128)
PetscBool PetscIsNanReal(PetscReal a)
{
  return isnanq(a) ? PETSC_TRUE : PETSC_FALSE;
}
#elif defined(PETSC_HAVE_ISNAN)
PetscBool PetscIsNanReal(PetscReal a)
{
  return isnan(a) ? PETSC_TRUE : PETSC_FALSE;
}
#elif defined(PETSC_HAVE__ISNAN)
#if defined(PETSC_HAVE_FLOAT_H)
#include <float.h>  /* Microsoft Windows defines _isnan() in float.h */
#endif
#if defined(PETSC_HAVE_IEEEFP_H)
#include <ieeefp.h>  /* Solaris prototypes these here */
#endif
PetscBool PetscIsNanReal(PetscReal a)
{
  return _isnan(a) ? PETSC_TRUE : PETSC_FALSE;
}
#else
PetscBool PetscIsNanReal(volatile PetscReal a)
{
  return (a != a) ? PETSC_TRUE : PETSC_FALSE;
}
#endif
