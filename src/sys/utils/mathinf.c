#if !defined(PETSC_SKIP_COMPLEX)
  #define PETSC_SKIP_COMPLEX
#endif

#include <petscsys.h>
/*@C
      PetscIsNormalReal - Returns `PETSC_TRUE` if the input value satisfies `isnormal()`

    Input Parameter:
.     a - the `PetscReal` Value

     Level: beginner

    Developer Notes:
    Uses the C99 standard `isnormal()` on systems where they exist.

    Uses `isnormalq()` with `__float128`

    Otherwise always returns true

.seealso: `PetscIsInfReal()`, `PetscIsNanReal()`
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
      PetscIsInfReal - Returns whether the `PetscReal` input is an infinity value.

    Input Parameter:
.     a - the floating point number

     Level: beginner

    Developer Notes:
    Uses the C99 standard `isinf()` on systems where it exists.

    Otherwise uses (a && a/2 == a), note that some optimizing compilers compile out this form, thus removing the check.

.seealso: `PetscIsNormalReal()`, `PetscIsNanReal()`
@*/
#if defined(PETSC_USE_REAL___FLOAT128)
PetscBool PetscIsInfReal(PetscReal a)
{
  return isinfq(a) ? PETSC_TRUE : PETSC_FALSE;
}
#elif defined(PETSC_HAVE_ISINF)
PetscBool PetscIsInfReal(PetscReal a)
{
  return isinf(a) ? PETSC_TRUE : PETSC_FALSE;
}
#elif defined(PETSC_HAVE__FINITE)
  #if defined(PETSC_HAVE_FLOAT_H)
    #include <float.h> /* Microsoft Windows defines _finite() in float.h */
  #endif
  #if defined(PETSC_HAVE_IEEEFP_H)
    #include <ieeefp.h> /* Solaris prototypes these here */
  #endif
PetscBool PetscIsInfReal(PetscReal a)
{
  return !_finite(a) ? PETSC_TRUE : PETSC_FALSE;
}
#else
PetscBool PetscIsInfReal(PetscReal a)
{
  return (a && a / 2 == a) ? PETSC_TRUE : PETSC_FALSE;
}
#endif

/*@C
      PetscIsNanReal - Returns whether the `PetscReal` input is a Not-a-Number (NaN) value.

    Input Parameter:
.     a - the floating point number

     Level: beginner

    Developer Notes:
    Uses the C99 standard `isnan()` on systems where it exists.

    Otherwise uses (a != a), note that some optimizing compilers compile
    out this form, thus removing the check.

.seealso: `PetscIsNormalReal()`, `PetscIsInfReal()`
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
    #include <float.h> /* Microsoft Windows defines _isnan() in float.h */
  #endif
  #if defined(PETSC_HAVE_IEEEFP_H)
    #include <ieeefp.h> /* Solaris prototypes these here */
  #endif
PetscBool PetscIsNanReal(PetscReal a)
{
  return _isnan(a) ? PETSC_TRUE : PETSC_FALSE;
}
#else
PetscBool PetscIsNanReal(PetscReal a)
{
  return (a != a) ? PETSC_TRUE : PETSC_FALSE;
}
#endif
