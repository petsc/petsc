#include <petscsys.h>

/*@C
    PetscIsCloseAtTol - Returns whether the two floating point numbers
       are close at a given relative and absolute tolerances.

    Input Parameters:
+     a - first floating point number
.     b - second floating point number
.     rtol - relative tolerance
-     atol - absolute tolerances

    Reference:
.   * -  https://www.python.org/dev/peps/pep-0485/

    Level: beginner

.seealso: `PetscEqualReal()`, `PetscEqualScalar()`
@*/
PetscBool PetscIsCloseAtTol(PetscReal a, PetscReal b, PetscReal rtol, PetscReal atol)
{
  PetscReal diff;
  /* NaN is not considered close to any other value, including NaN */
  if (PetscIsNanReal(a) || PetscIsNanReal(b)) return PETSC_FALSE;
  /* Fast path for exact equality or two infinities of same sign */
  if (a == b) return PETSC_TRUE;
  /* Handle two infinities of opposite sign */
  if (PetscIsInfReal(a) || PetscIsInfReal(b)) return PETSC_FALSE;
  /* Cannot error if tolerances are negative */
  rtol = PetscAbsReal(rtol);
  atol = PetscAbsReal(atol);
  /* The regular check for difference within tolerances */
  diff = PetscAbsReal(b - a);
  return ((diff <= PetscAbsReal(rtol * b)) || (diff <= PetscAbsReal(rtol * a)) || (diff <= atol)) ? PETSC_TRUE : PETSC_FALSE;
}
