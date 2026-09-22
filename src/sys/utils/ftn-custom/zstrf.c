#include <petsc/private/ftnimpl.h>

#if PetscDefined(HAVE_FORTRAN_CAPS)
  #define petscstrncpy_ PETSCSTRNCPY
#elif !PetscDefined(HAVE_FORTRAN_UNDERSCORE)
  #define petscstrncpy_ petscstrncpy
#endif

PETSC_EXTERN void petscstrncpy_(char *s, char *t, int *n, PetscErrorCode *ierr, PETSC_FORTRAN_CHARLEN_T len1, PETSC_FORTRAN_CHARLEN_T len2)
{
  char                   *t1, *t2;
  PETSC_FORTRAN_CHARLEN_T m;

  t1 = s;
  t2 = t;
  m  = (PETSC_FORTRAN_CHARLEN_T)*n;
  if (len1 < m) m = len1;
  if (len2 < m) m = len2;
  *ierr = PetscStrncpy(t1, t2, m);
}
