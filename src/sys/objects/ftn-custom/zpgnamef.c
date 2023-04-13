#include <petsc/private/fortranimpl.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
  #define petscobjectgetname_ PETSCOBJECTGETNAME
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
  #define petscobjectgetname_ petscobjectgetname
#endif

PETSC_EXTERN void petscobjectgetname_(PetscObject *obj, char *name, PetscErrorCode *ierr, PETSC_FORTRAN_CHARLEN_T len)
{
  const char *tmp;
  *ierr = PetscObjectGetName(*obj, &tmp);
  *ierr = PetscStrncpy(name, tmp, len);
  if (*ierr) return;
  FIXRETURNCHAR(PETSC_TRUE, name, len);
}
