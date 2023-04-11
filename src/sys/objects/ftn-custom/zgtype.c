#include <petsc/private/fortranimpl.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
  #define petscobjectgettype_ PETSCOBJECTGETTYPE
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
  #define petscobjectgettype_ petscobjectgettype
#endif

PETSC_EXTERN void petscobjectgettype_(PetscObject *obj, char *type, PetscErrorCode *ierr, PETSC_FORTRAN_CHARLEN_T len)
{
  const char *tmp;
  *ierr = PetscObjectGetType(*obj, &tmp);
  if (*ierr) return;
  *ierr = PetscStrncpy(type, tmp, len);
  if (*ierr) return;
  FIXRETURNCHAR(PETSC_TRUE, type, len);
}
