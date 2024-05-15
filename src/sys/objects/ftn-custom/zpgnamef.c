#include <petsc/private/fortranimpl.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
  #define petscgetversion_ PETSCGETVERSION
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
  #define petscgetversion_ petscgetversion
#endif

PETSC_EXTERN void petscgetversion_(char *version, int *ierr, PETSC_FORTRAN_CHARLEN_T len1)
{
  *ierr = PetscGetVersion(version, len1);
  FIXRETURNCHAR(PETSC_TRUE, version, len1);
}
