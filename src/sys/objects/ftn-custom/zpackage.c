#include <petsc/private/fortranimpl.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
  #define petschasexternalpackage_ PETSCHASEXTERNALPACKAGE
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
  #define petschasexternalpackage_ petschasexternalpackage
#endif

PETSC_EXTERN void petschasexternalpackage_(char *pkg, PetscBool *has, PetscErrorCode *ierr, PETSC_FORTRAN_CHARLEN_T len)
{
  char *t1;

  FIXCHAR(pkg, len, t1);
  *ierr = PetscHasExternalPackage(t1, has);
  if (*ierr) return;
  FREECHAR(pkg, t1);
}
