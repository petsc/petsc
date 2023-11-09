#include <petsc/private/fortranimpl.h>
#include <petscmatcoarsen.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
  #define matcoarsenviewfromoptions_ MATCOARSENVIEWFROMOPTIONS
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
  #define matcoarsenviewfromoptions_ matcoarsenviewfromoptions
#endif

PETSC_EXTERN void matcoarsenviewfromoptions_(MatCoarsen *a, PetscObject obj, char *type, PetscErrorCode *ierr, PETSC_FORTRAN_CHARLEN_T len)
{
  char *t;

  FIXCHAR(type, len, t);
  CHKFORTRANNULLOBJECT(obj);
  *ierr = MatCoarsenViewFromOptions(*a, obj, t);
  if (*ierr) return;
  FREECHAR(type, t);
}
