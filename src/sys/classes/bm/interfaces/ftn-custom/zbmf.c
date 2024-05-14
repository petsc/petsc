#include <petsc/private/fortranimpl.h>
#include <petscbm.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
  #define petscbmviewfromoptions_ PETSCBMVIEWFROMOPTIONS
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
  #define petscbmviewfromoptions_ petscbmviewfromoptions
#endif

PETSC_EXTERN void petscbmviewfromoptions_(PetscBench *bm, PetscObject obj, char *type, PetscErrorCode *ierr, PETSC_FORTRAN_CHARLEN_T len)
{
  char *t;

  FIXCHAR(type, len, t);
  CHKFORTRANNULLOBJECT(obj);
  *ierr = PetscBenchViewFromOptions(*bm, obj, t);
  if (*ierr) return;
  FREECHAR(type, t);
}
