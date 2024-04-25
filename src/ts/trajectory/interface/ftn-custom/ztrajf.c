#include <petsc/private/fortranimpl.h>
#include <petscts.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
  #define tstrajectoryviewfromoptions_ TSTRAJECTORYVIEWFROMOPTIONS
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
  #define tstrajectoryviewfromoptions_ tstrajectoryviewfromoptions
#endif

PETSC_EXTERN void tstrajectoryviewfromoptions_(TSTrajectory *ao, PetscObject obj, char *type, PetscErrorCode *ierr, PETSC_FORTRAN_CHARLEN_T len)
{
  char *t;

  FIXCHAR(type, len, t);
  CHKFORTRANNULLOBJECT(obj);
  *ierr = TSTrajectoryViewFromOptions(*ao, obj, t);
  if (*ierr) return;
  FREECHAR(type, t);
}
