#include <petsc/private/fortranimpl.h>
#include <petscts.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
  #define tstrajectorysetdirname_      TSTRAJECTORYSETDIRNAME
  #define tstrajectorysetfiletemplate_ TSTRAJECTORYSETFILETEMPLATE
  #define tstrajectoryviewfromoptions_ TSTRAJECTORYVIEWFROMOPTIONS
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
  #define tstrajectorysetdirname_      tstrajectorysetdirname
  #define tstrajectorysetfiletemplate_ tstrajectorysetfiletemplate
  #define tstrajectoryviewfromoptions_ tstrajectoryviewfromoptions
#endif

PETSC_EXTERN void tstrajectorysetdirname_(TSTrajectory *tj, char dirname[], int *ierr, PETSC_FORTRAN_CHARLEN_T len)
{
  char *t;
  FIXCHAR(dirname, len, t);
  *ierr = TSTrajectorySetDirname(*tj, t);
  if (*ierr) return;
  FREECHAR(dirname, t);
}

PETSC_EXTERN void tstrajectorysetfiletemplate_(TSTrajectory *tj, char filetemplate[], int *ierr, PETSC_FORTRAN_CHARLEN_T len)
{
  char *t;
  FIXCHAR(filetemplate, len, t);
  *ierr = TSTrajectorySetFiletemplate(*tj, t);
  FREECHAR(filetemplate, t);
}
PETSC_EXTERN void tstrajectoryviewfromoptions_(TSTrajectory *ao, PetscObject obj, char *type, PetscErrorCode *ierr, PETSC_FORTRAN_CHARLEN_T len)
{
  char *t;

  FIXCHAR(type, len, t);
  CHKFORTRANNULLOBJECT(obj);
  *ierr = TSTrajectoryViewFromOptions(*ao, obj, t);
  if (*ierr) return;
  FREECHAR(type, t);
}
