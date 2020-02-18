#include <petsc/private/fortranimpl.h>
#include <petscts.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define tstrajectorysetdirname_ TSTRAJECTORYSETDIRNAME
#define tstrajectorysetfiletemplate_ TSTRAJECTORYSETFILETEMPLATE
#define tstrajectoryviewfromoptions_ TSTRAJECTORYVIEWFROMOPTIONS
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define tstrajectorysetdirname_ tstrajectorysetdirname
#define tstrajectorysetfiletemplate_ tstrajectorysetfiletemplate
#define tstrajectoryviewfromoptions_ tstrajectoryviewfromoptions
#endif

PETSC_EXTERN void PETSC_STDCALL tstrajectorysetdirname_(TSTrajectory *tj,char dirname[] PETSC_MIXED_LEN(len),int *ierr PETSC_END_LEN(len))
{
  char *t;
  FIXCHAR(dirname,len,t);
  *ierr = TSTrajectorySetDirname(*tj,t);if (*ierr) return;
  FREECHAR(dirname,t);
}

PETSC_EXTERN void PETSC_STDCALL tstrajectorysetfiletemplate_(TSTrajectory *tj,char filetemplate[] PETSC_MIXED_LEN(len),int *ierr PETSC_END_LEN(len))
{
  char *t;
  FIXCHAR(filetemplate,len,t);
  *ierr = TSTrajectorySetFiletemplate(*tj,t);
  FREECHAR(filetemplate,t);
}
PETSC_EXTERN void PETSC_STDCALL tstrajectoryviewfromoptions_(TSTrajectory *ao,PetscObject obj,char* type PETSC_MIXED_LEN(len),PetscErrorCode *ierr PETSC_END_LEN(len))
{
  char *t;

  FIXCHAR(type,len,t);
  *ierr = TSTrajectoryViewFromOptions(*ao,obj,t);if (*ierr) return;
  FREECHAR(type,t);
}
