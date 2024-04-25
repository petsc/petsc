#include <petsc/private/fortranimpl.h>
#include <petscfv.h>
#include <petscviewer.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
  #define petscfvview_                 PETSCFVVIEW
  #define petscfvviewfromoptions_      PETSCFVVIEWFROMOPTIONS
  #define petsclimiterviewfromoptions_ PETSCLIMITERVIEWFROMOPTIONS
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
  #define petscfvview_                 petscfvview
  #define petscfvviewfromoptions_      petscfvviewfromoptions
  #define petsclimiterviewfromoptions_ petsclimiterviewfromoptions
#endif

PETSC_EXTERN void petscfvview_(PetscFV *fvm, PetscViewer *vin, PetscErrorCode *ierr)
{
  PetscViewer v;
  PetscPatchDefaultViewers_Fortran(vin, v);
  *ierr = PetscFVView(*fvm, v);
}

PETSC_EXTERN void petscfvviewfromoptions_(PetscFV *ao, PetscObject obj, char *type, PetscErrorCode *ierr, PETSC_FORTRAN_CHARLEN_T len)
{
  char *t;

  FIXCHAR(type, len, t);
  CHKFORTRANNULLOBJECT(obj);
  *ierr = PetscFVViewFromOptions(*ao, obj, t);
  if (*ierr) return;
  FREECHAR(type, t);
}

PETSC_EXTERN void petsclimiterviewfromoptions_(PetscLimiter *ao, PetscObject obj, char *type, PetscErrorCode *ierr, PETSC_FORTRAN_CHARLEN_T len)
{
  char *t;

  FIXCHAR(type, len, t);
  CHKFORTRANNULLOBJECT(obj);
  *ierr = PetscLimiterViewFromOptions(*ao, obj, t);
  if (*ierr) return;
  FREECHAR(type, t);
}
