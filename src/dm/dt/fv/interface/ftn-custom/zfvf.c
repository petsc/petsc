#include <petsc/private/fortranimpl.h>
#include <petscfv.h>
#include <petscviewer.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define petsclimiterviewfromoptions_  PETSCLIMITERVIEWFROMOPTIONS
#define petscfvviewfromoptions_ PETSCFVVIEWFROMOPTIONS
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define petsclimiterviewfromoptions_  petsclimiterviewfromoptions
#define petscfvviewfromoptions_ petscfvviewfromoptions
#endif

PETSC_EXTERN void petsclimiterviewfromoptions_(PetscLimiter *ao,PetscObject obj,char* type,PetscErrorCode *ierr,PETSC_FORTRAN_CHARLEN_T len)
{
  char *t;

  FIXCHAR(type,len,t);
  *ierr = PetscLimiterViewFromOptions(*ao,obj,t);if (*ierr) return;
  FREECHAR(type,t);
}

PETSC_EXTERN void petscfvviewfromoptions_(PetscFV *ao,PetscObject obj,char* type,PetscErrorCode *ierr,PETSC_FORTRAN_CHARLEN_T len)
{
  char *t;

  FIXCHAR(type,len,t);
  *ierr = PetscFVViewFromOptions(*ao,obj,t);if (*ierr) return;
  FREECHAR(type,t);
}
