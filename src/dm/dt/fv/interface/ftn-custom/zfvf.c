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

PETSC_EXTERN void PETSC_STDCALL petsclimiterviewfromoptions_(PetscLimiter *ao,PetscObject obj,char* type PETSC_MIXED_LEN(len),PetscErrorCode *ierr PETSC_END_LEN(len))
{
  char *t;

  FIXCHAR(type,len,t);
  *ierr = PetscLimiterViewFromOptions(*ao,obj,t);if (*ierr) return;
  FREECHAR(type,t);
}

PETSC_EXTERN void PETSC_STDCALL petscfvviewfromoptions_(PetscFV *ao,PetscObject obj,char* type PETSC_MIXED_LEN(len),PetscErrorCode *ierr PETSC_END_LEN(len))
{
  char *t;

  FIXCHAR(type,len,t);
  *ierr = PetscFVViewFromOptions(*ao,obj,t);if (*ierr) return;
  FREECHAR(type,t);
}
