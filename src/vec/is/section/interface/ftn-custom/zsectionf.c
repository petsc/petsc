#include <petsc/private/fortranimpl.h>
#include <petscsection.h>
#include <petscviewer.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define petscsectionviewfromoptions_  PETSCSECTIONVIEWFROMOPTIONS
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define petscsectionviewfromoptions_  petscsectionviewfromoptions
#endif

PETSC_EXTERN void PETSC_STDCALL petscsectionviewfromoptions_(PetscSection *ao,PetscObject obj,char* type PETSC_MIXED_LEN(len),PetscErrorCode *ierr PETSC_END_LEN(len))
{
  char *t;

  FIXCHAR(type,len,t);
  *ierr = PetscSectionViewFromOptions(*ao,obj,t);if (*ierr) return;
  FREECHAR(type,t);
}
