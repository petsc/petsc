#include <petsc/private/fortranimpl.h>
#include <petscsection.h>
#include <petscviewer.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define petscsectionviewfromoptions_  PETSCSECTIONVIEWFROMOPTIONS
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define petscsectionviewfromoptions_  petscsectionviewfromoptions
#endif

PETSC_EXTERN void petscsectionviewfromoptions_(PetscSection *ao,PetscObject obj,char* type,PetscErrorCode *ierr,PETSC_FORTRAN_CHARLEN_T len)
{
  char *t;

  FIXCHAR(type,len,t);
  *ierr = PetscSectionViewFromOptions(*ao,obj,t);if (*ierr) return;
  FREECHAR(type,t);
}
