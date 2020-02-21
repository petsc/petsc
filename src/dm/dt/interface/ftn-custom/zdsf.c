#include <petsc/private/fortranimpl.h>
#include <petscds.h>
#include <petscviewer.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define petscdsviewfromoptions_   PETSCDSVIEWFROMOPTIONS
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define petscdsviewfromoptions_   petscdsviewfromoptions
#endif

PETSC_EXTERN void petscdsviewfromoptions_(PetscDS *ao,PetscObject obj,char* type,PetscErrorCode *ierr,PETSC_FORTRAN_CHARLEN_T len)
{
  char *t;

  FIXCHAR(type,len,t);
  *ierr = PetscDSViewFromOptions(*ao,obj,t);if (*ierr) return;
  FREECHAR(type,t);
}

