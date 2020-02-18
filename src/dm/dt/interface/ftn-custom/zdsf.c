#include <petsc/private/fortranimpl.h>
#include <petscds.h>
#include <petscviewer.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define petscdsviewfromoptions_   PETSCDSVIEWFROMOPTIONS
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define petscdsviewfromoptions_   petscdsviewfromoptions
#endif

PETSC_EXTERN void PETSC_STDCALL petscdsviewfromoptions_(PetscDS *ao,PetscObject obj,char* type PETSC_MIXED_LEN(len),PetscErrorCode *ierr PETSC_END_LEN(len))
{
  char *t;

  FIXCHAR(type,len,t);
  *ierr = PetscDSViewFromOptions(*ao,obj,t);if (*ierr) return;
  FREECHAR(type,t);
}

