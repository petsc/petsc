#include <petsc/private/fortranimpl.h>
#include <petscpf.h>
#include <petscviewer.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define pfviewfromoptions_  PFVIEWFROMOPTIONS
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define pfviewfromoptions_  pfviewfromoptions
#endif

PETSC_EXTERN void PETSC_STDCALL pfviewfromoptions_(PF *ao,PetscObject obj,char* type PETSC_MIXED_LEN(len),PetscErrorCode *ierr PETSC_END_LEN(len))
{
  char *t;

  FIXCHAR(type,len,t);
  *ierr = PFViewFromOptions(*ao,obj,t);if (*ierr) return;
  FREECHAR(type,t);
}
