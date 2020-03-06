#include <petsc/private/fortranimpl.h>
#include <petscpf.h>
#include <petscviewer.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define pfviewfromoptions_  PFVIEWFROMOPTIONS
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define pfviewfromoptions_  pfviewfromoptions
#endif

PETSC_EXTERN void pfviewfromoptions_(PF *ao,PetscObject obj,char* type,PetscErrorCode *ierr,PETSC_FORTRAN_CHARLEN_T len)
{
  char *t;

  FIXCHAR(type,len,t);
  *ierr = PFViewFromOptions(*ao,obj,t);if (*ierr) return;
  FREECHAR(type,t);
}
