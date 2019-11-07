#include <petsc/private/fortranimpl.h>
#include <petscmat.h>
#include <petscviewer.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define matcoarsenviewfromoptions_  MATCOARSENVIEWFROMOPTIONS
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define matcoarsenviewfromoptions_  matcoarsenviewfromoptions
#endif

PETSC_EXTERN void PETSC_STDCALL matcoarsenviewfromoptions_(Mat *ao,PetscObject obj,char* type PETSC_MIXED_LEN(len),PetscErrorCode *ierr PETSC_END_LEN(len))
{
  char *t;

  FIXCHAR(type,len,t);
  *ierr = MatCoarsenViewFromOptions(*ao,obj,t);if (*ierr) return;
  FREECHAR(type,t);
}
