#include <petsc/private/fortranimpl.h>
#include <petscdraw.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define petscdrawlgsetoptionsprefix_  PETSCDRAWLGSETOPTIONSPREFIX
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define petscdrawlgsetoptionsprefix_  petscdrawlgsetoptionsprefix
#endif

PETSC_EXTERN void PETSC_STDCALL petscdrawlgsetoptionsprefix_(PetscDrawLG *lg,char* prefix PETSC_MIXED_LEN(len),PetscErrorCode *ierr PETSC_END_LEN(len))
{
  char *t;
  FIXCHAR(prefix,len,t);
  *ierr = PetscDrawLGSetOptionsPrefix(*lg,t);
  FREECHAR(prefix,t);
}
