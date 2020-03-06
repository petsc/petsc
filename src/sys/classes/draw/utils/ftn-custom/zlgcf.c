#include <petsc/private/fortranimpl.h>
#include <petscdraw.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define petscdrawlgsetoptionsprefix_  PETSCDRAWLGSETOPTIONSPREFIX
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define petscdrawlgsetoptionsprefix_  petscdrawlgsetoptionsprefix
#endif

PETSC_EXTERN void petscdrawlgsetoptionsprefix_(PetscDrawLG *lg,char* prefix,PetscErrorCode *ierr,PETSC_FORTRAN_CHARLEN_T len)
{
  char *t;
  FIXCHAR(prefix,len,t);
  *ierr = PetscDrawLGSetOptionsPrefix(*lg,t);if (*ierr) return;
  FREECHAR(prefix,t);
}
