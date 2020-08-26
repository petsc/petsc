#include <petsc/private/fortranimpl.h>
#include <petscdmda.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define dmdasetaotype_                 DMDASETAOTYPE
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define dmdagetglobalindices_          dmdagetglobalindices
#endif

PETSC_EXTERN void  dmdasetaotype_(DM *da,char* type, PetscErrorCode *ierr,PETSC_FORTRAN_CHARLEN_T len)
{
  char *t;
  FIXCHAR(type,len,t);
  *ierr = DMDASetAOType(*da,t);if (*ierr) return;
  FREECHAR(type,t);
}
