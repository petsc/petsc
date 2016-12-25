#include <petsc/private/fortranimpl.h>
#include <petscdmda.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define dmdasetaotype_                 DMDASETAOTYPE
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define dmdagetglobalindices_          dmdagetglobalindices
#endif

PETSC_EXTERN void PETSC_STDCALL  dmdasetaotype_(DM *da,char* type PETSC_MIXED_LEN(len), PetscErrorCode *ierr PETSC_END_LEN(len) )
{
  char *t;
  FIXCHAR(type,len,t);
  *ierr = DMDASetAOType(*da,t);
  FREECHAR(type,t);
}
