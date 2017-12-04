#include <petsc/private/fortranimpl.h>
#include <petscts.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define tsadaptdspsetfilter_ TSADAPTDSPSETFILTER
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define tsadaptdspsetfilter_ tsadaptdspsetfilter
#endif

PETSC_EXTERN void PETSC_STDCALL tsadaptdspsetfilter_(TSAdapt *tsadapt,char* name PETSC_MIXED_LEN(len),PetscErrorCode *ierr PETSC_END_LEN(len))
{
  char *t;

  FIXCHAR(name,len,t);
  *ierr = TSAdaptDSPSetFilter(*tsadapt,t);
  FREECHAR(name,t);
}
