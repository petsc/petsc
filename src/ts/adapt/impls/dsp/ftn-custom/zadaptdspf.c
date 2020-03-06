#include <petsc/private/fortranimpl.h>
#include <petscts.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define tsadaptdspsetfilter_ TSADAPTDSPSETFILTER
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define tsadaptdspsetfilter_ tsadaptdspsetfilter
#endif

PETSC_EXTERN void tsadaptdspsetfilter_(TSAdapt *tsadapt,char* name,PetscErrorCode *ierr,PETSC_FORTRAN_CHARLEN_T len)
{
  char *t;

  FIXCHAR(name,len,t);
  *ierr = TSAdaptDSPSetFilter(*tsadapt,t);if (*ierr) return;
  FREECHAR(name,t);
}
