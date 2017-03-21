
#include <petsc/private/fortranimpl.h>
#include <petscts.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define tsadaptsettype_TSADAPTSETTTYPE
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define tsadaptsettype_ tsadaptsettype
#endif

PETSC_EXTERN void PETSC_STDCALL tsadaptsettype_(TSAdapt *tsadapt,CHAR type PETSC_MIXED_LEN(len),PetscErrorCode *ierr PETSC_END_LEN(len))
{
  char *t;

  FIXCHAR(type,len,t);
  *ierr = TSAdaptSetType(*tsadapt,t);
  FREECHAR(type,t);
}
