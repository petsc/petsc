#include <petsc/private/fortranimpl.h>
#include <petscts.h>
#include <petsc/private/f90impl.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define tsrksettype_                     TSRKSETTYPE
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define tsrksettype_                     tsrksettype
#endif

PETSC_EXTERN void PETSC_STDCALL tsrksettype_(TS *ts,char* type PETSC_MIXED_LEN(len),PetscErrorCode *ierr PETSC_END_LEN(len))
{
  char *t;

  FIXCHAR(type,len,t);
  *ierr = TSRKSetType(*ts,t);
  FREECHAR(type,t);
}
