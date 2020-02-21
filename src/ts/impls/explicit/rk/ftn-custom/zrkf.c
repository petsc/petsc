#include <petsc/private/fortranimpl.h>
#include <petscts.h>
#include <petsc/private/f90impl.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define tsrksettype_                     TSRKSETTYPE
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define tsrksettype_                     tsrksettype
#endif

PETSC_EXTERN void tsrksettype_(TS *ts,char* type,PetscErrorCode *ierr,PETSC_FORTRAN_CHARLEN_T len)
{
  char *t;

  FIXCHAR(type,len,t);
  *ierr = TSRKSetType(*ts,t);if (*ierr) return;
  FREECHAR(type,t);
}
