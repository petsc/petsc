#include <petsc/private/fortranimpl.h>
#include <petscts.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
  #define tsadaptsettype_ TSADAPTSETTYPE
  #define tsadaptgettype_ TSADAPTGETTYPE
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
  #define tsadaptsettype_ tsadaptsettype
  #define tsadaptgettype_ tsadaptgettype
#endif

PETSC_EXTERN void tsadaptsettype_(TSAdapt *tsadapt, char *type, PetscErrorCode *ierr, PETSC_FORTRAN_CHARLEN_T len)
{
  char *t;

  FIXCHAR(type, len, t);
  *ierr = TSAdaptSetType(*tsadapt, t);
  if (*ierr) return;
  FREECHAR(type, t);
}

PETSC_EXTERN void tsadaptgettype_(TSAdapt *adapt, char *type, PetscErrorCode *ierr, PETSC_FORTRAN_CHARLEN_T len)
{
  const char *t;

  *ierr = TSAdaptGetType(*adapt, &t);
  *ierr = PetscStrncpy(type, t, len);
  FIXRETURNCHAR(PETSC_TRUE, type, len);
}
