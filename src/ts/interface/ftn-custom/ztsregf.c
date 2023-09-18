#include <petsc/private/fortranimpl.h>
#include <petscts.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
  #define tssettype_ TSSETTYPE
  #define tsgettype_ TSGETTYPE
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
  #define tssettype_ tssettype
  #define tsgettype_ tsgettype
#endif

PETSC_EXTERN void tssettype_(TS *ts, char *type, PetscErrorCode *ierr, PETSC_FORTRAN_CHARLEN_T len)
{
  char *t;

  FIXCHAR(type, len, t);
  *ierr = TSSetType(*ts, t);
  if (*ierr) return;
  FREECHAR(type, t);
}

PETSC_EXTERN void tsgettype_(TS *ts, char *name, PetscErrorCode *ierr, PETSC_FORTRAN_CHARLEN_T len)
{
  const char *tname;

  *ierr = TSGetType(*ts, &tname);
  *ierr = PetscStrncpy(name, tname, len);
  FIXRETURNCHAR(PETSC_TRUE, name, len);
}
