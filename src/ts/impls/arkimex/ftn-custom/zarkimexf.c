#include <petsc/private/fortranimpl.h>
#include <petscts.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
  #define tsarkimexsettype_ TSARKIMEXSETTYPE
  #define tsarkimexgettype_ TSARKIMEXGETTYPE
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
  #define tsarkimexsettype_ tsarkimexsettype
  #define tsarkimexgettype_ tsarkimexgettype
#endif

PETSC_EXTERN void tsarkimexsettype_(TS *ts, char *type, PetscErrorCode *ierr, PETSC_FORTRAN_CHARLEN_T len)
{
  char *t;

  FIXCHAR(type, len, t);
  *ierr = TSARKIMEXSetType(*ts, t);
  if (*ierr) return;
  FREECHAR(type, t);
}

PETSC_EXTERN void tsarkimexgettype_(TS *ts, char *name, PetscErrorCode *ierr, PETSC_FORTRAN_CHARLEN_T len)
{
  const char *tname;

  *ierr = TSARKIMEXGetType(*ts, &tname);
  *ierr = PetscStrncpy(name, tname, len);
  FIXRETURNCHAR(PETSC_TRUE, name, len);
}
