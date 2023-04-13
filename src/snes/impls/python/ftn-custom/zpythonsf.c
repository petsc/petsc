#include <petsc/private/fortranimpl.h>
#include <petscsnes.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
  #define snespythonsettype_ SNESPYTHONSETTYPE
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
  #define snespythonsettype_ snespythonsettype
#endif

PETSC_EXTERN void snespythonsettype_(SNES *snes, char *name, PetscErrorCode *ierr, PETSC_FORTRAN_CHARLEN_T len)
{
  char *t;
  FIXCHAR(name, len, t);
  *ierr = SNESPythonSetType(*snes, t);
  if (*ierr) return;
  FREECHAR(name, t);
}
