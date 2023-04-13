#include <petsc/private/fortranimpl.h>
#include <petscmat.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
  #define matpythonsettype_ MATPYTHONSETTYPE
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
  #define matpythonsettype_ matpythonsettype
#endif

PETSC_EXTERN void matpythonsettype_(Mat *mat, char *name, PetscErrorCode *ierr, PETSC_FORTRAN_CHARLEN_T len)
{
  char *t;
  FIXCHAR(name, len, t);
  *ierr = MatPythonSetType(*mat, t);
  if (*ierr) return;
  FREECHAR(name, t);
}
