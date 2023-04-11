#include <petsc/private/fortranimpl.h>
#include <petscpc.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
  #define pcpythonsettype_ PCPYTHONSETTYPE
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
  #define pcpythonsettype_ pcpythonsettype
#endif

PETSC_EXTERN void pcpythonsettype_(PC *pc, char *name, PetscErrorCode *ierr, PETSC_FORTRAN_CHARLEN_T len)
{
  char *t;
  FIXCHAR(name, len, t);
  *ierr = PCPythonSetType(*pc, t);
  if (*ierr) return;
  FREECHAR(name, t);
}
