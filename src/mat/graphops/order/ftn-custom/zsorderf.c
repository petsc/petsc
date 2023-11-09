#include <petsc/private/fortranimpl.h>
#include <petscmat.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
  #define matgetordering_ MATGETORDERING
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
  #define matgetordering_ matgetordering
#endif

PETSC_EXTERN void matgetordering_(Mat *mat, char *type, IS *rperm, IS *cperm, PetscErrorCode *ierr, PETSC_FORTRAN_CHARLEN_T len)
{
  char *t;
  FIXCHAR(type, len, t);
  *ierr = MatGetOrdering(*mat, t, rperm, cperm);
  if (*ierr) return;
  FREECHAR(type, t);
}
