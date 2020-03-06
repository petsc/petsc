#include <petsc/private/fortranimpl.h>
#include <petscmat.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define matcoloringsettype_ MATCOLORINGSETTYPE
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE) && !defined(FORTRANDOUBLEUNDERSCORE)
#define matcoloringsettype_ matcoloringsettype
#endif

PETSC_EXTERN void matcoloringsettype_(MatColoring *mc,char* type,PetscErrorCode *ierr,PETSC_FORTRAN_CHARLEN_T len)
{
  char *t;

  FIXCHAR(type,len,t);
  *ierr = MatColoringSetType(*mc,t);if (*ierr) return;
  FREECHAR(type,t);
}
