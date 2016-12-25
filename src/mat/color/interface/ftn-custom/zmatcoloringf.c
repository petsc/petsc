#include <petsc/private/fortranimpl.h>
#include <petscmat.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define matcoloringsettype_ MATCOLORINGSETTYPE
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE) && !defined(FORTRANDOUBLEUNDERSCORE)
#define matcoloringsettype_ matcoloringsettype
#endif

PETSC_EXTERN void PETSC_STDCALL matcoloringsettype_(MatColoring *mc,char* type PETSC_MIXED_LEN(len),PetscErrorCode *ierr PETSC_END_LEN(len))
{
  char *t;

  FIXCHAR(type,len,t);
  *ierr = MatColoringSetType(*mc,t);
  FREECHAR(type,t);
}
