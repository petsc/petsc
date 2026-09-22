#include <petsc/private/ftnimpl.h>
#include <petscmatlab.h>

#if PetscDefined(HAVE_FORTRAN_CAPS)
  #define petscmatlabengineevaluate_ PETSCMATLABENGINEEVALUATE
#elif !PetscDefined(HAVE_FORTRAN_UNDERSCORE)
  #define petscmatlabengineevaluate_ petscmatlabengineevaluate
#endif

PETSC_EXTERN void petscmatlabengineevaluate_(PetscMatlabEngine *mengine, char *string, PetscErrorCode *ierr, PETSC_FORTRAN_CHARLEN_T len)
{
  char *ms;
  FIXCHAR(string, len, ms);
  *ierr = PetscMatlabEngineEvaluate(*mengine, ms);
  if (*ierr) return;
  FREECHAR(string, ms);
}
