#include <petsc/private/fortranimpl.h>
#include <petscts.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define tspythonsettype_            TSPYTHONSETTYPE
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define tspythonsettype_            tspythonsettype
#endif

PETSC_EXTERN void tspythonsettype_(TS *ts, char* name,PetscErrorCode *ierr,PETSC_FORTRAN_CHARLEN_T len)
{
  char *t;
  FIXCHAR(name,len,t);
  *ierr = TSPythonSetType(*ts,t);if (*ierr) return;
  FREECHAR(name,t);
}
