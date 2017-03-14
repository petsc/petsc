#include <petsc/private/fortranimpl.h>
#include <petscts.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define tspythonsettype_            TSPYTHONSETTYPE
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define tspythonsettype_            tspythonsettype
#endif

PETSC_EXTERN void PETSC_STDCALL tspythonsettype_(TS *ts, char* name PETSC_MIXED_LEN(len),PetscErrorCode *ierr PETSC_END_LEN(len))
{
  char *t;
  FIXCHAR(name,len,t);
  *ierr = TSPythonSetType(*ts,t);
  FREECHAR(name,t);
}
