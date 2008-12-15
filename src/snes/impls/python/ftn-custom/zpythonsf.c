#include "private/fortranimpl.h"
#include "petscsnes.h"

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define snespythonsettype_            SNESPYTHONSETTYPE
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define snespythonsettype_            snespythonsettype
#endif


EXTERN_C_BEGIN

void PETSC_STDCALL  snespythonsettype_(SNES *snes, CHAR name PETSC_MIXED_LEN(len),PetscErrorCode *ierr PETSC_END_LEN(len) )
{
  char *t;
  FIXCHAR(name,len,t);
  *ierr = SNESPythonSetType(*snes,t);
  FREECHAR(name,t);
}


EXTERN_C_END
