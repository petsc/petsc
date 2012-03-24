#include <petsc-private/fortranimpl.h>
#include <petscpc.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define pcpythonsettype_            PCPYTHONSETTYPE
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define pcpythonsettype_            pcpythonsettype
#endif


EXTERN_C_BEGIN

void PETSC_STDCALL  pcpythonsettype_(PC *pc, CHAR name PETSC_MIXED_LEN(len),PetscErrorCode *ierr PETSC_END_LEN(len) )
{
  char *t;
  FIXCHAR(name,len,t);
  *ierr = PCPythonSetType(*pc,t);
  FREECHAR(name,t);
}


EXTERN_C_END
