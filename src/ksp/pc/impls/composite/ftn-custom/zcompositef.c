#include <petsc-private/fortranimpl.h>
#include <petscpc.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define pccompositeaddpc_          PCCOMPOSITEADDPC
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define pccompositeaddpc_          pccompositeaddpc
#endif

EXTERN_C_BEGIN
void PETSC_STDCALL pccompositeaddpc_(PC *pc,CHAR type PETSC_MIXED_LEN(len),PetscErrorCode *ierr PETSC_END_LEN(len))
{
  char *t;

  FIXCHAR(type,len,t);
  *ierr = PCCompositeAddPC(*pc,t);
  FREECHAR(type,t);
}

EXTERN_C_END
