#include <petsc/private/fortranimpl.h>
#include <petscpc.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define pccompositeaddpc_          PCCOMPOSITEADDPC
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define pccompositeaddpc_          pccompositeaddpc
#endif

PETSC_EXTERN void pccompositeaddpc_(PC *pc,char* type,PetscErrorCode *ierr,PETSC_FORTRAN_CHARLEN_T len)
{
  char *t;

  FIXCHAR(type,len,t);
  *ierr = PCCompositeAddPC(*pc,t);if (*ierr) return;
  FREECHAR(type,t);
}

