#include <petsc/private/fortranimpl.h>
#include <petscpc.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define pccompositeaddpctype_          PCCOMPOSITEADDPCTYPE
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define pccompositeaddpctype_          pccompositeaddpctype
#endif

PETSC_EXTERN void pccompositeaddpctype_(PC *pc,char* type,PetscErrorCode *ierr,PETSC_FORTRAN_CHARLEN_T len)
{
  char *t;

  FIXCHAR(type,len,t);
  *ierr = PCCompositeAddPCType(*pc,t);if (*ierr) return;
  FREECHAR(type,t);
}
