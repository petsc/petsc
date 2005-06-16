#include "zpetsc.h"
#include "petscpc.h"

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define pclusetmatordering_        PCLUSETMATORDERING
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define pclusetmatordering_        pclusetmatordering
#endif

EXTERN_C_BEGIN
void PETSC_STDCALL pclusetmatordering_(PC *pc,CHAR ordering PETSC_MIXED_LEN(len), PetscErrorCode *ierr PETSC_END_LEN(len)){
  char *t;

    FIXCHAR(ordering,len,t);
    *ierr = PCLUSetMatOrdering(*pc,t);
    FREECHAR(ordering,t);
}


EXTERN_C_END
