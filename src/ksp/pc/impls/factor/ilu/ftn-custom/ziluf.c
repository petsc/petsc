#include "zpetsc.h"
#include "petscpc.h"

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define pcilusetmatordering_       PCILUSETMATORDERING
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define pcilusetmatordering_       pcilusetmatordering
#endif

EXTERN_C_BEGIN
void PETSC_STDCALL pcilusetmatordering_(PC *pc,CHAR ordering PETSC_MIXED_LEN(len), PetscErrorCode *ierr PETSC_END_LEN(len)){
  char *t;

    FIXCHAR(ordering,len,t);
    *ierr = PCILUSetMatOrdering(*pc,t);
    FREECHAR(ordering,t);
}

EXTERN_C_END
