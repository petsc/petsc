#include "zpetsc.h"
#include "petscis.h"

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define islocaltoglobalmappingview_   ISLOCALTOGLOBALMAPPINGVIEW
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define islocaltoglobalmappingview_   islocaltoglobalmappingview
#endif

EXTERN_C_BEGIN

void PETSC_STDCALL islocaltoglobalmappingview_(ISLocalToGlobalMapping *mapping,PetscViewer *viewer,PetscErrorCode *ierr)
{
  PetscViewer v;
  PetscPatchDefaultViewers_Fortran(viewer,v);
  *ierr = ISLocalToGlobalMappingView(*mapping,v);
}

EXTERN_C_END
