#include "zpetsc.h"
#include "petscis.h"

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define islocaltoglobalmappingview_   ISLOCALTOGLOBALMAPPINGVIEW
#define islocaltoglobalmappingcreate_ ISLOCALTOGLOBALMAPPINGCREATE
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define islocaltoglobalmappingview_   islocaltoglobalmappingview
#define islocaltoglobalmappingcreate_ islocaltoglobalmappingcreate
#endif

EXTERN_C_BEGIN

void PETSC_STDCALL islocaltoglobalmappingview_(ISLocalToGlobalMapping *mapping,PetscViewer *viewer,PetscErrorCode *ierr)
{
  CHKFORTRANNULLOBJECTDEREFERENCE(viewer);
  *ierr = ISLocalToGlobalMappingView(*mapping,*viewer);
}

void PETSC_STDCALL islocaltoglobalmappingcreate_(MPI_Comm *comm,PetscInt *n,PetscInt *indices,ISLocalToGlobalMapping *mapping,PetscErrorCode *ierr)
{
  *ierr = ISLocalToGlobalMappingCreate((MPI_Comm)PetscToPointerComm(*comm),*n,indices,mapping);
}

EXTERN_C_END
