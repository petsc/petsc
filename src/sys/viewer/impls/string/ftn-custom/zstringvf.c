#include "zpetsc.h"
#include "petsc.h"

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define petscviewerstringopen_     PETSCVIEWERSTRINGOPEN
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define petscviewerstringopen_     petscviewerstringopen
#endif

EXTERN_C_BEGIN
void PETSC_STDCALL petscviewerstringopen_(MPI_Comm *comm,CHAR name PETSC_MIXED_LEN(len1),PetscInt *len,PetscViewer *str,
                                     PetscErrorCode *ierr PETSC_END_LEN(len1))
{
#if defined(PETSC_USES_CPTOFCD)
  *ierr = PetscViewerStringOpen((MPI_Comm)PetscToPointerComm(*comm),_fcdtocp(name),*len,str);
#else
  *ierr = PetscViewerStringOpen((MPI_Comm)PetscToPointerComm(*comm),name,*len,str);
#endif
}

EXTERN_C_END
