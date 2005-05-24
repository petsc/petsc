#include "zpetsc.h"
#include "petscis.h"

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define iscreategeneral_       ISCREATEGENERAL
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define iscreategeneral_       iscreategeneral
#endif

EXTERN_C_BEGIN

void PETSC_STDCALL iscreategeneral_(MPI_Comm *comm,PetscInt *n,PetscInt *idx,IS *is,PetscErrorCode *ierr)
{
  *ierr = ISCreateGeneral((MPI_Comm)PetscToPointerComm(*comm),*n,idx,is);
}

EXTERN_C_END
