#include "zpetsc.h"
#include "petscis.h"

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define iscreatestride_        ISCREATESTRIDE
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define iscreatestride_        iscreatestride
#endif

EXTERN_C_BEGIN

void PETSC_STDCALL iscreatestride_(MPI_Comm *comm,PetscInt *n,PetscInt *first,PetscInt *step,
                               IS *is,PetscErrorCode *ierr)
{
  *ierr = ISCreateStride((MPI_Comm)PetscToPointerComm(*comm),*n,*first,*step,is);
}

EXTERN_C_END
