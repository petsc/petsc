#include "zpetsc.h"
#include "petscts.h"

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define tscreate_                            TSCREATE
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define tscreate_                            tscreate
#endif

EXTERN_C_BEGIN
void PETSC_STDCALL tscreate_(MPI_Comm *comm,TS *outts,PetscErrorCode *ierr)
{
  *ierr = TSCreate((MPI_Comm)PetscToPointerComm(*comm),outts);
  *ierr = PetscMalloc(7*sizeof(void*),&((PetscObject)*outts)->fortran_func_pointers);
}

EXTERN_C_END
