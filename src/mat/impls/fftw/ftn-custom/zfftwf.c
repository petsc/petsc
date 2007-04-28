#include "zpetsc.h"
#include "petscmat.h"

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define matcreateseqfftw_        MATCREATESEQFFTW
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define matcreateseqfftw_        matcreateseqfftw
#endif

EXTERN_C_BEGIN
void PETSC_STDCALL matcreateseqfftw_(MPI_Comm *comm,PetscInt *ndim,PetscInt dim[],Mat* A,PetscErrorCode *ierr)
{ 
  *ierr = MatCreateSeqFFTW((MPI_Comm)PetscToPointerComm(*comm),*ndim,dim,A);
}

EXTERN_C_END
