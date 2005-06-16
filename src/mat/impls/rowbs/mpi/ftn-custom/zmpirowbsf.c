#include "zpetsc.h"
#include "petscmat.h"

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define matcreatempirowbs_               MATCREATEMPIROWBS
#define matmpirowbssetpreallocation_     MATMPIROWBSSETPREALLOCATION
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define matcreatempirowbs_               matcreatempirowbs
#define matmpirowbssetpreallocation_     matmpirowbssetpreallocation
#endif

EXTERN_C_BEGIN

/*  Fortran cannot pass in procinfo,hence ignored */
void PETSC_STDCALL matcreatempirowbs_(MPI_Comm *comm,PetscInt *m,PetscInt *M,PetscInt *nz,PetscInt *nnz,Mat *newmat,PetscErrorCode *ierr)
{
  CHKFORTRANNULLINTEGER(nnz);
  *ierr = MatCreateMPIRowbs((MPI_Comm)PetscToPointerComm(*comm),*m,*M,*nz,nnz,newmat);
}

void PETSC_STDCALL matmpirowbssetpreallocation_(Mat *mat,PetscInt *nz,PetscInt *nnz,PetscErrorCode *ierr)
{
  CHKFORTRANNULLINTEGER(nnz);
  *ierr = MatMPIRowbsSetPreallocation(*mat,*nz,nnz);
}

EXTERN_C_END
