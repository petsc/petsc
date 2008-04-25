#include "private/fortranimpl.h"
#include "petscmat.h"

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define matcreatempidense_               MATCREATEMPIDENSE
#define matmpidensesetpreallocation_     MATMPIDENSESETPREALLOCATION
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define matcreatempidense_               matcreatempidense
#define matmpidensesetpreallocation_     matmpidensesetpreallocation
#endif

EXTERN_C_BEGIN
void PETSC_STDCALL matcreatempidense_(MPI_Comm *comm,PetscInt *m,PetscInt *n,PetscInt *M,PetscInt *N,PetscScalar *data,Mat *newmat,
                        PetscErrorCode *ierr)
{
  CHKFORTRANNULLSCALAR(data);
  *ierr = MatCreateMPIDense(MPI_Comm_f2c(*(MPI_Fint *)&*comm),*m,*n,*M,*N,data,newmat);
}

void PETSC_STDCALL matmpidensesetpreallocation_(Mat *mat,PetscScalar *data,PetscErrorCode *ierr)
{
  CHKFORTRANNULLSCALAR(data);
  *ierr = MatMPIDenseSetPreallocation(*mat,data);
}

EXTERN_C_END
