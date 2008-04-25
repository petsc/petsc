#include "private/fortranimpl.h"
#include "petscmat.h"

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define matcreateseqdense_               MATCREATESEQDENSE
#define matseqdensesetpreallocation_     MATSEQDENSESETPREALLOCATION
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define matcreateseqdense_               matcreateseqdense
#define matseqdensesetpreallocation_     matseqdensesetpreallocation
#endif

EXTERN_C_BEGIN
void PETSC_STDCALL matcreateseqdense_(MPI_Comm *comm,PetscInt *m,PetscInt *n,PetscScalar *data,Mat *newmat,PetscErrorCode *ierr)
{
  CHKFORTRANNULLSCALAR(data);
  *ierr = MatCreateSeqDense(MPI_Comm_f2c(*(MPI_Fint *)&*comm),*m,*n,data,newmat);
}

void PETSC_STDCALL matseqdensesetpreallocation_(Mat *mat,PetscScalar *data,PetscErrorCode *ierr)
{
  CHKFORTRANNULLSCALAR(data);
  *ierr = MatSeqDenseSetPreallocation(*mat,data);
}

EXTERN_C_END
