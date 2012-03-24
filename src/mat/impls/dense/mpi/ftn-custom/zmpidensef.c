#include <petsc-private/fortranimpl.h>
#include <petscmat.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define matcreatedense_                  MATCREATEDENSE
#define matmpidensesetpreallocation_     MATMPIDENSESETPREALLOCATION
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define matcreatedense_                  matcreatedense
#define matmpidensesetpreallocation_     matmpidensesetpreallocation
#endif

EXTERN_C_BEGIN
void PETSC_STDCALL matcreatedense_(MPI_Comm *comm,PetscInt *m,PetscInt *n,PetscInt *M,PetscInt *N,PetscScalar *data,Mat *newmat,PetscErrorCode *ierr)
{
  CHKFORTRANNULLSCALAR(data);
  *ierr = MatCreateDense(MPI_Comm_f2c(*(MPI_Fint *)&*comm),*m,*n,*M,*N,data,newmat);
}

void PETSC_STDCALL matmpidensesetpreallocation_(Mat *mat,PetscScalar *data,PetscErrorCode *ierr)
{
  CHKFORTRANNULLSCALAR(data);
  *ierr = MatMPIDenseSetPreallocation(*mat,data);
}

EXTERN_C_END
