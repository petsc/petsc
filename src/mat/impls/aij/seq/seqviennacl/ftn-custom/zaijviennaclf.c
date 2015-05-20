#include <petsc/private/fortranimpl.h>
#include <petscmat.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define matcreateseqaijviennacl_                 MATCREATESEQAIJVIENNACL
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define matcreateseqaijviennacl_                 matcreateseqaijviennacl
#endif

PETSC_EXTERN void PETSC_STDCALL matcreateseqaijviennacl_(MPI_Comm *comm,PetscInt *m,PetscInt *n,PetscInt *nz,PetscInt *nnz,Mat *newmat,PetscErrorCode *ierr)
{
  CHKFORTRANNULLINTEGER(nnz);
  *ierr = MatCreateSeqAIJViennaCL(MPI_Comm_f2c(*(MPI_Fint*)&*comm),*m,*n,*nz,nnz,newmat);
}

