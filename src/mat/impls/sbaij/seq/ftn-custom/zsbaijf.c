#include <petsc-private/fortranimpl.h>
#include <petscmat.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define matcreateseqsbaij_               MATCREATESEQSBAIJ
#define matseqsbaijsetpreallocation_     MATSEQSBAIJSETPREALLOCATION
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define matcreateseqsbaij_               matcreateseqsbaij
#define matseqsbaijsetpreallocation_     matseqsbaijsetpreallocation
#endif

EXTERN_C_BEGIN

void PETSC_STDCALL matcreateseqsbaij_(MPI_Comm *comm,PetscInt *bs,PetscInt *m,PetscInt *n,PetscInt *nz,
                           PetscInt *nnz,Mat *newmat,PetscErrorCode *ierr)
{
  CHKFORTRANNULLINTEGER(nnz);
  *ierr = MatCreateSeqSBAIJ(MPI_Comm_f2c(*(MPI_Fint *)&*comm),*bs,*m,*n,*nz,nnz,newmat);
}

void PETSC_STDCALL matseqsbaijsetpreallocation_(Mat *mat,PetscInt *bs,PetscInt *nz,PetscInt *nnz,PetscErrorCode *ierr)
{
  CHKFORTRANNULLINTEGER(nnz);
  *ierr = MatSeqSBAIJSetPreallocation(*mat,*bs,*nz,nnz);
}
EXTERN_C_END
