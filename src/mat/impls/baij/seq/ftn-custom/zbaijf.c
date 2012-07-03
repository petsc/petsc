#include <petsc-private/fortranimpl.h>
#include <petscmat.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define matcreateseqbaij_                MATCREATESEQBAIJ
#define matseqbaijsetpreallocation_      MATSEQBAIJSETPREALLOCATION
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define matcreateseqbaij_                matcreateseqbaij
#define matseqbaijsetpreallocation_      matseqbaijsetpreallocation
#endif

EXTERN_C_BEGIN
void PETSC_STDCALL matcreateseqbaij_(MPI_Comm *comm,PetscInt *bs,PetscInt *m,PetscInt *n,PetscInt *nz,
                           PetscInt *nnz,Mat *newmat,PetscErrorCode *ierr)
{
  CHKFORTRANNULLINTEGER(nnz);
  *ierr = MatCreateSeqBAIJ(MPI_Comm_f2c(*(MPI_Fint *)&*comm),*bs,*m,*n,*nz,nnz,newmat);
}

void PETSC_STDCALL matseqbaijsetpreallocation_(Mat *mat,PetscInt *bs,PetscInt *nz,PetscInt *nnz,PetscErrorCode *ierr)
{
  CHKFORTRANNULLINTEGER(nnz);
  *ierr = MatSeqBAIJSetPreallocation(*mat,*bs,*nz,nnz);
}

EXTERN_C_END
