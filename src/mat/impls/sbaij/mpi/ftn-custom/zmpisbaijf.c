#include "private/fortranimpl.h"
#include "petscmat.h"

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define matcreatempisbaij_               MATCREATEMPISBAIJ
#define matmpisbaijsetpreallocation_     MATMPISBAIJSETPREALLOCATION
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define matcreatempisbaij_               matcreatempisbaij
#define matmpisbaijsetpreallocation_     matmpisbaijsetpreallocation
#endif

EXTERN_C_BEGIN
void PETSC_STDCALL matcreatempisbaij_(MPI_Comm *comm,PetscInt *bs,PetscInt *m,PetscInt *n,PetscInt *M,PetscInt *N,
         PetscInt *d_nz,PetscInt *d_nnz,PetscInt *o_nz,PetscInt *o_nnz,Mat *newmat,PetscErrorCode *ierr)
{
  CHKFORTRANNULLINTEGER(d_nnz);
  CHKFORTRANNULLINTEGER(o_nnz);
  *ierr = MatCreateMPISBAIJ(MPI_Comm_f2c(*(MPI_Fint *)&*comm),
                             *bs,*m,*n,*M,*N,*d_nz,d_nnz,*o_nz,o_nnz,newmat);
}

void PETSC_STDCALL matmpisbaijsetpreallocation_(Mat *mat,PetscInt *bs,PetscInt *d_nz,PetscInt *d_nnz,PetscInt *o_nz,PetscInt *o_nnz,PetscErrorCode *ierr)
{
  CHKFORTRANNULLINTEGER(d_nnz);
  CHKFORTRANNULLINTEGER(o_nnz);
  *ierr = MatMPISBAIJSetPreallocation(*mat,*bs,*d_nz,d_nnz,*o_nz,o_nnz);
}

EXTERN_C_END
