#include "private/fortranimpl.h"
#include "petscmat.h"

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define matmpiaijgetseqaij_              MATMPIAIJGETSEQAIJ
#define matcreatempiaij_                 MATCREATEMPIAIJ
#define matmpiaijsetpreallocation_       MATMPIAIJSETPREALLOCATION
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define matmpiaijgetseqaij_              matmpiaijgetseqaij
#define matcreatempiaij_                 matcreatempiaij
#define matmpiaijsetpreallocation_       matmpiaijsetpreallocation
#endif

EXTERN_C_BEGIN

void PETSC_STDCALL matmpiaijgetseqaij_(Mat *A,Mat *Ad,Mat *Ao,PetscInt *ic,size_t *iic,PetscErrorCode *ierr)
{
  PetscInt *i;
  *ierr = MatMPIAIJGetSeqAIJ(*A,Ad,Ao,&i);if (*ierr) return;
  *iic  = PetscIntAddressToFortran(ic,i);
}

void PETSC_STDCALL matcreatempiaij_(MPI_Comm *comm,PetscInt *m,PetscInt *n,PetscInt *M,PetscInt *N,
         PetscInt *d_nz,PetscInt *d_nnz,PetscInt *o_nz,PetscInt *o_nnz,Mat *newmat,PetscErrorCode *ierr)
{
  CHKFORTRANNULLINTEGER(d_nnz);
  CHKFORTRANNULLINTEGER(o_nnz);

  *ierr = MatCreateMPIAIJ(MPI_Comm_f2c(*(MPI_Fint *)&*comm),
                             *m,*n,*M,*N,*d_nz,d_nnz,*o_nz,o_nnz,newmat);
}

void PETSC_STDCALL matmpiaijsetpreallocation_(Mat *mat,PetscInt *d_nz,PetscInt *d_nnz,PetscInt *o_nz,PetscInt *o_nnz,PetscErrorCode *ierr)
{
  CHKFORTRANNULLINTEGER(d_nnz);
  CHKFORTRANNULLINTEGER(o_nnz);
  *ierr = MatMPIAIJSetPreallocation(*mat,*d_nz,d_nnz,*o_nz,o_nnz);
}

EXTERN_C_END
