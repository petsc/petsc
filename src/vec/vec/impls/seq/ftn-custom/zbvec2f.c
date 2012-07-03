#include <petsc-private/fortranimpl.h>
#include <petscvec.h>
#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define veccreateseqwitharray_    VECCREATESEQWITHARRAY
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define veccreateseqwitharray_    veccreateseqwitharray
#endif

EXTERN_C_BEGIN

void PETSC_STDCALL veccreateseqwitharray_(MPI_Comm *comm,PetscInt *bs,PetscInt *n,PetscScalar *s,Vec *V,PetscErrorCode *ierr)
{
  CHKFORTRANNULLSCALAR(s);
  *ierr = VecCreateSeqWithArray(MPI_Comm_f2c(*(MPI_Fint *)&*comm),*bs,*n,s,V);
}

EXTERN_C_END
