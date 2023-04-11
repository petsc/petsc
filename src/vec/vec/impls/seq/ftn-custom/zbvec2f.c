#include <petsc/private/fortranimpl.h>
#include <petscvec.h>
#if defined(PETSC_HAVE_FORTRAN_CAPS)
  #define veccreateseqwitharray0_ VECCREATESEQWITHARRAY0
  #define veccreateseqwitharray1_ VECCREATESEQWITHARRAY1
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
  #define veccreateseqwitharray0_ veccreateseqwitharray0
  #define veccreateseqwitharray1_ veccreateseqwitharray1
#endif

PETSC_EXTERN void veccreateseqwitharray0_(MPI_Comm *comm, int *bs, PetscInt *n, PetscScalar *s, Vec *V, PetscErrorCode *ierr)
{
  CHKFORTRANNULLSCALAR(s);
  *ierr = VecCreateSeqWithArray(MPI_Comm_f2c(*(MPI_Fint *)&*comm), *bs, *n, s, V);
}

PETSC_EXTERN void veccreateseqwitharray1_(MPI_Comm *comm, PetscInt64 *bs, PetscInt *n, PetscScalar *s, Vec *V, PetscErrorCode *ierr)
{
  CHKFORTRANNULLSCALAR(s);
  *ierr = VecCreateSeqWithArray(MPI_Comm_f2c(*(MPI_Fint *)&*comm), *bs, *n, s, V);
}
