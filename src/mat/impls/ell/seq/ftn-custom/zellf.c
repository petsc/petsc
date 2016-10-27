#include <petsc/private/fortranimpl.h>
#include <petscmat.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define matcreateseqell_                 MATCREATESEQELL
#define matseqellsetpreallocation_       MATSEQELLSETPREALLOCATION
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define matcreateseqell_                 matcreateseqell
#define matseqellsetpreallocation_       matseqellsetpreallocation
#endif

PETSC_EXTERN void PETSC_STDCALL matcreateseqell_(MPI_Comm *comm,PetscInt *m,PetscInt *n,PetscInt *maxrlenrow,PetscInt *rlen,Mat *newmat,PetscErrorCode *ierr)
{
  CHKFORTRANNULLINTEGER(rlen);
  *ierr = MatCreateSeqELL(MPI_Comm_f2c(*(MPI_Fint*)&*comm),*m,*n,*maxrlenrow,rlen,newmat);
}

PETSC_EXTERN void PETSC_STDCALL matseqellsetpreallocation_(Mat *mat,PetscInt *maxrlenrow,PetscInt *rlen,PetscErrorCode *ierr)
{
  CHKFORTRANNULLINTEGER(rlen);
  *ierr = MatSeqELLSetPreallocation(*mat,*maxrlenrow,rlen);
}

