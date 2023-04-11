#include <petsc/private/fortranimpl.h>
#include <petscmat.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
  #define matcreateseqsell_           MATCREATESEQSELL
  #define matseqsellsetpreallocation_ MATSEQSELLSETPREALLOCATION
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
  #define matcreateseqsell_           matcreateseqsell
  #define matseqsellsetpreallocation_ matseqsellsetpreallocation
#endif

PETSC_EXTERN void matcreateseqsell_(MPI_Comm *comm, PetscInt *m, PetscInt *n, PetscInt *maxrlenrow, PetscInt *rlen, Mat *newmat, PetscErrorCode *ierr)
{
  CHKFORTRANNULLINTEGER(rlen);
  *ierr = MatCreateSeqSELL(MPI_Comm_f2c(*(MPI_Fint *)&*comm), *m, *n, *maxrlenrow, rlen, newmat);
}

PETSC_EXTERN void matseqsellsetpreallocation_(Mat *mat, PetscInt *maxrlenrow, PetscInt *rlen, PetscErrorCode *ierr)
{
  CHKFORTRANNULLINTEGER(rlen);
  *ierr = MatSeqSELLSetPreallocation(*mat, *maxrlenrow, rlen);
}
