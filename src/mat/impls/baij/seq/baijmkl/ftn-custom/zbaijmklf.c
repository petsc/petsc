#include <petsc/private/fortranimpl.h>
#include <petscmat.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
  #define matcreateseqbaijmkl_ MATCREATESEQBAIJMKL
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
  #define matcreateseqbaijmkl_ matcreateseqbaijmkl
#endif

PETSC_EXTERN void matcreateseqbaijmkl_(MPI_Comm *comm, PetscInt *bs, PetscInt *m, PetscInt *n, PetscInt *nz, PetscInt *nnz, Mat *newmat, PetscErrorCode *ierr)
{
  CHKFORTRANNULLINTEGER(nnz);
  *ierr = MatCreateSeqBAIJMKL(MPI_Comm_f2c(*(MPI_Fint *)&*comm), *bs, *m, *n, *nz, nnz, newmat);
}
