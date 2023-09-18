#include <petsc/private/fortranimpl.h>
#include <petscmat.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
  #define matmpibaijgetseqbaij_       MATMPIBAIJGETSEQBAIJ
  #define matcreatebaij_              MATCREATEBAIJ
  #define matmpibaijsetpreallocation_ MATMPIBAIJSETPREALLOCATION
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
  #define matmpibaijgetseqbaij_       matmpibaijgetseqbaij
  #define matcreatebaij_              matcreatebaij
  #define matmpibaijsetpreallocation_ matmpibaijsetpreallocation
#endif

PETSC_EXTERN void matmpibaijgetseqbaij_(Mat *A, Mat *Ad, Mat *Ao, PetscInt *ic, size_t *iic, PetscErrorCode *ierr)
{
  const PetscInt *i;
  *ierr = MatMPIBAIJGetSeqBAIJ(*A, Ad, Ao, &i);
  if (*ierr) return;
  *iic = PetscIntAddressToFortran(ic, (PetscInt *)i);
}

PETSC_EXTERN void matcreatebaij_(MPI_Comm *comm, PetscInt *bs, PetscInt *m, PetscInt *n, PetscInt *M, PetscInt *N, PetscInt *d_nz, PetscInt *d_nnz, PetscInt *o_nz, PetscInt *o_nnz, Mat *newmat, PetscErrorCode *ierr)
{
  CHKFORTRANNULLINTEGER(d_nnz);
  CHKFORTRANNULLINTEGER(o_nnz);
  *ierr = MatCreateBAIJ(MPI_Comm_f2c(*(MPI_Fint *)&*comm), *bs, *m, *n, *M, *N, *d_nz, d_nnz, *o_nz, o_nnz, newmat);
}

PETSC_EXTERN void matmpibaijsetpreallocation_(Mat *mat, PetscInt *bs, PetscInt *d_nz, PetscInt *d_nnz, PetscInt *o_nz, PetscInt *o_nnz, PetscErrorCode *ierr)
{
  CHKFORTRANNULLINTEGER(d_nnz);
  CHKFORTRANNULLINTEGER(o_nnz);
  *ierr = MatMPIBAIJSetPreallocation(*mat, *bs, *d_nz, d_nnz, *o_nz, o_nnz);
}
