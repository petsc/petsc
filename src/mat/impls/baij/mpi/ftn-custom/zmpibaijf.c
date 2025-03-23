#include <petsc/private/ftnimpl.h>
#include <petscmat.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
  #define matmpibaijgetseqbaij_ MATMPIBAIJGETSEQBAIJ
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
  #define matmpibaijgetseqbaij_ matmpibaijgetseqbaij
#endif

PETSC_EXTERN void matmpibaijgetseqbaij_(Mat *A, Mat *Ad, Mat *Ao, PetscInt *ic, size_t *iic, PetscErrorCode *ierr)
{
  const PetscInt *i;
  *ierr = MatMPIBAIJGetSeqBAIJ(*A, Ad, Ao, &i);
  if (*ierr) return;
  *iic = PetscIntAddressToFortran(ic, (PetscInt *)i);
}
