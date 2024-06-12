#include <petsc/private/fortranimpl.h>
#include <petscmat.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
  #define matmpiaijgetseqaij_ MATMPIAIJGETSEQAIJ
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
  #define matmpiaijgetseqaij_ matmpiaijgetseqaij
#endif

PETSC_EXTERN void matmpiaijgetseqaij_(Mat *A, Mat *Ad, Mat *Ao, PetscInt *ic, size_t *iic, PetscErrorCode *ierr)
{
  const PetscInt *i;
  *ierr = MatMPIAIJGetSeqAIJ(*A, Ad, Ao, &i);
  if (*ierr) return;
  *iic = PetscIntAddressToFortran(ic, (PetscInt *)i);
}
