#include <petsc/private/ftnimpl.h>

#include <petscpc.h>
#if defined(PETSC_HAVE_FORTRAN_CAPS)
  #define pchpddmsetauxiliarymat_ PCHPDDMSETAUXILIARYMAT
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
  #define pchpddmsetauxiliarymat_ pchpddmsetauxiliarymat
#endif

PETSC_EXTERN void pchpddmsetauxiliarymat_(PC *pc, IS *is, Mat *A, PetscErrorCode (*setup)(Mat, PetscReal, Vec, Vec, PetscReal, IS, void *), PETSC_UNUSED void *setup_ctx, PetscErrorCode *ierr)
{
  if ((PetscFortranCallbackFn *)setup != (PetscFortranCallbackFn *)PETSC_NULL_FUNCTION_Fortran) {
    *ierr = PETSC_ERR_ARG_WRONG;
    return;
  }
  *ierr = PCHPDDMSetAuxiliaryMat(*pc, *is, *A, NULL, NULL);
}
