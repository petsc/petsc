#include <petsc/private/ftnimpl.h>
#include <petscksp.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
  #define kspgmressetorthogonalization_                  KSPGMRESSETORTHOGONALIZATION
  #define kspgmresmodifiedgramschmidtorthogonalization_  KSPGMRESMODIFIEDGRAMSCHMIDTORTHOGONALIZATION
  #define kspgmresclassicalgramschmidtorthogonalization_ KSPGMRESCLASSICALGRAMSCHMIDTORTHOGONALIZATION
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
  #define kspgmressetorthogonalization_                  kspgmressetorthogonalization
  #define kspgmresmodifiedgramschmidtorthogonalization_  kspgmresmodifiedgramschmidtorthogonalization
  #define kspgmresclassicalgramschmidtorthogonalization_ kspgmresclassicalgramschmidtorthogonalization
#endif

static struct {
  PetscFortranCallbackId orthog;
} _cb;

PETSC_EXTERN void kspgmresmodifiedgramschmidtorthogonalization_(KSP *, PetscInt *, PetscErrorCode *);
PETSC_EXTERN void kspgmresclassicalgramschmidtorthogonalization_(KSP *, PetscInt *, PetscErrorCode *);

static PetscErrorCode ourorthog(KSP ksp, PetscInt n)
{
  PetscObjectUseFortranCallback(ksp, _cb.orthog, (KSP *, PetscInt *, PetscErrorCode *), (&ksp, &n, &ierr));
}

PETSC_EXTERN void kspgmressetorthogonalization_(KSP *ksp, void (*orthog)(KSP *, PetscInt *, PetscErrorCode *), PetscErrorCode *ierr)
{
  if (orthog == kspgmresmodifiedgramschmidtorthogonalization_) {
    *ierr = KSPGMRESSetOrthogonalization(*ksp, KSPGMRESModifiedGramSchmidtOrthogonalization);
  } else if (orthog == kspgmresclassicalgramschmidtorthogonalization_) {
    *ierr = KSPGMRESSetOrthogonalization(*ksp, KSPGMRESClassicalGramSchmidtOrthogonalization);
  } else {
    *ierr = PetscObjectSetFortranCallback((PetscObject)*ksp, PETSC_FORTRAN_CALLBACK_CLASS, &_cb.orthog, (PetscFortranCallbackFn *)orthog, NULL);
    if (*ierr) return;
    *ierr = KSPGMRESSetOrthogonalization(*ksp, ourorthog);
  }
}
