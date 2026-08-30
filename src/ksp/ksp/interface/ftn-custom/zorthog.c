#include <petsc/private/ftnimpl.h>
#include <petscksp.h>

#if PetscDefined(HAVE_FORTRAN_CAPS)
  #define ksporthogonalizationset_                  KSPORTHOGONALIZATIONSET
  #define ksporthogonalizationmodifiedgramschmidt_  KSPORTHOGONALIZATIONMODIFIEDGRAMSCHMIDT
  #define ksporthogonalizationclassicalgramschmidt_ KSPORTHOGONALIZATIONCLASSICALGRAMSCHMIDT
#elif !PetscDefined(HAVE_FORTRAN_UNDERSCORE)
  #define ksporthogonalizationset_                  ksporthogonalizationset
  #define ksporthogonalizationmodifiedgramschmidt_  ksporthogonalizationmodifiedgramschmidt
  #define ksporthogonalizationclassicalgramschmidt_ ksporthogonalizationclassicalgramschmidt
#endif

static struct {
  PetscFortranCallbackId orthog;
} _cb;

PETSC_EXTERN void ksporthogonalizationmodifiedgramschmidt_(KSP *, Vec *, PetscInt *, Vec *, PetscScalar *, PetscErrorCode *);
PETSC_EXTERN void ksporthogonalizationclassicalgramschmidt_(KSP *, Vec *, PetscInt *, Vec *, PetscScalar *, PetscErrorCode *);

static PetscErrorCode ourorthog(KSP ksp, Vec V[], PetscInt n, Vec x, PetscScalar h[])
{
  PetscObjectUseFortranCallback(ksp, _cb.orthog, (KSP *, Vec *, PetscInt *, Vec *, PetscScalar *, PetscErrorCode *), (&ksp, V, &n, &x, h, &ierr));
}

PETSC_EXTERN void ksporthogonalizationset_(KSP *ksp, void (*orthog)(KSP *, Vec *, PetscInt *, Vec *, PetscScalar *, PetscErrorCode *), PetscErrorCode *ierr)
{
  if (orthog == ksporthogonalizationmodifiedgramschmidt_) {
    *ierr = KSPOrthogonalizationSet(*ksp, KSPOrthogonalizationModifiedGramSchmidt);
  } else if (orthog == ksporthogonalizationclassicalgramschmidt_) {
    *ierr = KSPOrthogonalizationSet(*ksp, KSPOrthogonalizationClassicalGramSchmidt);
  } else {
    *ierr = PetscObjectSetFortranCallback((PetscObject)*ksp, PETSC_FORTRAN_CALLBACK_CLASS, &_cb.orthog, (PetscFortranCallbackFn *)orthog, NULL);
    if (*ierr) return;
    *ierr = KSPOrthogonalizationSet(*ksp, ourorthog);
  }
}
