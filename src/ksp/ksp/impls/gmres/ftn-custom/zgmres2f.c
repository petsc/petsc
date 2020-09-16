#include <petsc/private/fortranimpl.h>
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

PETSC_EXTERN void kspgmresmodifiedgramschmidtorthogonalization_(KSP *ksp,PetscInt *n,PetscErrorCode *ierr)
{
  *ierr = KSPGMRESModifiedGramSchmidtOrthogonalization(*ksp,*n);
}

PETSC_EXTERN void kspgmresclassicalgramschmidtorthogonalization_(KSP *ksp,PetscInt *n,PetscErrorCode *ierr)
{
  *ierr = KSPGMRESClassicalGramSchmidtOrthogonalization(*ksp,*n);
}

static PetscErrorCode ourorthog(KSP ksp,PetscInt n)
{
  PetscObjectUseFortranCallback(ksp,_cb.orthog,(KSP*,PetscInt*,PetscErrorCode*),(&ksp,&n,&ierr));
}

PETSC_EXTERN void kspgmressetorthogonalization_(KSP *ksp,void (*orthog)(KSP*,PetscInt*,PetscErrorCode*),PetscErrorCode *ierr)
{

  if ((PetscVoidFunction)orthog == (PetscVoidFunction)kspgmresmodifiedgramschmidtorthogonalization_) {
    *ierr = KSPGMRESSetOrthogonalization(*ksp,KSPGMRESModifiedGramSchmidtOrthogonalization);
  } else if ((PetscVoidFunction)orthog == (PetscVoidFunction)kspgmresclassicalgramschmidtorthogonalization_) {
    *ierr = KSPGMRESSetOrthogonalization(*ksp,KSPGMRESClassicalGramSchmidtOrthogonalization);
  } else {
    *ierr = PetscObjectSetFortranCallback((PetscObject)*ksp,PETSC_FORTRAN_CALLBACK_CLASS,&_cb.orthog,(PetscVoidFunction)orthog,NULL); if (*ierr) return;
    *ierr = KSPGMRESSetOrthogonalization(*ksp,ourorthog);
  }
}

