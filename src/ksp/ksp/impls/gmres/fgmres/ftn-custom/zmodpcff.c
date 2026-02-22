#include <petsc/private/ftnimpl.h>
#include <petscksp.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
  #define kspflexiblesetmodifypc_      KSPFLEXIBLESETMODIFYPC
  #define kspflexiblemodifypcnochange_ KSPFLEXIBLEMODIFYPCNOCHANGE
  #define kspflexiblemodifypcksp_      KSPFLEXIBLEMODIFYPCKSP
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
  #define kspflexiblesetmodifypc_      kspflexiblesetmodifypc
  #define kspflexiblemodifypcnochange_ kspflexiblemodifypcnochange
  #define kspflexiblemodifypcksp_      kspflexiblemodifypcksp
#endif

static struct {
  PetscFortranCallbackId modify;
  PetscFortranCallbackId destroy;
} _cb;

static PetscErrorCode ourmodify(KSP ksp, PetscInt i, PetscInt i2, PetscReal d, PetscCtx ctx)
{
  PetscObjectUseFortranCallbackSubType(ksp, _cb.modify, (KSP *, PetscInt *, PetscInt *, PetscReal *, void *, PetscErrorCode *), (&ksp, &i, &i2, &d, _ctx, &ierr));
}

static PetscErrorCode ourmoddestroy(PetscCtxRt ctx)
{
  KSP ksp = *(KSP *)ctx;
  PetscObjectUseFortranCallbackSubType(ksp, _cb.destroy, (void *, PetscErrorCode *), (_ctx, &ierr));
}

PETSC_EXTERN void kspflexiblemodifypcnochange_(KSP *, PetscInt *, PetscInt *, PetscReal *, void *, PetscErrorCode *);
PETSC_EXTERN void kspflexiblemodifypcksp_(KSP *, PetscInt *, PetscInt *, PetscReal *, void *, PetscErrorCode *);

PETSC_EXTERN void kspflexiblesetmodifypc_(KSP *ksp, void (*fcn)(KSP *, PetscInt *, PetscInt *, PetscReal *, void *, PetscErrorCode *), PetscCtx ctx, void (*d)(void *, PetscErrorCode *), PetscErrorCode *ierr)
{
  CHKFORTRANNULLFUNCTION(d);
  if (fcn == kspflexiblemodifypcksp_) {
    *ierr = KSPFlexibleSetModifyPC(*ksp, KSPFlexibleModifyPCKSP, NULL, NULL);
  } else if (fcn == kspflexiblemodifypcnochange_) {
    *ierr = KSPFlexibleSetModifyPC(*ksp, KSPFlexibleModifyPCNoChange, NULL, NULL);
  } else {
    *ierr = PetscObjectSetFortranCallback((PetscObject)*ksp, PETSC_FORTRAN_CALLBACK_SUBTYPE, &_cb.modify, (PetscFortranCallbackFn *)fcn, ctx);
    if (*ierr) return;
    *ierr = PetscObjectSetFortranCallback((PetscObject)*ksp, PETSC_FORTRAN_CALLBACK_SUBTYPE, &_cb.destroy, (PetscFortranCallbackFn *)d, ctx);
    if (*ierr) return;
    *ierr = KSPFlexibleSetModifyPC(*ksp, ourmodify, *ksp, ourmoddestroy);
  }
}
