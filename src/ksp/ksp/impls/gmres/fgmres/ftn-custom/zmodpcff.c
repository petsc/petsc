#include <petsc/private/ftnimpl.h>
#include <petscksp.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
  #define kspfgmressetmodifypc_      KSPFGMRESSETMODIFYPC
  #define kspfgmresmodifypcnochange_ KSPFGMRESMODIFYPCNOCHANGE
  #define kspfgmresmodifypcksp_      KSPFGMRESMODIFYPCKSP
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
  #define kspfgmressetmodifypc_      kspfgmressetmodifypc
  #define kspfgmresmodifypcnochange_ kspfgmresmodifypcnochange
  #define kspfgmresmodifypcksp_      kspfgmresmodifypcksp
#endif

static struct {
  PetscFortranCallbackId modify;
  PetscFortranCallbackId destroy;
} _cb;

static PetscErrorCode ourmodify(KSP ksp, PetscInt i, PetscInt i2, PetscReal d, void *ctx)
{
  PetscObjectUseFortranCallbackSubType(ksp, _cb.modify, (KSP *, PetscInt *, PetscInt *, PetscReal *, void *, PetscErrorCode *), (&ksp, &i, &i2, &d, _ctx, &ierr));
}

static PetscErrorCode ourmoddestroy(void **ctx)
{
  KSP ksp = (KSP)*ctx;
  PetscObjectUseFortranCallbackSubType(ksp, _cb.destroy, (void *, PetscErrorCode *), (_ctx, &ierr));
}

PETSC_EXTERN void kspfgmresmodifypcnochange_(KSP *, PetscInt *, PetscInt *, PetscReal *, void *, PetscErrorCode *);
PETSC_EXTERN void kspfgmresmodifypcksp_(KSP *, PetscInt *, PetscInt *, PetscReal *, void *, PetscErrorCode *);

PETSC_EXTERN void kspfgmressetmodifypc_(KSP *ksp, void (*fcn)(KSP *, PetscInt *, PetscInt *, PetscReal *, void *, PetscErrorCode *), void *ctx, void (*d)(void *, PetscErrorCode *), PetscErrorCode *ierr)
{
  CHKFORTRANNULLFUNCTION(d);
  if ((PetscVoidFn *)fcn == (PetscVoidFn *)kspfgmresmodifypcksp_) {
    *ierr = KSPFGMRESSetModifyPC(*ksp, KSPFGMRESModifyPCKSP, NULL, NULL);
  } else if ((PetscVoidFn *)fcn == (PetscVoidFn *)kspfgmresmodifypcnochange_) {
    *ierr = KSPFGMRESSetModifyPC(*ksp, KSPFGMRESModifyPCNoChange, NULL, NULL);
  } else {
    *ierr = PetscObjectSetFortranCallback((PetscObject)*ksp, PETSC_FORTRAN_CALLBACK_SUBTYPE, &_cb.modify, (PetscVoidFn *)fcn, ctx);
    if (*ierr) return;
    *ierr = PetscObjectSetFortranCallback((PetscObject)*ksp, PETSC_FORTRAN_CALLBACK_SUBTYPE, &_cb.destroy, (PetscVoidFn *)d, ctx);
    if (*ierr) return;
    *ierr = KSPFGMRESSetModifyPC(*ksp, ourmodify, *ksp, ourmoddestroy);
  }
}
