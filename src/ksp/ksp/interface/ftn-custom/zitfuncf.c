#include <petsc/private/ftnimpl.h>
#include <petscksp.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
  #define kspmonitorset_               KSPMONITORSET
  #define kspconvergeddefaultcreate_   KSPCONVERGEDDEFAULTCREATE
  #define kspconvergeddefaultdestroy_  KSPCONVERGEDDEFAULTDESTROY
  #define kspsetconvergencetest_       KSPSETCONVERGENCETEST
  #define kspconvergeddefault_         KSPCONVERGEDDEFAULT
  #define kspconvergedskip_            KSPCONVERGEDSKIP
  #define kspgmresmonitorkrylov_       KSPGMRESMONITORKRYLOV
  #define kspmonitorresidual_          KSPMONITORRESIDUAL
  #define kspmonitortrueresidual_      KSPMONITORTRUERESIDUAL
  #define kspmonitorsolution_          KSPMONITORSOLUTION
  #define kspmonitorsingularvalue_     KSPMONITORSINGULARVALUE
  #define kspsetcomputerhs_            KSPSETCOMPUTERHS
  #define kspsetcomputeinitialguess_   KSPSETCOMPUTEINITIALGUESS
  #define kspsetcomputeoperators_      KSPSETCOMPUTEOPERATORS
  #define dmkspsetcomputerhs_          DMKSPSETCOMPUTERHS
  #define dmkspsetcomputeinitialguess_ DMKSPSETCOMPUTEINITIALGUESS
  #define dmkspsetcomputeoperators_    DMKSPSETCOMPUTEOPERATORS
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
  #define kspmonitorset_               kspmonitorset
  #define kspconvergeddefaultcreate_   kspconvergeddefaultcreate
  #define kspconvergeddefaultdestroy_  kspconvergeddefaultdestroy
  #define kspsetconvergencetest_       kspsetconvergencetest
  #define kspconvergeddefault_         kspconvergeddefault
  #define kspconvergedskip_            kspconvergedskip
  #define kspgmresmonitorkrylov_       kspgmresmonitorkrylov
  #define kspmonitorresidual_          kspmonitorresidual
  #define kspmonitortrueresidual_      kspmonitortrueresidual
  #define kspmonitorsolution_          kspmonitorsolution
  #define kspmonitorsingularvalue_     kspmonitorsingularvalue
  #define kspsetcomputerhs_            kspsetcomputerhs
  #define kspsetcomputeinitialguess_   kspsetcomputeinitialguess
  #define kspsetcomputeoperators_      kspsetcomputeoperators
  #define dmkspsetcomputerhs_          dmkspsetcomputerhs
  #define dmkspsetcomputeinitialguess_ dmkspsetcomputeinitialguess
  #define dmkspsetcomputeoperators_    dmkspsetcomputeoperators
#endif

/* These are defined in zdmkspf.c */
PETSC_EXTERN void dmkspsetcomputerhs_(DM *dm, void (*func)(KSP *, Vec *, void *, PetscErrorCode *), void *ctx, PetscErrorCode *ierr);
PETSC_EXTERN void dmkspsetcomputeinitialguess_(DM *dm, void (*func)(KSP *, Vec *, void *, PetscErrorCode *), void *ctx, PetscErrorCode *ierr);
PETSC_EXTERN void dmkspsetcomputeoperators_(DM *dm, void (*func)(KSP *, Vec *, void *, PetscErrorCode *), void *ctx, PetscErrorCode *ierr);

/*
        These cannot be called from Fortran but allow Fortran users to transparently set these monitors from .F code
*/

PETSC_EXTERN void kspconvergeddefault_(KSP *, PetscInt *, PetscReal *, KSPConvergedReason *, void *, PetscErrorCode *);
PETSC_EXTERN void kspconvergedskip_(KSP *, PetscInt *, PetscReal *, KSPConvergedReason *, void *, PetscErrorCode *);
PETSC_EXTERN void kspgmresmonitorkrylov_(KSP *, PetscInt *, PetscReal *, PetscViewerAndFormat *, PetscErrorCode *);
PETSC_EXTERN void kspmonitorresidual_(KSP *, PetscInt *, PetscReal *, PetscViewerAndFormat *, PetscErrorCode *);
PETSC_EXTERN void kspmonitorsingularvalue_(KSP *, PetscInt *, PetscReal *, PetscViewerAndFormat *, PetscErrorCode *);
PETSC_EXTERN void kspmonitortrueresidual_(KSP *, PetscInt *, PetscReal *, PetscViewerAndFormat *, PetscErrorCode *);
PETSC_EXTERN void kspmonitorsolution_(KSP *, PetscInt *, PetscReal *, PetscViewerAndFormat *, PetscErrorCode *);

static struct {
  PetscFortranCallbackId monitor;
  PetscFortranCallbackId monitordestroy;
  PetscFortranCallbackId test;
  PetscFortranCallbackId testdestroy;
} _cb;

static PetscErrorCode ourmonitor(KSP ksp, PetscInt i, PetscReal d, void *ctx)
{
  PetscObjectUseFortranCallback(ksp, _cb.monitor, (KSP *, PetscInt *, PetscReal *, void *, PetscErrorCode *), (&ksp, &i, &d, _ctx, &ierr));
}

static PetscErrorCode ourdestroy(void **ctx)
{
  KSP ksp = (KSP)*ctx;
  PetscObjectUseFortranCallback(ksp, _cb.monitordestroy, (void *, PetscErrorCode *), (_ctx, &ierr));
}

/* These are not extern C because they are passed into non-extern C user level functions */
static PetscErrorCode ourtest(KSP ksp, PetscInt i, PetscReal d, KSPConvergedReason *reason, void *ctx)
{
  PetscObjectUseFortranCallback(ksp, _cb.test, (KSP *, PetscInt *, PetscReal *, KSPConvergedReason *, void *, PetscErrorCode *), (&ksp, &i, &d, reason, _ctx, &ierr));
}

static PetscErrorCode ourtestdestroy(void **ctx)
{
  KSP ksp = (KSP)*ctx;
  PetscObjectUseFortranCallback(ksp, _cb.testdestroy, (void **, PetscErrorCode *), (&_ctx, &ierr));
}

/*
   For the built in monitors we ignore the monitordestroy that is passed in and use PetscViewerAndFormatDestroy()
*/
PETSC_EXTERN void kspmonitorset_(KSP *ksp, void (*monitor)(KSP *, PetscInt *, PetscReal *, void *, PetscErrorCode *), void *mctx, void (*monitordestroy)(void *, PetscErrorCode *), PetscErrorCode *ierr)
{
  CHKFORTRANNULLFUNCTION(monitordestroy);

  if ((PetscFortranCallbackFn *)monitor == (PetscFortranCallbackFn *)kspmonitorresidual_) {
    *ierr = KSPMonitorSet(*ksp, (KSPMonitorFn *)KSPMonitorResidual, *(PetscViewerAndFormat **)mctx, (PetscCtxDestroyFn *)PetscViewerAndFormatDestroy);
  } else if ((PetscFortranCallbackFn *)monitor == (PetscFortranCallbackFn *)kspmonitorsolution_) {
    *ierr = KSPMonitorSet(*ksp, (KSPMonitorFn *)KSPMonitorSolution, *(PetscViewerAndFormat **)mctx, (PetscCtxDestroyFn *)PetscViewerAndFormatDestroy);
  } else if ((PetscFortranCallbackFn *)monitor == (PetscFortranCallbackFn *)kspmonitortrueresidual_) {
    *ierr = KSPMonitorSet(*ksp, (KSPMonitorFn *)KSPMonitorTrueResidual, *(PetscViewerAndFormat **)mctx, (PetscCtxDestroyFn *)PetscViewerAndFormatDestroy);
  } else if ((PetscFortranCallbackFn *)monitor == (PetscFortranCallbackFn *)kspmonitorsingularvalue_) {
    *ierr = KSPMonitorSet(*ksp, (KSPMonitorFn *)KSPMonitorSingularValue, *(PetscViewerAndFormat **)mctx, (PetscCtxDestroyFn *)PetscViewerAndFormatDestroy);
  } else if ((PetscFortranCallbackFn *)monitor == (PetscFortranCallbackFn *)kspgmresmonitorkrylov_) {
    *ierr = KSPMonitorSet(*ksp, (KSPMonitorFn *)KSPGMRESMonitorKrylov, *(PetscViewerAndFormat **)mctx, (PetscCtxDestroyFn *)PetscViewerAndFormatDestroy);
  } else {
    *ierr = PetscObjectSetFortranCallback((PetscObject)*ksp, PETSC_FORTRAN_CALLBACK_CLASS, &_cb.monitor, (PetscFortranCallbackFn *)monitor, mctx);
    if (*ierr) return;
    *ierr = PetscObjectSetFortranCallback((PetscObject)*ksp, PETSC_FORTRAN_CALLBACK_CLASS, &_cb.monitordestroy, (PetscFortranCallbackFn *)monitordestroy, mctx);
    if (*ierr) return;
    *ierr = KSPMonitorSet(*ksp, ourmonitor, *ksp, ourdestroy);
  }
}

PETSC_EXTERN void kspconvergeddefaultdestroy_(void **ctx, PetscErrorCode *ierr)
{
  *ierr = KSPConvergedDefaultDestroy(ctx);
}

PETSC_EXTERN void kspsetconvergencetest_(KSP *ksp, void (*converge)(KSP *, PetscInt *, PetscReal *, KSPConvergedReason *, void *, PetscErrorCode *), void **cctx, void (*destroy)(void **, PetscErrorCode *), PetscErrorCode *ierr)
{
  CHKFORTRANNULLFUNCTION(destroy);

  if (converge == kspconvergeddefault_) {
    *ierr = KSPSetConvergenceTest(*ksp, KSPConvergedDefault, &cctx, KSPConvergedDefaultDestroy);
  } else if (converge == kspconvergedskip_) {
    *ierr = KSPSetConvergenceTest(*ksp, KSPConvergedSkip, NULL, NULL);
  } else {
    if (destroy == kspconvergeddefaultdestroy_) cctx = *(void ***)cctx;
    *ierr = PetscObjectSetFortranCallback((PetscObject)*ksp, PETSC_FORTRAN_CALLBACK_CLASS, &_cb.test, (PetscFortranCallbackFn *)converge, cctx);
    if (*ierr) return;
    *ierr = PetscObjectSetFortranCallback((PetscObject)*ksp, PETSC_FORTRAN_CALLBACK_CLASS, &_cb.testdestroy, (PetscFortranCallbackFn *)destroy, cctx);
    if (*ierr) return;
    *ierr = KSPSetConvergenceTest(*ksp, ourtest, *ksp, ourtestdestroy);
  }
}

PETSC_EXTERN void kspsetcomputerhs_(KSP *ksp, void (*func)(KSP *, Vec *, void *, PetscErrorCode *), void *ctx, PetscErrorCode *ierr)
{
  DM dm;
  *ierr = KSPGetDM(*ksp, &dm);
  if (!*ierr) dmkspsetcomputerhs_(&dm, func, ctx, ierr);
}

PETSC_EXTERN void kspsetcomputeinitialguess_(KSP *ksp, void (*func)(KSP *, Vec *, void *, PetscErrorCode *), void *ctx, PetscErrorCode *ierr)
{
  DM dm;
  *ierr = KSPGetDM(*ksp, &dm);
  if (!*ierr) dmkspsetcomputeinitialguess_(&dm, func, ctx, ierr);
}

PETSC_EXTERN void kspsetcomputeoperators_(KSP *ksp, void (*func)(KSP *, Vec *, void *, PetscErrorCode *), void *ctx, PetscErrorCode *ierr)
{
  DM dm;
  *ierr = KSPGetDM(*ksp, &dm);
  if (!*ierr) dmkspsetcomputeoperators_(&dm, func, ctx, ierr);
}

PETSC_EXTERN void kspconvergeddefaultcreate_(PetscFortranAddr *ctx, PetscErrorCode *ierr)
{
  *ierr = KSPConvergedDefaultCreate((void **)ctx);
}
