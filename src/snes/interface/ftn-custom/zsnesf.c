#include <petsc/private/ftnimpl.h>
#include <petscsnes.h>
#include <petscviewer.h>
#include <petsc/private/ftnimpl.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
  #define snessetpicard_                   SNESSETPICARD
  #define snessetpicardnointerface_        SNESSETPICARDNOINTERFACE
  #define snessolve_                       SNESSOLVE
  #define snescomputejacobiandefault_      SNESCOMPUTEJACOBIANDEFAULT
  #define snescomputejacobiandefaultcolor_ SNESCOMPUTEJACOBIANDEFAULTCOLOR
  #define snessetjacobian_                 SNESSETJACOBIAN
  #define snessetjacobiannointerface_      SNESSETJACOBIANNOINTERFACE
  #define snessetfunction_                 SNESSETFUNCTION
  #define snessetfunctionnointerface_      SNESSETFUNCTIONNOINTERFACE
  #define snessetobjective_                SNESSETOBJECTIVE
  #define snessetobjectivenointerface_     SNESSETOBJECTIVENOINTERFACE
  #define snessetngs_                      SNESSETNGS
  #define snessetupdate_                   SNESSETUPDATE
  #define snesgetfunction_                 SNESGETFUNCTION
  #define snesgetngs_                      SNESGETNGS
  #define snessetconvergencetest_          SNESSETCONVERGENCETEST
  #define snesconvergeddefault_            SNESCONVERGEDDEFAULT
  #define snesconvergedskip_               SNESCONVERGEDSKIP
  #define snesgetjacobian_                 SNESGETJACOBIAN
  #define snesmonitordefault_              SNESMONITORDEFAULT
  #define snesmonitorsolution_             SNESMONITORSOLUTION
  #define snesmonitorsolutionupdate_       SNESMONITORSOLUTIONUPDATE
  #define snesmonitorset_                  SNESMONITORSET
  #define snesnewtontrsetprecheck_         SNESNEWTONTRSETPRECHECK
  #define snesnewtontrsetpostcheck_        SNESNEWTONTRSETPOSTCHECK
  #define snesnewtontrdcsetprecheck_       SNESNEWTONTRDCSETPRECHECK
  #define snesnewtontrdcsetpostcheck_      SNESNEWTONTRDCSETPOSTCHECK
  #define matmffdcomputejacobian_          MATMFFDCOMPUTEJACOBIAN
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
  #define snessetpicard_                   snessetpicard
  #define snessetpicardnointerface_        snessetpicardnointerface
  #define snessolve_                       snessolve
  #define snescomputejacobiandefault_      snescomputejacobiandefault
  #define snescomputejacobiandefaultcolor_ snescomputejacobiandefaultcolor
  #define snessetjacobian_                 snessetjacobian
  #define snessetjacobiannointerface_      snessetjacobiannointerface
  #define snessetfunction_                 snessetfunction
  #define snessetfunctionnointerface_      snessetfunctionnointerface
  #define snessetobjective_                snessetobjective
  #define snessetobjectivenointerface_     snessetobjectivenointerface
  #define snessetngs_                      snessetngs
  #define snessetupdate_                   snessetupdate
  #define snesgetfunction_                 snesgetfunction
  #define snesgetngs_                      snesgetngs
  #define snessetconvergencetest_          snessetconvergencetest
  #define snesconvergeddefault_            snesconvergeddefault
  #define snesconvergedskip_               snesconvergedskip
  #define snesgetjacobian_                 snesgetjacobian
  #define snesmonitordefault_              snesmonitordefault
  #define snesmonitorsolution_             snesmonitorsolution
  #define snesmonitorsolutionupdate_       snesmonitorsolutionupdate
  #define snesmonitorset_                  snesmonitorset
  #define snesnewtontrsetprecheck_         snesnewtontrsetprecheck
  #define snesnewtontrsetpostcheck_        snesnewtontrsetpostcheck
  #define snesnewtontrdcsetprecheck_       snesnewtontrdcsetprecheck
  #define snesnewtontrdcsetpostcheck_      snesnewtontrdcsetpostcheck
  #define matmffdcomputejacobian_          matmffdcomputejacobian
#endif

static struct {
  PetscFortranCallbackId function;
  PetscFortranCallbackId objective;
  PetscFortranCallbackId test;
  PetscFortranCallbackId destroy;
  PetscFortranCallbackId jacobian;
  PetscFortranCallbackId monitor;
  PetscFortranCallbackId mondestroy;
  PetscFortranCallbackId ngs;
  PetscFortranCallbackId update;
  PetscFortranCallbackId trprecheck;
  PetscFortranCallbackId trpostcheck;
#if defined(PETSC_HAVE_F90_2PTR_ARG)
  PetscFortranCallbackId function_pgiptr;
  PetscFortranCallbackId objective_pgiptr;
  PetscFortranCallbackId trprecheck_pgiptr;
  PetscFortranCallbackId trpostcheck_pgiptr;
#endif
} _cb;

static PetscErrorCode ourtrprecheckfunction(SNES snes, Vec x, Vec y, PetscBool *changed_y, void *ctx)
{
#if defined(PETSC_HAVE_F90_2PTR_ARG)
  void *ptr;
  PetscCall(PetscObjectGetFortranCallback((PetscObject)snes, PETSC_FORTRAN_CALLBACK_CLASS, _cb.trprecheck_pgiptr, NULL, &ptr));
#endif
  PetscObjectUseFortranCallback(snes, _cb.trprecheck, (SNES *, Vec *, Vec *, PetscBool *, void *, PetscErrorCode *PETSC_F90_2PTR_PROTO_NOVAR), (&snes, &x, &y, changed_y, _ctx, &ierr PETSC_F90_2PTR_PARAM(ptr)));
}

PETSC_EXTERN void snesnewtontrsetprecheck_(SNES *snes, void (*func)(SNES, Vec, Vec, PetscBool *, void *), void *ctx, PetscErrorCode *ierr PETSC_F90_2PTR_PROTO(ptr))
{
  *ierr = PetscObjectSetFortranCallback((PetscObject)*snes, PETSC_FORTRAN_CALLBACK_CLASS, &_cb.trprecheck, (PetscFortranCallbackFn *)func, ctx);
  if (*ierr) return;
#if defined(PETSC_HAVE_F90_2PTR_ARG)
  *ierr = PetscObjectSetFortranCallback((PetscObject)*snes, PETSC_FORTRAN_CALLBACK_CLASS, &_cb.trprecheck_pgiptr, NULL, ptr);
  if (*ierr) return;
#endif
  *ierr = SNESNewtonTRSetPreCheck(*snes, ourtrprecheckfunction, NULL);
}

PETSC_EXTERN void snesnewtontrdcsetprecheck_(SNES *snes, void (*func)(SNES, Vec, Vec, PetscBool *, void *), void *ctx, PetscErrorCode *ierr PETSC_F90_2PTR_PROTO(ptr))
{
  *ierr = PetscObjectSetFortranCallback((PetscObject)*snes, PETSC_FORTRAN_CALLBACK_CLASS, &_cb.trprecheck, (PetscFortranCallbackFn *)func, ctx);
  if (*ierr) return;
#if defined(PETSC_HAVE_F90_2PTR_ARG)
  *ierr = PetscObjectSetFortranCallback((PetscObject)*snes, PETSC_FORTRAN_CALLBACK_CLASS, &_cb.trprecheck_pgiptr, NULL, ptr);
  if (*ierr) return;
#endif
  *ierr = SNESNewtonTRDCSetPreCheck(*snes, ourtrprecheckfunction, NULL);
}

static PetscErrorCode ourtrpostcheckfunction(SNES snes, Vec x, Vec y, Vec w, PetscBool *changed_y, PetscBool *changed_w, void *ctx)
{
#if defined(PETSC_HAVE_F90_2PTR_ARG)
  void *ptr;
  PetscCall(PetscObjectGetFortranCallback((PetscObject)snes, PETSC_FORTRAN_CALLBACK_CLASS, _cb.trpostcheck_pgiptr, NULL, &ptr));
#endif
  PetscObjectUseFortranCallback(snes, _cb.trpostcheck, (SNES *, Vec *, Vec *, Vec *, PetscBool *, PetscBool *, void *, PetscErrorCode *PETSC_F90_2PTR_PROTO_NOVAR), (&snes, &x, &y, &w, changed_y, changed_w, _ctx, &ierr PETSC_F90_2PTR_PARAM(ptr)));
}

PETSC_EXTERN void snesnewtontrsetpostcheck_(SNES *snes, void (*func)(SNES, Vec, Vec, Vec, PetscBool *, PetscBool *, void *), void *ctx, PetscErrorCode *ierr PETSC_F90_2PTR_PROTO(ptr))
{
  *ierr = PetscObjectSetFortranCallback((PetscObject)*snes, PETSC_FORTRAN_CALLBACK_CLASS, &_cb.trpostcheck, (PetscFortranCallbackFn *)func, ctx);
  if (*ierr) return;
#if defined(PETSC_HAVE_F90_2PTR_ARG)
  *ierr = PetscObjectSetFortranCallback((PetscObject)*snes, PETSC_FORTRAN_CALLBACK_CLASS, &_cb.trpostcheck_pgiptr, NULL, ptr);
  if (*ierr) return;
#endif
  *ierr = SNESNewtonTRSetPostCheck(*snes, ourtrpostcheckfunction, NULL);
}

PETSC_EXTERN void snesnewtontrdcsetpostcheck_(SNES *snes, void (*func)(SNES, Vec, Vec, Vec, PetscBool *, PetscBool *, void *), void *ctx, PetscErrorCode *ierr PETSC_F90_2PTR_PROTO(ptr))
{
  *ierr = PetscObjectSetFortranCallback((PetscObject)*snes, PETSC_FORTRAN_CALLBACK_CLASS, &_cb.trpostcheck, (PetscFortranCallbackFn *)func, ctx);
  if (*ierr) return;
#if defined(PETSC_HAVE_F90_2PTR_ARG)
  *ierr = PetscObjectSetFortranCallback((PetscObject)*snes, PETSC_FORTRAN_CALLBACK_CLASS, &_cb.trpostcheck_pgiptr, NULL, ptr);
  if (*ierr) return;
#endif
  *ierr = SNESNewtonTRDCSetPostCheck(*snes, ourtrpostcheckfunction, NULL);
}

static PetscErrorCode oursnesfunction(SNES snes, Vec x, Vec f, void *ctx)
{
#if defined(PETSC_HAVE_F90_2PTR_ARG)
  void *ptr;
  PetscCall(PetscObjectGetFortranCallback((PetscObject)snes, PETSC_FORTRAN_CALLBACK_CLASS, _cb.function_pgiptr, NULL, &ptr));
#endif
  PetscObjectUseFortranCallback(snes, _cb.function, (SNES *, Vec *, Vec *, void *, PetscErrorCode *PETSC_F90_2PTR_PROTO_NOVAR), (&snes, &x, &f, _ctx, &ierr PETSC_F90_2PTR_PARAM(ptr)));
}

static PetscErrorCode oursnesobjective(SNES snes, Vec x, PetscReal *v, void *ctx)
{
#if defined(PETSC_HAVE_F90_2PTR_ARG)
  void *ptr;
  PetscCall(PetscObjectGetFortranCallback((PetscObject)snes, PETSC_FORTRAN_CALLBACK_CLASS, _cb.objective_pgiptr, NULL, &ptr));
#endif
  PetscObjectUseFortranCallback(snes, _cb.objective, (SNES *, Vec *, PetscReal *, void *, PetscErrorCode *PETSC_F90_2PTR_PROTO_NOVAR), (&snes, &x, v, _ctx, &ierr PETSC_F90_2PTR_PARAM(ptr)));
}

static PetscErrorCode oursnestest(SNES snes, PetscInt it, PetscReal a, PetscReal d, PetscReal c, SNESConvergedReason *reason, void *ctx)
{
  PetscObjectUseFortranCallback(snes, _cb.test, (SNES *, PetscInt *, PetscReal *, PetscReal *, PetscReal *, SNESConvergedReason *, void *, PetscErrorCode *), (&snes, &it, &a, &d, &c, reason, _ctx, &ierr));
}

static PetscErrorCode ourdestroy(void *ctx)
{
  PetscObjectUseFortranCallback(ctx, _cb.destroy, (void *, PetscErrorCode *), (_ctx, &ierr));
}

static PetscErrorCode oursnesjacobian(SNES snes, Vec x, Mat m, Mat p, void *ctx)
{
  PetscObjectUseFortranCallback(snes, _cb.jacobian, (SNES *, Vec *, Mat *, Mat *, void *, PetscErrorCode *), (&snes, &x, &m, &p, _ctx, &ierr));
}

static PetscErrorCode oursnesupdate(SNES snes, PetscInt i)
{
  PetscObjectUseFortranCallback(snes, _cb.update, (SNES *, PetscInt *, PetscErrorCode *), (&snes, &i, &ierr));
}
static PetscErrorCode oursnesngs(SNES snes, Vec x, Vec b, void *ctx)
{
  PetscObjectUseFortranCallback(snes, _cb.ngs, (SNES *, Vec *, Vec *, void *, PetscErrorCode *), (&snes, &x, &b, _ctx, &ierr));
}
static PetscErrorCode oursnesmonitor(SNES snes, PetscInt i, PetscReal d, void *ctx)
{
  PetscObjectUseFortranCallback(snes, _cb.monitor, (SNES *, PetscInt *, PetscReal *, void *, PetscErrorCode *), (&snes, &i, &d, _ctx, &ierr));
}
static PetscErrorCode ourmondestroy(void **ctx)
{
  SNES snes = (SNES)*ctx;
  PetscObjectUseFortranCallback(snes, _cb.mondestroy, (void *, PetscErrorCode *), (_ctx, &ierr));
}

PETSC_EXTERN void snescomputejacobiandefault_(SNES *, Vec *, Mat *, Mat *, void *, PetscErrorCode *);
PETSC_EXTERN void snescomputejacobiandefaultcolor_(SNES *, Vec *, Mat *, Mat *, void *, PetscErrorCode *);
PETSC_EXTERN void matmffdcomputejacobian_(SNES *, Vec *, Mat *, Mat *, void *, PetscErrorCode *);

PETSC_EXTERN void snessetjacobian_(SNES *snes, Mat *A, Mat *B, void (*func)(SNES *, Vec *, Mat *, Mat *, void *, PetscErrorCode *), void *ctx, PetscErrorCode *ierr PETSC_F90_2PTR_PROTO(ptr))
{
  CHKFORTRANNULLFUNCTION(func);
  if (func == snescomputejacobiandefault_) {
    *ierr = SNESSetJacobian(*snes, *A, *B, SNESComputeJacobianDefault, ctx);
  } else if (func == snescomputejacobiandefaultcolor_) {
    if (!ctx) {
      *ierr = PETSC_ERR_ARG_NULL;
      return;
    }
    *ierr = SNESSetJacobian(*snes, *A, *B, SNESComputeJacobianDefaultColor, *(MatFDColoring *)ctx);
  } else if (func == matmffdcomputejacobian_) {
    *ierr = SNESSetJacobian(*snes, *A, *B, MatMFFDComputeJacobian, ctx);
  } else {
    *ierr = PetscObjectSetFortranCallback((PetscObject)*snes, PETSC_FORTRAN_CALLBACK_CLASS, &_cb.jacobian, (PetscFortranCallbackFn *)func, ctx);
    if (!*ierr) *ierr = SNESSetJacobian(*snes, *A, *B, oursnesjacobian, NULL);
  }
}

PETSC_EXTERN void snessetjacobiannointerface_(SNES *snes, Mat *A, Mat *B, void (*func)(SNES *, Vec *, Mat *, Mat *, void *, PetscErrorCode *), void *ctx, PetscErrorCode *ierr PETSC_F90_2PTR_PROTO(ptr))
{
  snessetjacobian_(snes, A, B, func, ctx, ierr PETSC_F90_2PTR_PARAM(ptr));
}

/*  func is currently ignored from Fortran */
PETSC_EXTERN void snesgetjacobian_(SNES *snes, Mat *A, Mat *B, int *func, void **ctx, PetscErrorCode *ierr)
{
  SNESJacobianFn *jfunc;
  void           *jctx;

  CHKFORTRANNULL(ctx);
  CHKFORTRANNULLOBJECT(A);
  CHKFORTRANNULLOBJECT(B);
  *ierr = SNESGetJacobian(*snes, A, B, &jfunc, &jctx);
  if (*ierr) return;
  if (jfunc == SNESComputeJacobianDefault || jfunc == SNESComputeJacobianDefaultColor || jfunc == MatMFFDComputeJacobian) {
    if (ctx) *ctx = jctx;
  } else {
    *ierr = PetscObjectGetFortranCallback((PetscObject)*snes, PETSC_FORTRAN_CALLBACK_CLASS, _cb.jacobian, NULL, ctx);
  }
}

static PetscErrorCode oursnespicardfunction(SNES snes, Vec x, Vec f, void *ctx)
{
#if defined(PETSC_HAVE_F90_2PTR_ARG)
  void *ptr;
  PetscCall(PetscObjectGetFortranCallback((PetscObject)snes, PETSC_FORTRAN_CALLBACK_CLASS, _cb.function_pgiptr, NULL, &ptr));
#endif
  PetscObjectUseFortranCallback(snes, _cb.function, (SNES *, Vec *, Vec *, void *, PetscErrorCode *PETSC_F90_2PTR_PROTO_NOVAR), (&snes, &x, &f, _ctx, &ierr PETSC_F90_2PTR_PARAM(ptr)));
}

static PetscErrorCode oursnespicardjacobian(SNES snes, Vec x, Mat m, Mat p, void *ctx)
{
  PetscObjectUseFortranCallback(snes, _cb.jacobian, (SNES *, Vec *, Mat *, Mat *, void *, PetscErrorCode *), (&snes, &x, &m, &p, _ctx, &ierr));
}

PETSC_EXTERN void snessetpicard_(SNES *snes, Vec *r, void (*func)(SNES, Vec, Vec, void *, PetscErrorCode *), Mat *A, Mat *B, void (*J)(SNES *, Vec *, Mat *, Mat *, void *, PetscErrorCode *), void *ctx, PetscErrorCode *ierr PETSC_F90_2PTR_PROTO(ptr))
{
  *ierr = PetscObjectSetFortranCallback((PetscObject)*snes, PETSC_FORTRAN_CALLBACK_CLASS, &_cb.function, (PetscFortranCallbackFn *)func, ctx);
#if defined(PETSC_HAVE_F90_2PTR_ARG)
  *ierr = PetscObjectSetFortranCallback((PetscObject)*snes, PETSC_FORTRAN_CALLBACK_CLASS, &_cb.function_pgiptr, NULL, ptr);
  if (*ierr) return;
#endif
  *ierr = PetscObjectSetFortranCallback((PetscObject)*snes, PETSC_FORTRAN_CALLBACK_CLASS, &_cb.jacobian, (PetscFortranCallbackFn *)J, ctx);
  if (!*ierr) *ierr = SNESSetPicard(*snes, *r, oursnespicardfunction, *A, *B, oursnespicardjacobian, NULL);
}

PETSC_EXTERN void snessetpicardnointerface_(SNES *snes, Vec *r, void (*func)(SNES, Vec, Vec, void *, PetscErrorCode *), Mat *A, Mat *B, void (*J)(SNES *, Vec *, Mat *, Mat *, void *, PetscErrorCode *), void *ctx, PetscErrorCode *ierr PETSC_F90_2PTR_PROTO(ptr))
{
  snessetpicard_(snes, r, func, A, B, J, ctx, ierr PETSC_F90_2PTR_PARAM(ptr));
}

/*
   These are not usually called from Fortran but allow Fortran users
   to transparently set these monitors from .F code
*/

PETSC_EXTERN void snessetfunction_(SNES *snes, Vec *r, void (*func)(SNES, Vec, Vec, void *, PetscErrorCode *), void *ctx, PetscErrorCode *ierr PETSC_F90_2PTR_PROTO(ptr))
{
  *ierr = PetscObjectSetFortranCallback((PetscObject)*snes, PETSC_FORTRAN_CALLBACK_CLASS, &_cb.function, (PetscFortranCallbackFn *)func, ctx);
  if (*ierr) return;
#if defined(PETSC_HAVE_F90_2PTR_ARG)
  *ierr = PetscObjectSetFortranCallback((PetscObject)*snes, PETSC_FORTRAN_CALLBACK_CLASS, &_cb.function_pgiptr, NULL, ptr);
  if (*ierr) return;
#endif
  *ierr = SNESSetFunction(*snes, *r, oursnesfunction, NULL);
}

PETSC_EXTERN void snessetfunctionnointerface_(SNES *snes, Vec *r, void (*func)(SNES, Vec, Vec, void *, PetscErrorCode *), void *ctx, PetscErrorCode *ierr PETSC_F90_2PTR_PROTO(ptr))
{
  snessetfunction_(snes, r, func, ctx, ierr PETSC_F90_2PTR_PARAM(ptr));
}

PETSC_EXTERN void snessetobjective_(SNES *snes, SNESObjectiveFn func, void *ctx, PetscErrorCode *ierr PETSC_F90_2PTR_PROTO(ptr))
{
  *ierr = PetscObjectSetFortranCallback((PetscObject)*snes, PETSC_FORTRAN_CALLBACK_CLASS, &_cb.objective, (PetscFortranCallbackFn *)func, ctx);
  if (*ierr) return;
#if defined(PETSC_HAVE_F90_2PTR_ARG)
  *ierr = PetscObjectSetFortranCallback((PetscObject)*snes, PETSC_FORTRAN_CALLBACK_CLASS, &_cb.objective_pgiptr, NULL, ptr);
  if (*ierr) return;
#endif
  *ierr = SNESSetObjective(*snes, oursnesobjective, NULL);
}

PETSC_EXTERN void snessetobjectivenointerface_(SNES *snes, SNESObjectiveFn func, void *ctx, PetscErrorCode *ierr PETSC_F90_2PTR_PROTO(ptr))
{
  snessetobjective_(snes, func, ctx, ierr PETSC_F90_2PTR_PARAM(ptr));
}

PETSC_EXTERN void snessetngs_(SNES *snes, void (*func)(SNES *, Vec *, Vec *, void *, PetscErrorCode *), void *ctx, PetscErrorCode *ierr)
{
  *ierr = PetscObjectSetFortranCallback((PetscObject)*snes, PETSC_FORTRAN_CALLBACK_CLASS, &_cb.ngs, (PetscFortranCallbackFn *)func, ctx);
  if (*ierr) return;
  *ierr = SNESSetNGS(*snes, oursnesngs, NULL);
}
PETSC_EXTERN void snessetupdate_(SNES *snes, void (*func)(SNES *, PetscInt *, PetscErrorCode *), PetscErrorCode *ierr)
{
  *ierr = PetscObjectSetFortranCallback((PetscObject)*snes, PETSC_FORTRAN_CALLBACK_CLASS, &_cb.update, (PetscFortranCallbackFn *)func, NULL);
  if (*ierr) return;
  *ierr = SNESSetUpdate(*snes, oursnesupdate);
}

/* the func argument is ignored */
PETSC_EXTERN void snesgetfunction_(SNES *snes, Vec *r, void (*func)(SNES, Vec, Vec, void *, PetscErrorCode *), void **ctx, PetscErrorCode *ierr)
{
  CHKFORTRANNULLOBJECT(r);
  *ierr = SNESGetFunction(*snes, r, NULL, NULL);
  if (*ierr) return;
  if ((PetscFortranCallbackFn *)func == (PetscFortranCallbackFn *)PETSC_NULL_FUNCTION_Fortran) return;
  *ierr = PetscObjectGetFortranCallback((PetscObject)*snes, PETSC_FORTRAN_CALLBACK_CLASS, _cb.function, NULL, ctx);
}

PETSC_EXTERN void snesgetngs_(SNES *snes, void *func, void **ctx, PetscErrorCode *ierr)
{
  *ierr = PetscObjectGetFortranCallback((PetscObject)*snes, PETSC_FORTRAN_CALLBACK_CLASS, _cb.ngs, NULL, ctx);
}

PETSC_EXTERN void snesconvergeddefault_(SNES *, PetscInt *, PetscReal *, PetscReal *, PetscReal *, SNESConvergedReason *, void *, PetscErrorCode *);
PETSC_EXTERN void snesconvergedskip_(SNES *, PetscInt *, PetscReal *, PetscReal *, PetscReal *, SNESConvergedReason *, void *, PetscErrorCode *);

PETSC_EXTERN void snessetconvergencetest_(SNES *snes, void (*func)(SNES *, PetscInt *, PetscReal *, PetscReal *, PetscReal *, SNESConvergedReason *, void *, PetscErrorCode *), void *cctx, void (*destroy)(void *), PetscErrorCode *ierr)
{
  CHKFORTRANNULLFUNCTION(destroy);

  if (func == snesconvergeddefault_) {
    *ierr = SNESSetConvergenceTest(*snes, SNESConvergedDefault, NULL, NULL);
  } else if (func == snesconvergedskip_) {
    *ierr = SNESSetConvergenceTest(*snes, SNESConvergedSkip, NULL, NULL);
  } else {
    *ierr = PetscObjectSetFortranCallback((PetscObject)*snes, PETSC_FORTRAN_CALLBACK_CLASS, &_cb.test, (PetscFortranCallbackFn *)func, cctx);
    if (*ierr) return;
    *ierr = PetscObjectSetFortranCallback((PetscObject)*snes, PETSC_FORTRAN_CALLBACK_CLASS, &_cb.destroy, (PetscFortranCallbackFn *)destroy, cctx);
    if (*ierr) return;
    *ierr = SNESSetConvergenceTest(*snes, oursnestest, *snes, ourdestroy);
  }
}

PETSC_EXTERN void snesmonitordefault_(SNES *, PetscInt *, PetscReal *, PetscViewerAndFormat **, PetscErrorCode *);

PETSC_EXTERN void snesmonitorsolution_(SNES *snes, PetscInt *its, PetscReal *fgnorm, PetscViewerAndFormat **dummy, PetscErrorCode *ierr);

PETSC_EXTERN void snesmonitorsolutionupdate_(SNES *snes, PetscInt *its, PetscReal *fgnorm, PetscViewerAndFormat **dummy, PetscErrorCode *ierr);

PETSC_EXTERN void snesmonitorset_(SNES *snes, void (*func)(SNES *, PetscInt *, PetscReal *, void *, PetscErrorCode *), void *mctx, void (*mondestroy)(void *, PetscErrorCode *), PetscErrorCode *ierr)
{
  CHKFORTRANNULLFUNCTION(mondestroy);
  if ((PetscFortranCallbackFn *)func == (PetscFortranCallbackFn *)snesmonitordefault_) {
    *ierr = SNESMonitorSet(*snes, (PetscErrorCode (*)(SNES, PetscInt, PetscReal, void *))SNESMonitorDefault, *(PetscViewerAndFormat **)mctx, (PetscCtxDestroyFn *)PetscViewerAndFormatDestroy);
  } else if ((PetscFortranCallbackFn *)func == (PetscFortranCallbackFn *)snesmonitorsolution_) {
    *ierr = SNESMonitorSet(*snes, (PetscErrorCode (*)(SNES, PetscInt, PetscReal, void *))SNESMonitorSolution, *(PetscViewerAndFormat **)mctx, (PetscCtxDestroyFn *)PetscViewerAndFormatDestroy);
  } else if ((PetscFortranCallbackFn *)func == (PetscFortranCallbackFn *)snesmonitorsolutionupdate_) {
    *ierr = SNESMonitorSet(*snes, (PetscErrorCode (*)(SNES, PetscInt, PetscReal, void *))SNESMonitorSolutionUpdate, *(PetscViewerAndFormat **)mctx, (PetscCtxDestroyFn *)PetscViewerAndFormatDestroy);
  } else {
    *ierr = PetscObjectSetFortranCallback((PetscObject)*snes, PETSC_FORTRAN_CALLBACK_CLASS, &_cb.monitor, (PetscFortranCallbackFn *)func, mctx);
    if (*ierr) return;
    *ierr = PetscObjectSetFortranCallback((PetscObject)*snes, PETSC_FORTRAN_CALLBACK_CLASS, &_cb.mondestroy, (PetscFortranCallbackFn *)mondestroy, mctx);
    if (*ierr) return;
    *ierr = SNESMonitorSet(*snes, oursnesmonitor, *snes, ourmondestroy);
  }
}
