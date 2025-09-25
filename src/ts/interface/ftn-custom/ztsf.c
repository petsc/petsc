#include <petsc/private/ftnimpl.h>
#include <petscts.h>
#include <petscviewer.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
  #define tsmonitorlgsettransform_      TSMONITORLGSETTRANSFORM
  #define tssetrhsfunction_             TSSETRHSFUNCTION
  #define tsgetrhsfunction_             TSGETRHSFUNCTION
  #define tssetrhsjacobian_             TSSETRHSJACOBIAN
  #define tsgetrhsjacobian_             TSGETRHSJACOBIAN
  #define tssetifunction_               TSSETIFUNCTION
  #define tsgetifunction_               TSGETIFUNCTION
  #define tssetijacobian_               TSSETIJACOBIAN
  #define tsgetijacobian_               TSGETIJACOBIAN
  #define tsmonitorset_                 TSMONITORSET
  #define tssetrhsjacobianp_            TSSETRHSJACOBIANP
  #define tsgetrhsjacobianp_            TSGETRHSJACOBIANP
  #define tssetijacobianp_              TSSETIJACOBIANP
  #define tsgetijacobianp_              TSGETIJACOBIANP
  #define tscomputerhsfunctionlinear_   TSCOMPUTERHSFUNCTIONLINEAR
  #define tscomputerhsjacobianconstant_ TSCOMPUTERHSJACOBIANCONSTANT
  #define tscomputeifunctionlinear_     TSCOMPUTEIFUNCTIONLINEAR
  #define tscomputeijacobianconstant_   TSCOMPUTEIJACOBIANCONSTANT
  #define tsmonitordefault_             TSMONITORDEFAULT
  #define tssetprestep_                 TSSETPRESTEP
  #define tssetpoststep_                TSSETPOSTSTEP
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
  #define tsmonitorlgsettransform_      tsmonitorlgsettransform
  #define tssetrhsfunction_             tssetrhsfunction
  #define tsgetrhsfunction_             tsgetrhsfunction
  #define tssetrhsjacobian_             tssetrhsjacobian
  #define tsgetrhsjacobian_             tsgetrhsjacobian
  #define tssetifunction_               tssetifunction
  #define tsgetifunction_               tsgetifunction
  #define tssetijacobian_               tssetijacobian
  #define tsgetijacobian_               tsgetijacobian
  #define tssetijacobianp_              tssetijacobianp
  #define tsgetijacobianp_              tsgetijacobianp
  #define tssetrhsjacobianp_            tssetrhsjacobianp
  #define tsgetrhsjacobianp_            tsgetrhsjacobianp
  #define tsmonitorset_                 tsmonitorset
  #define tscomputerhsfunctionlinear_   tscomputerhsfunctionlinear
  #define tscomputerhsjacobianconstant_ tscomputerhsjacobianconstant
  #define tscomputeifunctionlinear_     tscomputeifunctionlinear
  #define tscomputeijacobianconstant_   tscomputeijacobianconstant
  #define tsmonitordefault_             tsmonitordefault
  #define tssetprestep_                 tssetprestep
  #define tssetpoststep_                tssetpoststep
#endif

static struct {
  PetscFortranCallbackId prestep;
  PetscFortranCallbackId poststep;
  PetscFortranCallbackId rhsfunction;
  PetscFortranCallbackId rhsjacobian;
  PetscFortranCallbackId ifunction;
  PetscFortranCallbackId ijacobian;
  PetscFortranCallbackId rhsjacobianp;
  PetscFortranCallbackId ijacobianp;
  PetscFortranCallbackId monitor;
  PetscFortranCallbackId mondestroy;
  PetscFortranCallbackId transform;
#if defined(PETSC_HAVE_F90_2PTR_ARG)
  PetscFortranCallbackId function_pgiptr;
#endif
} _cb;

static PetscErrorCode ourprestep(TS ts)
{
#if defined(PETSC_HAVE_F90_2PTR_ARG) && defined(foo)
  void *ptr;
  PetscCall(PetscObjectGetFortranCallback((PetscObject)ts, PETSC_FORTRAN_CALLBACK_CLASS, _cb.function_pgiptr, NULL, &ptr));
#endif
  PetscObjectUseFortranCallback(ts, _cb.prestep, (TS *, PetscErrorCode * /* PETSC_F90_2PTR_PROTO_NOVAR */), (&ts, &ierr /* PETSC_F90_2PTR_PARAM(ptr) */));
}
static PetscErrorCode ourpoststep(TS ts)
{
#if defined(PETSC_HAVE_F90_2PTR_ARG) && defined(foo)
  void *ptr;
  PetscCall(PetscObjectGetFortranCallback((PetscObject)ts, PETSC_FORTRAN_CALLBACK_CLASS, _cb.function_pgiptr, NULL, &ptr));
#endif
  PetscObjectUseFortranCallback(ts, _cb.poststep, (TS *, PetscErrorCode * /* PETSC_F90_2PTR_PROTO_NOVAR */), (&ts, &ierr /* PETSC_F90_2PTR_PARAM(ptr) */));
}
static PetscErrorCode ourrhsfunction(TS ts, PetscReal d, Vec x, Vec f, void *ctx)
{
#if defined(PETSC_HAVE_F90_2PTR_ARG) && defined(foo)
  void *ptr;
  PetscCall(PetscObjectGetFortranCallback((PetscObject)ts, PETSC_FORTRAN_CALLBACK_CLASS, _cb.function_pgiptr, NULL, &ptr));
#endif
  PetscObjectUseFortranCallback(ts, _cb.rhsfunction, (TS *, PetscReal *, Vec *, Vec *, void *, PetscErrorCode * /* PETSC_F90_2PTR_PROTO_NOVAR */), (&ts, &d, &x, &f, _ctx, &ierr /* PETSC_F90_2PTR_PARAM(ptr) */));
}
static PetscErrorCode ourifunction(TS ts, PetscReal d, Vec x, Vec xdot, Vec f, void *ctx)
{
#if defined(PETSC_HAVE_F90_2PTR_ARG) && defined(foo)
  void *ptr;
  PetscCall(PetscObjectGetFortranCallback((PetscObject)ts, PETSC_FORTRAN_CALLBACK_CLASS, _cb.function_pgiptr, NULL, &ptr));
#endif
  PetscObjectUseFortranCallback(ts, _cb.ifunction, (TS *, PetscReal *, Vec *, Vec *, Vec *, void *, PetscErrorCode * /* PETSC_F90_2PTR_PROTO_NOVAR */), (&ts, &d, &x, &xdot, &f, _ctx, &ierr /* PETSC_F90_2PTR_PARAM(ptr) */));
}
static PetscErrorCode ourrhsjacobian(TS ts, PetscReal d, Vec x, Mat m, Mat p, void *ctx)
{
#if defined(PETSC_HAVE_F90_2PTR_ARG) && defined(foo)
  void *ptr;
  PetscCall(PetscObjectGetFortranCallback((PetscObject)ts, PETSC_FORTRAN_CALLBACK_CLASS, _cb.function_pgiptr, NULL, &ptr));
#endif
  PetscObjectUseFortranCallback(ts, _cb.rhsjacobian, (TS *, PetscReal *, Vec *, Mat *, Mat *, void *, PetscErrorCode * /* PETSC_F90_2PTR_PROTO_NOVAR */), (&ts, &d, &x, &m, &p, _ctx, &ierr /* PETSC_F90_2PTR_PARAM(ptr) */));
}
static PetscErrorCode ourijacobian(TS ts, PetscReal d, Vec x, Vec xdot, PetscReal shift, Mat m, Mat p, void *ctx)
{
#if defined(PETSC_HAVE_F90_2PTR_ARG) && defined(foo)
  void *ptr;
  PetscCall(PetscObjectGetFortranCallback((PetscObject)ts, PETSC_FORTRAN_CALLBACK_CLASS, _cb.function_pgiptr, NULL, &ptr));
#endif
  PetscObjectUseFortranCallback(ts, _cb.ijacobian, (TS *, PetscReal *, Vec *, Vec *, PetscReal *, Mat *, Mat *, void *, PetscErrorCode * /* PETSC_F90_2PTR_PROTO_NOVAR */), (&ts, &d, &x, &xdot, &shift, &m, &p, _ctx, &ierr /* PETSC_F90_2PTR_PARAM(ptr) */));
}
static PetscErrorCode ourijacobianp(TS ts, PetscReal d, Vec x, Vec xdot, PetscReal shift, Mat m, void *ctx)
{
#if defined(PETSC_HAVE_F90_2PTR_ARG) && defined(foo)
  void *ptr;
  PetscCall(PetscObjectGetFortranCallback((PetscObject)ts, PETSC_FORTRAN_CALLBACK_CLASS, _cb.function_pgiptr, NULL, &ptr));
#endif
  PetscObjectUseFortranCallback(ts, _cb.ijacobianp, (TS *, PetscReal *, Vec *, Vec *, PetscReal *, Mat *, void *, PetscErrorCode * /* PETSC_F90_2PTR_PROTO_NOVAR */), (&ts, &d, &x, &xdot, &shift, &m, _ctx, &ierr /* PETSC_F90_2PTR_PARAM(ptr) */));
}
static PetscErrorCode ourrhsjacobianp(TS ts, PetscReal d, Vec x, Mat m, void *ctx)
{
#if defined(PETSC_HAVE_F90_2PTR_ARG) && defined(foo)
  void *ptr;
  PetscCall(PetscObjectGetFortranCallback((PetscObject)ts, PETSC_FORTRAN_CALLBACK_CLASS, _cb.function_pgiptr, NULL, &ptr));
#endif
  PetscObjectUseFortranCallback(ts, _cb.rhsjacobianp, (TS *, PetscReal *, Vec *, Mat *, void *, PetscErrorCode * /* PETSC_F90_2PTR_PROTO_NOVAR */), (&ts, &d, &x, &m, _ctx, &ierr /* PETSC_F90_2PTR_PARAM(ptr) */));
}

static PetscErrorCode ourmonitordestroy(void **ctx)
{
  TS ts = (TS)*ctx;
  PetscObjectUseFortranCallback(ts, _cb.mondestroy, (void *, PetscErrorCode *), (_ctx, &ierr));
}

/*
   Note ctx is the same as ts so we need to get the Fortran context out of the TS
*/
static PetscErrorCode ourmonitor(TS ts, PetscInt i, PetscReal d, Vec v, void *ctx)
{
  PetscObjectUseFortranCallback(ts, _cb.monitor, (TS *, PetscInt *, PetscReal *, Vec *, void *, PetscErrorCode *), (&ts, &i, &d, &v, _ctx, &ierr));
}

/*
   Currently does not handle destroy or context
*/
static PetscErrorCode ourtransform(void *ctx, Vec x, Vec *xout)
{
  PetscObjectUseFortranCallback((TS)ctx, _cb.transform, (void *, Vec *, Vec *, PetscErrorCode *), (_ctx, &x, xout, &ierr));
}

PETSC_EXTERN void tsmonitorlgsettransform_(TS *ts, void (*f)(void *, Vec *, Vec *, PetscErrorCode *), PetscErrorCode (*d)(void *, PetscErrorCode *), void *ctx, PetscErrorCode *ierr)
{
  *ierr = TSMonitorLGSetTransform(*ts, ourtransform, NULL, NULL);
  if (*ierr) return;
  *ierr = PetscObjectSetFortranCallback((PetscObject)*ts, PETSC_FORTRAN_CALLBACK_CLASS, &_cb.transform, (PetscFortranCallbackFn *)f, ctx);
}

PETSC_EXTERN void tssetprestep_(TS *ts, PetscErrorCode (*f)(TS *, PetscErrorCode *), PetscErrorCode *ierr)
{
  *ierr = TSSetPreStep(*ts, ourprestep);
  if (*ierr) return;
  *ierr = PetscObjectSetFortranCallback((PetscObject)*ts, PETSC_FORTRAN_CALLBACK_CLASS, &_cb.prestep, (PetscFortranCallbackFn *)f, NULL);
}

PETSC_EXTERN void tssetpoststep_(TS *ts, PetscErrorCode (*f)(TS *, PetscErrorCode *), PetscErrorCode *ierr)
{
  *ierr = TSSetPostStep(*ts, ourpoststep);
  if (*ierr) return;
  *ierr = PetscObjectSetFortranCallback((PetscObject)*ts, PETSC_FORTRAN_CALLBACK_CLASS, &_cb.poststep, (PetscFortranCallbackFn *)f, NULL);
}

PETSC_EXTERN void tscomputerhsfunctionlinear_(TS *, PetscReal *, Vec *, Vec *, void *, PetscErrorCode *);

PETSC_EXTERN void tssetrhsfunction_(TS *ts, Vec *r, void (*f)(TS *, PetscReal *, Vec *, Vec *, void *, PetscErrorCode *), void *fP, PetscErrorCode *ierr)
{
  Vec R;
  CHKFORTRANNULLOBJECT(r);
  CHKFORTRANNULLFUNCTION(f);
  R = r ? *r : (Vec)NULL;
  if (f == tscomputerhsfunctionlinear_) {
    *ierr = TSSetRHSFunction(*ts, R, TSComputeRHSFunctionLinear, fP);
  } else {
    *ierr = PetscObjectSetFortranCallback((PetscObject)*ts, PETSC_FORTRAN_CALLBACK_CLASS, &_cb.rhsfunction, (PetscFortranCallbackFn *)f, fP);
    *ierr = TSSetRHSFunction(*ts, R, ourrhsfunction, NULL);
  }
}
PETSC_EXTERN void tsgetrhsfunction_(TS *ts, Vec *r, void *func, void **ctx, PetscErrorCode *ierr)
{
  CHKFORTRANNULLINTEGER(ctx);
  CHKFORTRANNULLOBJECT(r);
  *ierr = TSGetRHSFunction(*ts, r, NULL, ctx);
}

PETSC_EXTERN void tscomputeifunctionlinear_(TS *ts, PetscReal *t, Vec *X, Vec *Xdot, Vec *F, void *ctx, PetscErrorCode *ierr);

PETSC_EXTERN void tssetifunction_(TS *ts, Vec *r, void (*f)(TS *, PetscReal *, Vec *, Vec *, Vec *, void *, PetscErrorCode *), void *fP, PetscErrorCode *ierr)
{
  Vec R;
  CHKFORTRANNULLOBJECT(r);
  CHKFORTRANNULLFUNCTION(f);
  R = r ? *r : (Vec)NULL;
  if (f == tscomputeifunctionlinear_) {
    *ierr = TSSetIFunction(*ts, R, TSComputeIFunctionLinear, fP);
  } else {
    *ierr = PetscObjectSetFortranCallback((PetscObject)*ts, PETSC_FORTRAN_CALLBACK_CLASS, &_cb.ifunction, (PetscFortranCallbackFn *)f, fP);
    *ierr = TSSetIFunction(*ts, R, ourifunction, NULL);
  }
}
PETSC_EXTERN void tsgetifunction_(TS *ts, Vec *r, void *func, void **ctx, PetscErrorCode *ierr)
{
  CHKFORTRANNULLINTEGER(ctx);
  CHKFORTRANNULLOBJECT(r);
  *ierr = TSGetIFunction(*ts, r, NULL, ctx);
}

/* ---------------------------------------------------------*/
PETSC_EXTERN void tscomputerhsjacobianconstant_(TS *, PetscReal *, Vec *, Mat *, Mat *, void *, PetscErrorCode *);

PETSC_EXTERN void tssetrhsjacobian_(TS *ts, Mat *A, Mat *B, void (*f)(TS *, PetscReal *, Vec *, Mat *, Mat *, void *, PetscErrorCode *), void *fP, PetscErrorCode *ierr)
{
  CHKFORTRANNULLFUNCTION(f);
  if (f == tscomputerhsjacobianconstant_) {
    *ierr = TSSetRHSJacobian(*ts, *A, *B, TSComputeRHSJacobianConstant, fP);
  } else {
    *ierr = PetscObjectSetFortranCallback((PetscObject)*ts, PETSC_FORTRAN_CALLBACK_CLASS, &_cb.rhsjacobian, (PetscFortranCallbackFn *)f, fP);
    *ierr = TSSetRHSJacobian(*ts, *A, *B, ourrhsjacobian, NULL);
  }
}

PETSC_EXTERN void tscomputeijacobianconstant_(TS *ts, PetscReal *t, Vec *X, Vec *Xdot, PetscReal *shift, Mat *A, Mat *B, void *ctx, PetscErrorCode *ierr);

PETSC_EXTERN void tssetijacobian_(TS *ts, Mat *A, Mat *B, void (*f)(TS *, PetscReal *, Vec *, Vec *, PetscReal *, Mat *, Mat *, void *, PetscErrorCode *), void *fP, PetscErrorCode *ierr)
{
  CHKFORTRANNULLFUNCTION(f);
  if (f == tscomputeijacobianconstant_) {
    *ierr = TSSetIJacobian(*ts, *A, *B, TSComputeIJacobianConstant, fP);
  } else {
    *ierr = PetscObjectSetFortranCallback((PetscObject)*ts, PETSC_FORTRAN_CALLBACK_CLASS, &_cb.ijacobian, (PetscFortranCallbackFn *)f, fP);
    *ierr = TSSetIJacobian(*ts, *A, *B, ourijacobian, NULL);
  }
}
PETSC_EXTERN void tsgetijacobian_(TS *ts, Mat *J, Mat *M, int *func, void **ctx, PetscErrorCode *ierr)
{
  CHKFORTRANNULLINTEGER(ctx);
  CHKFORTRANNULLOBJECT(J);
  CHKFORTRANNULLOBJECT(M);
  *ierr = TSGetIJacobian(*ts, J, M, NULL, ctx);
}
PETSC_EXTERN void tssetijacobianp_(TS *ts, Mat *A, void (*f)(TS *, PetscReal *, Vec *, Vec *, PetscReal, Mat *, void *, PetscErrorCode *), void *fP, PetscErrorCode *ierr)
{
  CHKFORTRANNULLFUNCTION(f);
  *ierr = PetscObjectSetFortranCallback((PetscObject)*ts, PETSC_FORTRAN_CALLBACK_CLASS, &_cb.ijacobianp, (PetscFortranCallbackFn *)f, fP);
  *ierr = TSSetIJacobianP(*ts, *A, ourijacobianp, NULL);
}
PETSC_EXTERN void tsgetijacobianp_(TS *ts, Mat *J, int *func, void **ctx, PetscErrorCode *ierr)
{
  CHKFORTRANNULLINTEGER(ctx);
  CHKFORTRANNULLOBJECT(J);
  *ierr = TSGetIJacobianP(*ts, J, NULL, ctx);
}
PETSC_EXTERN void tssetrhsjacobianp_(TS *ts, Mat *A, void (*f)(TS *, PetscReal *, Vec *, Mat *, void *, PetscErrorCode *), void *fP, PetscErrorCode *ierr)
{
  CHKFORTRANNULLFUNCTION(f);
  *ierr = PetscObjectSetFortranCallback((PetscObject)*ts, PETSC_FORTRAN_CALLBACK_CLASS, &_cb.rhsjacobianp, (PetscFortranCallbackFn *)f, fP);
  *ierr = TSSetRHSJacobianP(*ts, *A, ourrhsjacobianp, NULL);
}
PETSC_EXTERN void tsgetrhsjacobianp_(TS *ts, Mat *J, int *func, void **ctx, PetscErrorCode *ierr)
{
  CHKFORTRANNULLINTEGER(ctx);
  CHKFORTRANNULLOBJECT(J);
  *ierr = TSGetRHSJacobianP(*ts, J, NULL, ctx);
}

PETSC_EXTERN void tsmonitordefault_(TS *, PetscInt *, PetscReal *, Vec *, PetscViewerAndFormat **, PetscErrorCode *);

/* ---------------------------------------------------------*/

/* PETSC_EXTERN void tsmonitordefault_(TS*,PetscInt*,PetscReal*,Vec*,void*,PetscErrorCode*); */

PETSC_EXTERN void tsmonitorset_(TS *ts, void (*func)(TS *, PetscInt *, PetscReal *, Vec *, void *, PetscErrorCode *), void *mctx, void (*d)(void *, PetscErrorCode *), PetscErrorCode *ierr)
{
  CHKFORTRANNULLFUNCTION(d);
  if ((PetscFortranCallbackFn *)func == (PetscFortranCallbackFn *)tsmonitordefault_) {
    *ierr = TSMonitorSet(*ts, (PetscErrorCode (*)(TS, PetscInt, PetscReal, Vec, void *))TSMonitorDefault, *(PetscViewerAndFormat **)mctx, (PetscCtxDestroyFn *)PetscViewerAndFormatDestroy);
  } else {
    *ierr = PetscObjectSetFortranCallback((PetscObject)*ts, PETSC_FORTRAN_CALLBACK_CLASS, &_cb.monitor, (PetscFortranCallbackFn *)func, mctx);
    *ierr = PetscObjectSetFortranCallback((PetscObject)*ts, PETSC_FORTRAN_CALLBACK_CLASS, &_cb.mondestroy, (PetscFortranCallbackFn *)d, mctx);
    *ierr = TSMonitorSet(*ts, ourmonitor, *ts, ourmonitordestroy);
  }
}

/* ---------------------------------------------------------*/
/*  func is currently ignored from Fortran */
PETSC_EXTERN void tsgetrhsjacobian_(TS *ts, Mat *J, Mat *M, int *func, void **ctx, PetscErrorCode *ierr)
{
  *ierr = TSGetRHSJacobian(*ts, J, M, NULL, ctx);
}
