#include <petsc/private/ftnimpl.h>
#include <petsc/private/taoimpl.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
  #define taosetobjective_                    TAOSETOBJECTIVE
  #define taosetgradient_                     TAOSETGRADIENT
  #define taosetobjectiveandgradient_         TAOSETOBJECTIVEANDGRADIENT
  #define taosethessian_                      TAOSETHESSIAN
  #define taosetresidualroutine_              TAOSETRESIDUALROUTINE
  #define taosetjacobianresidualroutine_      TAOSETJACOBIANRESIDUALROUTINE
  #define taosetjacobianroutine_              TAOSETJACOBIANROUTINE
  #define taosetjacobianstateroutine_         TAOSETJACOBIANSTATEROUTINE
  #define taosetjacobiandesignroutine_        TAOSETJACOBIANDESIGNROUTINE
  #define taosetjacobianinequalityroutine_    TAOSETJACOBIANINEQUALITYROUTINE
  #define taosetjacobianequalityroutine_      TAOSETJACOBIANEQUALITYROUTINE
  #define taosetinequalityconstraintsroutine_ TAOSETINEQUALITYCONSTRAINTSROUTINE
  #define taosetequalityconstraintsroutine_   TAOSETEQUALITYCONSTRAINTSROUTINE
  #define taosetvariableboundsroutine_        TAOSETVARIABLEBOUNDSROUTINE
  #define taosetconstraintsroutine_           TAOSETCONSTRAINTSROUTINE
  #define taomonitorset_                      TAOMONITORSET
  #define taogetconvergencehistory_           TAOGETCONVERGENCEHISTORY
  #define taosetconvergencetest_              TAOSETCONVERGENCETEST
  #define taosetupdate_                       TAOSETUPDATE
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
  #define taosetobjective_                    taosetobjective
  #define taosetgradient_                     taosetgradient
  #define taosetobjectiveandgradient_         taosetobjectiveandgradient
  #define taosethessian_                      taosethessian
  #define taosetresidualroutine_              taosetresidualroutine
  #define taosetjacobianresidualroutine_      taosetjacobianresidualroutine
  #define taosetjacobianroutine_              taosetjacobianroutine
  #define taosetjacobianstateroutine_         taosetjacobianstateroutine
  #define taosetjacobiandesignroutine_        taosetjacobiandesignroutine
  #define taosetjacobianinequalityroutine_    taosetjacobianinequalityroutine
  #define taosetjacobianequalityroutine_      taosetjacobianequalityroutine
  #define taosetinequalityconstraintsroutine_ taosetinequalityconstraintsroutine
  #define taosetequalityconstraintsroutine_   taosetequalityconstraintsroutine
  #define taosetvariableboundsroutine_        taosetvariableboundsroutine
  #define taosetconstraintsroutine_           taosetconstraintsroutine
  #define taomonitorset_                      taomonitorset
  #define taogetconvergencehistory_           taogetconvergencehistory
  #define taosetconvergencetest_              taosetconvergencetest
  #define taosetupdate_                       taosetupdate
#endif

static struct {
  PetscFortranCallbackId obj;
  PetscFortranCallbackId grad;
  PetscFortranCallbackId objgrad;
  PetscFortranCallbackId hess;
  PetscFortranCallbackId lsres;
  PetscFortranCallbackId lsjac;
  PetscFortranCallbackId jac;
  PetscFortranCallbackId jacstate;
  PetscFortranCallbackId jacdesign;
  PetscFortranCallbackId bounds;
  PetscFortranCallbackId mon;
  PetscFortranCallbackId mondestroy;
  PetscFortranCallbackId convtest;
  PetscFortranCallbackId constraints;
  PetscFortranCallbackId jacineq;
  PetscFortranCallbackId jaceq;
  PetscFortranCallbackId conineq;
  PetscFortranCallbackId coneq;
  PetscFortranCallbackId nfuncs;
  PetscFortranCallbackId update;
#if defined(PETSC_HAVE_F90_2PTR_ARG)
  PetscFortranCallbackId function_pgiptr;
#endif
} _cb;

static PetscErrorCode ourtaoobjectiveroutine(Tao tao, Vec x, PetscReal *f, void *ctx)
{
  PetscObjectUseFortranCallback(tao, _cb.obj, (Tao *, Vec *, PetscReal *, void *, PetscErrorCode *), (&tao, &x, f, _ctx, &ierr));
}

static PetscErrorCode ourtaogradientroutine(Tao tao, Vec x, Vec g, void *ctx)
{
  PetscObjectUseFortranCallback(tao, _cb.grad, (Tao *, Vec *, Vec *, void *, PetscErrorCode *), (&tao, &x, &g, _ctx, &ierr));
}

static PetscErrorCode ourtaoobjectiveandgradientroutine(Tao tao, Vec x, PetscReal *f, Vec g, void *ctx)
{
  PetscObjectUseFortranCallback(tao, _cb.objgrad, (Tao *, Vec *, PetscReal *, Vec *, void *, PetscErrorCode *), (&tao, &x, f, &g, _ctx, &ierr));
}

static PetscErrorCode ourtaohessianroutine(Tao tao, Vec x, Mat H, Mat Hpre, void *ctx)
{
  PetscObjectUseFortranCallback(tao, _cb.hess, (Tao *, Vec *, Mat *, Mat *, void *, PetscErrorCode *), (&tao, &x, &H, &Hpre, _ctx, &ierr));
}

static PetscErrorCode ourtaojacobianroutine(Tao tao, Vec x, Mat H, Mat Hpre, void *ctx)
{
  PetscObjectUseFortranCallback(tao, _cb.jac, (Tao *, Vec *, Mat *, Mat *, void *, PetscErrorCode *), (&tao, &x, &H, &Hpre, _ctx, &ierr));
}

static PetscErrorCode ourtaojacobianstateroutine(Tao tao, Vec x, Mat H, Mat Hpre, Mat Hinv, void *ctx)
{
  PetscObjectUseFortranCallback(tao, _cb.jacstate, (Tao *, Vec *, Mat *, Mat *, Mat *, void *, PetscErrorCode *), (&tao, &x, &H, &Hpre, &Hinv, _ctx, &ierr));
}

static PetscErrorCode ourtaojacobiandesignroutine(Tao tao, Vec x, Mat H, void *ctx)
{
  PetscObjectUseFortranCallback(tao, _cb.jacdesign, (Tao *, Vec *, Mat *, void *, PetscErrorCode *), (&tao, &x, &H, _ctx, &ierr));
}

static PetscErrorCode ourtaoboundsroutine(Tao tao, Vec xl, Vec xu, void *ctx)
{
  PetscObjectUseFortranCallback(tao, _cb.bounds, (Tao *, Vec *, Vec *, void *, PetscErrorCode *), (&tao, &xl, &xu, _ctx, &ierr));
}
static PetscErrorCode ourtaoresidualroutine(Tao tao, Vec x, Vec f, void *ctx)
{
  PetscObjectUseFortranCallback(tao, _cb.lsres, (Tao *, Vec *, Vec *, void *, PetscErrorCode *), (&tao, &x, &f, _ctx, &ierr));
}

static PetscErrorCode ourtaojacobianresidualroutine(Tao tao, Vec x, Mat J, Mat Jpre, void *ctx)
{
  PetscObjectUseFortranCallback(tao, _cb.lsjac, (Tao *, Vec *, Mat *, Mat *, void *, PetscErrorCode *), (&tao, &x, &J, &Jpre, _ctx, &ierr));
}

static PetscErrorCode ourtaomonitor(Tao tao, void *ctx)
{
  PetscObjectUseFortranCallback(tao, _cb.mon, (Tao *, void *, PetscErrorCode *), (&tao, _ctx, &ierr));
}

static PetscErrorCode ourtaomondestroy(void **ctx)
{
  Tao tao = (Tao)*ctx;
  PetscObjectUseFortranCallback(tao, _cb.mondestroy, (void *, PetscErrorCode *), (_ctx, &ierr));
}
static PetscErrorCode ourtaoconvergencetest(Tao tao, void *ctx)
{
  PetscObjectUseFortranCallback(tao, _cb.convtest, (Tao *, void *, PetscErrorCode *), (&tao, _ctx, &ierr));
}

static PetscErrorCode ourtaoconstraintsroutine(Tao tao, Vec x, Vec c, void *ctx)
{
  PetscObjectUseFortranCallback(tao, _cb.constraints, (Tao *, Vec *, Vec *, void *, PetscErrorCode *), (&tao, &x, &c, _ctx, &ierr));
}

static PetscErrorCode ourtaojacobianinequalityroutine(Tao tao, Vec x, Mat J, Mat Jpre, void *ctx)
{
  PetscObjectUseFortranCallback(tao, _cb.jacineq, (Tao *, Vec *, Mat *, Mat *, void *, PetscErrorCode *), (&tao, &x, &J, &Jpre, _ctx, &ierr));
}

static PetscErrorCode ourtaojacobianequalityroutine(Tao tao, Vec x, Mat J, Mat Jpre, void *ctx)
{
  PetscObjectUseFortranCallback(tao, _cb.jaceq, (Tao *, Vec *, Mat *, Mat *, void *, PetscErrorCode *), (&tao, &x, &J, &Jpre, _ctx, &ierr));
}

static PetscErrorCode ourtaoinequalityconstraintsroutine(Tao tao, Vec x, Vec c, void *ctx)
{
  PetscObjectUseFortranCallback(tao, _cb.conineq, (Tao *, Vec *, Vec *, void *, PetscErrorCode *), (&tao, &x, &c, _ctx, &ierr));
}

static PetscErrorCode ourtaoequalityconstraintsroutine(Tao tao, Vec x, Vec c, void *ctx)
{
  PetscObjectUseFortranCallback(tao, _cb.coneq, (Tao *, Vec *, Vec *, void *, PetscErrorCode *), (&tao, &x, &c, _ctx, &ierr));
}

static PetscErrorCode ourtaoupdateroutine(Tao tao, PetscInt iter, void *ctx)
{
  PetscObjectUseFortranCallback(tao, _cb.update, (Tao *, PetscInt *, void *), (&tao, &iter, _ctx));
}

PETSC_EXTERN void taosetobjective_(Tao *tao, void (*func)(Tao *, Vec *, PetscReal *, void *, PetscErrorCode *), void *ctx, PetscErrorCode *ierr)
{
  CHKFORTRANNULLFUNCTION(func);
  *ierr = PetscObjectSetFortranCallback((PetscObject)*tao, PETSC_FORTRAN_CALLBACK_CLASS, &_cb.obj, (PetscFortranCallbackFn *)func, ctx);
  if (!*ierr) *ierr = TaoSetObjective(*tao, ourtaoobjectiveroutine, ctx);
}

PETSC_EXTERN void taosetgradient_(Tao *tao, Vec *g, void (*func)(Tao *, Vec *, Vec *, void *, PetscErrorCode *), void *ctx, PetscErrorCode *ierr)
{
  CHKFORTRANNULLFUNCTION(func);
  *ierr = PetscObjectSetFortranCallback((PetscObject)*tao, PETSC_FORTRAN_CALLBACK_CLASS, &_cb.grad, (PetscFortranCallbackFn *)func, ctx);
  if (!*ierr) *ierr = TaoSetGradient(*tao, *g, ourtaogradientroutine, ctx);
}

PETSC_EXTERN void taosetobjectiveandgradient_(Tao *tao, Vec *g, void (*func)(Tao *, Vec *, PetscReal *, Vec *, void *, PetscErrorCode *), void *ctx, PetscErrorCode *ierr)
{
  CHKFORTRANNULLFUNCTION(func);
  *ierr = PetscObjectSetFortranCallback((PetscObject)*tao, PETSC_FORTRAN_CALLBACK_CLASS, &_cb.objgrad, (PetscFortranCallbackFn *)func, ctx);
  if (!*ierr) *ierr = TaoSetObjectiveAndGradient(*tao, *g, ourtaoobjectiveandgradientroutine, ctx);
}

PETSC_EXTERN void taosethessian_(Tao *tao, Mat *J, Mat *Jp, void (*func)(Tao *, Vec *, Mat *, Mat *, void *, PetscErrorCode *), void *ctx, PetscErrorCode *ierr)
{
  CHKFORTRANNULLFUNCTION(func);
  *ierr = PetscObjectSetFortranCallback((PetscObject)*tao, PETSC_FORTRAN_CALLBACK_CLASS, &_cb.hess, (PetscFortranCallbackFn *)func, ctx);
  if (!*ierr) *ierr = TaoSetHessian(*tao, *J, *Jp, ourtaohessianroutine, ctx);
}

PETSC_EXTERN void taosetresidualroutine_(Tao *tao, Vec *F, void (*func)(Tao *, Vec *, Vec *, void *, PetscErrorCode *), void *ctx, PetscErrorCode *ierr)
{
  CHKFORTRANNULLFUNCTION(func);
  *ierr = PetscObjectSetFortranCallback((PetscObject)*tao, PETSC_FORTRAN_CALLBACK_CLASS, &_cb.lsres, (PetscFortranCallbackFn *)func, ctx);
  if (!*ierr) *ierr = TaoSetResidualRoutine(*tao, *F, ourtaoresidualroutine, ctx);
}

PETSC_EXTERN void taosetjacobianresidualroutine_(Tao *tao, Mat *J, Mat *Jpre, void (*func)(Tao *, Vec *, Mat *, Mat *, void *, PetscErrorCode *), void *ctx, PetscErrorCode *ierr)
{
  CHKFORTRANNULLFUNCTION(func);
  *ierr = PetscObjectSetFortranCallback((PetscObject)*tao, PETSC_FORTRAN_CALLBACK_CLASS, &_cb.lsjac, (PetscFortranCallbackFn *)func, ctx);
  if (!*ierr) *ierr = TaoSetJacobianResidualRoutine(*tao, *J, *Jpre, ourtaojacobianresidualroutine, ctx);
}

PETSC_EXTERN void taosetjacobianroutine_(Tao *tao, Mat *J, Mat *Jp, void (*func)(Tao *, Vec *, Mat *, Mat *, void *, PetscErrorCode *), void *ctx, PetscErrorCode *ierr)
{
  CHKFORTRANNULLFUNCTION(func);
  *ierr = PetscObjectSetFortranCallback((PetscObject)*tao, PETSC_FORTRAN_CALLBACK_CLASS, &_cb.jac, (PetscFortranCallbackFn *)func, ctx);
  if (!*ierr) *ierr = TaoSetJacobianRoutine(*tao, *J, *Jp, ourtaojacobianroutine, ctx);
}

PETSC_EXTERN void taosetjacobianstateroutine_(Tao *tao, Mat *J, Mat *Jp, Mat *Jinv, void (*func)(Tao *, Vec *, Mat *, Mat *, Mat *, void *, PetscErrorCode *), void *ctx, PetscErrorCode *ierr)
{
  CHKFORTRANNULLFUNCTION(func);
  *ierr = PetscObjectSetFortranCallback((PetscObject)*tao, PETSC_FORTRAN_CALLBACK_CLASS, &_cb.jacstate, (PetscFortranCallbackFn *)func, ctx);
  if (!*ierr) *ierr = TaoSetJacobianStateRoutine(*tao, *J, *Jp, *Jinv, ourtaojacobianstateroutine, ctx);
}

PETSC_EXTERN void taosetjacobiandesignroutine_(Tao *tao, Mat *J, void (*func)(Tao *, Vec *, Mat *, void *, PetscErrorCode *), void *ctx, PetscErrorCode *ierr)
{
  CHKFORTRANNULLFUNCTION(func);
  *ierr = PetscObjectSetFortranCallback((PetscObject)tao, PETSC_FORTRAN_CALLBACK_CLASS, &_cb.jacdesign, (PetscFortranCallbackFn *)func, ctx);
  if (!*ierr) *ierr = TaoSetJacobianDesignRoutine(*tao, *J, ourtaojacobiandesignroutine, ctx);
}

PETSC_EXTERN void taosetvariableboundsroutine_(Tao *tao, void (*func)(Tao *, Vec *, Vec *, void *, PetscErrorCode *), void *ctx, PetscErrorCode *ierr)
{
  CHKFORTRANNULLFUNCTION(func);
  *ierr = PetscObjectSetFortranCallback((PetscObject)*tao, PETSC_FORTRAN_CALLBACK_CLASS, &_cb.bounds, (PetscFortranCallbackFn *)func, ctx);
  if (!*ierr) *ierr = TaoSetVariableBoundsRoutine(*tao, ourtaoboundsroutine, ctx);
}

PETSC_EXTERN void taomonitorset_(Tao *tao, void (*func)(Tao *, void *, PetscErrorCode *), void *ctx, void (*mondestroy)(void *, PetscErrorCode *), PetscErrorCode *ierr)
{
  CHKFORTRANNULLFUNCTION(mondestroy);
  *ierr = PetscObjectSetFortranCallback((PetscObject)*tao, PETSC_FORTRAN_CALLBACK_CLASS, &_cb.mon, (PetscFortranCallbackFn *)func, ctx);
  if (*ierr) return;
  *ierr = PetscObjectSetFortranCallback((PetscObject)*tao, PETSC_FORTRAN_CALLBACK_CLASS, &_cb.mondestroy, (PetscFortranCallbackFn *)mondestroy, ctx);
  if (*ierr) return;
  *ierr = TaoMonitorSet(*tao, ourtaomonitor, *tao, ourtaomondestroy);
}

PETSC_EXTERN void taosetconvergencetest_(Tao *tao, void (*func)(Tao *, void *, PetscErrorCode *), void *ctx, PetscErrorCode *ierr)
{
  CHKFORTRANNULLFUNCTION(func);
  *ierr = PetscObjectSetFortranCallback((PetscObject)*tao, PETSC_FORTRAN_CALLBACK_CLASS, &_cb.convtest, (PetscFortranCallbackFn *)func, ctx);
  if (!*ierr) *ierr = TaoSetConvergenceTest(*tao, ourtaoconvergencetest, ctx);
}

PETSC_EXTERN void taosetconstraintsroutine_(Tao *tao, Vec *C, void (*func)(Tao *, Vec *, Vec *, void *, PetscErrorCode *), void *ctx, PetscErrorCode *ierr)
{
  CHKFORTRANNULLFUNCTION(func);
  *ierr = PetscObjectSetFortranCallback((PetscObject)*tao, PETSC_FORTRAN_CALLBACK_CLASS, &_cb.constraints, (PetscFortranCallbackFn *)func, ctx);
  if (!*ierr) *ierr = TaoSetConstraintsRoutine(*tao, *C, ourtaoconstraintsroutine, ctx);
}

PETSC_EXTERN void taogetconvergencehistory_(Tao *tao, PetscInt *nhist, PetscErrorCode *ierr)
{
  *ierr = TaoGetConvergenceHistory(*tao, NULL, NULL, NULL, NULL, nhist);
}

PETSC_EXTERN void taosetjacobianinequalityroutine_(Tao *tao, Mat *J, Mat *Jp, void (*func)(Tao *, Vec *, Mat *, Mat *, void *, PetscErrorCode *), void *ctx, PetscErrorCode *ierr)
{
  CHKFORTRANNULLFUNCTION(func);
  *ierr = PetscObjectSetFortranCallback((PetscObject)*tao, PETSC_FORTRAN_CALLBACK_CLASS, &_cb.jacineq, (PetscFortranCallbackFn *)func, ctx);
  if (!*ierr) *ierr = TaoSetJacobianInequalityRoutine(*tao, *J, *Jp, ourtaojacobianinequalityroutine, ctx);
}

PETSC_EXTERN void taosetjacobianequalityroutine_(Tao *tao, Mat *J, Mat *Jp, void (*func)(Tao *, Vec *, Mat *, Mat *, void *, PetscErrorCode *), void *ctx, PetscErrorCode *ierr)
{
  CHKFORTRANNULLFUNCTION(func);
  *ierr = PetscObjectSetFortranCallback((PetscObject)*tao, PETSC_FORTRAN_CALLBACK_CLASS, &_cb.jaceq, (PetscFortranCallbackFn *)func, ctx);
  if (!*ierr) *ierr = TaoSetJacobianEqualityRoutine(*tao, *J, *Jp, ourtaojacobianequalityroutine, ctx);
}

PETSC_EXTERN void taosetinequalityconstraintsroutine_(Tao *tao, Vec *C, void (*func)(Tao *, Vec *, Vec *, void *, PetscErrorCode *), void *ctx, PetscErrorCode *ierr)
{
  CHKFORTRANNULLFUNCTION(func);
  *ierr = PetscObjectSetFortranCallback((PetscObject)*tao, PETSC_FORTRAN_CALLBACK_CLASS, &_cb.conineq, (PetscFortranCallbackFn *)func, ctx);
  if (!*ierr) *ierr = TaoSetInequalityConstraintsRoutine(*tao, *C, ourtaoinequalityconstraintsroutine, ctx);
}

PETSC_EXTERN void taosetequalityconstraintsroutine_(Tao *tao, Vec *C, void (*func)(Tao *, Vec *, Vec *, void *, PetscErrorCode *), void *ctx, PetscErrorCode *ierr)
{
  CHKFORTRANNULLFUNCTION(func);
  *ierr = PetscObjectSetFortranCallback((PetscObject)*tao, PETSC_FORTRAN_CALLBACK_CLASS, &_cb.coneq, (PetscFortranCallbackFn *)func, ctx);
  if (!*ierr) *ierr = TaoSetEqualityConstraintsRoutine(*tao, *C, ourtaoequalityconstraintsroutine, ctx);
}

PETSC_EXTERN void taosetupdate_(Tao *tao, void (*func)(Tao *, PetscInt *, PetscErrorCode *), void *ctx, PetscErrorCode *ierr)
{
  CHKFORTRANNULLFUNCTION(func);
  *ierr = PetscObjectSetFortranCallback((PetscObject)*tao, PETSC_FORTRAN_CALLBACK_CLASS, &_cb.update, (PetscFortranCallbackFn *)func, ctx);
  if (!*ierr) *ierr = TaoSetUpdate(*tao, ourtaoupdateroutine, ctx);
}
