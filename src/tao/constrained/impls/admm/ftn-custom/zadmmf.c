#include <petsc/private/fortranimpl.h>
#include <petsc/private/f90impl.h>
#include <petsc/private/taoimpl.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define taoadmmsetmisfitobjectiveandgradientroutine_       TAOADMMSETMISFITOBJECTIVEANDGRADIENTROUTINE
#define taoadmmsetmisfithessianroutine_                    TAOADMMSETMISFITHESSIANROUTINE
#define taoadmmsetmisfitconstraintjacobian_                TAOADMMSETMISFITCONSTRAINTJACOBIAN
#define taoadmmsetregularizerobjectiveandgradientroutine_  TAOADMMSETREGULARIZEROBJECTIVEANDGRADIENTROUTINE
#define taoadmmsetregularizerhessianroutine_               TAOADMMSETREGULARIZERHESSIANROUTINE
#define taoadmmsetregularizerconstraintjacobian_           TAOADMMSETREGULARIZERCONSTRAINTJACOBIAN
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define taoadmmsetmisfitobjectiveandgradientroutine_       taoadmmsetmisfitobjectiveandgradientroutine
#define taoadmmsetmisfithessianroutine_                    taoadmmsetmisfithessianroutine
#define taoadmmsetmisfitconstraintjacobian_                taoadmmsetmisfitconstraintjacobian
#define taoadmmsetregularizerobjectiveandgradientroutine_  taoadmmsetregularizerobjectiveandgradientroutine
#define taoadmmsetregularizerhessianroutine_               taoadmmsetregularizerhessianroutine
#define taoadmmsetregularizerconstraintjacobian_           taoadmmsetregularizerconstraintjacobian
#endif

static struct {
  PetscFortranCallbackId misfitobjgrad;
  PetscFortranCallbackId misfithess;
  PetscFortranCallbackId misfitjacobian;
  PetscFortranCallbackId regobjgrad;
  PetscFortranCallbackId reghess;
  PetscFortranCallbackId regjacobian;
} _cb;

static PetscErrorCode ourtaoadmmmisfitobjgradroutine(Tao tao, Vec x, PetscReal *f, Vec g, void* ctx)
{
    PetscObjectUseFortranCallback(tao,_cb.misfitobjgrad,(Tao*,Vec*,PetscReal*,Vec*,void*,PetscErrorCode*),(&tao,&x,f,&g,_ctx,&ierr));
}

static PetscErrorCode ourtaoadmmmisfithessroutine(Tao tao, Vec x, Mat H, Mat Hpre, void *ctx)
{
    PetscObjectUseFortranCallback(tao,_cb.misfithess,(Tao*,Vec*,Mat*,Mat*,void*,PetscErrorCode*),(&tao,&x,&H,&Hpre,_ctx,&ierr));
}

static PetscErrorCode ourtaoadmmmisfitconstraintjacobian(Tao tao, Vec x, Mat J, Mat Jpre, void *ctx)
{
    PetscObjectUseFortranCallback(tao,_cb.misfitjacobian,(Tao*,Vec*,Mat*,Mat*,void*,PetscErrorCode*),(&tao,&x,&J,&Jpre,_ctx,&ierr));
}

static PetscErrorCode ourtaoadmmregularizerobjgradroutine(Tao tao, Vec x, PetscReal *f, Vec g, void* ctx)
{
    PetscObjectUseFortranCallback(tao,_cb.regobjgrad,(Tao*,Vec*,PetscReal*,Vec*,void*,PetscErrorCode*),(&tao,&x,f,&g,_ctx,&ierr));
}

static PetscErrorCode ourtaoadmmregularizerhessroutine(Tao tao, Vec x, Mat H, Mat Hpre, void *ctx)
{
    PetscObjectUseFortranCallback(tao,_cb.reghess,(Tao*,Vec*,Mat*,Mat*,void*,PetscErrorCode*),(&tao,&x,&H,&Hpre,_ctx,&ierr));
}

static PetscErrorCode ourtaoadmmregularizerconstraintjacobian(Tao tao, Vec x, Mat J, Mat Jpre, void *ctx)
{
    PetscObjectUseFortranCallback(tao,_cb.regjacobian,(Tao*,Vec*,Mat*,Mat*,void*,PetscErrorCode*),(&tao,&x,&J,&Jpre,_ctx,&ierr));
}

PETSC_EXTERN void taoadmmsetmisfitobjectiveandgradientroutine_(Tao *tao, void (*func)(Tao*, Vec *, PetscReal *, Vec *, void *, PetscErrorCode *), void *ctx, PetscErrorCode *ierr)
{
    CHKFORTRANNULLFUNCTION(func);
    *ierr = PetscObjectSetFortranCallback((PetscObject)*tao,PETSC_FORTRAN_CALLBACK_CLASS,&_cb.misfitobjgrad,(PetscVoidFunction)func,ctx);
    if (!*ierr) *ierr = TaoADMMSetMisfitObjectiveAndGradientRoutine(*tao,ourtaoadmmmisfitobjgradroutine,ctx);
}

PETSC_EXTERN void taoadmmsetmisfithessianroutine_(Tao *tao, Mat *H, Mat *Hpre, void (*func)(Tao*, Vec *, Mat *, Mat *, void *, PetscErrorCode *), void *ctx, PetscErrorCode *ierr)
{
    CHKFORTRANNULLFUNCTION(func);
    *ierr = PetscObjectSetFortranCallback((PetscObject)*tao,PETSC_FORTRAN_CALLBACK_CLASS,&_cb.misfithess,(PetscVoidFunction)func,ctx);
    if (!*ierr) *ierr = TaoADMMSetMisfitHessianRoutine(*tao,*H,*Hpre,ourtaoadmmmisfithessroutine,ctx);
}

PETSC_EXTERN void taoadmmsetmisfitconstraintjacobian_(Tao *tao, Mat *J, Mat *Jpre, void (*func)(Tao*, Vec *, Mat *, Mat *, void *, PetscErrorCode *), void *ctx, PetscErrorCode *ierr)
{
    CHKFORTRANNULLFUNCTION(func);
    *ierr = PetscObjectSetFortranCallback((PetscObject)*tao,PETSC_FORTRAN_CALLBACK_CLASS,&_cb.misfitjacobian,(PetscVoidFunction)func,ctx);
    if (!*ierr) *ierr = TaoADMMSetMisfitConstraintJacobian(*tao,*J,*Jpre, ourtaoadmmmisfitconstraintjacobian,ctx);
}

PETSC_EXTERN void taoadmmsetregularizerobjectiveandgradientroutine_(Tao *tao, void (*func)(Tao*, Vec *, PetscReal *, Vec *, void *, PetscErrorCode *), void *ctx, PetscErrorCode *ierr)
{
    CHKFORTRANNULLFUNCTION(func);
    *ierr = PetscObjectSetFortranCallback((PetscObject)*tao,PETSC_FORTRAN_CALLBACK_CLASS,&_cb.regobjgrad,(PetscVoidFunction)func,ctx);
    if (!*ierr) *ierr = TaoADMMSetRegularizerObjectiveAndGradientRoutine(*tao,ourtaoadmmregularizerobjgradroutine,ctx);
}

PETSC_EXTERN void taoadmmsetregularizerhessianroutine_(Tao *tao, Mat *H, Mat *Hpre, void (*func)(Tao*, Vec *, Mat *, Mat *,void *, PetscErrorCode *), void *ctx, PetscErrorCode *ierr)
{
    CHKFORTRANNULLFUNCTION(func);
    *ierr = PetscObjectSetFortranCallback((PetscObject)*tao,PETSC_FORTRAN_CALLBACK_CLASS,&_cb.reghess,(PetscVoidFunction)func,ctx);
    if (!*ierr) *ierr = TaoADMMSetRegularizerHessianRoutine(*tao,*H,*Hpre, ourtaoadmmregularizerhessroutine,ctx);
}

PETSC_EXTERN void taoadmmsetregularizerconstraintjacobian_(Tao *tao, Mat *J, Mat *Jpre, void (*func)(Tao*, Vec *, Mat *, Mat *, void *, PetscErrorCode *), void *ctx, PetscErrorCode *ierr)
{
    CHKFORTRANNULLFUNCTION(func);
    *ierr = PetscObjectSetFortranCallback((PetscObject)*tao,PETSC_FORTRAN_CALLBACK_CLASS,&_cb.misfitjacobian,(PetscVoidFunction)func,ctx);
    if (!*ierr) *ierr = TaoADMMSetRegularizerConstraintJacobian(*tao,*J,*Jpre, ourtaoadmmregularizerconstraintjacobian,ctx);
}
