#include <petsc/private/fortranimpl.h>
#include <petsc/private/f90impl.h>
#include <petsc/private/taoimpl.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define taobrgnsetregularizerobjectiveandgradientroutine_ TAOBRGNSETREGULARIZEROBJECTIVEANDGRADIENTROUTINE
#define taobrgnsetregularizerhessianroutine_              TAOBRGNSETREGULARIZERHESSIANROUTINE
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define taobrgnsetregularizerobjectiveandgradientroutine_ taobrgnsetregularizerobjectiveandgradientroutine
#define taobrgnsetregularizerhessianroutine_              taobrgnsetregularizerhessianroutine
#endif

static struct {
  PetscFortranCallbackId objgrad;
  PetscFortranCallbackId hess;
} _cb;

static PetscErrorCode ourtaobrgnregobjgradroutine(Tao tao, Vec x, PetscReal *f, Vec g, void* ctx)
{
    PetscObjectUseFortranCallback(tao,_cb.objgrad,(Tao*,Vec*,PetscReal*,Vec*,void*,PetscErrorCode*),(&tao,&x,f,&g,_ctx,&ierr));
}

static PetscErrorCode ourtaobrgnreghessroutine(Tao tao, Vec x, Mat H, void *ctx)
{
    PetscObjectUseFortranCallback(tao,_cb.hess,(Tao*,Vec*,Mat*,void*,PetscErrorCode*),(&tao,&x,&H,_ctx,&ierr));
}

PETSC_EXTERN void PETSC_STDCALL taobrgnsetregularizerobjectiveandgradientroutine_(Tao *tao, void (PETSC_STDCALL *func)(Tao*, Vec *, PetscReal *, Vec *, void *, PetscErrorCode *), void *ctx, PetscErrorCode *ierr)
{
    CHKFORTRANNULLFUNCTION(func);
    *ierr = PetscObjectSetFortranCallback((PetscObject)*tao,PETSC_FORTRAN_CALLBACK_CLASS,&_cb.objgrad,(PetscVoidFunction)func,ctx);
    if(!*ierr) *ierr = TaoBRGNSetRegularizerObjectiveAndGradientRoutine(*tao,ourtaobrgnregobjgradroutine,ctx);
}

PETSC_EXTERN void PETSC_STDCALL taobrgnsetregularizerhessianroutine_(Tao *tao, Mat *H, void (PETSC_STDCALL *func)(Tao*, Vec *, Mat *,void *, PetscErrorCode *), void *ctx, PetscErrorCode *ierr)
{
    CHKFORTRANNULLFUNCTION(func);
    *ierr = PetscObjectSetFortranCallback((PetscObject)*tao,PETSC_FORTRAN_CALLBACK_CLASS,&_cb.hess,(PetscVoidFunction)func,ctx);
    if(!*ierr) *ierr = TaoBRGNSetRegularizerHessianRoutine(*tao,*H, ourtaobrgnreghessroutine,ctx);
}
