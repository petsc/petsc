#include <petsc/private/ftnimpl.h>
#include <petsc/private/taolinesearchimpl.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
  #define taolinesearchsetobjectiveroutine_            TAOLINESEARCHSETOBJECTIVEROUTINE
  #define taolinesearchsetgradientroutine_             TAOLINESEARCHSETGRADIENTROUTINE
  #define taolinesearchsetobjectiveandgradientroutine_ TAOLINESEARCHSETOBJECTIVEANDGRADIENTROUTINE
  #define taolinesearchsetobjectiveandgtsroutine_      TAOLINESEARCHSETOBJECTIVEANDGTSROUTINE
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
  #define taolinesearchsetobjectiveroutine_            taolinesearchsetobjectiveroutine
  #define taolinesearchsetgradientroutine_             taolinesearchsetgradientroutine
  #define taolinesearchsetobjectiveandgradientroutine_ taolinesearchsetobjectiveandgradientroutine
  #define taolinesearchsetobjectiveandgtsroutine_      taolinesearchsetobjectiveandgtsroutine
#endif

static int    OBJ     = 0;
static int    GRAD    = 1;
static int    OBJGRAD = 2;
static int    OBJGTS  = 3;
static size_t NFUNCS  = 4;

static PetscErrorCode ourtaolinesearchobjectiveroutine(TaoLineSearch ls, Vec x, PetscReal *f, void *ctx)
{
  PetscCallFortranVoidFunction((*(void (*)(TaoLineSearch *, Vec *, PetscReal *, void *, PetscErrorCode *))(((PetscObject)ls)->fortran_func_pointers[OBJ]))(&ls, &x, f, ctx, &ierr));
  return PETSC_SUCCESS;
}

static PetscErrorCode ourtaolinesearchgradientroutine(TaoLineSearch ls, Vec x, Vec g, void *ctx)
{
  PetscCallFortranVoidFunction((*(void (*)(TaoLineSearch *, Vec *, Vec *, void *, PetscErrorCode *))(((PetscObject)ls)->fortran_func_pointers[GRAD]))(&ls, &x, &g, ctx, &ierr));
  return PETSC_SUCCESS;
}

static PetscErrorCode ourtaolinesearchobjectiveandgradientroutine(TaoLineSearch ls, Vec x, PetscReal *f, Vec g, void *ctx)
{
  PetscCallFortranVoidFunction((*(void (*)(TaoLineSearch *, Vec *, PetscReal *, Vec *, void *, PetscErrorCode *))(((PetscObject)ls)->fortran_func_pointers[OBJGRAD]))(&ls, &x, f, &g, ctx, &ierr));
  return PETSC_SUCCESS;
}

static PetscErrorCode ourtaolinesearchobjectiveandgtsroutine(TaoLineSearch ls, Vec x, Vec s, PetscReal *f, PetscReal *gts, void *ctx)
{
  PetscCallFortranVoidFunction((*(void (*)(TaoLineSearch *, Vec *, Vec *, PetscReal *, PetscReal *, void *, PetscErrorCode *))(((PetscObject)ls)->fortran_func_pointers[OBJGTS]))(&ls, &x, &s, f, gts, ctx, &ierr));
  return PETSC_SUCCESS;
}

PETSC_EXTERN void taolinesearchsetobjectiveroutine_(TaoLineSearch *ls, void (*func)(TaoLineSearch *, Vec *, PetscReal *, void *, PetscErrorCode *), void *ctx, PetscErrorCode *ierr)
{
  PetscObjectAllocateFortranPointers(*ls, NFUNCS);
  if (!func) {
    *ierr = TaoLineSearchSetObjectiveRoutine(*ls, NULL, ctx);
  } else {
    ((PetscObject)*ls)->fortran_func_pointers[OBJ] = (PetscFortranCallbackFn *)func;
    *ierr                                          = TaoLineSearchSetObjectiveRoutine(*ls, ourtaolinesearchobjectiveroutine, ctx);
  }
}

PETSC_EXTERN void taolinesearchsetgradientroutine_(TaoLineSearch *ls, void (*func)(TaoLineSearch *, Vec *, Vec *, void *, PetscErrorCode *), void *ctx, PetscErrorCode *ierr)
{
  PetscObjectAllocateFortranPointers(*ls, NFUNCS);
  if (!func) {
    *ierr = TaoLineSearchSetGradientRoutine(*ls, NULL, ctx);
  } else {
    ((PetscObject)*ls)->fortran_func_pointers[GRAD] = (PetscFortranCallbackFn *)func;
    *ierr                                           = TaoLineSearchSetGradientRoutine(*ls, ourtaolinesearchgradientroutine, ctx);
  }
}

PETSC_EXTERN void taolinesearchsetobjectiveandgradientroutine_(TaoLineSearch *ls, void (*func)(TaoLineSearch *, Vec *, PetscReal *, Vec *, void *, PetscErrorCode *), void *ctx, PetscErrorCode *ierr)
{
  PetscObjectAllocateFortranPointers(*ls, NFUNCS);
  if (!func) {
    *ierr = TaoLineSearchSetObjectiveAndGradientRoutine(*ls, NULL, ctx);
  } else {
    ((PetscObject)*ls)->fortran_func_pointers[OBJGRAD] = (PetscFortranCallbackFn *)func;
    *ierr                                              = TaoLineSearchSetObjectiveAndGradientRoutine(*ls, ourtaolinesearchobjectiveandgradientroutine, ctx);
  }
}

PETSC_EXTERN void taolinesearchsetobjectiveandgtsroutine_(TaoLineSearch *ls, void (*func)(TaoLineSearch *, Vec *, Vec *, PetscReal *, PetscReal *, void *, PetscErrorCode *), void *ctx, PetscErrorCode *ierr)
{
  PetscObjectAllocateFortranPointers(*ls, NFUNCS);
  if (!func) {
    *ierr = TaoLineSearchSetObjectiveAndGTSRoutine(*ls, NULL, ctx);
  } else {
    ((PetscObject)*ls)->fortran_func_pointers[OBJGTS] = (PetscFortranCallbackFn *)func;
    *ierr                                             = TaoLineSearchSetObjectiveAndGTSRoutine(*ls, ourtaolinesearchobjectiveandgtsroutine, ctx);
  }
}
