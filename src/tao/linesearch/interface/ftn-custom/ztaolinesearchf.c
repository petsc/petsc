#include <petsc/private/fortranimpl.h>
#include <petsc/private/taolinesearchimpl.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define taolinesearchsetobjectiveroutine_            TAOLINESEARCHSETOBJECTIVEROUTINE
#define taolinesearchsetgradientroutine_             TAOLINESEARCHSETGRADIENTROUTINE
#define taolinesearchsetobjectiveandgradientroutine_ TAOLINESEARCHSETOBJECTIVEANDGRADIENTROUTINE
#define taolinesearchsetobjectiveandgtsroutine_      TAOLINESEARCHSETOBJECTIVEANDGTSROUTINE
#define taolinesearchview_                           TAOLINESEARCHVIEW
#define taolinesearchsettype_                        TAOLINESEARCHSETTYPE
#define taolinesearchviewfromoptions_                TAOLINESEARCHVIEWFROMOPTIONS
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)

#define taolinesearchsetobjectiveroutine_            taolinesearchsetobjectiveroutine
#define taolinesearchsetgradientroutine_             taolinesearchsetgradientroutine
#define taolinesearchsetobjectiveandgradientroutine_ taolinesearchsetobjectiveandgradientroutine
#define taolinesearchsetobjectiveandgtsroutine_      taolinesearchsetobjectiveandgtsroutine
#define taolinesearchview_                           taolinesearchview
#define taolinesearchsettype_                        taolinesearchsettype
#define taolinesearchviewfromoptions_                taolinesearchviewfromoptions
#endif

static int OBJ=0;
static int GRAD=1;
static int OBJGRAD=2;
static int OBJGTS=3;
static size_t NFUNCS=4;

static PetscErrorCode ourtaolinesearchobjectiveroutine(TaoLineSearch ls, Vec x, PetscReal *f, void *ctx)
{
    CHKERR_FORTRAN_VOID_FUNCTION((*(void (*)(TaoLineSearch*,Vec*,PetscReal*,void*,PetscErrorCode*)) (((PetscObject)ls)->fortran_func_pointers[OBJ]))(&ls,&x,f,ctx,&ierr));
    return 0;
}

static PetscErrorCode ourtaolinesearchgradientroutine(TaoLineSearch ls, Vec x, Vec g, void *ctx)
{
    CHKERR_FORTRAN_VOID_FUNCTION((*(void (*)(TaoLineSearch*,Vec*,Vec*,void*,PetscErrorCode*)) (((PetscObject)ls)->fortran_func_pointers[GRAD]))(&ls,&x,&g,ctx,&ierr));
    return 0;
}

static PetscErrorCode ourtaolinesearchobjectiveandgradientroutine(TaoLineSearch ls, Vec x, PetscReal *f, Vec g, void* ctx)
{
    CHKERR_FORTRAN_VOID_FUNCTION((*(void (*)(TaoLineSearch*,Vec*,PetscReal*,Vec*,void*,PetscErrorCode*))(((PetscObject)ls)->fortran_func_pointers[OBJGRAD]))(&ls,&x,f,&g,ctx,&ierr));
    return 0;
}

static PetscErrorCode ourtaolinesearchobjectiveandgtsroutine(TaoLineSearch ls, Vec x, Vec s, PetscReal *f, PetscReal *gts, void* ctx)
{
    CHKERR_FORTRAN_VOID_FUNCTION((*(void (*)(TaoLineSearch*,Vec*,Vec*,PetscReal*,PetscReal*,void*,PetscErrorCode*)) (((PetscObject)ls)->fortran_func_pointers[OBJGTS]))(&ls,&x,&s,f,gts,ctx,&ierr));
    return 0;
}

PETSC_EXTERN void taolinesearchsetobjectiveroutine_(TaoLineSearch *ls, void (*func)(TaoLineSearch*, Vec *, PetscReal *, void *, PetscErrorCode *), void *ctx, PetscErrorCode *ierr)
{
    PetscObjectAllocateFortranPointers(*ls,NFUNCS);
    if (!func) {
        *ierr = TaoLineSearchSetObjectiveRoutine(*ls,0,ctx);
    } else {
        ((PetscObject)*ls)->fortran_func_pointers[OBJ] = (PetscVoidFunction)func;
        *ierr = TaoLineSearchSetObjectiveRoutine(*ls, ourtaolinesearchobjectiveroutine,ctx);
    }
}

PETSC_EXTERN void taolinesearchsetgradientroutine_(TaoLineSearch *ls, void (*func)(TaoLineSearch*, Vec *, Vec *, void *, PetscErrorCode *), void *ctx, PetscErrorCode *ierr)
{
    PetscObjectAllocateFortranPointers(*ls,NFUNCS);
    if (!func) {
        *ierr = TaoLineSearchSetGradientRoutine(*ls,0,ctx);
    } else {
        ((PetscObject)*ls)->fortran_func_pointers[GRAD] = (PetscVoidFunction)func;
        *ierr = TaoLineSearchSetGradientRoutine(*ls, ourtaolinesearchgradientroutine,ctx);
    }
}

PETSC_EXTERN void taolinesearchsetobjectiveandgradientroutine_(TaoLineSearch *ls, void (*func)(TaoLineSearch*, Vec *, PetscReal *, Vec *, void *, PetscErrorCode *), void *ctx, PetscErrorCode *ierr)
{
    PetscObjectAllocateFortranPointers(*ls,NFUNCS);
    if (!func) {
        *ierr = TaoLineSearchSetObjectiveAndGradientRoutine(*ls,0,ctx);
    } else {
        ((PetscObject)*ls)->fortran_func_pointers[OBJGRAD] = (PetscVoidFunction)func;
        *ierr = TaoLineSearchSetObjectiveAndGradientRoutine(*ls, ourtaolinesearchobjectiveandgradientroutine,ctx);
    }
}

PETSC_EXTERN void taolinesearchsetobjectiveandgtsroutine_(TaoLineSearch *ls, void (*func)(TaoLineSearch*, Vec *, Vec *, PetscReal*, PetscReal*,void*, PetscErrorCode *), void *ctx, PetscErrorCode *ierr)
{
    PetscObjectAllocateFortranPointers(*ls,NFUNCS);
    if (!func) {
        *ierr = TaoLineSearchSetObjectiveAndGTSRoutine(*ls,0,ctx);
    } else {
        ((PetscObject)*ls)->fortran_func_pointers[OBJGTS] = (PetscVoidFunction)func;
        *ierr = TaoLineSearchSetObjectiveAndGTSRoutine(*ls, ourtaolinesearchobjectiveandgtsroutine,ctx);
    }
}

PETSC_EXTERN void taolinesearchsettype_(TaoLineSearch *ls, char* type_name, PetscErrorCode *ierr,PETSC_FORTRAN_CHARLEN_T len)

{
    char *t;

    FIXCHAR(type_name,len,t);
    *ierr = TaoLineSearchSetType(*ls,t);if (*ierr) return;
    FREECHAR(type_name,t);

}

PETSC_EXTERN void taolinesearchview_(TaoLineSearch *ls, PetscViewer *viewer, PetscErrorCode *ierr)
{
    PetscViewer v;
    PetscPatchDefaultViewers_Fortran(viewer,v);
    *ierr = TaoLineSearchView(*ls,v);
}

PETSC_EXTERN void taolinesearchgetoptionsprefix_(TaoLineSearch *ls, char* prefix, PetscErrorCode *ierr,PETSC_FORTRAN_CHARLEN_T len)
{
  const char *name;
  *ierr = TaoLineSearchGetOptionsPrefix(*ls,&name);
  *ierr = PetscStrncpy(prefix,name,len); if (*ierr) return;
  FIXRETURNCHAR(PETSC_TRUE,prefix,len);

}

PETSC_EXTERN void taolinesearchappendoptionsprefix_(TaoLineSearch *ls, char* prefix,PetscErrorCode *ierr,PETSC_FORTRAN_CHARLEN_T len)
{
  char *name;
  FIXCHAR(prefix,len,name);
  *ierr = TaoLineSearchAppendOptionsPrefix(*ls,name);if (*ierr) return;
  FREECHAR(prefix,name);
}

PETSC_EXTERN void taolinesearchsetoptionsprefix_(TaoLineSearch *ls, char* prefix,PetscErrorCode *ierr,PETSC_FORTRAN_CHARLEN_T len)
{
  char *t;
  FIXCHAR(prefix,len,t);
  *ierr = TaoLineSearchSetOptionsPrefix(*ls,t);if (*ierr) return;
  FREECHAR(prefix,t);
}

PETSC_EXTERN void taolinesearchgettype_(TaoLineSearch *ls, char* name, PetscErrorCode *ierr,PETSC_FORTRAN_CHARLEN_T len)
{
  const char *tname;
  *ierr = TaoLineSearchGetType(*ls,&tname);
  *ierr = PetscStrncpy(name,tname,len); if (*ierr) return;
  FIXRETURNCHAR(PETSC_TRUE,name,len);

}
PETSC_EXTERN void taolinesearchviewfromoptions_(TaoLineSearch *ao,PetscObject obj,char* type,PetscErrorCode *ierr,PETSC_FORTRAN_CHARLEN_T len)
{
  char *t;

  FIXCHAR(type,len,t);
  CHKFORTRANNULLOBJECT(obj);
  *ierr = TaoLineSearchViewFromOptions(*ao,obj,t);if (*ierr) return;
  FREECHAR(type,t);
}
