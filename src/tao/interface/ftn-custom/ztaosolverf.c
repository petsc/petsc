#include <petsc/private/fortranimpl.h>
#include <petsc/private/taoimpl.h>


#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define taosetobjectiveroutine_             TAOSETOBJECTIVEROUTINE
#define taosetgradientroutine_              TAOSETGRADIENTROUTINE
#define taosetobjectiveandgradientroutine_  TAOSETOBJECTIVEANDGRADIENTROUTINE
#define taosethessianroutine_               TAOSETHESSIANROUTINE
#define taosetseparableobjectiveroutine_    TAOSETSEPARABLEOBJECTIVEROUTINE
#define taosetjacobianroutine_              TAOSETJACOBIANROUTINE
#define taosetjacobianstateroutine_         TAOSETJACOBIANSTATEROUTINE
#define taosetjacobiandesignroutine_        TAOSETJACOBIANDESIGNROUTINE
#define taosetjacobianinequalityroutine_    TAOSETJACOBIANINEQUALITYROUTINE
#define taosetjacobianequalityroutine_      TAOSETJACOBIANEQUALITYROUTINE
#define taosetinequalityconstraintsroutine_ TAOSETINEQUALITYCONSTRAINTSROUTINE
#define taosetequalityconstraintsroutine_   TAOSETEQUALITYCONSTRAINTSROUTINE
#define taosetvariableboundsroutine_        TAOSETVARIABLEBOUNDSROUTINE
#define taosetconstraintsroutine_           TAOSETCONSTRAINTSROUTINE
#define taosetmonitor_                      TAOSETMONITOR
#define taosettype_                         TAOSETTYPE
#define taoview_                            TAOVIEW
#define taogetconvergencehistory_           TAOGETCONVERGENCEHISTORY
#define taosetconvergencetest_              TAOSETCONVERGENCETEST
#define taogetoptionsprefix_                TAOGETOPTIONSPREFIX
#define taosetoptionsprefix_                TAOSETOPTIONSPREFIX
#define taoappendoptionsprefix_             TAOAPPENDOPTIONSPREFIX
#define taogettype_                         TAOGETTYPE
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)

#define taosetobjectiveroutine_             taosetobjectiveroutine
#define taosetgradientroutine_              taosetgradientroutine
#define taosetobjectiveandgradientroutine_  taosetobjectiveandgradientroutine
#define taosethessianroutine_               taosethessianroutine
#define taosetseparableobjectiveroutine_    taosetseparableobjectiveroutine
#define taosetjacobianroutine_              taosetjacobianroutine
#define taosetjacobianstateroutine_         taosetjacobianstateroutine
#define taosetjacobiandesignroutine_        taosetjacobiandesignroutine
#define taosetjacobianinequalityroutine_    taosetjacobianinequalityroutine
#define taosetjacobianequalityroutine_      taosetjacobianequalityroutine
#define taosetinequalityconstraintsroutine_ taosetinequalityconstraintsroutine
#define taosetequalityconstraintsroutine_   taosetequalityconstraintsroutine
#define taosetvariableboundsroutine_        taosetvariableboundsroutine
#define taosetconstraintsroutine_           taosetconstraintsroutine
#define taosetmonitor_                      taosetmonitor
#define taosettype_                         taosettype
#define taoview_                            taoview
#define taogetconvergencehistory_           taogetconvergencehistory
#define taosetconvergencetest_              taosetconvergencetest
#define taogetoptionsprefix_                taogetoptionsprefix
#define taosetoptionsprefix_                taosetoptionsprefix
#define taoappendoptionsprefix_             taoappendoptionsprefix
#define taogettype_                         taogettype
#endif

static int OBJ=0;       /*  objective routine index */
static int GRAD=1;      /*  gradient routine index */
static int OBJGRAD=2;   /*  objective and gradient routine */
static int HESS=3;      /*  hessian routine index */
static int SEPOBJ=4;    /*  separable objective routine index */
static int JAC=5;       /*  jacobian routine index */
static int JACSTATE=6;  /*  jacobian state routine index */
static int JACDESIGN=7; /*  jacobian design routine index */
static int BOUNDS=8;
static int MON=9;       /*  monitor routine index */
static int MONCTX=10;       /*  monitor routine index */
static int MONDESTROY=11; /*  monitor destroy index */
static int CONVTEST=12;  /*  */
static int CONSTRAINTS=13;
static int JACINEQ=14;
static int JACEQ=15;
static int CONINEQ=16;
static int CONEQ=17;
static int NFUNCS=18;

static PetscErrorCode ourtaoobjectiveroutine(Tao tao, Vec x, PetscReal *f, void *ctx)
{
    PetscErrorCode ierr = 0;
    (*(void (PETSC_STDCALL *)(Tao*,Vec*,PetscReal*,void*,PetscErrorCode*))
        (((PetscObject)tao)->fortran_func_pointers[OBJ]))(&tao,&x,f,ctx,&ierr);
    CHKERRQ(ierr);
    return 0;
}

static PetscErrorCode ourtaogradientroutine(Tao tao, Vec x, Vec g, void *ctx)
{
    PetscErrorCode ierr = 0;
    (*(void (PETSC_STDCALL *)(Tao*,Vec*,Vec*,void*,PetscErrorCode*))
       (((PetscObject)tao)->fortran_func_pointers[GRAD]))(&tao,&x,&g,ctx,&ierr);
    CHKERRQ(ierr);
    return 0;

}

static PetscErrorCode ourtaoobjectiveandgradientroutine(Tao tao, Vec x, PetscReal *f, Vec g, void* ctx)
{
    PetscErrorCode ierr = 0;
    (*(void (PETSC_STDCALL *)(Tao*,Vec*,PetscReal*,Vec*,void*,PetscErrorCode*))
     (((PetscObject)tao)->fortran_func_pointers[OBJGRAD]))(&tao,&x,f,&g,ctx,&ierr);
    CHKERRQ(ierr);
    return 0;
}

static PetscErrorCode ourtaohessianroutine(Tao tao, Vec x, Mat H, Mat Hpre, void *ctx)
{
    PetscErrorCode ierr = 0;
    (*(void (PETSC_STDCALL *)(Tao*,Vec*,Mat*,Mat*,void*,PetscErrorCode*))
     (((PetscObject)tao)->fortran_func_pointers[HESS]))(&tao,&x,&H,&Hpre,ctx,&ierr); CHKERRQ(ierr);
    return 0;
}

static PetscErrorCode ourtaojacobianroutine(Tao tao, Vec x, Mat H, Mat Hpre, void *ctx)
{
    PetscErrorCode ierr = 0;
    (*(void (PETSC_STDCALL *)(Tao*,Vec*,Mat*,Mat*,void*,PetscErrorCode*))
     (((PetscObject)tao)->fortran_func_pointers[JAC]))(&tao,&x,&H,&Hpre,ctx,&ierr); CHKERRQ(ierr);
    return 0;
}

static PetscErrorCode ourtaojacobianstateroutine(Tao tao, Vec x, Mat H, Mat Hpre, Mat Hinv, void *ctx)
{
    PetscErrorCode ierr = 0;
    (*(void (PETSC_STDCALL *)(Tao*,Vec*,Mat*,Mat*,Mat*,void*,PetscErrorCode*))
     (((PetscObject)tao)->fortran_func_pointers[JACSTATE]))(&tao,&x,&H,&Hpre,&Hinv,ctx,&ierr); CHKERRQ(ierr);
    return 0;
}

static PetscErrorCode ourtaojacobiandesignroutine(Tao tao, Vec x, Mat H, void *ctx)
{
    PetscErrorCode ierr = 0;
    (*(void (PETSC_STDCALL *)(Tao*,Vec*,Mat*,void*,PetscErrorCode*))
     (((PetscObject)tao)->fortran_func_pointers[JACDESIGN]))(&tao,&x,&H,ctx,&ierr); CHKERRQ(ierr);
    return 0;
}

static PetscErrorCode ourtaoboundsroutine(Tao tao, Vec xl, Vec xu, void *ctx)
{
    PetscErrorCode ierr = 0;
    (*(void (PETSC_STDCALL *)(Tao*,Vec*,Vec*,void*,PetscErrorCode*))
     (((PetscObject)tao)->fortran_func_pointers[BOUNDS]))(&tao,&xl,&xu,ctx,&ierr); CHKERRQ(ierr);
    return 0;
}
static PetscErrorCode ourtaoseparableobjectiveroutine(Tao tao, Vec x, Vec f, void *ctx)
{
    PetscErrorCode ierr = 0;
    (*(void (PETSC_STDCALL *)(Tao*,Vec*,Vec*,void*,PetscErrorCode*))
     (((PetscObject)tao)->fortran_func_pointers[SEPOBJ]))(&tao,&x,&f,ctx,&ierr);
    return 0;
}

static PetscErrorCode ourtaomonitor(Tao tao, void *ctx)
{
    PetscErrorCode ierr = 0;
    (*(void (PETSC_STDCALL *)(Tao *, void*, PetscErrorCode*))
     (((PetscObject)tao)->fortran_func_pointers[MON]))(&tao,ctx,&ierr);
    CHKERRQ(ierr);
    return 0;
}

static PetscErrorCode ourtaomondestroy(void **ctx)
{
    PetscErrorCode ierr = 0;
    Tao tao = *(Tao*)ctx;
    if (((PetscObject)tao)->fortran_func_pointers[MONDESTROY]) {
      (*(void (PETSC_STDCALL *)(void*,PetscErrorCode*))(((PetscObject)tao)->fortran_func_pointers[MONDESTROY]))
        ((void*)(PETSC_UINTPTR_T)((PetscObject)tao)->fortran_func_pointers[MONCTX],&ierr);
      CHKERRQ(ierr);
    }
    return 0;
}
static PetscErrorCode ourtaoconvergencetest(Tao tao, void *ctx)
{
    PetscErrorCode ierr = 0;
    (*(void (PETSC_STDCALL *)(Tao *, void*, PetscErrorCode*))
     (((PetscObject)tao)->fortran_func_pointers[CONVTEST]))(&tao,ctx,&ierr);
    CHKERRQ(ierr);
    return 0;
}


static PetscErrorCode ourtaoconstraintsroutine(Tao tao, Vec x, Vec c, void *ctx)
{
    PetscErrorCode ierr = 0;
    (*(void (PETSC_STDCALL *)(Tao*,Vec*,Vec*,void*,PetscErrorCode*))
       (((PetscObject)tao)->fortran_func_pointers[CONSTRAINTS]))(&tao,&x,&c,ctx,&ierr);
    CHKERRQ(ierr);
    return 0;

}

static PetscErrorCode ourtaojacobianinequalityroutine(Tao tao, Vec x, Mat J, Mat Jpre, void *ctx)
{
    PetscErrorCode ierr = 0;
    (*(void (PETSC_STDCALL *)(Tao*,Vec*,Mat*,Mat*,void*,PetscErrorCode*))
     (((PetscObject)tao)->fortran_func_pointers[JACINEQ]))(&tao,&x,&J,&Jpre,ctx,&ierr); CHKERRQ(ierr);
    return 0;
}

static PetscErrorCode ourtaojacobianequalityroutine(Tao tao, Vec x, Mat J, Mat Jpre, void *ctx)
{
    PetscErrorCode ierr = 0;
    (*(void (PETSC_STDCALL *)(Tao*,Vec*,Mat*,Mat*,void*,PetscErrorCode*))
     (((PetscObject)tao)->fortran_func_pointers[JACEQ]))(&tao,&x,&J,&Jpre,ctx,&ierr); CHKERRQ(ierr);
    return 0;
}

static PetscErrorCode ourtaoinequalityconstraintsroutine(Tao tao, Vec x, Vec c, void *ctx)
{
    PetscErrorCode ierr = 0;
    (*(void (PETSC_STDCALL *)(Tao*,Vec*,Vec*,void*,PetscErrorCode*))
       (((PetscObject)tao)->fortran_func_pointers[CONINEQ]))(&tao,&x,&c,ctx,&ierr);
    CHKERRQ(ierr);
    return 0;

}

static PetscErrorCode ourtaoequalityconstraintsroutine(Tao tao, Vec x, Vec c, void *ctx)
{
    PetscErrorCode ierr = 0;
    (*(void (PETSC_STDCALL *)(Tao*,Vec*,Vec*,void*,PetscErrorCode*))
       (((PetscObject)tao)->fortran_func_pointers[CONEQ]))(&tao,&x,&c,ctx,&ierr);
    CHKERRQ(ierr);
    return 0;

}


EXTERN_C_BEGIN


PETSC_EXTERN void PETSC_STDCALL taosetobjectiveroutine_(Tao *tao, void (PETSC_STDCALL *func)(Tao*, Vec *, PetscReal *, void *, PetscErrorCode *), void *ctx, PetscErrorCode *ierr)
{
    PetscObjectAllocateFortranPointers(*tao,NFUNCS);
    if (!func) {
        *ierr = TaoSetObjectiveRoutine(*tao,0,ctx);
    } else {
        ((PetscObject)*tao)->fortran_func_pointers[OBJ] = (PetscVoidFunction)func;
        *ierr = TaoSetObjectiveRoutine(*tao, ourtaoobjectiveroutine,ctx);
    }
}

PETSC_EXTERN void PETSC_STDCALL taosetgradientroutine_(Tao *tao, void (PETSC_STDCALL *func)(Tao*, Vec *, Vec *, void *, PetscErrorCode *), void *ctx, PetscErrorCode *ierr)
{
    PetscObjectAllocateFortranPointers(*tao,NFUNCS);
    if (!func) {
        *ierr = TaoSetGradientRoutine(*tao,0,ctx);
    } else {
        ((PetscObject)*tao)->fortran_func_pointers[GRAD] = (PetscVoidFunction)func;
        *ierr = TaoSetGradientRoutine(*tao, ourtaogradientroutine,ctx);
    }
}

PETSC_EXTERN void PETSC_STDCALL taosetobjectiveandgradientroutine_(Tao *tao, void (PETSC_STDCALL *func)(Tao*, Vec *, PetscReal *, Vec *, void *, PetscErrorCode *), void *ctx, PetscErrorCode *ierr)
{
    PetscObjectAllocateFortranPointers(*tao,NFUNCS);
    if (!func) {
        *ierr = TaoSetObjectiveAndGradientRoutine(*tao,0,ctx);
    } else {
        ((PetscObject)*tao)->fortran_func_pointers[OBJGRAD] = (PetscVoidFunction)func;
        *ierr = TaoSetObjectiveAndGradientRoutine(*tao, ourtaoobjectiveandgradientroutine,ctx);
    }
}




PETSC_EXTERN void PETSC_STDCALL taosetseparableobjectiveroutine_(Tao *tao, Vec *F, void (PETSC_STDCALL *func)(Tao*, Vec *, Vec *, void *, PetscErrorCode *), void *ctx, PetscErrorCode *ierr)
{
    PetscObjectAllocateFortranPointers(*tao,NFUNCS);
    if (!func) {
        *ierr = TaoSetSeparableObjectiveRoutine(*tao,*F,0,ctx);
    } else {
        ((PetscObject)*tao)->fortran_func_pointers[SEPOBJ] = (PetscVoidFunction)func;
        *ierr = TaoSetSeparableObjectiveRoutine(*tao,*F, ourtaoseparableobjectiveroutine,ctx);
    }
}



PETSC_EXTERN void PETSC_STDCALL taosetjacobianroutine_(Tao *tao, Mat *J, Mat *Jp, void (PETSC_STDCALL *func)(Tao*, Vec *, Mat *, Mat *, void *, PetscErrorCode *), void *ctx, PetscErrorCode *ierr)
{
    PetscObjectAllocateFortranPointers(*tao,NFUNCS);
    if (!func) {
        *ierr = TaoSetJacobianRoutine(*tao,*J,*Jp,0,ctx);
    } else {
        ((PetscObject)*tao)->fortran_func_pointers[JAC] = (PetscVoidFunction)func;
        *ierr = TaoSetJacobianRoutine(*tao,*J, *Jp, ourtaojacobianroutine,ctx);
    }
}

PETSC_EXTERN void PETSC_STDCALL taosetjacobianstateroutine_(Tao *tao, Mat *J, Mat *Jp, Mat*Jinv, void (PETSC_STDCALL *func)(Tao*, Vec *, Mat *, Mat *, Mat*, void *, PetscErrorCode *), void *ctx, PetscErrorCode *ierr)
{
    PetscObjectAllocateFortranPointers(*tao,NFUNCS);
    if (!func) {
      *ierr = TaoSetJacobianStateRoutine(*tao,*J,*Jp,*Jinv,0,ctx);
    } else {
      ((PetscObject)*tao)->fortran_func_pointers[JACSTATE] = (PetscVoidFunction)func;
      *ierr = TaoSetJacobianStateRoutine(*tao,*J, *Jp, *Jinv, ourtaojacobianstateroutine,ctx);
    }
}

PETSC_EXTERN void PETSC_STDCALL taosetjacobiandesignroutine_(Tao *tao, Mat *J, void (PETSC_STDCALL *func)(Tao*, Vec *, Mat *, void *, PetscErrorCode *), void *ctx, PetscErrorCode *ierr)
{
    PetscObjectAllocateFortranPointers(*tao,NFUNCS);
    if (!func) {
        *ierr = TaoSetJacobianDesignRoutine(*tao,*J,0,ctx);
    } else {
        ((PetscObject)*tao)->fortran_func_pointers[JACDESIGN] = (PetscVoidFunction)func;
        *ierr = TaoSetJacobianDesignRoutine(*tao,*J, ourtaojacobiandesignroutine,ctx);
    }
}


PETSC_EXTERN void PETSC_STDCALL taosethessianroutine_(Tao *tao, Mat *J, Mat *Jp, void (PETSC_STDCALL *func)(Tao*, Vec *, Mat *, Mat *,void *, PetscErrorCode *), void *ctx, PetscErrorCode *ierr)
{
    PetscObjectAllocateFortranPointers(*tao,NFUNCS);
    if (!func) {
        *ierr = TaoSetHessianRoutine(*tao,*J,*Jp,0,ctx);
    } else {
        ((PetscObject)*tao)->fortran_func_pointers[HESS] = (PetscVoidFunction)func;
        *ierr = TaoSetHessianRoutine(*tao,*J, *Jp, ourtaohessianroutine,ctx);
    }
}

PETSC_EXTERN void PETSC_STDCALL taosetvariableboundsroutine_(Tao *tao, void (PETSC_STDCALL *func)(Tao*,Vec*,Vec*,void*,PetscErrorCode*),void *ctx, PetscErrorCode *ierr)
{
    PetscObjectAllocateFortranPointers(*tao,NFUNCS);
    if (func) {
        ((PetscObject)*tao)->fortran_func_pointers[BOUNDS] = (PetscVoidFunction)func;
        *ierr = TaoSetVariableBoundsRoutine(*tao,ourtaoboundsroutine,ctx);
    } else {
        *ierr = TaoSetVariableBoundsRoutine(*tao,0,ctx);
    }

}
PETSC_EXTERN void PETSC_STDCALL taosetmonitor_(Tao *tao, void (PETSC_STDCALL *func)(Tao*,void*,PetscErrorCode*),void *ctx, void (PETSC_STDCALL *mondestroy)(void*,PetscErrorCode*),PetscErrorCode *ierr)
{
    PetscObjectAllocateFortranPointers(*tao,NFUNCS);
    if (func) {
        ((PetscObject)*tao)->fortran_func_pointers[MON] = (PetscVoidFunction)func;
        CHKFORTRANNULLFUNCTION(mondestroy);
        ((PetscObject)*tao)->fortran_func_pointers[MONDESTROY] = (PetscVoidFunction)mondestroy;
        *ierr = TaoSetMonitor(*tao,ourtaomonitor,ctx,ourtaomondestroy);
    }
}

PETSC_EXTERN void PETSC_STDCALL taosetconvergencetest_(Tao *tao, void (PETSC_STDCALL *func)(Tao*,void*,PetscErrorCode*),void *ctx, PetscErrorCode *ierr)
{
    PetscObjectAllocateFortranPointers(*tao,NFUNCS);
    if (!func) {
        *ierr = TaoSetConvergenceTest(*tao,0,ctx);
    } else {
        ((PetscObject)*tao)->fortran_func_pointers[CONVTEST] = (PetscVoidFunction)func;
        *ierr = TaoSetConvergenceTest(*tao,ourtaoconvergencetest,ctx);
    }
}


PETSC_EXTERN void PETSC_STDCALL taosetconstraintsroutine_(Tao *tao, Vec *C, void (PETSC_STDCALL *func)(Tao*, Vec *, Vec *, void *, PetscErrorCode *), void *ctx, PetscErrorCode *ierr)
{
    PetscObjectAllocateFortranPointers(*tao,NFUNCS);
    if (!func) {
      *ierr = TaoSetConstraintsRoutine(*tao,*C,0,ctx);
    } else {
        ((PetscObject)*tao)->fortran_func_pointers[CONSTRAINTS] = (PetscVoidFunction)func;
        *ierr = TaoSetConstraintsRoutine(*tao, *C, ourtaoconstraintsroutine,ctx);
    }
}


PETSC_EXTERN void PETSC_STDCALL taosettype_(Tao *tao, char* type_name PETSC_MIXED_LEN(len), PetscErrorCode *ierr PETSC_END_LEN(len))

{
    char *t;

    FIXCHAR(type_name,len,t);
    *ierr = TaoSetType(*tao,t);
    FREECHAR(type_name,t);

}

PETSC_EXTERN void PETSC_STDCALL taoview_(Tao *tao, PetscViewer *viewer, PetscErrorCode *ierr)
{
    PetscViewer v;
    PetscPatchDefaultViewers_Fortran(viewer,v);
    *ierr = TaoView(*tao,v);
}

PETSC_EXTERN void PETSC_STDCALL taogetconvergencehistory_(Tao *tao, PetscInt *nhist, PetscErrorCode *ierr)
{
  *ierr = TaoGetConvergenceHistory(*tao,NULL,NULL,NULL,NULL,nhist);
}

PETSC_EXTERN void PETSC_STDCALL taogetoptionsprefix_(Tao *tao, char* prefix PETSC_MIXED_LEN(len), PetscErrorCode *ierr PETSC_END_LEN(len))
{
  const char *name;
  *ierr = TaoGetOptionsPrefix(*tao,&name);
  *ierr = PetscStrncpy(prefix,name,len); if (*ierr) return;
  FIXRETURNCHAR(PETSC_TRUE,prefix,len);

}

PETSC_EXTERN void PETSC_STDCALL taoappendoptionsprefix_(Tao *tao, char* prefix PETSC_MIXED_LEN(len),PetscErrorCode *ierr PETSC_END_LEN(len))
{
  char *name;
  FIXCHAR(prefix,len,name);
  *ierr = TaoAppendOptionsPrefix(*tao,name);
  FREECHAR(prefix,name);
}

PETSC_EXTERN void PETSC_STDCALL taosetoptionsprefix_(Tao *tao, char* prefix PETSC_MIXED_LEN(len),PetscErrorCode *ierr PETSC_END_LEN(len))
{
  char *t;
  FIXCHAR(prefix,len,t);
  *ierr = TaoSetOptionsPrefix(*tao,t);
  FREECHAR(prefix,t);
}

PETSC_EXTERN void PETSC_STDCALL taogettype_(Tao *tao, char* name PETSC_MIXED_LEN(len), PetscErrorCode *ierr  PETSC_END_LEN(len))
{
  const char *tname;
  *ierr = TaoGetType(*tao,&tname);
  *ierr = PetscStrncpy(name,tname,len); if (*ierr) return;
  FIXRETURNCHAR(PETSC_TRUE,name,len);

}


PETSC_EXTERN void PETSC_STDCALL taosetjacobianinequalityroutine_(Tao *tao, Mat *J, Mat *Jp, void (PETSC_STDCALL *func)(Tao*, Vec *, Mat *, Mat *,void *, PetscErrorCode *), void *ctx, PetscErrorCode *ierr)
{
    PetscObjectAllocateFortranPointers(*tao,NFUNCS);
    if (!func) {
        *ierr = TaoSetJacobianInequalityRoutine(*tao,*J,*Jp,0,ctx);
    } else {
        ((PetscObject)*tao)->fortran_func_pointers[JACINEQ] = (PetscVoidFunction)func;
        *ierr = TaoSetJacobianInequalityRoutine(*tao,*J, *Jp, ourtaojacobianinequalityroutine,ctx);
    }
}

PETSC_EXTERN void PETSC_STDCALL taosetjacobianequalityroutine_(Tao *tao, Mat *J, Mat *Jp, void (PETSC_STDCALL *func)(Tao*, Vec *, Mat *, Mat *, void *, PetscErrorCode *), void *ctx, PetscErrorCode *ierr)
{
    PetscObjectAllocateFortranPointers(*tao,NFUNCS);
    if (!func) {
        *ierr = TaoSetJacobianEqualityRoutine(*tao,*J,*Jp,0,ctx);
    } else {
        ((PetscObject)*tao)->fortran_func_pointers[JACEQ] = (PetscVoidFunction)func;
        *ierr = TaoSetJacobianEqualityRoutine(*tao,*J, *Jp, ourtaojacobianequalityroutine,ctx);
    }
}


PETSC_EXTERN void PETSC_STDCALL taosetinequalityconstraintsroutine_(Tao *tao, Vec *C, void (PETSC_STDCALL *func)(Tao*, Vec *, Vec *, void *, PetscErrorCode *), void *ctx, PetscErrorCode *ierr)
{
    PetscObjectAllocateFortranPointers(*tao,NFUNCS);
    if (!func) {
      *ierr = TaoSetInequalityConstraintsRoutine(*tao,*C,0,ctx);
    } else {
        ((PetscObject)*tao)->fortran_func_pointers[CONINEQ] = (PetscVoidFunction)func;
        *ierr = TaoSetInequalityConstraintsRoutine(*tao, *C, ourtaoinequalityconstraintsroutine,ctx);
    }
}

PETSC_EXTERN void PETSC_STDCALL taosetequalityconstraintsroutine_(Tao *tao, Vec *C, void (PETSC_STDCALL *func)(Tao*, Vec *, Vec *, void *, PetscErrorCode *), void *ctx, PetscErrorCode *ierr)
{
    PetscObjectAllocateFortranPointers(*tao,NFUNCS);
    if (!func) {
      *ierr = TaoSetEqualityConstraintsRoutine(*tao,*C,0,ctx);
    } else {
        ((PetscObject)*tao)->fortran_func_pointers[CONEQ] = (PetscVoidFunction)func;
        *ierr = TaoSetEqualityConstraintsRoutine(*tao, *C, ourtaoequalityconstraintsroutine,ctx);
    }
}


EXTERN_C_END


