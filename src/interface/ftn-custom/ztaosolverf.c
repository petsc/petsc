#include "private/fortranimpl.h"
#include "private/taosolver_impl.h"


#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define taosolversetobjectiveroutine_            TAOSOLVERSETOBJECTIVEROUTINE
#define taosolversetgradientroutine_             TAOSOLVERSETGRADIENTROUTINE
#define taosolversetobjectiveandgradientroutine_ TAOSOLVERSETOBJECTIVEANDGRADIENTROUTINE
#define taosolversethessianroutine_              TAOSOLVERSETHESSIANROUTINE
#define taosolversetseparableobjectiveroutine_   TAOSOLVERSETSEPARABLEOBJECTIVEROUTINE
#define taosolversetjacobianroutine_             TAOSOLVERSETJACOBIANROUTINE
#define taosolversetjacobianstateroutine_        TAOSOLVERSETJACOBIANSTATEROUTINE
#define taosolversetjacobiandesignroutine_       TAOSOLVERSETJACOBIANDESIGNROUTINE
#define taosolversetvariableboundsroutine_       TAOSOLVERSETVARIABLEBOUNDSROUTINE
#define taosolversetconstraintsroutine_          TAOSOLVERSETCONSTRAINTSROUTINE
#define taosolversetmonitor_                     TAOSOLVERSETMONITOR
#define taosolversettype_                        TAOSOLVERSETTYPE
#define taosolverview_                           TAOSOLVERVIEW
#define taosolvergethistory_                     TAOSOLVERGETHISTORY
#define taosolversetconvergencetest_             TAOSOLVERSETCONVERGENCETEST
#define taosolvergetoptionsprefix_               TAOSOLVERGETOPTIONSPREFIX
#define taosolversetoptionsprefix_               TAOSOLVERSETOPTIONSPREFIX
#define taosolverappendoptionsprefix_            TAOSOLVERAPPENDOPTIONSPREFIX
#define taosolvergettype_                        TAOSOLVERGETTYPE
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)

#define taosolversetobjectiveroutine_            taosolversetobjectiveroutine
#define taosolversetgradientroutine_             taosolversetgradientroutine
#define taosolversetobjectiveandgradientroutine_ taosolversetobjectiveandgradientroutine
#define taosolversethessianroutine_              taosolversethessianroutine
#define taosolversetseparableobjectiveroutine_   taosolversetseparableobjectiveroutine
#define taosolversetjacobianroutine_             taosolversetjacobianroutine
#define taosolversetjacobianstateroutine_        taosolversetjacobianstateroutine
#define taosolversetjacobiandesignroutine_       taosolversetjacobiandesignroutine
#define taosolversetvariableboundsroutine_       taosolversetvariableboundsroutine
#define taosolversetconstraintsroutine_          taosolversetconstraintsroutine
#define taosolversetmonitor_                     taosolversetmonitor
#define taosolversettype_                        taosolversettype
#define taosolverview_                           taosolverview
#define taosolvergethistory_                     taosolvergethistory
#define taosolversetconvergencetest_             taosolversetconvergencetest
#define taosolvergetoptionsprefix_               taosolvergetoptionsprefix
#define taosolversetoptionsprefix_               taosolversetoptionsprefix
#define taosolverappendoptionsprefix_            taosolverappendoptionsprefix
#define taosolvergettype_                        taosolvergettype
#endif

static int OBJ=0;       // objective routine index
static int GRAD=1;      // gradient routine index
static int OBJGRAD=2;   // objective and gradient routine
static int HESS=3;      // hessian routine index
static int SEPOBJ=4;    // separable objective routine index
static int JAC=5;       // jacobian routine index
static int JACSTATE=6;  // jacobian state routine index
static int JACDESIGN=7; // jacobian design routine index
static int BOUNDS=8;
static int MON=9;       // monitor routine index
static int MONCTX=10;       // monitor routine index
static int MONDESTROY=11; // monitor destroy index
static int CONVTEST=12;  //
static int CONSTRAINTS=13;
static int NFUNCS=14;

static PetscErrorCode ourtaosolverobjectiveroutine(TaoSolver tao, Vec x, PetscReal *f, void *ctx)
{
    PetscErrorCode ierr = 0;
    (*(void (PETSC_STDCALL *)(TaoSolver*,Vec*,PetscReal*,void*,PetscErrorCode*))
	(((PetscObject)tao)->fortran_func_pointers[OBJ]))(&tao,&x,f,ctx,&ierr);
    CHKERRQ(ierr);
    return 0;
}

static PetscErrorCode ourtaosolvergradientroutine(TaoSolver tao, Vec x, Vec g, void *ctx)
{
    PetscErrorCode ierr = 0;
    (*(void (PETSC_STDCALL *)(TaoSolver*,Vec*,Vec*,void*,PetscErrorCode*))
       (((PetscObject)tao)->fortran_func_pointers[GRAD]))(&tao,&x,&g,ctx,&ierr);
    CHKERRQ(ierr);
    return 0;
    
}

static PetscErrorCode ourtaosolverobjectiveandgradientroutine(TaoSolver tao, Vec x, PetscReal *f, Vec g, void* ctx)
{
    PetscErrorCode ierr = 0;
    (*(void (PETSC_STDCALL *)(TaoSolver*,Vec*,PetscReal*,Vec*,void*,PetscErrorCode*))
     (((PetscObject)tao)->fortran_func_pointers[OBJGRAD]))(&tao,&x,f,&g,ctx,&ierr);
    CHKERRQ(ierr);
    return 0;
}

static PetscErrorCode ourtaosolverhessianroutine(TaoSolver tao, Vec x, Mat *H, Mat *Hpre, MatStructure *type, void *ctx) 
{
    PetscErrorCode ierr = 0;
    (*(void (PETSC_STDCALL *)(TaoSolver*,Vec*,Mat*,Mat*,MatStructure*,void*,PetscErrorCode*))
     (((PetscObject)tao)->fortran_func_pointers[HESS]))(&tao,&x,H,Hpre,type,ctx,&ierr); CHKERRQ(ierr);
    return 0;
}

static PetscErrorCode ourtaosolverjacobianroutine(TaoSolver tao, Vec x, Mat *H, Mat *Hpre, MatStructure *type, void *ctx) 
{
    PetscErrorCode ierr = 0;
    (*(void (PETSC_STDCALL *)(TaoSolver*,Vec*,Mat*,Mat*,MatStructure*,void*,PetscErrorCode*))
     (((PetscObject)tao)->fortran_func_pointers[JAC]))(&tao,&x,H,Hpre,type,ctx,&ierr); CHKERRQ(ierr);
    return 0;
}

static PetscErrorCode ourtaosolverjacobianstateroutine(TaoSolver tao, Vec x, Mat *H, Mat *Hpre, Mat *Hinv, MatStructure *type, void *ctx) 
{
    PetscErrorCode ierr = 0;
    (*(void (PETSC_STDCALL *)(TaoSolver*,Vec*,Mat*,Mat*,Mat*,MatStructure*,void*,PetscErrorCode*))
     (((PetscObject)tao)->fortran_func_pointers[JACSTATE]))(&tao,&x,H,Hpre,Hinv,type,ctx,&ierr); CHKERRQ(ierr);
    return 0;
}

static PetscErrorCode ourtaosolverjacobiandesignroutine(TaoSolver tao, Vec x, Mat *H, void *ctx) 
{
    PetscErrorCode ierr = 0;
    (*(void (PETSC_STDCALL *)(TaoSolver*,Vec*,Mat*,void*,PetscErrorCode*))
     (((PetscObject)tao)->fortran_func_pointers[JACDESIGN]))(&tao,&x,H,ctx,&ierr); CHKERRQ(ierr);
    return 0;
}

static PetscErrorCode ourtaosolverboundsroutine(TaoSolver tao, Vec xl, Vec xu, void *ctx)
{
    PetscErrorCode ierr = 0;
    (*(void (PETSC_STDCALL *)(TaoSolver*,Vec*,Vec*,void*,PetscErrorCode*))
     (((PetscObject)tao)->fortran_func_pointers[BOUNDS]))(&tao,&xl,&xu,ctx,&ierr); CHKERRQ(ierr);
    return 0;
}
static PetscErrorCode ourtaosolverseparableobjectiveroutine(TaoSolver tao, Vec x, Vec f, void *ctx) 
{
    PetscErrorCode ierr = 0;
    (*(void (PETSC_STDCALL *)(TaoSolver*,Vec*,Vec*,void*,PetscErrorCode*))
     (((PetscObject)tao)->fortran_func_pointers[SEPOBJ]))(&tao,&x,&f,ctx,&ierr);
    return 0;
}

static PetscErrorCode ourtaosolvermonitor(TaoSolver tao, void *ctx)
{
    PetscErrorCode ierr = 0;
    (*(void (PETSC_STDCALL *)(TaoSolver *, void*, PetscErrorCode*))
     (((PetscObject)tao)->fortran_func_pointers[MON]))(&tao,ctx,&ierr);
    CHKERRQ(ierr);
    return 0;
}

static PetscErrorCode ourtaomondestroy(void **ctx) 
{
    PetscErrorCode ierr = 0;
    TaoSolver tao = *(TaoSolver*)ctx;
    void *mctx = (void*)((PetscObject)tao)->fortran_func_pointers[MONCTX];
    (*(void (PETSC_STDCALL *)(void*,PetscErrorCode*))(((PetscObject)tao)->fortran_func_pointers[MONDESTROY]))(mctx,&ierr); CHKERRQ(ierr);
    return 0;
}
static PetscErrorCode ourtaosolverconvergencetest(TaoSolver tao, void *ctx)
{
    PetscErrorCode ierr = 0;
    (*(void (PETSC_STDCALL *)(TaoSolver *, void*, PetscErrorCode*))
     (((PetscObject)tao)->fortran_func_pointers[CONVTEST]))(&tao,ctx,&ierr);
    CHKERRQ(ierr);
    return 0;
}


static PetscErrorCode ourtaosolverconstraintsroutine(TaoSolver tao, Vec x, Vec c, void *ctx)
{
    PetscErrorCode ierr = 0;
    (*(void (PETSC_STDCALL *)(TaoSolver*,Vec*,Vec*,void*,PetscErrorCode*))
       (((PetscObject)tao)->fortran_func_pointers[CONSTRAINTS]))(&tao,&x,&c,ctx,&ierr);
    CHKERRQ(ierr);
    return 0;
    
}

EXTERN_C_BEGIN


void PETSC_STDCALL taosolversetobjectiveroutine_(TaoSolver *tao, void (PETSC_STDCALL *func)(TaoSolver*, Vec *, PetscReal *, void *, PetscErrorCode *), void *ctx, PetscErrorCode *ierr)
{
    CHKFORTRANNULLOBJECT(ctx);
    PetscObjectAllocateFortranPointers(*tao,NFUNCS);
    if (!func) {
	*ierr = TaoSolverSetObjectiveRoutine(*tao,0,ctx);
    } else {
	((PetscObject)*tao)->fortran_func_pointers[OBJ] = (PetscVoidFunction)func;
	*ierr = TaoSolverSetObjectiveRoutine(*tao, ourtaosolverobjectiveroutine,ctx);
    }
}

void PETSC_STDCALL taosolversetgradientroutine_(TaoSolver *tao, void (PETSC_STDCALL *func)(TaoSolver*, Vec *, Vec *, void *, PetscErrorCode *), void *ctx, PetscErrorCode *ierr)
{
    CHKFORTRANNULLOBJECT(ctx);
    PetscObjectAllocateFortranPointers(*tao,NFUNCS);
    if (!func) {
	*ierr = TaoSolverSetGradientRoutine(*tao,0,ctx);
    } else {
	((PetscObject)*tao)->fortran_func_pointers[GRAD] = (PetscVoidFunction)func;
	*ierr = TaoSolverSetGradientRoutine(*tao, ourtaosolvergradientroutine,ctx);
    }
}

void PETSC_STDCALL taosolversetobjectiveandgradientroutine_(TaoSolver *tao, void (PETSC_STDCALL *func)(TaoSolver*, Vec *, PetscReal *, Vec *, void *, PetscErrorCode *), void *ctx, PetscErrorCode *ierr)
{
    CHKFORTRANNULLOBJECT(ctx);
    PetscObjectAllocateFortranPointers(*tao,NFUNCS);
    if (!func) {
	*ierr = TaoSolverSetObjectiveAndGradientRoutine(*tao,0,ctx);
    } else {
	((PetscObject)*tao)->fortran_func_pointers[OBJGRAD] = (PetscVoidFunction)func;
	*ierr = TaoSolverSetObjectiveAndGradientRoutine(*tao, ourtaosolverobjectiveandgradientroutine,ctx);
    }
}




void PETSC_STDCALL taosolversetseparableobjectiveroutine_(TaoSolver *tao, Vec *F, void (PETSC_STDCALL *func)(TaoSolver*, Vec *, Vec *, void *, PetscErrorCode *), void *ctx, PetscErrorCode *ierr)
{
    CHKFORTRANNULLOBJECT(ctx);
    PetscObjectAllocateFortranPointers(*tao,NFUNCS);
    if (!func) {
	*ierr = TaoSolverSetSeparableObjectiveRoutine(*tao,*F,0,ctx);
    } else {
	((PetscObject)*tao)->fortran_func_pointers[SEPOBJ] = (PetscVoidFunction)func;
	*ierr = TaoSolverSetSeparableObjectiveRoutine(*tao,*F, ourtaosolverseparableobjectiveroutine,ctx);
    }
}



void PETSC_STDCALL taosolversetjacobianroutine_(TaoSolver *tao, Mat *J, Mat *Jp, void (PETSC_STDCALL *func)(TaoSolver*, Vec *, Mat *, Mat *, MatStructure *,void *, PetscErrorCode *), void *ctx, PetscErrorCode *ierr)
{
    CHKFORTRANNULLOBJECT(ctx);
    PetscObjectAllocateFortranPointers(*tao,NFUNCS);
    if (!func) {
	*ierr = TaoSolverSetJacobianRoutine(*tao,*J,*Jp,0,ctx);
    } else {
	((PetscObject)*tao)->fortran_func_pointers[JAC] = (PetscVoidFunction)func;
	*ierr = TaoSolverSetJacobianRoutine(*tao,*J, *Jp, ourtaosolverjacobianroutine,ctx);
    }
}

void PETSC_STDCALL taosolversetjacobianstateroutine_(TaoSolver *tao, Mat *J, Mat *Jp, Mat*Jinv, void (PETSC_STDCALL *func)(TaoSolver*, Vec *, Mat *, Mat *, Mat*, MatStructure *,void *, PetscErrorCode *), void *ctx, PetscErrorCode *ierr)
{
    CHKFORTRANNULLOBJECT(ctx);
    PetscObjectAllocateFortranPointers(*tao,NFUNCS);
    if (!func) {
      *ierr = TaoSolverSetJacobianStateRoutine(*tao,*J,*Jp,*Jinv,0,ctx);
    } else {
      ((PetscObject)*tao)->fortran_func_pointers[JACSTATE] = (PetscVoidFunction)func;
      *ierr = TaoSolverSetJacobianStateRoutine(*tao,*J, *Jp, *Jinv, ourtaosolverjacobianstateroutine,ctx);
    }
}

void PETSC_STDCALL taosolversetjacobiandesignroutine_(TaoSolver *tao, Mat *J, void (PETSC_STDCALL *func)(TaoSolver*, Vec *, Mat *, void *, PetscErrorCode *), void *ctx, PetscErrorCode *ierr)
{
    CHKFORTRANNULLOBJECT(ctx);
    PetscObjectAllocateFortranPointers(*tao,NFUNCS);
    if (!func) {
	*ierr = TaoSolverSetJacobianDesignRoutine(*tao,*J,0,ctx);
    } else {
	((PetscObject)*tao)->fortran_func_pointers[JACDESIGN] = (PetscVoidFunction)func;
	*ierr = TaoSolverSetJacobianDesignRoutine(*tao,*J, ourtaosolverjacobiandesignroutine,ctx);
    }
}

void PETSC_STDCALL taosolversethessianroutine_(TaoSolver *tao, Mat *J, Mat *Jp, void (PETSC_STDCALL *func)(TaoSolver*, Vec *, Mat *, Mat *, MatStructure *,void *, PetscErrorCode *), void *ctx, PetscErrorCode *ierr)
{
    CHKFORTRANNULLOBJECT(ctx);
    PetscObjectAllocateFortranPointers(*tao,NFUNCS);
    if (!func) {
	*ierr = TaoSolverSetHessianRoutine(*tao,*J,*Jp,0,ctx);
    } else {
	((PetscObject)*tao)->fortran_func_pointers[HESS] = (PetscVoidFunction)func;
	*ierr = TaoSolverSetHessianRoutine(*tao,*J, *Jp, ourtaosolverhessianroutine,ctx);
    }
}

void PETSC_STDCALL taosolversetvariableboundsroutine_(TaoSolver *tao, void (PETSC_STDCALL *func)(TaoSolver*,Vec*,Vec*,void*,PetscErrorCode*),void *ctx, PetscErrorCode *ierr)
{
    CHKFORTRANNULLOBJECT(ctx);
    PetscObjectAllocateFortranPointers(*tao,NFUNCS);
    if (func) {
	((PetscObject)*tao)->fortran_func_pointers[BOUNDS] = (PetscVoidFunction)func;
	*ierr = TaoSolverSetVariableBoundsRoutine(*tao,ourtaosolverboundsroutine,ctx);
    } else {
	*ierr = TaoSolverSetVariableBoundsRoutine(*tao,0,ctx); 
    }
    
}    
void PETSC_STDCALL taosolversetmonitor_(TaoSolver *tao, void (PETSC_STDCALL *func)(TaoSolver*,void*,PetscErrorCode*),void *ctx, void (PETSC_STDCALL *mondestroy)(void*,PetscErrorCode*),PetscErrorCode *ierr)
{
    CHKFORTRANNULLOBJECT(ctx);
    PetscObjectAllocateFortranPointers(*tao,NFUNCS);
    if (func) {
	((PetscObject)*tao)->fortran_func_pointers[MON] = (PetscVoidFunction)func;
	if (FORTRANNULLFUNCTION(mondestroy)){
	  *ierr = TaoSolverSetMonitor(*tao,ourtaosolvermonitor,*tao,PETSC_NULL);
	} else {
	  *ierr = TaoSolverSetMonitor(*tao,ourtaosolvermonitor,*tao,ourtaomondestroy);
	}
    }
}

void PETSC_STDCALL taosolversetconvergencetest_(TaoSolver *tao, void (PETSC_STDCALL *func)(TaoSolver*,void*,PetscErrorCode*),void *ctx, PetscErrorCode *ierr)
{
    CHKFORTRANNULLOBJECT(ctx);
    PetscObjectAllocateFortranPointers(*tao,NFUNCS);
    if (!func) {
	*ierr = TaoSolverSetConvergenceTest(*tao,0,ctx);
    } else {
	((PetscObject)*tao)->fortran_func_pointers[CONVTEST] = (PetscVoidFunction)func;
	*ierr = TaoSolverSetConvergenceTest(*tao,ourtaosolverconvergencetest,ctx);
    }
}

	
void PETSC_STDCALL taosolversetconstraintsroutine_(TaoSolver *tao, Vec *C, void (PETSC_STDCALL *func)(TaoSolver*, Vec *, Vec *, void *, PetscErrorCode *), void *ctx, PetscErrorCode *ierr)
{
    CHKFORTRANNULLOBJECT(ctx);
    PetscObjectAllocateFortranPointers(*tao,NFUNCS);
    if (!func) {
      *ierr = TaoSolverSetConstraintsRoutine(*tao,*C,0,ctx);
    } else {
	((PetscObject)*tao)->fortran_func_pointers[CONSTRAINTS] = (PetscVoidFunction)func;
	*ierr = TaoSolverSetConstraintsRoutine(*tao, *C, ourtaosolverconstraintsroutine,ctx);
    }
}
    

void PETSC_STDCALL taosolversettype_(TaoSolver *tao, CHAR type_name PETSC_MIXED_LEN(len), PetscErrorCode *ierr PETSC_END_LEN(len))

{
    char *t;
    
    FIXCHAR(type_name,len,t);
    *ierr = TaoSolverSetType(*tao,t);
    FREECHAR(type_name,t);
	
}

void PETSC_STDCALL taosolverview_(TaoSolver *tao, PetscViewer *viewer, PetscErrorCode *ierr)
{
    PetscViewer v;
    PetscPatchDefaultViewers_Fortran(viewer,v);
    *ierr = TaoSolverView(*tao,v);
}

void PETSC_STDCALL taosolvergethistory_(TaoSolver *tao, PetscInt *nhist, PetscErrorCode *ierr) 
{
  *nhist  = (*tao)->hist_len;
  *ierr = 0;
}

void PETSC_STDCALL taosolvergetoptionsprefix_(TaoSolver *tao, CHAR prefix PETSC_MIXED_LEN(len), PetscErrorCode *ierr PETSC_END_LEN(len))
{
  const char *name;
  *ierr = TaoSolverGetOptionsPrefix(*tao,&name);
  *ierr = PetscStrncpy(prefix,name,len); if (*ierr) return;
  FIXRETURNCHAR(PETSC_TRUE,prefix,len);

}

void PETSC_STDCALL taosolverappendoptionsprefix_(TaoSolver *tao, CHAR prefix PETSC_MIXED_LEN(len),PetscErrorCode *ierr PETSC_END_LEN(len))
{
  char *name;
  FIXCHAR(prefix,len,name);
  *ierr = TaoSolverAppendOptionsPrefix(*tao,name);
  FREECHAR(prefix,name);
}

void PETSC_STDCALL taosolversetoptionsprefix_(TaoSolver *tao, CHAR prefix PETSC_MIXED_LEN(len),PetscErrorCode *ierr PETSC_END_LEN(len))
{
  char *t;
  FIXCHAR(prefix,len,t);
  *ierr = TaoSolverSetOptionsPrefix(*tao,t);
  FREECHAR(prefix,t);
}

void PETSC_STDCALL taosolvergettype_(TaoSolver *tao, CHAR name PETSC_MIXED_LEN(len), PetscErrorCode *ierr  PETSC_END_LEN(len))
{
  const char *tname;
  *ierr = TaoSolverGetType(*tao,&tname);
  *ierr = PetscStrncpy(name,tname,len); if (*ierr) return;
  FIXRETURNCHAR(PETSC_TRUE,name,len);
  
}
EXTERN_C_END


