#include "private/fortranimpl.h"
#include "taosolver.h"

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define taosolversetobjectiveroutine_            TAOSOLVERSETOBJECTIVEROUTINE
#define taosolversetgradientroutine_             TAOSOLVERSETGRADIENTROUTINE
#define taosolversetobjectiveandgradientroutine_ TAOSOLVERSETOBJECTIVEANDGRADIENTROUTINE
#define taosolversethessianroutine_              TAOSOLVERSETHESSIANROUTINE
#define taosolversetseparableobjectiveroutine_   TAOSOLVERSETSEPARABLEOBJECTIVEROUTINE
#define taosolversetjacobianroutine_             TAOSOLVERSETJACOBIANROUTINE
#define taosolversetvariableboundsroutine_       TAOSOLVERSETVARIABLEBOUNDSROUTINE
#define taosolversetmonitor_                     TAOSOLVERSETMONITOR
#define taosolversettype_                        TAOSOLVERSETTYPE
#define taosolverview_                           TAOSOLVERVIEW
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)

#define taosolversetobjectiveroutine_            taosolversetobjectiveroutine
#define taosolversetgradientroutine_             taosolversetgradientroutine
#define taosolversetobjectiveandgradientroutine_ taosolversetobjectiveandgradientroutine
#define taosolversethessianroutine_              taosolversethessianroutine
#define taosolversetseparableobjectiveroutine_   taosolversetseparableobjectiveroutine
#define taosolversetjacobianroutine_             taosolversetjacobianroutine
#define taosolversetvariableboundsroutine_       taosolversetvariableboundsroutine
#define taosolversetmonitor_                     taosolversetmonitor
#define taosolversettype_                        taosolversettype
#define taosolverview_                           taosolverview
#endif

static int OBJ=0;       // objective routine index
static int GRAD=1;      // gradient routine index
static int OBJGRAD=2;   // objective and gradient routine
static int HESS=3;      // hessian routine index
static int SEPOBJ=4;    // separable objective routine index
static int JAC=5;       // jacobian routine index
static int BOUNDS=6;
static int MON=7;       // monitor routine index
static int NFUNCS=8;

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


EXTERN_C_BEGIN


void PETSC_STDCALL taosolversetobjectiveroutine_(TaoSolver *tao, void (PETSC_STDCALL *func)(TaoSolver*, Vec *, PetscReal *, void *, PetscErrorCode *), void *ctx, PetscErrorCode *ierr)
{
    CHKFORTRANNULLOBJECT(ctx);
    CHKFORTRANNULLFUNCTION(func);
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
    CHKFORTRANNULLFUNCTION(func);
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
    CHKFORTRANNULLFUNCTION(func);
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
    CHKFORTRANNULLFUNCTION(func);
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
    CHKFORTRANNULLFUNCTION(func);
    PetscObjectAllocateFortranPointers(*tao,NFUNCS);
    if (!func) {
	*ierr = TaoSolverSetJacobianRoutine(*tao,*J,*Jp,0,ctx);
    } else {
	((PetscObject)*tao)->fortran_func_pointers[JAC] = (PetscVoidFunction)func;
	*ierr = TaoSolverSetJacobianRoutine(*tao,*J, *Jp, ourtaosolverjacobianroutine,ctx);
    }
}

void PETSC_STDCALL taosolversethessianroutine_(TaoSolver *tao, Mat *J, Mat *Jp, void (PETSC_STDCALL *func)(TaoSolver*, Vec *, Mat *, Mat *, MatStructure *,void *, PetscErrorCode *), void *ctx, PetscErrorCode *ierr)
{
    CHKFORTRANNULLOBJECT(ctx);
    CHKFORTRANNULLFUNCTION(func);
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
    CHKFORTRANNULLFUNCTION(func);
    PetscObjectAllocateFortranPointers(*tao,NFUNCS);
    if (func) {
	((PetscObject)*tao)->fortran_func_pointers[BOUNDS] = (PetscVoidFunction)func;
	*ierr = TaoSolverSetVariableBoundsRoutine(*tao,ourtaosolverboundsroutine,ctx);
    } else {
	*ierr = TaoSolverSetVariableBoundsRoutine(*tao,0,ctx); 
    }
    
}    
void PETSC_STDCALL taosolversetmonitor_(TaoSolver *tao, void (PETSC_STDCALL *func)(TaoSolver*,void*,PetscErrorCode*),void *ctx, PetscErrorCode *ierr)
{
    CHKFORTRANNULLOBJECT(ctx);
    CHKFORTRANNULLFUNCTION(func);
    PetscObjectAllocateFortranPointers(*tao,NFUNCS);
    if (!func) {
	*ierr = TaoSolverSetMonitor(*tao,0,ctx);
    } else {
	((PetscObject)*tao)->fortran_func_pointers[MON] = (PetscVoidFunction)func;
	*ierr = TaoSolverSetMonitor(*tao,ourtaosolvermonitor,ctx);
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
EXTERN_C_END
