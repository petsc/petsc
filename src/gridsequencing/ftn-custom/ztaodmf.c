#include "private/fortranimpl.h"
#include "taosolver.h"
#include "private/taodm_impl.h"

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define taodmsetobjectiveandgradientroutine_          TAODMSETOBJECTIVEANDGRADIENTROUTINE
#define taodmsetobjectiveroutine_                     TAODMSETOBJECTIVEROUTINE
#define taodmsetgradientroutine_                      TAODMSETGRADIENTROUTINE
#define taodmsethessianroutine_                       TAODMSETHESSIANROUTINE
#define taodmsetpremonitor_                           TAODMSETPREMONITOR
#define taodmsetpostmonitor_                          TAODMSETPOSTMONITOR
#define taodmsetlocalobjectiveandgradientroutine_     TAODMSETLOCALOBJECTIVEANDGRADIENTROUTINE
#define taodmsetlocalobjectiveroutine_                TAODMSETLOCALOBJECTIVEROUTINE
#define taodmsetlocalgradientroutine_                 TAODMSETLOCALGRADIENTROUTINE
#define taodmsetlocalhessianroutine_                  TAODMSETLOCALHESSIANROUTINE
#define taodmsetinitialguessroutine_                  TAODMSETINITIALGUESSROUTINE
#define taodmsetvariableboundsroutine_                TAODMSETVARIABLEBOUNDSROUTINE
#define taodmview_                                    TAODMVIEW
#define taodmsetsolvertype_                           TAODMSETSOLVERTYPE
#define taodmcreate_                                  TAODMCREATE
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)

#define taodmsetobjectiveandgradientroutine_          taodmsetobjectiveandgradientroutine
#define taodmsetobjectiveroutine_                     taodmsetobjectiveroutine
#define taodmsetgradientroutine_                      taodmsetgradientroutine
#define taodmsethessianroutine_                       taodmsethessianroutine
#define taodmsetpremonitor_                           taodmsetpremonitor
#define taodmsetpostmonitor_                          taodmsetpostmonitor
#define taodmsetlocalobjectiveandgradientroutine_     taodmsetlocalobjectiveandgradientroutine
#define taodmsetlocalobjectiveroutine_                taodmsetlocalobjectiveroutine
#define taodmsetlocalgradientroutine_                 taodmsetlocalgradientroutine
#define taodmsetlocalhessianroutine_                  taodmsetlocalhessianroutine
#define taodmsetinitialguessroutine_                  taodmsetinitialguessroutine
#define taodmsetvariableboundsroutine_                taodmsetvariableboundsroutine
#define taodmview_                                    taodmview
#define taodmsetsolvertype_                           taodmsetsolvertype
#define taodmcreate_                                  taodmcreate

#endif


static int OBJ=0;
static int GRAD=0;
static int OBJGRAD=2;
static int HESS=3;
static int LOBJ=4;
static int LGRAD=5;
static int LOBJGRAD=6;
static int LHESS=7;
static int BOUNDS=8;
static int GUESS=9;
static int NFUNCS=10;

static PetscErrorCode ourtaodmobjectiveroutine(TaoSolver tao, Vec X, PetscScalar *f, void *ctx) {
  PetscErrorCode ierr = 0;
  TaoDM taodm = (TaoDM)ctx; 
  TaoDM taodm0 = (TaoDM)(taodm->coarselevel);
  (*(void (PETSC_STDCALL*)(TaoSolver*, Vec*, PetscScalar*, void*, PetscErrorCode*))
     (((PetscObject)taodm0)->fortran_func_pointers[OBJ]))(&tao,&X,f,ctx,&ierr);
   CHKERRQ(ierr);
   return 0;

}

EXTERN_C_BEGIN
void PETSC_STDCALL taodmsetobjectiveroutine_(TaoDM *taodm, void (PETSC_STDCALL *func)(TaoSolver*, Vec *, PetscScalar*, void*), void *ctx, PetscErrorCode *ierr)
{
  CHKFORTRANNULLOBJECT(ctx);
  CHKFORTRANNULLFUNCTION(func);
  PetscObjectAllocateFortranPointers(taodm,NFUNCS);
  if (!func) {
    *ierr = TaoDMSetObjectiveRoutine(taodm,0);
  } else {
    ((PetscObject)*taodm)->fortran_func_pointers[OBJ] = (PetscVoidFunction)func;
    *ierr = TaoDMSetObjectiveRoutine(taodm,ourtaodmobjectiveroutine);
  }
}
EXTERN_C_END
  

static PetscErrorCode ourtaodmgradientroutine(TaoSolver tao, Vec X, Vec G, void *ctx) {
  PetscErrorCode ierr = 0;
  TaoDM taodm = (TaoDM)ctx; 
  TaoDM taodm0 = (TaoDM)(taodm->coarselevel);
  (*(void (PETSC_STDCALL*)(TaoSolver*, Vec*, Vec*, void*, PetscErrorCode*))
     (((PetscObject)taodm0)->fortran_func_pointers[GRAD]))(&tao,&X,&G,ctx,&ierr);
   CHKERRQ(ierr);
   return 0;
}

EXTERN_C_BEGIN
void PETSC_STDCALL taodmsetgradientroutine_(TaoDM *taodm, void (PETSC_STDCALL *func)(TaoSolver*, Vec *, Vec*, void*), void *ctx, PetscErrorCode *ierr)
{
  CHKFORTRANNULLOBJECT(ctx);
  CHKFORTRANNULLFUNCTION(func);
  PetscObjectAllocateFortranPointers(taodm,NFUNCS);
  if (!func) {
    *ierr = TaoDMSetGradientRoutine(taodm,0);
  } else {
    ((PetscObject)*taodm)->fortran_func_pointers[GRAD] = (PetscVoidFunction)func;
    *ierr = TaoDMSetGradientRoutine(taodm,ourtaodmgradientroutine);
  }
}
EXTERN_C_END



static PetscErrorCode ourtaodmobjectiveandgradientroutine(TaoSolver tao, Vec X, PetscScalar *f, Vec G, void *ctx) {
  PetscErrorCode ierr = 0;
  TaoDM taodm = (TaoDM)ctx; 
  TaoDM taodm0 = (TaoDM)(taodm->coarselevel);
  (*(void (PETSC_STDCALL*)(TaoSolver*, Vec*, PetscScalar *, Vec*, void*, PetscErrorCode*))
   (((PetscObject)taodm0)->fortran_func_pointers[GRAD]))(&tao,&X,f,&G,ctx,&ierr);
  CHKERRQ(ierr);
  return 0;
}


EXTERN_C_BEGIN
void PETSC_STDCALL taodmsetobjectiveandgradientroutine_(TaoDM *taodm, void (PETSC_STDCALL *func)(TaoSolver*, Vec *, PetscScalar *, Vec*, void*), void *ctx, PetscErrorCode *ierr)
{
  CHKFORTRANNULLOBJECT(ctx);
  CHKFORTRANNULLFUNCTION(func);
  PetscObjectAllocateFortranPointers(taodm,NFUNCS);
  if (!func) {
    *ierr = TaoDMSetObjectiveAndGradientRoutine(taodm,0);
  } else {
    ((PetscObject)*taodm)->fortran_func_pointers[GRAD] = (PetscVoidFunction)func;
    *ierr = TaoDMSetObjectiveAndGradientRoutine(taodm,ourtaodmobjectiveandgradientroutine);
  }
}
EXTERN_C_END

static PetscErrorCode ourtaodminitialguessroutine(TaoDM taodm, Vec X) {
  PetscErrorCode ierr;
  (*(void (PETSC_STDCALL*)(TaoDM*, Vec*, PetscErrorCode*))
   (((PetscObject)taodm)->fortran_func_pointers[GUESS]))(&taodm,&X,&ierr);
  CHKERRQ(ierr);
  return 0;
}

EXTERN_C_BEGIN
void PETSC_STDCALL taodmsetinitialguessroutine_(TaoDM *taodm, void (PETSC_STDCALL *func)(TaoDM*, Vec*, PetscErrorCode *), PetscErrorCode *ierr)
{
  CHKFORTRANNULLFUNCTION(func);
  PetscObjectAllocateFortranPointers(taodm,NFUNCS);
  if (!func) {
    *ierr = TaoDMSetInitialGuessRoutine(taodm,0);
  } else {
    ((PetscObject)*taodm)->fortran_func_pointers[GUESS] = (PetscVoidFunction)func;
    *ierr = TaoDMSetInitialGuessRoutine(taodm,ourtaodminitialguessroutine);
  }
}
EXTERN_C_END
  

static PetscErrorCode ourtaodmvariableboundsroutine(TaoDM taodm, Vec XL, Vec XU) {
  PetscErrorCode ierr;
  (*(void (PETSC_STDCALL*)(TaoDM*, Vec*, Vec*, PetscErrorCode *))
   (((PetscObject)taodm)->fortran_func_pointers[BOUNDS]))(&taodm,&XL,&XU,&ierr);
  CHKERRQ(ierr);
  return 0;
}
EXTERN_C_BEGIN
void PETSC_STDCALL taodmsetvariableboundsroutine_(TaoDM *taodm, void (PETSC_STDCALL *func)(TaoDM *, Vec *, Vec *, PetscErrorCode *), PetscErrorCode *ierr)
{
  CHKFORTRANNULLFUNCTION(func);
  PetscObjectAllocateFortranPointers(taodm,NFUNCS);
  if (!func) {
    *ierr = TaoDMSetVariableBoundsRoutine(taodm,0);
  } else {
    ((PetscObject)*taodm)->fortran_func_pointers[BOUNDS] = (PetscVoidFunction)func;
    *ierr = TaoDMSetVariableBoundsRoutine(taodm,ourtaodmvariableboundsroutine);
  }
}
EXTERN_C_END


EXTERN_C_BEGIN
void PETSC_STDCALL  taodmcreate_(MPI_Fint * comm,PetscInt *nlevels,void*user,TaoDM **taodm, int *__ierr ){
*__ierr = TaoDMCreate(
  MPI_Comm_f2c( *(comm) ),*nlevels,user,taodm);

}
EXTERN_C_END
