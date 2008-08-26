#include "private/fortranimpl.h"
#include "petscksp.h"

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define kspmonitorset_             KSPMONITORSET
#define kspsetconvergencetest_     KSPSETCONVERGENCETEST
#define kspgetresidualhistory_     KSPGETRESIDUALHISTORY

#define kspdefaultconverged_       KSPDEFAULTCONVERGED
#define kspdefaultconvergedcreate_  KSPDEFAULTCONVERGEDCREATE
#define kspskipconverged_          KSPSKIPCONVERGED
#define kspgmresmonitorkrylov_     KSPGMRESMONITORKRYLOV
#define kspmonitordefault_         KSPMONITORDEFAULT
#define kspmonitortrueresidualnorm_    KSPMONITORTRUERESIDUALNORM
#define kspmonitorsolution_        KSPMONITORSOLUTION
#define kspmonitorlg_              KSPMONITORLG
#define kspmonitorlgtrueresidualnorm_  KSPMONITORLGTRUERESIDUALNORM
#define kspmonitorsingularvalue_   KSPMONITORSINGULARVALUE
#define kspfgmresmodifypcksp_      KSPFGMRESMODIFYPCKSP
#define kspfgmresmodifypcnochange_ KSPFGMRESMODIFYPCNOCHANGE
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define kspmonitorset_             kspmonitorset
#define kspsetconvergencetest_     kspsetconvergencetest
#define kspgetresidualhistory_     kspgetresidualhistory
#define kspdefaultconverged_       kspdefaultconverged
#define kspdefaultconvergedcreate_  kspdefaultconvergedcreate
#define kspskipconverged_          kspskipconverged
#define kspmonitorsingularvalue_   kspmonitorsingularvalue
#define kspgmresmonitorkrylov_     kspgmresmonitorkrylov
#define kspmonitordefault_         kspmonitordefault
#define kspmonitortrueresidualnorm_    kspmonitortrueresidualnorm
#define kspmonitorsolution_        kspmonitorsolution
#define kspmonitorlg_              kspmonitorlg
#define kspmonitorlgtrueresidualnorm_  kspmonitorlgtrueresidualnorm
#define kspfgmresmodifypcksp_      kspfgmresmodifypcksp
#define kspfgmresmodifypcnochange_ kspfgmresmodifypcnochange
#endif


EXTERN_C_BEGIN

/*
        These are not usually called from Fortran but allow Fortran users 
   to transparently set these monitors from .F code
   
   functions, hence no STDCALL
*/

void kspdefaultconverged_(KSP *ksp,PetscInt *n,PetscReal *rnorm,KSPConvergedReason *flag,void *dummy,PetscErrorCode *ierr)
{
  CHKFORTRANNULLOBJECT(dummy);
  *ierr = KSPDefaultConverged(*ksp,*n,*rnorm,flag,dummy);
}

void kspskipconverged_(KSP *ksp,PetscInt *n,PetscReal *rnorm,KSPConvergedReason *flag,void *dummy,PetscErrorCode *ierr)
{
  CHKFORTRANNULLOBJECT(dummy);
  *ierr = KSPSkipConverged(*ksp,*n,*rnorm,flag,dummy);
}

void kspgmresmonitorkrylov_(KSP *ksp,PetscInt *it,PetscReal *norm,void *ctx,PetscErrorCode *ierr)
{
  *ierr = KSPGMRESMonitorKrylov(*ksp,*it,*norm,ctx);
}

void  kspmonitordefault_(KSP *ksp,PetscInt *it,PetscReal *norm,void *ctx,PetscErrorCode *ierr)
{
  *ierr = KSPMonitorDefault(*ksp,*it,*norm,ctx);
}
 
void  kspmonitorsingularvalue_(KSP *ksp,PetscInt *it,PetscReal *norm,void *ctx,PetscErrorCode *ierr)
{
  *ierr = KSPMonitorSingularValue(*ksp,*it,*norm,ctx);
}

void  kspmonitorlg_(KSP *ksp,PetscInt *it,PetscReal *norm,void *ctx,PetscErrorCode *ierr)
{
  *ierr = KSPMonitorLG(*ksp,*it,*norm,ctx);
}

void  kspmonitorlgtrueresidualnorm_(KSP *ksp,PetscInt *it,PetscReal *norm,void *ctx,PetscErrorCode *ierr)
{
  *ierr = KSPMonitorLGTrueResidualNorm(*ksp,*it,*norm,ctx);
}

void  kspmonitortrueresidualnorm_(KSP *ksp,PetscInt *it,PetscReal *norm,void *ctx,PetscErrorCode *ierr)
{
  *ierr = KSPMonitorTrueResidualNorm(*ksp,*it,*norm,ctx);
}

void  kspmonitorsolution_(KSP *ksp,PetscInt *it,PetscReal *norm,void *ctx,PetscErrorCode *ierr)
{
  *ierr = KSPMonitorSolution(*ksp,*it,*norm,ctx);
}

EXTERN_C_END

static PetscErrorCode ourmonitor(KSP ksp,PetscInt i,PetscReal d,void* ctx)
{
  PetscErrorCode ierr = 0;
  void           *mctx = (void*) ((PetscObject)ksp)->fortran_func_pointers[1];
  (*(void (PETSC_STDCALL *)(KSP*,PetscInt*,PetscReal*,void*,PetscErrorCode*))(((PetscObject)ksp)->fortran_func_pointers[0]))(&ksp,&i,&d,mctx,&ierr);CHKERRQ(ierr);
  return 0;
}

static PetscErrorCode ourdestroy(void* ctx)
{
  PetscErrorCode ierr = 0;
  KSP            ksp = (KSP)ctx;
  void           *mctx = (void*) ((PetscObject)ksp)->fortran_func_pointers[1];
  (*(void (PETSC_STDCALL *)(void*,PetscErrorCode*))(((PetscObject)ksp)->fortran_func_pointers[2]))(mctx,&ierr);CHKERRQ(ierr);
  return 0;
}

/* These are not extern C because they are passed into non-extern C user level functions */
static PetscErrorCode ourtest(KSP ksp,PetscInt i,PetscReal d,KSPConvergedReason *reason,void* ctx)
{
  PetscErrorCode ierr;
  void           *mctx = (void*) ((PetscObject)ksp)->fortran_func_pointers[4];
  (*(void (PETSC_STDCALL *)(KSP*,PetscInt*,PetscReal*,KSPConvergedReason*,void*,PetscErrorCode*))(((PetscObject)ksp)->fortran_func_pointers[3]))(&ksp,&i,&d,reason,mctx,&ierr);CHKERRQ(ierr);
  return 0;
}
static PetscErrorCode ourtestdestroy(void* ctx)
{
  PetscErrorCode ierr;
  KSP            ksp = (KSP)ctx;
  void           *mctx = (void*) ((PetscObject)ksp)->fortran_func_pointers[4];
  (*(void (PETSC_STDCALL *)(void*,PetscErrorCode*))(((PetscObject)ksp)->fortran_func_pointers[5]))(mctx,&ierr);CHKERRQ(ierr);
  return 0;
}


EXTERN_C_BEGIN
void PETSC_STDCALL kspmonitorset_(KSP *ksp,void (PETSC_STDCALL *monitor)(KSP*,PetscInt*,PetscReal*,void*,PetscErrorCode*),
                    void *mctx,void (PETSC_STDCALL *monitordestroy)(void*,PetscErrorCode*),PetscErrorCode *ierr)
{
  CHKFORTRANNULLFUNCTION(monitordestroy);

  PetscObjectAllocateFortranPointers(*ksp,6);
  if ((PetscVoidFunction)monitor == (PetscVoidFunction)kspmonitordefault_) {
    *ierr = KSPMonitorSet(*ksp,KSPMonitorDefault,0,0);
  } else if ((PetscVoidFunction)monitor == (PetscVoidFunction)kspmonitorlg_) {
    *ierr = KSPMonitorSet(*ksp,KSPMonitorLG,0,0);
  } else if ((PetscVoidFunction)monitor == (PetscVoidFunction)kspmonitorlgtrueresidualnorm_) {
    *ierr = KSPMonitorSet(*ksp,KSPMonitorLGTrueResidualNorm,0,0);
  } else if ((PetscVoidFunction)monitor == (PetscVoidFunction)kspmonitorsolution_) {
    *ierr = KSPMonitorSet(*ksp,KSPMonitorSolution,0,0);
  } else if ((PetscVoidFunction)monitor == (PetscVoidFunction)kspmonitortrueresidualnorm_) {
    *ierr = KSPMonitorSet(*ksp,KSPMonitorTrueResidualNorm,0,0);
  } else if ((PetscVoidFunction)monitor == (PetscVoidFunction)kspmonitorsingularvalue_) {
    *ierr = KSPMonitorSet(*ksp,KSPMonitorSingularValue,0,0);
  } else if ((PetscVoidFunction)monitor == (PetscVoidFunction)kspgmresmonitorkrylov_) {
    *ierr = KSPMonitorSet(*ksp,KSPGMRESMonitorKrylov,0,0);
  } else {
    ((PetscObject)*ksp)->fortran_func_pointers[0] = (PetscVoidFunction)monitor;
    ((PetscObject)*ksp)->fortran_func_pointers[1] = (PetscVoidFunction)mctx;
    if (!monitordestroy) {
      *ierr = KSPMonitorSet(*ksp,ourmonitor,*ksp,0);
    } else {
      ((PetscObject)*ksp)->fortran_func_pointers[2] = (PetscVoidFunction)monitordestroy;
      *ierr = KSPMonitorSet(*ksp,ourmonitor,*ksp,ourdestroy);
    }
  }
}

void PETSC_STDCALL kspsetconvergencetest_(KSP *ksp,
      void (PETSC_STDCALL *converge)(KSP*,PetscInt*,PetscReal*,KSPConvergedReason*,void*,PetscErrorCode*),void *cctx,
      void (PETSC_STDCALL *destroy)(void*,PetscErrorCode*),PetscErrorCode *ierr)
{
  CHKFORTRANNULLOBJECT(cctx);
  CHKFORTRANNULLFUNCTION(destroy);

  PetscObjectAllocateFortranPointers(*ksp,6);
  if ((PetscVoidFunction)converge == (PetscVoidFunction)kspdefaultconverged_) {
    *ierr = KSPSetConvergenceTest(*ksp,KSPDefaultConverged,cctx,KSPDefaultConvergedDestroy);
  } else if ((PetscVoidFunction)converge == (PetscVoidFunction)kspskipconverged_) {
    *ierr = KSPSetConvergenceTest(*ksp,KSPSkipConverged,0,0);
  } else {
    ((PetscObject)*ksp)->fortran_func_pointers[3] = (PetscVoidFunction)converge;
    ((PetscObject)*ksp)->fortran_func_pointers[4] = (PetscVoidFunction)cctx;
    if (!destroy) {
      *ierr = KSPSetConvergenceTest(*ksp,ourtest,*ksp,0);
    } else {
      ((PetscObject)*ksp)->fortran_func_pointers[5] = (PetscVoidFunction)destroy;
      *ierr = KSPSetConvergenceTest(*ksp,ourtest,*ksp,ourtestdestroy);
    }
  }
}

void PETSC_STDCALL kspdefaultconvergedcreate_(PetscFortranAddr *ctx,PetscErrorCode *ierr)
{
  *ierr = KSPDefaultConvergedCreate((void**)ctx);
}

void PETSC_STDCALL kspgetresidualhistory_(KSP *ksp,PetscInt *na,PetscErrorCode *ierr)
{
  *ierr = KSPGetResidualHistory(*ksp,PETSC_NULL,na);
}

EXTERN_C_END
