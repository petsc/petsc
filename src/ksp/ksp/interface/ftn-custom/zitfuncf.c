#include "zpetsc.h"
#include "petscksp.h"

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define kspmonitorset_             KSPMONITORSET
#define kspsetconvergencetest_     KSPSETCONVERGENCETEST
#define kspgetresidualhistory_     KSPGETRESIDUALHISTORY

#define kspdefaultconverged_       KSPDEFAULTCONVERGED
#define kspskipconverged_          KSPSKIPCONVERGED
#define kspgmresmonitorkrylov_     KSPGMRESKMONITORKRYLOV
#define kspmonitordefault_         KSPMONITORDEFAULT
#define kspmonitortrueresidualnorm_    KSPMONITORTRUERESIDUALNORM
#define kspmonitorsolution_        KSPMONITORSOLUTION
#define kspmonitorlg_              KSPLGMONITORLG
#define kspmonitorlgtrueresidualnorm_  KSPLGTRUEMONITORLGTRUERESIDUALNORM
#define kspmonitorsingularvalue_   KSPMONITORSINGULARVALUE
#define kspfgmresmodifypcksp_      KSPFGMRESMODIFYPCKSP
#define kspfgmresmodifypcnochange_ KSPFGMRESMODIFYPCNOCHANGE
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define kspmonitorset_             kspmonitorset
#define kspsetconvergencetest_     kspsetconvergencetest
#define kspgetresidualhistory_     kspgetresidualhistory
#define kspdefaultconverged_       kspdefaultconverged
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
static void (PETSC_STDCALL *f1)(KSP*,PetscInt*,PetscReal*,void*,PetscErrorCode*);
static void (PETSC_STDCALL *f21)(void*,PetscErrorCode*);
static void (PETSC_STDCALL *f2)(KSP*,PetscInt*,PetscReal*,KSPConvergedReason*,void*,PetscErrorCode*);

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
  (*f1)(&ksp,&i,&d,ctx,&ierr);CHKERRQ(ierr);
  return 0;
}

static PetscErrorCode ourdestroy(void* ctx)
{
  PetscErrorCode ierr = 0;
  (*f21)(ctx,&ierr);CHKERRQ(ierr);
  return 0;
}

/* These are not extern C because they are passed into non-extern C user level functions */
static PetscErrorCode ourtest(KSP ksp,PetscInt i,PetscReal d,KSPConvergedReason *reason,void* ctx)
{
  PetscErrorCode ierr;
  (*f2)(&ksp,&i,&d,reason,ctx,&ierr);CHKERRQ(ierr);
  return 0;
}


EXTERN_C_BEGIN
void PETSC_STDCALL kspmonitorset_(KSP *ksp,void (PETSC_STDCALL *monitor)(KSP*,PetscInt*,PetscReal*,void*,PetscErrorCode*),
                    void *mctx,void (PETSC_STDCALL *monitordestroy)(void*,PetscErrorCode*),PetscErrorCode *ierr)
{
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
    f1  = monitor;
    if (FORTRANNULLFUNCTION(monitordestroy)) {
      *ierr = KSPMonitorSet(*ksp,ourmonitor,mctx,0);
    } else {
      f21 = monitordestroy;
      *ierr = KSPMonitorSet(*ksp,ourmonitor,mctx,ourdestroy);
    }
  }
}

void PETSC_STDCALL kspsetconvergencetest_(KSP *ksp,
      void (PETSC_STDCALL *converge)(KSP*,PetscInt*,PetscReal*,KSPConvergedReason*,void*,PetscErrorCode*),void *cctx,PetscErrorCode *ierr)
{
  if ((PetscVoidFunction)converge == (PetscVoidFunction)kspdefaultconverged_) {
    *ierr = KSPSetConvergenceTest(*ksp,KSPDefaultConverged,0);
  } else if ((PetscVoidFunction)converge == (PetscVoidFunction)kspskipconverged_) {
    *ierr = KSPSetConvergenceTest(*ksp,KSPSkipConverged,0);
  } else {
    f2 = converge;
    *ierr = KSPSetConvergenceTest(*ksp,ourtest,cctx);
  }
}

void PETSC_STDCALL kspgetresidualhistory_(KSP *ksp,PetscInt *na,PetscErrorCode *ierr)
{
  *ierr = KSPGetResidualHistory(*ksp,PETSC_NULL,na);
}

EXTERN_C_END
