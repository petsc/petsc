#include "zpetsc.h"
#include "petscksp.h"

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define kspsetmonitor_             KSPSETMONITOR
#define kspsetconvergencetest_     KSPSETCONVERGENCETEST
#define kspgetresidualhistory_     KSPGETRESIDUALHISTORY

#define kspdefaultconverged_       KSPDEFAULTCONVERGED
#define kspskipconverged_          KSPSKIPCONVERGED
#define kspgmreskrylovmonitor_     KSPGMRESKRYLOVMONITOR
#define kspdefaultmonitor_         KSPDEFAULTMONITOR
#define ksptruemonitor_            KSPTRUEMONITOR
#define kspvecviewmonitor_         KSPVECVIEWMONITOR
#define ksplgmonitor_              KSPLGMONITOR
#define ksplgtruemonitor_          KSPLGTRUEMONITOR
#define kspsingularvaluemonitor_   KSPSINGULARVALUEMONITOR
#define kspfgmresmodifypcksp_      KSPFGMRESMODIFYPCKSP
#define kspfgmresmodifypcnochange_ KSPFGMRESMODIFYPCNOCHANGE
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define kspsetmonitor_             kspsetmonitor
#define kspsetconvergencetest_     kspsetconvergencetest
#define kspgetresidualhistory_     kspgetresidualhistory
#define kspdefaultconverged_       kspdefaultconverged
#define kspskipconverged_          kspskipconverged
#define kspsingularvaluemonitor_   kspsingularvaluemonitor
#define kspgmreskrylovmonitor_     kspgmreskrylovmonitor
#define kspdefaultmonitor_         kspdefaultmonitor
#define ksptruemonitor_            ksptruemonitor
#define kspvecviewmonitor_         kspvecviewmonitor
#define ksplgmonitor_              ksplgmonitor
#define ksplgtruemonitor_          ksplgtruemonitor
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

void kspgmreskrylovmonitor_(KSP *ksp,PetscInt *it,PetscReal *norm,void *ctx,PetscErrorCode *ierr)
{
  *ierr = KSPGMRESKrylovMonitor(*ksp,*it,*norm,ctx);
}

void  kspdefaultmonitor_(KSP *ksp,PetscInt *it,PetscReal *norm,void *ctx,PetscErrorCode *ierr)
{
  *ierr = KSPDefaultMonitor(*ksp,*it,*norm,ctx);
}
 
void  kspsingularvaluemonitor_(KSP *ksp,PetscInt *it,PetscReal *norm,void *ctx,PetscErrorCode *ierr)
{
  *ierr = KSPSingularValueMonitor(*ksp,*it,*norm,ctx);
}

void  ksplgmonitor_(KSP *ksp,PetscInt *it,PetscReal *norm,void *ctx,PetscErrorCode *ierr)
{
  *ierr = KSPLGMonitor(*ksp,*it,*norm,ctx);
}

void  ksplgtruemonitor_(KSP *ksp,PetscInt *it,PetscReal *norm,void *ctx,PetscErrorCode *ierr)
{
  *ierr = KSPLGTrueMonitor(*ksp,*it,*norm,ctx);
}

void  ksptruemonitor_(KSP *ksp,PetscInt *it,PetscReal *norm,void *ctx,PetscErrorCode *ierr)
{
  *ierr = KSPTrueMonitor(*ksp,*it,*norm,ctx);
}

void  kspvecviewmonitor_(KSP *ksp,PetscInt *it,PetscReal *norm,void *ctx,PetscErrorCode *ierr)
{
  *ierr = KSPVecViewMonitor(*ksp,*it,*norm,ctx);
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
void PETSC_STDCALL kspsetmonitor_(KSP *ksp,void (PETSC_STDCALL *monitor)(KSP*,PetscInt*,PetscReal*,void*,PetscErrorCode*),
                    void *mctx,void (PETSC_STDCALL *monitordestroy)(void*,PetscErrorCode*),PetscErrorCode *ierr)
{
  if ((FCNVOID)monitor == (FCNVOID)kspdefaultmonitor_) {
    *ierr = KSPSetMonitor(*ksp,KSPDefaultMonitor,0,0);
  } else if ((FCNVOID)monitor == (FCNVOID)ksplgmonitor_) {
    *ierr = KSPSetMonitor(*ksp,KSPLGMonitor,0,0);
  } else if ((FCNVOID)monitor == (FCNVOID)ksplgtruemonitor_) {
    *ierr = KSPSetMonitor(*ksp,KSPLGTrueMonitor,0,0);
  } else if ((FCNVOID)monitor == (FCNVOID)kspvecviewmonitor_) {
    *ierr = KSPSetMonitor(*ksp,KSPVecViewMonitor,0,0);
  } else if ((FCNVOID)monitor == (FCNVOID)ksptruemonitor_) {
    *ierr = KSPSetMonitor(*ksp,KSPTrueMonitor,0,0);
  } else if ((FCNVOID)monitor == (FCNVOID)kspsingularvaluemonitor_) {
    *ierr = KSPSetMonitor(*ksp,KSPSingularValueMonitor,0,0);
  } else if ((FCNVOID)monitor == (FCNVOID)kspgmreskrylovmonitor_) {
    *ierr = KSPSetMonitor(*ksp,KSPGMRESKrylovMonitor,0,0);
  } else {
    f1  = monitor;
    if (FORTRANNULLFUNCTION(monitordestroy)) {
      *ierr = KSPSetMonitor(*ksp,ourmonitor,mctx,0);
    } else {
      f21 = monitordestroy;
      *ierr = KSPSetMonitor(*ksp,ourmonitor,mctx,ourdestroy);
    }
  }
}

void PETSC_STDCALL kspsetconvergencetest_(KSP *ksp,
      void (PETSC_STDCALL *converge)(KSP*,PetscInt*,PetscReal*,KSPConvergedReason*,void*,PetscErrorCode*),void *cctx,PetscErrorCode *ierr)
{
  if ((FCNVOID)converge == (FCNVOID)kspdefaultconverged_) {
    *ierr = KSPSetConvergenceTest(*ksp,KSPDefaultConverged,0);
  } else if ((FCNVOID)converge == (FCNVOID)kspskipconverged_) {
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
