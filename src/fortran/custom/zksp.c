
#include "src/fortran/custom/zpetsc.h"
#include "petscksp.h"

#ifdef PETSC_HAVE_FORTRAN_CAPS
#define kspsetnullspace_           KSPSETNULLSPACE
#define kspgetnullspace_           KSPGETNULLSPACE
#define kspgetresidualnorm_        KSPGETRESIDUALNORM
#define kspgetconvergedreason_     KSPGETCONVERGEDREASON
#define kspfgmressetmodifypc_      KSPFGMRESSETMODIFYPC
#define kspfgmresmodifypcksp_      KSPFGMRESMODIFYPCKSP
#define kspfgmresmodifypcnochange_ KSPFGMRESMODIFYPCNOCHANGE
#define kspdefaultconverged_       KSPDEFAULTCONVERGED
#define kspskipconverged_          KSPSKIPCONVERGED
#define kspgmreskrylovmonitor_     KSPGMRESKRYLOVMONITOR
#define kspdefaultmonitor_         KSPDEFAULTMONITOR
#define ksptruemonitor_            KSPTRUEMONITOR
#define kspvecviewmonitor_         KSPVECVIEWMONITOR
#define ksplgmonitor_              KSPLGMONITOR
#define ksplgtruemonitor_          KSPLGTRUEMONITOR
#define kspsingularvaluemonitor_   KSPSINGULARVALUEMONITOR
#define kspregisterdestroy_        KSPREGISTERDESTROY
#define kspdestroy_                KSPDESTROY
#define ksplgmonitordestroy_       KSPLGMONITORDESTROY
#define ksplgmonitorcreate_        KSPLGMONITORCREATE
#define kspgetrhs_                 KSPGETRHS
#define kspgetsolution_            KSPGETSOLUTION
#define kspgetpc_                  KSPGETPC
#define kspsetmonitor_             KSPSETMONITOR
#define kspsetconvergencetest_     KSPSETCONVERGENCETEST
#define kspcreate_                 KSPCREATE
#define kspsetoptionsprefix_       KSPSETOPTIONSPREFIX
#define kspappendoptionsprefix_    KSPAPPENDOPTIONSPREFIX
#define kspgettype_                KSPGETTYPE
#define kspgetpreconditionerside_  KSPGETPRECONDITIONERSIDE
#define kspbuildsolution_          KSPBUILDSOLUTION 
#define kspbuildresidual_          KSPBUILDRESIDUAL
#define kspsettype_                KSPSETTYPE           
#define kspgetresidualhistory_     KSPGETRESIDUALHISTORY
#define kspgetoptionsprefix_       KSPGETOPTIONSPREFIX
#define kspview_                   KSPVIEW
#define kspgmressetrestart_        KSPGMRESSETRESTART
#define kspsetnormtype_            KSPSETNORMTYPE
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define kspsetnullspace_           kspsetnullspace
#define kspgetnullspace_           kspgetnullspace
#define kspgetconvergedreason_     kspgetconvergedreason
#define kspfgmressetmodifypc_      kspfgmressetmodifypc
#define kspfgmresmodifypcksp_      kspfgmresmodifypcksp
#define kspfgmresmodifypcnochange_ kspfgmresmodifypcnochange
#define kspdefaultconverged_       kspdefaultconverged
#define kspskipconverged_          kspskipconverged
#define kspsingularvaluemonitor_   kspsingularvaluemonitor
#define kspgmreskrylovmonitor_     kspgmreskrylovmonitor
#define kspdefaultmonitor_         kspdefaultmonitor
#define ksptruemonitor_            ksptruemonitor
#define kspvecviewmonitor_         kspvecviewmonitor
#define ksplgmonitor_              ksplgmonitor
#define ksplgtruemonitor_          ksplgtruemonitor
#define kspgetresidualhistory_     kspgetresidualhistory
#define kspsettype_                kspsettype
#define kspregisterdestroy_        kspregisterdestroy
#define kspdestroy_                kspdestroy
#define ksplgmonitordestroy_       ksplgmonitordestroy
#define ksplgmonitorcreate_        ksplgmonitorcreate
#define kspgetrhs_                 kspgetrhs
#define kspgetsolution_            kspgetsolution
#define kspgetpc_                  kspgetpc
#define kspsetmonitor_             kspsetmonitor
#define kspsetconvergencetest_     kspsetconvergencetest
#define kspcreate_                 kspcreate
#define kspsetoptionsprefix_       kspsetoptionsprefix
#define kspappendoptionsprefix_    kspappendoptionsprefix
#define kspgettype_                kspgettype
#define kspgetpreconditionerside_  kspgetpreconditionerside
#define kspbuildsolution_          kspbuildsolution
#define kspbuildresidual_          kspbuildresidual
#define kspgetoptionsprefix_       kspgetoptionsprefix
#define kspview_                   kspview
#define kspgetresidualnorm_        kspgetresidualnorm
#define kspgmressetrestart_        kspgmressetrestart
#define kspsetnormtype_            kspsetnormtype
#endif

EXTERN_C_BEGIN
static void (PETSC_STDCALL *f2)(KSP*,PetscInt*,PetscReal*,KSPConvergedReason*,void*,PetscErrorCode*);
static void (PETSC_STDCALL *f1)(KSP*,PetscInt*,PetscReal*,void*,PetscErrorCode*);
static void (PETSC_STDCALL *f21)(void*,PetscErrorCode*);
static void (PETSC_STDCALL *f109)(KSP*,PetscInt*,PetscInt*,PetscReal*,void*,PetscErrorCode*);
static void (PETSC_STDCALL *f210)(void*,PetscErrorCode*);
EXTERN_C_END

/* These are not extern C because they are passed into non-extern C user level functions */
static PetscErrorCode ourtest(KSP ksp,PetscInt i,PetscReal d,KSPConvergedReason *reason,void* ctx)
{
  PetscErrorCode ierr;
  (*f2)(&ksp,&i,&d,reason,ctx,&ierr);CHKERRQ(ierr);
  return 0;
}

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

static PetscErrorCode ourmodify(KSP ksp,PetscInt i,PetscInt i2,PetscReal d,void* ctx)
{
  PetscErrorCode ierr = 0;
  (*f109)(&ksp,&i,&i2,&d,ctx,&ierr);CHKERRQ(ierr);
  return 0;
}

static PetscErrorCode ourmoddestroy(void* ctx)
{
  PetscErrorCode ierr = 0;
  (*f210)(ctx,&ierr);CHKERRQ(ierr);
  return 0;
}

EXTERN_C_BEGIN

void PETSC_STDCALL kspgmressetrestart_(KSP *ksp,PetscInt *max_k, PetscErrorCode *ierr)
{
  *ierr = KSPGMRESSetRestart(*ksp,*max_k);
}

void PETSC_STDCALL kspgetresidualnorm_(KSP *ksp,PetscReal *rnorm,PetscErrorCode *ierr)
{
  *ierr = KSPGetResidualNorm(*ksp,rnorm);
}

void PETSC_STDCALL kspgetconvergedreason_(KSP *ksp,KSPConvergedReason *reason,PetscErrorCode *ierr)
{
  *ierr = KSPGetConvergedReason(*ksp,reason);
}

/* function */
void PETSC_STDCALL kspsetnullspace_(KSP *ksp,MatNullSpace *sp, PetscErrorCode *ierr)
{
  *ierr = KSPSetNullSpace(*ksp,*sp);
}

void PETSC_STDCALL kspgetnullspace_(KSP *ksp,MatNullSpace *sp, PetscErrorCode *ierr)
{
  *ierr = KSPGetNullSpace(*ksp,sp);
}

void PETSC_STDCALL kspview_(KSP *ksp,PetscViewer *viewer, PetscErrorCode *ierr)
{
  PetscViewer v;
  PetscPatchDefaultViewers_Fortran(viewer,v);
  *ierr = KSPView(*ksp,v);
}

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

void PETSC_STDCALL kspgetresidualhistory_(KSP *ksp,PetscInt *na,PetscErrorCode *ierr)
{
  *ierr = KSPGetResidualHistory(*ksp,PETSC_NULL,na);
}

void PETSC_STDCALL kspsettype_(KSP *ksp,CHAR type PETSC_MIXED_LEN(len),PetscErrorCode *ierr PETSC_END_LEN(len))
{
  char *t;

  FIXCHAR(type,len,t);
  *ierr = KSPSetType(*ksp,t);
  FREECHAR(type,t);
}

void PETSC_STDCALL kspgettype_(KSP *ksp,CHAR name PETSC_MIXED_LEN(len),PetscErrorCode *ierr PETSC_END_LEN(len))
{
  const char *tname;

  *ierr = KSPGetType(*ksp,&tname);if (*ierr) return;
#if defined(PETSC_USES_CPTOFCD)
  {
    char *t = _fcdtocp(name); int len1 = _fcdlen(name);
    *ierr = PetscStrncpy(t,tname,len1); 
  }
#else
  *ierr = PetscStrncpy(name,tname,len);
#endif
  FIXRETURNCHAR(name,len);

}

void PETSC_STDCALL kspgetpreconditionerside_(KSP *ksp,PCSide *side,PetscErrorCode *ierr){
*ierr = KSPGetPreconditionerSide(*ksp,side);
}

void PETSC_STDCALL kspsetoptionsprefix_(KSP *ksp,CHAR prefix PETSC_MIXED_LEN(len),
                                        PetscErrorCode *ierr PETSC_END_LEN(len))
{
  char *t;

  FIXCHAR(prefix,len,t);
  *ierr = KSPSetOptionsPrefix(*ksp,t);
  FREECHAR(prefix,t);
}

void PETSC_STDCALL kspappendoptionsprefix_(KSP *ksp,CHAR prefix PETSC_MIXED_LEN(len),
                                           PetscErrorCode *ierr PETSC_END_LEN(len))
{
  char *t;

  FIXCHAR(prefix,len,t);
  *ierr = KSPAppendOptionsPrefix(*ksp,t);
  FREECHAR(prefix,t);
}

void PETSC_STDCALL kspcreate_(MPI_Comm *comm,KSP *ksp,PetscErrorCode *ierr){
  *ierr = KSPCreate((MPI_Comm)PetscToPointerComm(*comm),ksp);
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

/*
        These are not usually called from Fortran but allow Fortran users 
   to transparently set these monitors from .F code
   
   functions, hence no STDCALL
*/
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

void PETSC_STDCALL kspgetpc_(KSP *ksp,PC *B,PetscErrorCode *ierr)
{
  *ierr = KSPGetPC(*ksp,B);
}

void PETSC_STDCALL kspgetsolution_(KSP *ksp,Vec *v,PetscErrorCode *ierr)
{
  *ierr = KSPGetSolution(*ksp,v);
}

void PETSC_STDCALL kspgetrhs_(KSP *ksp,Vec *r,PetscErrorCode *ierr)
{
  *ierr = KSPGetRhs(*ksp,r);
}

/*
   Possible bleeds memory but cannot be helped.
*/
void PETSC_STDCALL ksplgmonitorcreate_(CHAR host PETSC_MIXED_LEN(len1),
                    CHAR label PETSC_MIXED_LEN(len2),int *x,int *y,int *m,int *n,PetscDrawLG *ctx,
                    PetscErrorCode *ierr PETSC_END_LEN(len1) PETSC_END_LEN(len2))
{
  char   *t1,*t2;

  FIXCHAR(host,len1,t1);
  FIXCHAR(label,len2,t2);
  *ierr = KSPLGMonitorCreate(t1,t2,*x,*y,*m,*n,ctx);
}

void PETSC_STDCALL ksplgmonitordestroy_(PetscDrawLG *ctx,PetscErrorCode *ierr)
{
  *ierr = KSPLGMonitorDestroy(*ctx);
}

void PETSC_STDCALL kspdestroy_(KSP *ksp,PetscErrorCode *ierr)
{
  *ierr = KSPDestroy(*ksp);
}

void PETSC_STDCALL kspregisterdestroy_(PetscErrorCode* ierr)
{
  *ierr = KSPRegisterDestroy();
}

void PETSC_STDCALL kspbuildsolution_(KSP *ctx,Vec *v,Vec *V,PetscErrorCode *ierr)
{
  *ierr = KSPBuildSolution(*ctx,*v,V);
}

void PETSC_STDCALL kspbuildresidual_(KSP *ctx,Vec *t,Vec *v,Vec *V,PetscErrorCode *ierr)
{
  *ierr = KSPBuildResidual(*ctx,*t,*v,V);
}

void PETSC_STDCALL kspgetoptionsprefix_(KSP *ksp,CHAR prefix PETSC_MIXED_LEN(len),PetscErrorCode *ierr PETSC_END_LEN(len))
{
  const char *tname;

  *ierr = KSPGetOptionsPrefix(*ksp,&tname);
#if defined(PETSC_USES_CPTOFCD)
  {
    char *t = _fcdtocp(prefix); int len1 = _fcdlen(prefix);
    *ierr = PetscStrncpy(t,tname,len1); if (*ierr) return;
  }
#else
  *ierr = PetscStrncpy(prefix,tname,len); if (*ierr) return;
#endif
}

void PETSC_STDCALL kspfgmresmodifypcnochange_(KSP *ksp,PetscInt *total_its,PetscInt *loc_its,PetscReal *res_norm,void* dummy,PetscErrorCode *ierr)
{
  *ierr = KSPFGMRESModifyPCNoChange(*ksp,*total_its,*loc_its,*res_norm,dummy);
}

void PETSC_STDCALL kspfgmresmodifypcksp_(KSP *ksp,PetscInt *total_its,PetscInt *loc_its,PetscReal *res_norm,void*dummy,PetscErrorCode *ierr)
{
  *ierr = KSPFGMRESModifyPCKSP(*ksp,*total_its,*loc_its,*res_norm,dummy);
}

void PETSC_STDCALL kspfgmressetmodifypc_(KSP *ksp,void (PETSC_STDCALL *fcn)(KSP*,PetscInt*,PetscInt*,PetscReal*,void*,PetscErrorCode*),void* ctx,void (PETSC_STDCALL *d)(void*,PetscErrorCode*),PetscErrorCode *ierr)
{
  if ((FCNVOID)fcn == (FCNVOID)kspfgmresmodifypcksp_) {
    *ierr = KSPFGMRESSetModifyPC(*ksp,KSPFGMRESModifyPCKSP,0,0);
  } else if ((FCNVOID)fcn == (FCNVOID)kspfgmresmodifypcnochange_) {
    *ierr = KSPFGMRESSetModifyPC(*ksp,KSPFGMRESModifyPCNoChange,0,0);
  } else {
    f109 = fcn;
    if (FORTRANNULLFUNCTION(d)) {
      *ierr = KSPFGMRESSetModifyPC(*ksp,ourmodify,ctx,0);
    } else {
      f210 = d;
      *ierr = KSPFGMRESSetModifyPC(*ksp,ourmodify,ctx,ourmoddestroy);
    }
  }
}

void PETSC_STDCALL kspsetnormtype_(KSP *ksp,KSPNormType *type,PetscErrorCode *ierr)
{
  *ierr = KSPSetNormType(*ksp,*type);
}

EXTERN_C_END
