#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: zksp.c,v 1.20 1998/03/06 00:06:09 bsmith Exp balay $";
#endif

#include "src/fortran/custom/zpetsc.h"
#include "ksp.h"
#include "pinclude/petscfix.h"

#ifdef HAVE_FORTRAN_CAPS
#define kspregisterdestroy_       KSPREGISTERDESTROY
#define kspregisterall_           KSPREGISTERALL
#define kspdestroy_               KSPDESTROY
#define ksplgmonitordestroy_      KSPLGMONITORDESTROY
#define ksplgmonitorcreate_       KSPLGMONITORCREATE
#define kspgetrhs_                KSPGETRHS
#define kspgetsolution_           KSPGETSOLUTION
#define kspgetpc_                 KSPGETPC
#define kspsetmonitor_            KSPSETMONITOR
#define kspsetconvergencetest_    KSPSETCONVERGENCETEST
#define kspcreate_                KSPCREATE
#define kspsetoptionsprefix_      KSPSETOPTIONSPREFIX
#define kspappendoptionsprefix_   KSPAPPENDOPTIONSPREFIX
#define kspgettype_               KSPGETTYPE
#define kspgetpreconditionerside_ KSPGETPRECONDITIONERSIDE
#define kspbuildsolution_         KSPBUILDSOLUTION
#define kspsettype_               KSPSETTYPE           
#elif !defined(HAVE_FORTRAN_UNDERSCORE)
#define kspsettype_               kspsettype
#define kspregisterdestroy_       kspregisterdestroy
#define kspregisterall_           kspregisterall
#define kspdestroy_               kspdestroy
#define ksplgmonitordestroy_      ksplgmonitordestroy
#define ksplgmonitorcreate_       ksplgmonitorcreate
#define kspgetrhs_                kspgetrhs
#define kspgetsolution_           kspgetsolution
#define kspgetpc_                 kspgetpc
#define kspsetmonitor_            kspsetmonitor
#define kspsetconvergencetest_    kspsetconvergencetest
#define kspcreate_                kspcreate
#define kspsetoptionsprefix_      kspsetoptionsprefix
#define kspappendoptionsprefix_   kspappendoptionsprefix
#define kspgettype_               kspgettype
#define kspgetpreconditionerside_ kspgetpreconditionerside
#define kspbuildsolution_         kspbuildsolution
#endif

#if defined(__cplusplus)
extern "C" {
#endif

void kspsettype_(KSP ksp,CHAR itmethod, int *__ierr,int len )
{
  char *t;

  FIXCHAR(itmethod,len,t);
  *__ierr = KSPSetType((KSP)PetscToPointer(ksp),t);
  FREECHAR(itmethod,t);
}

void kspgettype_(KSP ksp,CHAR name,int *__ierr,int len)
{
  char *tname;

  *__ierr = KSPGetType((KSP)PetscToPointer(ksp),&tname);
#if defined(USES_CPTOFCD)
  {
    char *t = _fcdtocp(name); int len1 = _fcdlen(name);
    PetscStrncpy(t,tname,len1);
  }
#else
  PetscStrncpy(name,tname,len);
#endif
}

void kspgetpreconditionerside_(KSP ksp,PCSide *side, int *__ierr ){
*__ierr = KSPGetPreconditionerSide(
	(KSP)PetscToPointer(ksp),side );
}

void kspsetoptionsprefix_(KSP ksp,CHAR prefix, int *__ierr,int len ){
  char *t;

  FIXCHAR(prefix,len,t);
  *__ierr = KSPSetOptionsPrefix((KSP)PetscToPointer(ksp),t);
  FREECHAR(prefix,t);
}

void kspappendoptionsprefix_(KSP ksp,CHAR prefix, int *__ierr,int len ){
  char *t;

  FIXCHAR(prefix,len,t);
  *__ierr = KSPAppendOptionsPrefix((KSP)PetscToPointer(ksp),t);
  FREECHAR(prefix,t);
}

void kspcreate_(MPI_Comm *comm,KSP *ksp, int *__ierr ){
  KSP tmp;
  *__ierr = KSPCreate((MPI_Comm)PetscToPointerComm( *comm ),&tmp);
  *(PetscFortranAddr*)ksp =  PetscFromPointer(tmp);
}

static int (*f2)(PetscFortranAddr*,int*,double*,void*,int*);
static int ourtest(KSP ksp,int i,double d,void* ctx)
{
  int              ierr = 0;
  PetscFortranAddr s1;

  s1 = PetscFromPointer(ksp);
  (*f2)(&s1,&i,&d,ctx,&ierr); CHKERRQ(ierr);
  PetscRmPointer(&s1);
  return 0;
}
void kspsetconvergencetest_(KSP ksp,
      int (*converge)(PetscFortranAddr*,int*,double*,void*,int*),void *cctx, int *__ierr){
  f2 = converge;
  *__ierr = KSPSetConvergenceTest(
	(KSP)PetscToPointer(ksp),ourtest,cctx);
}

static int (*f1)(PetscFortranAddr*,int*,double*,void*,int*);
static int ourmonitor(KSP ksp,int i,double d,void* ctx)
{
  int              ierr = 0;
  PetscFortranAddr s1;

  s1 = PetscFromPointer(ksp);
  (*f1)(&s1,&i,&d,ctx,&ierr); CHKERRQ(ierr);
  PetscRmPointer(&s1);
  return 0;
}
void kspsetmonitor_(KSP ksp,int (*monitor)(PetscFortranAddr*,int*,double*,void*,int*),
                    void *mctx, int *__ierr ){
  f1 = monitor;
  *__ierr = KSPSetMonitor((KSP)PetscToPointer(ksp),ourmonitor,mctx);
}

void kspgetpc_(KSP ksp,PC *B, int *__ierr ){
  PC pc;
  *__ierr = KSPGetPC((KSP)PetscToPointer(ksp),&pc);
  *(PetscFortranAddr*) B = PetscFromPointer(pc);
}

void kspgetsolution_(KSP ksp,Vec *v, int *__ierr ){
  Vec vv;
  *__ierr = KSPGetSolution((KSP)PetscToPointer(ksp),&vv);
  *(PetscFortranAddr*) v =  PetscFromPointer(vv);
}
void kspgetrhs_(KSP ksp,Vec *r, int *__ierr ){
  Vec vv;
  *__ierr = KSPGetRhs((KSP)PetscToPointer(ksp),&vv);
  *(PetscFortranAddr*) r =  PetscFromPointer(vv);
}

/*
   Possible bleeds memory but cannot be helped.
*/
void ksplgmonitorcreate_(CHAR host,CHAR label,int *x,int *y,int *m,
                       int *n,DrawLG *ctx, int *__ierr,int len1,int len2){
  char   *t1,*t2;
  DrawLG lg;

  FIXCHAR(host,len1,t1);
  FIXCHAR(label,len2,t2);
  *__ierr = KSPLGMonitorCreate(t1,t2,*x,*y,*m,*n,&lg);
  *(PetscFortranAddr*) ctx = PetscFromPointer(lg);
}

void ksplgmonitordestroy_(DrawLG ctx, int *__ierr ){
  *__ierr = KSPLGMonitorDestroy((DrawLG)PetscToPointer(ctx));
  PetscRmPointer(ctx);
}


void kspdestroy_(KSP ksp, int *__ierr ){
  *__ierr = KSPDestroy((KSP)PetscToPointer(ksp));
  PetscRmPointer(ksp);
}

void kspregisterdestroy_(int* __ierr)
{
  *__ierr = KSPRegisterDestroy();
}

void kspbuildsolution_(KSP ctx,Vec v,Vec *V, int *__ierr ){
  Vec vv;
  *__ierr = KSPBuildSolution(
      (KSP)PetscToPointer(ctx),
      (Vec)PetscToPointer(v),&vv);
  *(PetscFortranAddr*) V = PetscFromPointer(vv);
}
void kspbuildresidual_(KSP ctx,Vec t,Vec v,Vec *V, int *__ierr ){
  Vec vv;
  *__ierr = KSPBuildResidual(
    (KSP)PetscToPointer(ctx),
    (Vec)PetscToPointer(t),
    (Vec)PetscToPointer(v),&vv);
  *(PetscFortranAddr*) V = PetscFromPointer(vv);
}

#if defined(__cplusplus)
}
#endif
