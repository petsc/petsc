#ifndef lint
static char vcid[] = "$Id: zksp.c,v 1.11 1996/01/30 00:40:19 bsmith Exp bsmith $";
#endif

#include "zpetsc.h"
#include "draw.h"
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
#define kspgetbinv_               KSPGETBINV
#define kspsetmonitor_            KSPSETMONITOR
#define kspsetconvergencetest_    KSPSETCONVERGENCETEST
#define kspcreate_                KSPCREATE
#define kspsetoptionsprefix_      KSPSETOPTIONSPREFIX
#define kspappendoptionsprefix_      KSPAPPENDOPTIONSPREFIX
#define kspgettype_               KSPGETTYPE
#define kspgetpreconditionerside_ KSPGETPRECONDITIONERSIDE
#elif !defined(HAVE_FORTRAN_UNDERSCORE)
#define kspregisterdestroy_       kspregisterdestroy
#define kspregisterall_           kspregisterall
#define kspdestroy_               kspdestroy
#define ksplgmonitordestroy_      ksplgmonitordestroy
#define ksplgmonitorcreate_       ksplgmonitorcreate
#define kspgetrhs_                kspgetrhs
#define kspgetsolution_           kspgetsolution
#define kspgetbinv_               kspgetbinv
#define kspsetmonitor_            kspsetmonitor
#define kspsetconvergencetest_    kspsetconvergencetest
#define kspcreate_                kspcreate
#define kspsetoptionsprefix_      kspsetoptionsprefix
#define kspappendoptionsprefix_   kspappendoptionsprefix
#define kspgettype_               kspgettype
#define kspgetpreconditionerside_ kspgetpreconditionerside
#endif

#if defined(__cplusplus)
extern "C" {
#endif

void kspgettype_(KSP ksp,KSPType *type,CHAR name,int *__ierr,int len)
{
  char *tname;
  if (FORTRANNULL(type)) type = PETSC_NULL;
  *__ierr = KSPGetType((KSP)MPIR_ToPointer(*(int*)ksp),type,&tname);
#if defined(PARCH_t3d)
  {
  char *t = _fcdtocp(name); int len1 = _fcdlen(name);
  if (t != PETSC_NULL_CHAR_Fortran) PetscStrncpy(t,tname,len1);
  }
#else
  if (name != PETSC_NULL_CHAR_Fortran) PetscStrncpy(name,tname,len);
#endif
}

void kspgetpreconditionerside_(KSP itP,PCSide *side, int *__ierr ){
*__ierr = KSPGetPreconditionerSide(
	(KSP)MPIR_ToPointer( *(int*)(itP) ),side );
}

void kspsetoptionsprefix_(KSP ksp,CHAR prefix, int *__ierr,int len ){
  char *t;

  FIXCHAR(prefix,len,t);
  *__ierr = KSPSetOptionsPrefix((KSP)MPIR_ToPointer( *(int*)(ksp) ),t);
  FREECHAR(prefix,t);
}

void kspappendoptionsprefix_(KSP ksp,CHAR prefix, int *__ierr,int len ){
  char *t;

  FIXCHAR(prefix,len,t);
  *__ierr = KSPAppendOptionsPrefix((KSP)MPIR_ToPointer( *(int*)(ksp) ),t);
  FREECHAR(prefix,t);
}

void kspcreate_(MPI_Comm comm,KSP *ksp, int *__ierr ){
  KSP tmp;
  *__ierr = KSPCreate((MPI_Comm)MPIR_ToPointer_Comm( *(int*)(comm) ),&tmp);
  *(int*)ksp =  MPIR_FromPointer(tmp);
}

static int (*f2)(int*,int*,double*,void*,int*);
static int ourtest(KSP ksp,int i,double d,void* ctx)
{
  int s1, ierr = 0;
  s1 = MPIR_FromPointer(ksp);
  (*f2)(&s1,&i,&d,ctx,&ierr); CHKERRQ(ierr);
  MPIR_RmPointer(s1);
  return 0;
}
void kspsetconvergencetest_(KSP itP,
      int (*converge)(int*,int*,double*,void*,int*),void *cctx, int *__ierr){
  f2 = converge;
  *__ierr = KSPSetConvergenceTest(
	(KSP)MPIR_ToPointer( *(int*)(itP) ),ourtest,cctx);
}

static int (*f1)(int*,int*,double*,void*,int*);
static int ourmonitor(KSP ksp,int i,double d,void* ctx)
{
  int s1, ierr = 0;
  s1 = MPIR_FromPointer(ksp);
  (*f1)(&s1,&i,&d,ctx,&ierr); CHKERRQ(ierr);
  MPIR_RmPointer(s1);
  return 0;
}
void kspsetmonitor_(KSP itP,int (*monitor)(int*,int*,double*,void*,int*),
                    void *mctx, int *__ierr ){
  f1 = monitor;
  *__ierr = KSPSetMonitor((KSP)MPIR_ToPointer(*(int*)(itP)),ourmonitor,mctx);
}

void kspgetbinv_(KSP itP,PC *B, int *__ierr ){
  PC pc;
  *__ierr = KSPGetBinv((KSP)MPIR_ToPointer( *(int*)(itP) ),&pc);
  *(int*) B = MPIR_FromPointer(pc);
}

void kspgetsolution_(KSP itP,Vec *v, int *__ierr ){
  Vec vv;
  *__ierr = KSPGetSolution((KSP)MPIR_ToPointer( *(int*)(itP) ),&vv);
  *(int*) v =  MPIR_FromPointer(vv);
}
void kspgetrhs_(KSP itP,Vec *r, int *__ierr ){
  Vec vv;
  *__ierr = KSPGetRhs((KSP)MPIR_ToPointer( *(int*)(itP) ),&vv);
  *(int*) r =  MPIR_FromPointer(vv);
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
  *(int*) ctx = MPIR_FromPointer(lg);
}

void ksplgmonitordestroy_(DrawLG ctx, int *__ierr ){
  *__ierr = KSPLGMonitorDestroy((DrawLG)MPIR_ToPointer( *(int*)(ctx) ));
  MPIR_RmPointer(*(int*)(ctx) );
}


void kspdestroy_(KSP itP, int *__ierr ){
  *__ierr = KSPDestroy((KSP)MPIR_ToPointer( *(int*)(itP) ));
  MPIR_RmPointer(*(int*)(itP) );
}

void kspregisterdestroy_(int* MPIR_ierr)
{
  *MPIR_ierr = KSPRegisterDestroy();
}

void kspregisterall_(int* MPIR_ierr)
{
  *MPIR_ierr = KSPRegisterAll();
}

void kspbuildsolution_(KSP ctx,Vec v,Vec *V, int *__ierr ){
  Vec vv;
  *__ierr = KSPBuildSolution(
      (KSP)MPIR_ToPointer( *(int*)(ctx) ),
      (Vec)MPIR_ToPointer( *(int*)(v) ),&vv);
  *(int*) V = MPIR_FromPointer(vv);
}
void kspbuildresidual_(KSP ctx,Vec t,Vec v,Vec *V, int *__ierr ){
  Vec vv;
  *__ierr = KSPBuildResidual(
    (KSP)MPIR_ToPointer( *(int*)(ctx) ),
    (Vec)MPIR_ToPointer( *(int*)(t) ),
    (Vec)MPIR_ToPointer( *(int*)(v) ),&vv);
  *(int*) V = MPIR_FromPointer(vv);
}

#if defined(__cplusplus)
}
#endif
