#ifndef lint
static char vcid[] = "$Id: zksp.c,v 1.7 1995/12/14 14:31:41 curfman Exp bsmith $";
#endif

#include "zpetsc.h"
#include "draw.h"
#include "ksp.h"
#include "pinclude/petscfix.h"

#ifdef FORTRANCAPS
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
#define kspgettype_               KSPGETTYPE
#define kspgetpreconditionerside_ KSPGETPRECONDITIONERSIDE
#elif !defined(FORTRANUNDERSCORE) && !defined(FORTRANDOUBLEUNDERSCORE)
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
#define kspgettype_               kspgettype
#define kspgetpreconditionerside_ kspgetpreconditionerside
#endif

void kspgettype_(KSP ksp,KSPType *type,char *name,int *__ierr,int len)
{
  char *tname;
  if (type == PETSC_NULL_Fortran) type = PETSC_NULL;
  *__ierr = KSPGetType((KSP)MPIR_ToPointer(*(int*)ksp),type,&tname);
  if (name != PETSC_NULL_Fortran) PetscStrncpy(name,tname,len);
}

void kspgetpreconditionerside_(KSP itP,PCSide *side, int *__ierr ){
*__ierr = KSPGetPreconditionerSide(
	(KSP)MPIR_ToPointer( *(int*)(itP) ),side );
}

void kspsetoptionsprefix_(KSP ksp,char *prefix, int *__ierr,int len ){
  char *t;
  if (prefix[len] != 0) {
    t = (char *) PetscMalloc( (len+1)*sizeof(char) ); 
    PetscStrncpy(t,prefix,len);
    t[len] = 0;
  }
  else t = prefix;
  *__ierr = KSPSetOptionsPrefix((KSP)MPIR_ToPointer( *(int*)(ksp) ),t);
}

void kspcreate_(MPI_Comm comm,KSP *ksp, int *__ierr ){
  KSP tmp;
  *__ierr = KSPCreate((MPI_Comm)MPIR_ToPointer( *(int*)(comm) ),&tmp);
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
void ksplgmonitorcreate_(char *host,char *label,int *x,int *y,int *m,
                       int *n,DrawLG *ctx, int *__ierr,int len1,int len2){
  char *t1,*t2;
  DrawLG lg;
  if (host[len1] != 0) {
    t1 = (char *) PetscMalloc( (len1+1)*sizeof(char) ); 
    PetscStrncpy(t1,host,len1);
    t1[len1] = 0;
  }
  else t1 = host;
  if (label[len2] != 0) {
    t2 = (char *) PetscMalloc( (len2+1)*sizeof(char) ); 
    PetscStrncpy(t2,label,len2);
    t2[len2] = 0;
  }
  else t2 = label;
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
