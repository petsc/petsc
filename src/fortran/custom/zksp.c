#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: zksp.c,v 1.18 1997/07/01 19:32:56 bsmith Exp balay $";
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
#elif !defined(HAVE_FORTRAN_UNDERSCORE)
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

void kspgettype_(KSP ksp,KSPType *type,CHAR name,int *__ierr,int len)
{
  char *tname;
  if (FORTRANNULL(type)) type = PETSC_NULL;
  *__ierr = KSPGetType((KSP)PetscToPointer(*(int*)ksp),type,&tname);
#if defined(USES_CPTOFCD)
  {
  char *t = _fcdtocp(name); int len1 = _fcdlen(name);
  if (t != PETSC_NULL_CHARACTER_Fortran) PetscStrncpy(t,tname,len1);
  }
#else
  if (name != PETSC_NULL_CHARACTER_Fortran) PetscStrncpy(name,tname,len);
#endif
}

void kspgetpreconditionerside_(KSP ksp,PCSide *side, int *__ierr ){
*__ierr = KSPGetPreconditionerSide(
	(KSP)PetscToPointer( *(int*)(ksp) ),side );
}

void kspsetoptionsprefix_(KSP ksp,CHAR prefix, int *__ierr,int len ){
  char *t;

  FIXCHAR(prefix,len,t);
  *__ierr = KSPSetOptionsPrefix((KSP)PetscToPointer( *(int*)(ksp) ),t);
  FREECHAR(prefix,t);
}

void kspappendoptionsprefix_(KSP ksp,CHAR prefix, int *__ierr,int len ){
  char *t;

  FIXCHAR(prefix,len,t);
  *__ierr = KSPAppendOptionsPrefix((KSP)PetscToPointer( *(int*)(ksp) ),t);
  FREECHAR(prefix,t);
}

void kspcreate_(MPI_Comm *comm,KSP *ksp, int *__ierr ){
  KSP tmp;
  *__ierr = KSPCreate((MPI_Comm)PetscToPointerComm( *comm ),&tmp);
  *(int*)ksp =  PetscFromPointer(tmp);
}

static int (*f2)(int*,int*,double*,void*,int*);
static int ourtest(KSP ksp,int i,double d,void* ctx)
{
  int s1, ierr = 0;
  s1 = PetscFromPointer(ksp);
  (*f2)(&s1,&i,&d,ctx,&ierr); CHKERRQ(ierr);
  PetscRmPointer(s1);
  return 0;
}
void kspsetconvergencetest_(KSP ksp,
      int (*converge)(int*,int*,double*,void*,int*),void *cctx, int *__ierr){
  f2 = converge;
  *__ierr = KSPSetConvergenceTest(
	(KSP)PetscToPointer( *(int*)(ksp) ),ourtest,cctx);
}

static int (*f1)(int*,int*,double*,void*,int*);
static int ourmonitor(KSP ksp,int i,double d,void* ctx)
{
  int s1, ierr = 0;
  s1 = PetscFromPointer(ksp);
  (*f1)(&s1,&i,&d,ctx,&ierr); CHKERRQ(ierr);
  PetscRmPointer(s1);
  return 0;
}
void kspsetmonitor_(KSP ksp,int (*monitor)(int*,int*,double*,void*,int*),
                    void *mctx, int *__ierr ){
  f1 = monitor;
  *__ierr = KSPSetMonitor((KSP)PetscToPointer(*(int*)(ksp)),ourmonitor,mctx);
}

void kspgetpc_(KSP ksp,PC *B, int *__ierr ){
  PC pc;
  *__ierr = KSPGetPC((KSP)PetscToPointer( *(int*)(ksp) ),&pc);
  *(int*) B = PetscFromPointer(pc);
}

void kspgetsolution_(KSP ksp,Vec *v, int *__ierr ){
  Vec vv;
  *__ierr = KSPGetSolution((KSP)PetscToPointer( *(int*)(ksp) ),&vv);
  *(int*) v =  PetscFromPointer(vv);
}
void kspgetrhs_(KSP ksp,Vec *r, int *__ierr ){
  Vec vv;
  *__ierr = KSPGetRhs((KSP)PetscToPointer( *(int*)(ksp) ),&vv);
  *(int*) r =  PetscFromPointer(vv);
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
  *(int*) ctx = PetscFromPointer(lg);
}

void ksplgmonitordestroy_(DrawLG ctx, int *__ierr ){
  *__ierr = KSPLGMonitorDestroy((DrawLG)PetscToPointer( *(int*)(ctx) ));
  PetscRmPointer(*(int*)(ctx) );
}


void kspdestroy_(KSP ksp, int *__ierr ){
  *__ierr = KSPDestroy((KSP)PetscToPointer( *(int*)(ksp) ));
  PetscRmPointer(*(int*)(ksp) );
}

void kspregisterdestroy_(int* __ierr)
{
  *__ierr = KSPRegisterDestroy();
}

void kspregisterall_(int* __ierr)
{
  *__ierr = KSPRegisterAll();
}

void kspbuildsolution_(KSP ctx,Vec v,Vec *V, int *__ierr ){
  Vec vv;
  *__ierr = KSPBuildSolution(
      (KSP)PetscToPointer( *(int*)(ctx) ),
      (Vec)PetscToPointer( *(int*)(v) ),&vv);
  *(int*) V = PetscFromPointer(vv);
}
void kspbuildresidual_(KSP ctx,Vec t,Vec v,Vec *V, int *__ierr ){
  Vec vv;
  *__ierr = KSPBuildResidual(
    (KSP)PetscToPointer( *(int*)(ctx) ),
    (Vec)PetscToPointer( *(int*)(t) ),
    (Vec)PetscToPointer( *(int*)(v) ),&vv);
  *(int*) V = PetscFromPointer(vv);
}

#if defined(__cplusplus)
}
#endif
