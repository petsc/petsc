#ifndef lint
static char vcid[] = "$Id: zsnes.c,v 1.8 1996/01/29 21:51:54 bsmith Exp bsmith $";
#endif

#include "zpetsc.h"
#include "draw.h"
#include "snes.h"

#ifdef HAVE_FORTRAN_CAPS
#define snesregisterdestroy_         SNESREGISTERDESTROY
#define snesregisterall_             SNESREGISTERALL
#define snessetjacobian_             SNESSETJACOBIAN
#define snescreate_                  SNESCREATE
#define snessetfunction_             SNESSETFUNCTION
#define snessetminimizationfunction_ SNESSETMINIMIZATIONFUNCTION
#define snesgetsles_                 SNESGETSLES
#define snessetgradient_             SNESSETGRADIENT
#define snessethessian_              SNESSETHESSIAN
#define snessetmonitor_              SNESSETMONITOR
#define snessetconvergencetest_      SNESSETCONVERGENCETEST
#define snesregisterdestroy_         SNESREGISTERDESTROY
#define snesgetsolution_             SNESGETSOLUTION
#define snesgetsolutionupdate_       SNESGETSOLUTIONUPDATE
#define snesgetfunction_             SNESGETFUNCTION
#define snesgetgradient_             SNESGETGRADIENT
#define snesdestroy_                 SNESDESTROY
#define snesgettype_                 SNESGETTYPE
#define snessetoptionsprefix_        SNESSETOPTIONSPREFIX 
#define snesappendoptionsprefix_     SNESAPPENDOPTIONSPREFIX 
#define snesdefaultmatrixfreematcreate_ SNESDEFAULTMATRIXFREEMATCREATE
#elif !defined(HAVE_FORTRAN_UNDERSCORE)
#define snesregisterdestroy_         snesregisterdestroy
#define snesregisterall_             snesregisterall
#define snessetjacobian_             snessetjacobian
#define snescreate_                  snescreate
#define snessetfunction_             snessetfunction
#define snessethessian_              snessethessian
#define snessetgradient_             snessetgradient
#define snesgetsles_                 snesgetsles
#define snessetminimizationfunction_ snessetminimizationfunction
#define snesdestroy_                 snesdestroy
#define snessetmonitor_              snessetmonitor
#define snessetconvergencetest_      snessetconvergencetest
#define snesregisterdestroy_         snesregisterdestroy
#define snesgetsolution_             snesgetsolution
#define snesgetsolutionupdate_       snesgetsolutionupdate
#define snesgetfunction_             snesgetfunction
#define snesgetgradient_             snesgetgradient
#define snesgettype_                 snesgettype
#define snessetoptionsprefix_        snessetoptionsprefix 
#define snesappendoptionsprefix_     snesappendoptionsprefix
#define snesdefaultmatrixfreematcreate_ snesdefaultmatrixfreematcreate
#endif

#if defined(__cplusplus)
extern "C" {
#endif

void snessetoptionsprefix_(SNES snes,CHAR prefix, int *__ierr,int len ){
  char *t;

  FIXCHAR(prefix,len,t);
  *__ierr = SNESSetOptionsPrefix((SNES)MPIR_ToPointer( *(int*)(snes) ),t);
  FREECHAR(prefix,t);
}

void snesappendoptionsprefix_(SNES snes,CHAR prefix, int *__ierr,int len ){
  char *t;

  FIXCHAR(prefix,len,t);
  *__ierr = SNESAppendOptionsPrefix((SNES)MPIR_ToPointer( *(int*)(snes) ),t);
  FREECHAR(prefix,t);
}

void snesdefaultmatrixfreematcreate_(SNES snes,Vec x,Mat *J, int *__ierr ){
  Mat mm;
  *__ierr = SNESDefaultMatrixFreeMatCreate(
	(SNES)MPIR_ToPointer( *(int*)(snes) ),
	(Vec)MPIR_ToPointer( *(int*)(x) ),&mm);
  *(int*) J = MPIR_FromPointer(mm);
}

static int (*f7)(int*,int*,double*,void*,int*);
static int oursnesmonitor(SNES snes,int i,double d,void*ctx)
{
  int ierr = 0, s1;
  s1 = MPIR_FromPointer(snes);
  (*f7)(&s1,&i,&d,ctx,&ierr); CHKERRQ(ierr);
  MPIR_RmPointer(s1);
  return 0;
}
void snessetmonitor_(SNES snes,int (*func)(int*,int*,double*,void*,int*),
                    void *mctx, int *__ierr ){
  f7 = func;
  *__ierr = SNESSetMonitor(
	(SNES)MPIR_ToPointer( *(int*)(snes) ),oursnesmonitor,mctx);
}
static int (*f8)(int*,double*,double*,double*,void*,int*);
static int oursnestest(SNES snes,double a,double d,double c,void*ctx)
{
  int ierr = 0, s1;
  s1 = MPIR_FromPointer(snes);
  (*f8)(&s1,&a,&d,&c,ctx,&ierr); CHKERRQ(ierr);
  MPIR_RmPointer(s1);
  return 0;
}
void snessetconvergencetest_(SNES snes,
       int (*func)(int*,double*,double*,double*,void*,int*),
       void *cctx, int *__ierr ){
  f8 = func;
  *__ierr = SNESSetConvergenceTest(
	(SNES)MPIR_ToPointer( *(int*)(snes) ),oursnestest,cctx);
}
void snesgetsolution_(SNES snes,Vec *x, int *__ierr ){
  Vec rr;
  *__ierr = SNESGetSolution((SNES)MPIR_ToPointer( *(int*)(snes) ),&rr);
  *(int*) x = MPIR_FromPointer(rr);  
}
void snesgetsolutionupdate_(SNES snes,Vec *x, int *__ierr ){
  Vec rr;
  *__ierr = SNESGetSolutionUpdate((SNES)MPIR_ToPointer( *(int*)(snes) ),&rr);
  *(int*) x = MPIR_FromPointer(rr);  
}
void snesgetfunction_(SNES snes,Vec *r, int *__ierr ){
  Vec rr;
  *__ierr = SNESGetFunction((SNES)MPIR_ToPointer( *(int*)(snes) ),&rr);
  *(int*) r = MPIR_FromPointer(rr);
}
void snesgetgradient_(SNES snes,Vec *r, int *__ierr ){
  Vec rr;
  *__ierr = SNESGetGradient((SNES)MPIR_ToPointer( *(int*)(snes) ),&rr);
  *(int*) r = MPIR_FromPointer(rr);
}

void snesdestroy_(SNES snes, int *__ierr ){
  *__ierr = SNESDestroy((SNES)MPIR_ToPointer( *(int*)(snes) ));
  MPIR_RmPointer(*(int*)(snes));
}
void snesgetsles_(SNES snes,SLES *sles, int *__ierr ){
  SLES s;
  *__ierr = SNESGetSLES((SNES)MPIR_ToPointer( *(int*)(snes) ),&s);
  *(int*) sles = MPIR_FromPointer(s);
}
static int (*f6)(int*,int*,int*,int*,int*,void*,int*);
static int oursneshessianfunction(SNES snes,Vec x,Mat* mat,Mat* pmat,
                                  MatStructure* st,void *ctx)
{
  int ierr = 0, s1, s2,s3, s4,o3,o4;
  s1 = MPIR_FromPointer(snes);
  s2 = MPIR_FromPointer(x);
  o3 = s3 = MPIR_FromPointer(*mat);
  o4 = s4 = MPIR_FromPointer(*pmat);
  (*f6)(&s1,&s2,&s3,&s4,(int*)st,ctx,&ierr); CHKERRQ(ierr);
  if (o3 != s3) *mat  = (Mat) MPIR_ToPointer(s3);
  if (o4 != s4) *pmat = (Mat) MPIR_ToPointer(s4);
  MPIR_RmPointer(s1);
  MPIR_RmPointer(s2);
  MPIR_RmPointer(o3);
  MPIR_RmPointer(o3);
  return 0;
}
void snessethessian_(SNES snes,Mat A,Mat B,int (*func)(int*,int*,int*,int*,
                     int*,void*,int*),void *ctx, int *__ierr ){
  f6 = func;
  *__ierr = SNESSetHessian(
	(SNES)MPIR_ToPointer( *(int*)(snes) ),
	(Mat)MPIR_ToPointer( *(int*)(A) ),
	(Mat)MPIR_ToPointer( *(int*)(B) ),oursneshessianfunction,ctx);
}

static int (*f5)(int*,int*,int*,void*,int*);
static int oursnesgradientfunction(SNES snes,Vec x,Vec d,void *ctx)
{
  int ierr = 0, s1, s2,s3;
  s1 = MPIR_FromPointer(snes);
  s2 = MPIR_FromPointer(x);
  s3 = MPIR_FromPointer(d);
  (*f5)(&s1,&s2,&s3,ctx,&ierr); CHKERRQ(ierr);
  MPIR_RmPointer(s1);
  MPIR_RmPointer(s2);
  MPIR_RmPointer(s3);
  return 0;
}
void snessetgradient_(SNES snes,Vec r,int (*func)(int*,int*,int*,void*,int*),
                     void *ctx, int *__ierr ){
  f5 = func;
  *__ierr = SNESSetGradient(
	(SNES)MPIR_ToPointer( *(int*)(snes) ),
	(Vec)MPIR_ToPointer( *(int*)(r) ),oursnesgradientfunction,ctx);
}

static int (*f4)(int*,int*,double*,void*,int*);
static int oursnesminfunction(SNES snes,Vec x,double* d,void *ctx)
{
  int ierr = 0, s1, s2;
  s1 = MPIR_FromPointer(snes);
  s2 = MPIR_FromPointer(x);
  (*f4)(&s1,&s2,d,ctx,&ierr); CHKERRQ(ierr);
  MPIR_RmPointer(s1);
  MPIR_RmPointer(s2);
  return 0;
}
void snessetminimizationfunction_(SNES snes,
          int (*func)(int*,int*,double*,void*,int*),void *ctx, int *__ierr ){
  f4 = func;
  *__ierr = SNESSetMinimizationFunction(
	(SNES)MPIR_ToPointer( *(int*)(snes) ),oursnesminfunction,ctx);
}

static int (*f2)(int*,int*,int*,void*,int*);
static int oursnesfunction(SNES snes,Vec x,Vec f,void *ctx)
{
  int ierr = 0, s1, s2, s3;
  s1 = MPIR_FromPointer(snes);
  s2 = MPIR_FromPointer(x);
  s3 = MPIR_FromPointer(f);
  (*f2)(&s1,&s2,&s3,ctx,&ierr); CHKERRQ(ierr);
  MPIR_RmPointer(s1);
  MPIR_RmPointer(s2);
  MPIR_RmPointer(s3);
  return 0;
}
void snessetfunction_(SNES snes,Vec r,int (*func)(int*,int*,int*,void*,int*),
                     void *ctx, int *__ierr ){
   f2 = func;
*__ierr = SNESSetFunction(
	(SNES)MPIR_ToPointer( *(int*)(snes) ),
	(Vec)MPIR_ToPointer( *(int*)(r) ),oursnesfunction,ctx);
}
/* ---------------------------------------------------------*/
void snescreate_(MPI_Comm comm,SNESProblemType *type,SNES *outsnes, int *__ierr ){
  SNES snes;
*__ierr = SNESCreate(
	(MPI_Comm)MPIR_ToPointer_Comm( *(int*)(comm) ),*type,&snes);
  *(int*)outsnes = MPIR_FromPointer(snes);
}
/* ---------------------------------------------------------*/
static int (*f3)(int*,int*,int*,int*,MatStructure*,void*,int*);
static int oursnesjacobian(SNES snes,Vec x,Mat* m,Mat* p,MatStructure* type,
                          void*ctx)
{
  int ierr = 0, s1,s2,s3,s4,s3_o,s4_o;
  s1 = MPIR_FromPointer(snes);
  s2 = MPIR_FromPointer(x);
  s3 = s3_o = MPIR_FromPointer(*m);
  s4 = s4_o = MPIR_FromPointer(*p);
  (*f3)(&s1,&s2,&s3,&s4,type,ctx,&ierr); CHKERRQ(ierr);
  if (s3_o != s3) *m = (Mat) MPIR_ToPointer(s3);
  if (s4_o != s4) *p = (Mat) MPIR_ToPointer(s4);
  MPIR_RmPointer(s1);
  MPIR_RmPointer(s2);
  MPIR_RmPointer(s3);
  MPIR_RmPointer(s4);
  return 0;
}

void snessetjacobian_(SNES snes,Mat A,Mat B,int (*func)(int*,int*,int*,int*,
                      MatStructure*,void*,int*),void *ctx, int *__ierr )
{
  f3 = func;
  *__ierr = SNESSetJacobian(
	(SNES)MPIR_ToPointer( *(int*)(snes) ),
	(Mat)MPIR_ToPointer( *(int*)(A) ),
	(Mat)MPIR_ToPointer( *(int*)(B) ),oursnesjacobian,ctx);
}

/* -------------------------------------------------------------*/

void snesregisterdestroy_(int *__ierr)
{
  *__ierr = SNESRegisterDestroy();
}

void snesregisterall_(int *__ierr)
{
  *__ierr = SNESRegisterAll();
}

void snesgettype_(SNES snes,SNESType *type,CHAR name,int *__ierr,int len)
{
  char *tname;

  if (FORTRANNULL(type)) type = PETSC_NULL;
  *__ierr = SNESGetType((SNES)MPIR_ToPointer(*(int*)snes),type,&tname);
#if defined(PARCH_t3d)
  {
  char *t = _fcdtocp(name); int len1 = _fcdlen(name);
  if (t != PETSC_NULL_CHAR_Fortran) PetscStrncpy(t,tname,len1);
  }
#else
  if (name != PETSC_NULL_CHAR_Fortran) PetscStrncpy(name,tname,len);
#endif
}

#if defined(__cplusplus)
}
#endif

