#ifndef lint
static char vcid[] = "$Id: zsnes.c,v 1.10 1996/03/04 21:50:23 bsmith Exp bsmith $";
#endif

#include "src/fortran/custom/zpetsc.h"
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
  *__ierr = SNESSetOptionsPrefix((SNES)PetscToPointer( *(int*)(snes) ),t);
  FREECHAR(prefix,t);
}

void snesappendoptionsprefix_(SNES snes,CHAR prefix, int *__ierr,int len ){
  char *t;

  FIXCHAR(prefix,len,t);
  *__ierr = SNESAppendOptionsPrefix((SNES)PetscToPointer( *(int*)(snes) ),t);
  FREECHAR(prefix,t);
}

void snesdefaultmatrixfreematcreate_(SNES snes,Vec x,Mat *J, int *__ierr ){
  Mat mm;
  *__ierr = SNESDefaultMatrixFreeMatCreate(
	(SNES)PetscToPointer( *(int*)(snes) ),
	(Vec)PetscToPointer( *(int*)(x) ),&mm);
  *(int*) J = PetscFromPointer(mm);
}

static int (*f7)(int*,int*,double*,void*,int*);
static int oursnesmonitor(SNES snes,int i,double d,void*ctx)
{
  int ierr = 0, s1;
  s1 = PetscFromPointer(snes);
  (*f7)(&s1,&i,&d,ctx,&ierr); CHKERRQ(ierr);
  PetscRmPointer(s1);
  return 0;
}
void snessetmonitor_(SNES snes,int (*func)(int*,int*,double*,void*,int*),
                    void *mctx, int *__ierr ){
  f7 = func;
  *__ierr = SNESSetMonitor(
	(SNES)PetscToPointer( *(int*)(snes) ),oursnesmonitor,mctx);
}
static int (*f8)(int*,double*,double*,double*,void*,int*);
static int oursnestest(SNES snes,double a,double d,double c,void*ctx)
{
  int ierr = 0, s1;
  s1 = PetscFromPointer(snes);
  (*f8)(&s1,&a,&d,&c,ctx,&ierr); CHKERRQ(ierr);
  PetscRmPointer(s1);
  return 0;
}
void snessetconvergencetest_(SNES snes,
       int (*func)(int*,double*,double*,double*,void*,int*),
       void *cctx, int *__ierr ){
  f8 = func;
  *__ierr = SNESSetConvergenceTest(
	(SNES)PetscToPointer( *(int*)(snes) ),oursnestest,cctx);
}
void snesgetsolution_(SNES snes,Vec *x, int *__ierr ){
  Vec rr;
  *__ierr = SNESGetSolution((SNES)PetscToPointer( *(int*)(snes) ),&rr);
  *(int*) x = PetscFromPointer(rr);  
}
void snesgetsolutionupdate_(SNES snes,Vec *x, int *__ierr ){
  Vec rr;
  *__ierr = SNESGetSolutionUpdate((SNES)PetscToPointer( *(int*)(snes) ),&rr);
  *(int*) x = PetscFromPointer(rr);  
}
void snesgetfunction_(SNES snes,Vec *r, int *__ierr ){
  Vec rr;
  *__ierr = SNESGetFunction((SNES)PetscToPointer( *(int*)(snes) ),&rr);
  *(int*) r = PetscFromPointer(rr);
}
void snesgetgradient_(SNES snes,Vec *r, int *__ierr ){
  Vec rr;
  *__ierr = SNESGetGradient((SNES)PetscToPointer( *(int*)(snes) ),&rr);
  *(int*) r = PetscFromPointer(rr);
}

void snesdestroy_(SNES snes, int *__ierr ){
  *__ierr = SNESDestroy((SNES)PetscToPointer( *(int*)(snes) ));
  PetscRmPointer(*(int*)(snes));
}
void snesgetsles_(SNES snes,SLES *sles, int *__ierr ){
  SLES s;
  *__ierr = SNESGetSLES((SNES)PetscToPointer( *(int*)(snes) ),&s);
  *(int*) sles = PetscFromPointer(s);
}
static int (*f6)(int*,int*,int*,int*,int*,void*,int*);
static int oursneshessianfunction(SNES snes,Vec x,Mat* mat,Mat* pmat,
                                  MatStructure* st,void *ctx)
{
  int ierr = 0, s1, s2,s3, s4,o3,o4;
  s1 = PetscFromPointer(snes);
  s2 = PetscFromPointer(x);
  o3 = s3 = PetscFromPointer(*mat);
  o4 = s4 = PetscFromPointer(*pmat);
  (*f6)(&s1,&s2,&s3,&s4,(int*)st,ctx,&ierr); CHKERRQ(ierr);
  if (o3 != s3) *mat  = (Mat) PetscToPointer(s3);
  if (o4 != s4) *pmat = (Mat) PetscToPointer(s4);
  PetscRmPointer(s1);
  PetscRmPointer(s2);
  PetscRmPointer(o3);
  PetscRmPointer(o3);
  return 0;
}
void snessethessian_(SNES snes,Mat A,Mat B,int (*func)(int*,int*,int*,int*,
                     int*,void*,int*),void *ctx, int *__ierr ){
  f6 = func;
  *__ierr = SNESSetHessian(
	(SNES)PetscToPointer( *(int*)(snes) ),
	(Mat)PetscToPointer( *(int*)(A) ),
	(Mat)PetscToPointer( *(int*)(B) ),oursneshessianfunction,ctx);
}

static int (*f5)(int*,int*,int*,void*,int*);
static int oursnesgradientfunction(SNES snes,Vec x,Vec d,void *ctx)
{
  int ierr = 0, s1, s2,s3;
  s1 = PetscFromPointer(snes);
  s2 = PetscFromPointer(x);
  s3 = PetscFromPointer(d);
  (*f5)(&s1,&s2,&s3,ctx,&ierr); CHKERRQ(ierr);
  PetscRmPointer(s1);
  PetscRmPointer(s2);
  PetscRmPointer(s3);
  return 0;
}
void snessetgradient_(SNES snes,Vec r,int (*func)(int*,int*,int*,void*,int*),
                     void *ctx, int *__ierr ){
  f5 = func;
  *__ierr = SNESSetGradient(
	(SNES)PetscToPointer( *(int*)(snes) ),
	(Vec)PetscToPointer( *(int*)(r) ),oursnesgradientfunction,ctx);
}

static int (*f4)(int*,int*,double*,void*,int*);
static int oursnesminfunction(SNES snes,Vec x,double* d,void *ctx)
{
  int ierr = 0, s1, s2;
  s1 = PetscFromPointer(snes);
  s2 = PetscFromPointer(x);
  (*f4)(&s1,&s2,d,ctx,&ierr); CHKERRQ(ierr);
  PetscRmPointer(s1);
  PetscRmPointer(s2);
  return 0;
}
void snessetminimizationfunction_(SNES snes,
          int (*func)(int*,int*,double*,void*,int*),void *ctx, int *__ierr ){
  f4 = func;
  *__ierr = SNESSetMinimizationFunction(
	(SNES)PetscToPointer( *(int*)(snes) ),oursnesminfunction,ctx);
}

static int (*f2)(int*,int*,int*,void*,int*);
static int oursnesfunction(SNES snes,Vec x,Vec f,void *ctx)
{
  int ierr = 0, s1, s2, s3;
  s1 = PetscFromPointer(snes);
  s2 = PetscFromPointer(x);
  s3 = PetscFromPointer(f);
  (*f2)(&s1,&s2,&s3,ctx,&ierr); CHKERRQ(ierr);
  PetscRmPointer(s1);
  PetscRmPointer(s2);
  PetscRmPointer(s3);
  return 0;
}
void snessetfunction_(SNES snes,Vec r,int (*func)(int*,int*,int*,void*,int*),
                     void *ctx, int *__ierr ){
   f2 = func;
*__ierr = SNESSetFunction(
	(SNES)PetscToPointer( *(int*)(snes) ),
	(Vec)PetscToPointer( *(int*)(r) ),oursnesfunction,ctx);
}
/* ---------------------------------------------------------*/
void snescreate_(MPI_Comm comm,SNESProblemType *type,SNES *outsnes, int *__ierr ){
  SNES snes;
*__ierr = SNESCreate(
	(MPI_Comm)PetscToPointerComm( *(int*)(comm) ),*type,&snes);
  *(int*)outsnes = PetscFromPointer(snes);
}
/* ---------------------------------------------------------*/
static int (*f3)(int*,int*,int*,int*,MatStructure*,void*,int*);
static int oursnesjacobian(SNES snes,Vec x,Mat* m,Mat* p,MatStructure* type,
                          void*ctx)
{
  int ierr = 0, s1,s2,s3,s4,s3_o,s4_o;
  s1 = PetscFromPointer(snes);
  s2 = PetscFromPointer(x);
  s3 = s3_o = PetscFromPointer(*m);
  s4 = s4_o = PetscFromPointer(*p);
  (*f3)(&s1,&s2,&s3,&s4,type,ctx,&ierr); CHKERRQ(ierr);
  if (s3_o != s3) *m = (Mat) PetscToPointer(s3);
  if (s4_o != s4) *p = (Mat) PetscToPointer(s4);
  PetscRmPointer(s1);
  PetscRmPointer(s2);
  PetscRmPointer(s3);
  PetscRmPointer(s4);
  return 0;
}

void snessetjacobian_(SNES snes,Mat A,Mat B,int (*func)(int*,int*,int*,int*,
                      MatStructure*,void*,int*),void *ctx, int *__ierr )
{
  f3 = func;
  *__ierr = SNESSetJacobian(
	(SNES)PetscToPointer( *(int*)(snes) ),
	(Mat)PetscToPointer( *(int*)(A) ),
	(Mat)PetscToPointer( *(int*)(B) ),oursnesjacobian,ctx);
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
  *__ierr = SNESGetType((SNES)PetscToPointer(*(int*)snes),type,&tname);
#if defined(USES_CPTOFCD)
  {
  char *t = _fcdtocp(name); int len1 = _fcdlen(name);
  if (t != PETSC_NULL_CHARACTER_Fortran) PetscStrncpy(t,tname,len1);
  }
#else
  if (name != PETSC_NULL_CHARACTER_Fortran) PetscStrncpy(name,tname,len);
#endif
}

#if defined(__cplusplus)
}
#endif

