#ifndef lint
static char vcid[] = "$Id: zts.c,v 1.3 1997/04/04 19:10:23 bsmith Exp bsmith $";
#endif

#include "src/fortran/custom/zpetsc.h"
#include "ts.h"

#ifdef HAVE_FORTRAN_CAPS
#define tssetrhsfunction_ TSSETRHSFUNCTION
#define tssetrhsmatrix_   TSSETRHSMATRIX
#define tssetrhsjacobian_ TSSETRHSJACOBIAN
#define tscreate_         TSCREATE
#define tsgetsolution_    TSGETSOLUTION
#define tsgetsnes_        TSGETSNES
#define tsgetsles_        TSGETSLES
#define tsgettype_        TSGETTYPE
#define tsdestroy_        TSDESTROY
#define tssetmonitor_     TSSETMONITOR
#elif !defined(HAVE_FORTRAN_UNDERSCORE)
#define tssetrhsfunction_ tssetrhsfunction
#define tssetrhsmatrix_   tssetrhsmatrix
#define tssetrhsjacobian_ tssetrhsjacobian
#define tscreate_         tscreate
#define tsgetsolution_    tsgetsolution
#define tsgetsnes_        tsgetsnes
#define tsgetsles_        tsgetsles
#define tsgettype_        tsgettype
#define tsdestroy_        tsdestroy
#define tssetmonitor_     tssetmonitor
#endif

#if defined(__cplusplus)
extern "C" {
#endif

static int (*f2)(int*,double*,int*,int*,void*,int*);
static int ourtsfunction(TS ts,double d,Vec x,Vec f,void *ctx)
{
  int ierr = 0, s1, s2, s3;
  s1 = PetscFromPointer(ts);
  s2 = PetscFromPointer(x);
  s3 = PetscFromPointer(f);
  (*f2)(&s1,&d,&s2,&s3,ctx,&ierr); CHKERRQ(ierr);
  PetscRmPointer(s1);
  PetscRmPointer(s2);
  PetscRmPointer(s3);
  return 0;
}

void tssetrhsfunction_(TS ts,int (*f)(int*,double*,int*,int*,void*,int*),void*fP, int *__ierr )
{
  f2 = f;
  *__ierr = TSSetRHSFunction((TS)PetscToPointer( *(int*)(ts) ),ourtsfunction,fP);
}


/* ---------------------------------------------------------*/
static int (*f3)(int*,double*,int*,int*,MatStructure*,void*,int*);
static int ourtsmatrix(TS ts,double d,Mat* m,Mat* p,MatStructure* type,void*ctx)
{
  int ierr = 0, s1,s3,s4,s3_o,s4_o;
  s1 = PetscFromPointer(ts);
  s3 = s3_o = PetscFromPointer(*m);
  s4 = s4_o = PetscFromPointer(*p);
  (*f3)(&s1,&d,&s3,&s4,type,ctx,&ierr); CHKERRQ(ierr);
  if (s3_o != s3) *m = (Mat) PetscToPointer(s3);
  if (s4_o != s4) *p = (Mat) PetscToPointer(s4);
  PetscRmPointer(s1);
  PetscRmPointer(s3);
  PetscRmPointer(s4);
  return 0;
}

void tssetrhsmatrix_(TS ts,Mat A,Mat B,int (*f)(int*,double*,int*,int*,
                    MatStructure*,void*,int *),void*fP, int *__ierr )
{
  if (FORTRANNULL(f)) {
    *__ierr = TSSetRHSMatrix((TS)PetscToPointer( *(int*)(ts) ),
	                   (Mat)PetscToPointer( *(int*)(A) ),
	                   (Mat)PetscToPointer( *(int*)(B) ),PETSC_NULL,fP);
  } else {
    f3 = f;
    *__ierr = TSSetRHSMatrix((TS)PetscToPointer( *(int*)(ts) ),
	                   (Mat)PetscToPointer( *(int*)(A) ),
	                   (Mat)PetscToPointer( *(int*)(B) ),ourtsmatrix,fP);
  }
}

/* ---------------------------------------------------------*/
static int (*f4)(int*,double*,int*,int*,int*,MatStructure*,void*,int*);
static int ourtsjacobian(TS ts,double d,Vec x,Mat* m,Mat* p,MatStructure* type,void*ctx)
{
  int ierr = 0, s1,s2,s3,s4,s3_o,s4_o;
  s1 = PetscFromPointer(ts);
  s2 = PetscFromPointer(x);
  s3 = s3_o = PetscFromPointer(*m);
  s4 = s4_o = PetscFromPointer(*p);
  (*f4)(&s1,&d,&s2,&s3,&s4,type,ctx,&ierr); CHKERRQ(ierr);
  if (s3_o != s3) *m = (Mat) PetscToPointer(s3);
  if (s4_o != s4) *p = (Mat) PetscToPointer(s4);
  PetscRmPointer(s1);
  PetscRmPointer(s2);
  PetscRmPointer(s3);
  PetscRmPointer(s4);
  return 0;
}

void tssetrhsjacobian_(TS ts,Mat A,Mat B,int (*f)(int*,double*,int*,int*,int*,
                     MatStructure*,void*,int*),void*fP, int *__ierr )
{
  if (FORTRANNULL(f)) {
    *__ierr = TSSetRHSJacobian((TS)PetscToPointer( *(int*)(ts) ),
	                       (Mat)PetscToPointer( *(int*)(A) ),
	                       (Mat)PetscToPointer( *(int*)(B) ),PETSC_NULL,fP);
  } else {
    f4 = f;
    *__ierr = TSSetRHSJacobian((TS)PetscToPointer( *(int*)(ts) ),
	                       (Mat)PetscToPointer( *(int*)(A) ),
	                       (Mat)PetscToPointer( *(int*)(B) ),ourtsjacobian,fP);
  }
}

void tsgetsolution_(TS ts,Vec *v, int *__ierr )
{
  Vec rr;
  *__ierr = TSGetSolution((TS)PetscToPointer( *(int*)(ts) ),&rr);
  *(int*) v = PetscFromPointer(rr);
}

void tscreate_(MPI_Comm *comm,TSProblemType *problemtype,TS *outts, int *__ierr )
{
  TS s;
  *__ierr = TSCreate((MPI_Comm)PetscToPointerComm( *comm ),*problemtype,&s);
  *(int*)outts = PetscFromPointer(s);
}

void tsgetsnes_(TS ts,SNES *snes, int *__ierr )
{
  SNES s;
  *__ierr = TSGetSNES((TS)PetscToPointer( *(int*)(ts) ),&s);
  *(int*) snes = PetscFromPointer(s);
}

void tsgetsles_(TS ts,SLES *sles, int *__ierr )
{
  SLES s;
  *__ierr = TSGetSLES((TS)PetscToPointer( *(int*)(ts) ),&s);
  *(int*) sles = PetscFromPointer(s);
}

void tsgettype_(TS ts,TSType *type,CHAR name,int *__ierr,int len)
{
  char *tname;
  if (FORTRANNULL(type)) type = PETSC_NULL;
  *__ierr = TSGetType((TS)PetscToPointer(*(int*)ts),type,&tname);
#if defined(USES_CPTOFCD)
  {
  char *t = _fcdtocp(name); int len1 = _fcdlen(name);
  if (t != PETSC_NULL_CHARACTER_Fortran) PetscStrncpy(t,tname,len1);
  }
#else
  if (name != PETSC_NULL_CHARACTER_Fortran) PetscStrncpy(name,tname,len);
#endif
}

void tsdestroy_(TS ts, int *__ierr ){
  *__ierr = TSDestroy((TS)PetscToPointer( *(int*)(ts) ));
  PetscRmPointer(*(int*)(ts));
}

static int (*f7)(int*,int*,double*,int*,void*,int*);
static int ourtsmonitor(TS ts,int i,double d,Vec v,void*ctx)
{
  int ierr = 0, s1,s2;
  s1 = PetscFromPointer(ts);
  s2 = PetscFromPointer(v);
  (*f7)(&s1,&i,&d,&s2,ctx,&ierr); CHKERRQ(ierr);
  PetscRmPointer(s1);
  PetscRmPointer(s2);
  return 0;
}
void tssetmonitor_(TS ts,int (*func)(int*,int*,double*,int*,void*,int*),
                    void *mctx, int *__ierr ){
  f7 = func;
  *__ierr = TSSetMonitor((TS)PetscToPointer( *(int*)(ts) ),ourtsmonitor,mctx);
}

#if defined(__cplusplus)
}
#endif

