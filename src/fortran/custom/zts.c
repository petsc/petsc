#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: zts.c,v 1.11 1998/04/08 22:29:25 balay Exp balay $";
#endif

#include "src/fortran/custom/zpetsc.h"
#include "ts.h"

#ifdef HAVE_FORTRAN_CAPS
#define tssetrhsfunction_                    TSSETRHSFUNCTION
#define tssetrhsmatrix_                      TSSETRHSMATRIX
#define tssetrhsjacobian_                    TSSETRHSJACOBIAN
#define tscreate_                            TSCREATE
#define tsgetsolution_                       TSGETSOLUTION
#define tsgetsnes_                           TSGETSNES
#define tsgetsles_                           TSGETSLES
#define tsgettype_                           TSGETTYPE
#define tsdestroy_                           TSDESTROY
#define tssetmonitor_                        TSSETMONITOR
#define tssetrhsjacobiandefault_             TSSETRHSJACOBIANDEFAULT
#define tssettype_                           TSSETTYPE
#define tspvodegetiterations_                TSPVODEGETITERATIONS
#elif !defined(HAVE_FORTRAN_UNDERSCORE)
#define tspvodegetiterations_                 tspvodegetiterations
#define tssetrhsfunction_                     tssetrhsfunction
#define tssetrhsmatrix_                       tssetrhsmatrix
#define tssetrhsjacobian_                     tssetrhsjacobian
#define tscreate_                             tscreate
#define tsgetsolution_                        tsgetsolution
#define tsgetsnes_                            tsgetsnes
#define tsgetsles_                            tsgetsles
#define tsgettype_                            tsgettype
#define tsdestroy_                            tsdestroy
#define tssetmonitor_                         tssetmonitor
#define tssetrhsjacobiandefault_              tssetrhsjacobiandefault
#define tssettype_                            tssettype
#endif

#if defined(__cplusplus)
extern "C" {
#endif

void tssettype_(TS ts,CHAR itmethod, int *__ierr,int len )
{
  char *t;

  FIXCHAR(itmethod,len,t);
  *__ierr = TSSetType((TS)PetscToPointer(ts),t);
  FREECHAR(itmethod,t);
}


static int (*f2)(PetscFortranAddr*,double*,PetscFortranAddr*,PetscFortranAddr*,void*,int*);
static int ourtsfunction(TS ts,double d,Vec x,Vec f,void *ctx)
{
  int              ierr = 0;
  PetscFortranAddr s1,s2,s3;

  s1 = PetscFromPointer(ts);
  s2 = PetscFromPointer(x);
  s3 = PetscFromPointer(f);
  (*f2)(&s1,&d,&s2,&s3,ctx,&ierr); CHKERRQ(ierr);
  PetscRmPointer(&s1);
  PetscRmPointer(&s2);
  PetscRmPointer(&s3);
  return 0;
}

void tssetrhsfunction_(TS ts,int (*f)(PetscFortranAddr*,double*,PetscFortranAddr*,
                 PetscFortranAddr*,void*,int*),void*fP, int *__ierr )
{
  f2 = f;
  *__ierr = TSSetRHSFunction((TS)PetscToPointer(ts),ourtsfunction,fP);
}


/* ---------------------------------------------------------*/
static int (*f3)(PetscFortranAddr*,double*,PetscFortranAddr*,PetscFortranAddr*,
                 MatStructure*,void*,int*);
static int ourtsmatrix(TS ts,double d,Mat* m,Mat* p,MatStructure* type,void*ctx)
{
  int              ierr = 0;
  PetscFortranAddr s1,s3,s4,s3_o,s4_o;

  s1 = PetscFromPointer(ts);
  s3 = s3_o = PetscFromPointer(*m);
  s4 = s4_o = PetscFromPointer(*p);
  (*f3)(&s1,&d,&s3,&s4,type,ctx,&ierr); CHKERRQ(ierr);
  if (s3_o != s3) *m = (Mat) PetscToPointer(&s3);
  if (s4_o != s4) *p = (Mat) PetscToPointer(&s4);
  PetscRmPointer(&s1);
  PetscRmPointer(&s3);
  PetscRmPointer(&s4);
  return 0;
}

void tssetrhsmatrix_(TS ts,Mat A,Mat B,int (*f)(PetscFortranAddr*,double*,PetscFortranAddr*,
            PetscFortranAddr*,MatStructure*,void*,int *),void*fP, int *__ierr )
{
  if (FORTRANNULLFUNCTION(f)) {
    *__ierr = TSSetRHSMatrix((TS)PetscToPointer(ts),
	                   (Mat)PetscToPointer(A),
	                   (Mat)PetscToPointer(B),PETSC_NULL,fP);
  } else {
    f3 = f;
    *__ierr = TSSetRHSMatrix((TS)PetscToPointer(ts),
	                   (Mat)PetscToPointer(A),
	                   (Mat)PetscToPointer(B),ourtsmatrix,fP);
  }
}

/* ---------------------------------------------------------*/
static int (*f4)(PetscFortranAddr*,double*,PetscFortranAddr*,PetscFortranAddr*,PetscFortranAddr*,
                 MatStructure*,void*,int*);
static int ourtsjacobian(TS ts,double d,Vec x,Mat* m,Mat* p,MatStructure* type,void*ctx)
{
  int              ierr = 0;
  PetscFortranAddr s1,s2,s3,s4,s3_o,s4_o;

  s1 = PetscFromPointer(ts);
  s2 = PetscFromPointer(x);
  s3 = s3_o = PetscFromPointer(*m);
  s4 = s4_o = PetscFromPointer(*p);
  (*f4)(&s1,&d,&s2,&s3,&s4,type,ctx,&ierr); CHKERRQ(ierr);
  if (s3_o != s3) *m = (Mat) PetscToPointer(&s3);
  if (s4_o != s4) *p = (Mat) PetscToPointer(&s4);
  PetscRmPointer(&s1);
  PetscRmPointer(&s2);
  PetscRmPointer(&s3);
  PetscRmPointer(&s4);
  return 0;
}

void tssetrhsjacobian_(TS ts,Mat A,Mat B,int (*f)(PetscFortranAddr*,double*,PetscFortranAddr*,
             PetscFortranAddr*,PetscFortranAddr*,MatStructure*,void*,int*),void*fP, int *__ierr )
{
  if (FORTRANNULLFUNCTION(f)) {
    *__ierr = TSSetRHSJacobian((TS)PetscToPointer(ts),
	                       (Mat)PetscToPointer(A),
	                       (Mat)PetscToPointer(B),PETSC_NULL,fP);
  } else {
    f4 = f;
    *__ierr = TSSetRHSJacobian((TS)PetscToPointer(ts),
	                       (Mat)PetscToPointer(A),
	                       (Mat)PetscToPointer(B),ourtsjacobian,fP);
  }
}

void tsgetsolution_(TS ts,Vec *v, int *__ierr )
{
  Vec rr;
  *__ierr = TSGetSolution((TS)PetscToPointer(ts),&rr);
  *(PetscFortranAddr*) v = PetscFromPointer(rr);
}

void tscreate_(MPI_Comm *comm,TSProblemType *problemtype,TS *outts, int *__ierr )
{
  TS s;
  *__ierr = TSCreate((MPI_Comm)PetscToPointerComm( *comm ),*problemtype,&s);
  *(PetscFortranAddr*)outts = PetscFromPointer(s);
}

void tsgetsnes_(TS ts,SNES *snes, int *__ierr )
{
  SNES s;
  *__ierr = TSGetSNES((TS)PetscToPointer(ts),&s);
  *(PetscFortranAddr*) snes = PetscFromPointer(s);
}

void tsgetsles_(TS ts,SLES *sles, int *__ierr )
{
  SLES s;
  *__ierr = TSGetSLES((TS)PetscToPointer(ts),&s);
  *(PetscFortranAddr*) sles = PetscFromPointer(s);
}

void tsgettype_(TS ts,CHAR name,int *__ierr,int len)
{
  char *tname;

  *__ierr = TSGetType((TS)PetscToPointer(ts),(TSType *)&tname);
#if defined(USES_CPTOFCD)
  {
    char *t = _fcdtocp(name); int len1 = _fcdlen(name);
    PetscStrncpy(t,tname,len1);
  }
#else
  PetscStrncpy(name,tname,len);
#endif
}

#if defined(HAVE_PVODE)  && !defined(__cplusplus)
void tspvodegetiterations_(TS ts,int *nonlin, int *lin, int *__ierr)
{
  if (FORTRANNULLINTEGER(nonlin)) nonlin = PETSC_NULL;
  if (FORTRANNULLINTEGER(lin))    lin    = PETSC_NULL;
  *__ierr = TSPVodeGetIterations((TS)PetscToPointer(ts),nonlin,lin);
}
#endif

void tsdestroy_(TS ts, int *__ierr ){
  *__ierr = TSDestroy((TS)PetscToPointer(ts));
  PetscRmPointer(ts);
}

static int (*f7)(PetscFortranAddr*,int*,double*,PetscFortranAddr*,void*,int*);
static int ourtsmonitor(TS ts,int i,double d,Vec v,void*ctx)
{
  int              ierr = 0;
  PetscFortranAddr s1,s2;

  s1 = PetscFromPointer(ts);
  s2 = PetscFromPointer(v);
  (*f7)(&s1,&i,&d,&s2,ctx,&ierr); CHKERRQ(ierr);
  PetscRmPointer(&s1);
  PetscRmPointer(&s2);
  return 0;
}
void tssetmonitor_(TS ts,int (*func)(PetscFortranAddr*,int*,double*,PetscFortranAddr*,void*,int*),
                    void *mctx, int *__ierr ){
  f7 = func;
  *__ierr = TSSetMonitor((TS)PetscToPointer(ts),ourtsmonitor,mctx);
}

void tssetrhsjacobiandefault_(TS ts,MatFDColoring fd,Mat A,Mat B,int *err)
{
  *err = TSSetRHSJacobianDefault((TS)PetscToPointer(ts),
                                             (MatFDColoring)PetscToPointer(fd),
                                             (Mat)PetscToPointer(A),
	                                     (Mat)PetscToPointer(B)); 
}

#if defined(__cplusplus)
}
#endif

