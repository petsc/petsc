#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: zts.c,v 1.13 1998/10/05 20:43:23 balay Exp bsmith $";
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

EXTERN_C_BEGIN

void tssettype_(TS *ts,CHAR itmethod, int *__ierr,int len )
{
  char *t;

  FIXCHAR(itmethod,len,t);
  *__ierr = TSSetType(*ts,t);
  FREECHAR(itmethod,t);
}


static int (*f2)(TS*,double*,Vec*,Vec*,void*,int*);
static int ourtsfunction(TS ts,double d,Vec x,Vec f,void *ctx)
{
  int ierr = 0;
  (*f2)(&ts,&d,&x,&f,ctx,&ierr); CHKERRQ(ierr);
  return 0;
}

void tssetrhsfunction_(TS *ts,int (*f)(TS*,double*,Vec*,Vec*,void*,int*),void*fP, int *__ierr )
{
  f2 = f;
  *__ierr = TSSetRHSFunction(*ts,ourtsfunction,fP);
}


/* ---------------------------------------------------------*/
static int (*f3)(TS*,double*,Mat*,Mat*,MatStructure*,void*,int*);
static int ourtsmatrix(TS ts,double d,Mat* m,Mat* p,MatStructure* type,void*ctx)
{
  int ierr = 0;
  (*f3)(&ts,&d,m,p,type,ctx,&ierr); CHKERRQ(ierr);
  return 0;
}

void tssetrhsmatrix_(TS *ts,Mat *A,Mat *B,int (*f)(TS*,double*,Mat*,Mat*,MatStructure*,
                                                   void*,int *),void*fP, int *__ierr )
{
  if (FORTRANNULLFUNCTION(f)) {
    *__ierr = TSSetRHSMatrix(*ts,*A,*B,PETSC_NULL,fP);
  } else {
    f3 = f;
    *__ierr = TSSetRHSMatrix(*ts,*A,*B,ourtsmatrix,fP);
  }
}

/* ---------------------------------------------------------*/
static int (*f4)(TS*,double*,Vec*,Mat*,Mat*,MatStructure*,void*,int*);
static int ourtsjacobian(TS ts,double d,Vec x,Mat* m,Mat* p,MatStructure* type,void*ctx)
{
  int ierr = 0;
  (*f4)(&ts,&d,&x,m,p,type,ctx,&ierr); CHKERRQ(ierr);
  return 0;
}

void tssetrhsjacobian_(TS *ts,Mat *A,Mat *B,int (*f)(TS*,double*,Vec*,Mat*,Mat*,MatStructure*,
               void*,int*),void*fP, int *__ierr )
{
  if (FORTRANNULLFUNCTION(f)) {
    *__ierr = TSSetRHSJacobian(*ts,*A,*B,PETSC_NULL,fP);
  } else {
    f4 = f;
    *__ierr = TSSetRHSJacobian(*ts,*A,*B,ourtsjacobian,fP);
  }
}

void tsgetsolution_(TS *ts,Vec *v, int *__ierr )
{
  *__ierr = TSGetSolution(*ts,v);
}

void tscreate_(MPI_Comm *comm,TSProblemType *problemtype,TS *outts, int *__ierr )
{
  *__ierr = TSCreate((MPI_Comm)PetscToPointerComm( *comm ),*problemtype,outts);
}

void tsgetsnes_(TS *ts,SNES *snes, int *__ierr )
{
  *__ierr = TSGetSNES(*ts,snes);
}

void tsgetsles_(TS *ts,SLES *sles, int *__ierr )
{
  *__ierr = TSGetSLES(*ts,sles);
}

void tsgettype_(TS *ts,CHAR name,int *__ierr,int len)
{
  char *tname;

  *__ierr = TSGetType(*ts,(TSType *)&tname);
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
void tspvodegetiterations_(TS *ts,int *nonlin, int *lin, int *__ierr)
{
  if (FORTRANNULLINTEGER(nonlin)) nonlin = PETSC_NULL;
  if (FORTRANNULLINTEGER(lin))    lin    = PETSC_NULL;
  *__ierr = TSPVodeGetIterations(*ts,nonlin,lin);
}
#endif

void tsdestroy_(TS *ts, int *__ierr ){
  *__ierr = TSDestroy(*ts);
}

static int (*f7)(TS*,int*,double*,Vec*,void*,int*);
static int ourtsmonitor(TS ts,int i,double d,Vec v,void*ctx)
{
  int              ierr = 0;
  (*f7)(&ts,&i,&d,&v,ctx,&ierr); CHKERRQ(ierr);
  return 0;
}
void tssetmonitor_(TS *ts,int (*func)(TS*,int*,double*,Vec*,void*,int*),
                    void *mctx, int *__ierr ){
  f7 = func;
  *__ierr = TSSetMonitor(*ts,ourtsmonitor,mctx);
}

void tssetrhsjacobiandefault_(TS *ts,MatFDColoring *fd,Mat *A,Mat *B,int *err)
{
  *err = TSSetRHSJacobianDefault(*ts,*fd,*A,*B); 
}

EXTERN_C_END

