/*$Id: zts.c,v 1.21 1999/10/24 14:04:19 bsmith Exp bsmith $*/

#include "src/fortran/custom/zpetsc.h"
#include "ts.h"

#ifdef PETSC_HAVE_FORTRAN_CAPS
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
#define tssettype_                           TSSETTYPE
#define tspvodegetiterations_                TSPVODEGETITERATIONS
#define tsdefaultcomputejacobian_            TSDEFAULTCOMPUTEJACOBIAN
#define tsdefaultcomputejacobiancolor_       TSDEFAULTCOMPUTEJACOBIANCOLOR
#define tsgetoptionsprefix_                  TSGETOPTIONSPREFIX
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define tsdefaultcomputejacobian_            tsdefaultcomputejacobian
#define tsdefaultcomputejacobiancolor_       tsdefaultcomputejacobiancolor
#define tspvodegetiterations_                tspvodegetiterations
#define tssetrhsfunction_                    tssetrhsfunction
#define tssetrhsmatrix_                      tssetrhsmatrix
#define tssetrhsjacobian_                    tssetrhsjacobian
#define tscreate_                            tscreate
#define tsgetsolution_                       tsgetsolution
#define tsgetsnes_                           tsgetsnes
#define tsgetsles_                           tsgetsles
#define tsgettype_                           tsgettype
#define tsdestroy_                           tsdestroy
#define tssetmonitor_                        tssetmonitor
#define tssettype_                           tssettype
#define tsgetoptionsprefix_                  tsgetoptionsprefix
#endif

EXTERN_C_BEGIN

void PETSC_STDCALL tsdefaultcomputejacobian_(TS *ts,double *t,Vec *xx1,Mat *J,Mat *B,MatStructure *flag,void *ctx,int *__ierr)
{
  *__ierr = TSDefaultComputeJacobian(*ts,*t,*xx1,J,B,flag,ctx);
}

void PETSC_STDCALL tsdefaultcomputejacobiancolor_(TS *ts,double *t,Vec *xx1,Mat *J,Mat *B,MatStructure *flag,void *ctx,int *__ierr)
{
  *__ierr = TSDefaultComputeJacobianColor(*ts,*t,*xx1,J,B,flag,*(MatFDColoring*)ctx);
}

void PETSC_STDCALL tssettype_(TS *ts,CHAR type, int *__ierr,int len )
{
  char *t;

  FIXCHAR(type,len,t);
  *__ierr = TSSetType(*ts,t);
  FREECHAR(type,t);
}


static int (*f2)(TS*,double*,Vec*,Vec*,void*,int*);
static int ourtsfunction(TS ts,double d,Vec x,Vec f,void *ctx)
{
  int ierr = 0;
  (*f2)(&ts,&d,&x,&f,ctx,&ierr);CHKERRQ(ierr);
  return 0;
}

void PETSC_STDCALL tssetrhsfunction_(TS *ts,int (*f)(TS*,double*,Vec*,Vec*,void*,int*),void*fP, int *__ierr )
{
  f2 = f;
  *__ierr = TSSetRHSFunction(*ts,ourtsfunction,fP);
}


/* ---------------------------------------------------------*/
static int (*f3)(TS*,double*,Mat*,Mat*,MatStructure*,void*,int*);
static int ourtsmatrix(TS ts,double d,Mat* m,Mat* p,MatStructure* type,void*ctx)
{
  int ierr = 0;
  (*f3)(&ts,&d,m,p,type,ctx,&ierr);CHKERRQ(ierr);
  return 0;
}

void PETSC_STDCALL tssetrhsmatrix_(TS *ts,Mat *A,Mat *B,int (*f)(TS*,double*,Mat*,Mat*,MatStructure*,
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
static void (*f4)(TS*,double*,Vec*,Mat*,Mat*,MatStructure*,void*,int*);
static int ourtsjacobian(TS ts,double d,Vec x,Mat* m,Mat* p,MatStructure* type,void*ctx)
{
  int ierr = 0;
  (*f4)(&ts,&d,&x,m,p,type,ctx,&ierr);CHKERRQ(ierr);
  return 0;
}

void PETSC_STDCALL tssetrhsjacobian_(TS *ts,Mat *A,Mat *B,void (*f)(TS*,double*,Vec*,Mat*,Mat*,MatStructure*,
               void*,int*),void*fP, int *__ierr )
{
  if (FORTRANNULLFUNCTION(f)) {
    *__ierr = TSSetRHSJacobian(*ts,*A,*B,PETSC_NULL,fP);
  } else if ((void*)f == (void*)tsdefaultcomputejacobian_) {
    *__ierr = TSSetRHSJacobian(*ts,*A,*B,TSDefaultComputeJacobian,fP);
  } else if ((void*)f == (void*)tsdefaultcomputejacobiancolor_) {
    *__ierr = TSSetRHSJacobian(*ts,*A,*B,TSDefaultComputeJacobianColor,*(MatFDColoring*)fP);
  } else {
    f4 = f;
    *__ierr = TSSetRHSJacobian(*ts,*A,*B,ourtsjacobian,fP);
  }
}

void PETSC_STDCALL tsgetsolution_(TS *ts,Vec *v, int *__ierr )
{
  *__ierr = TSGetSolution(*ts,v);
}

void PETSC_STDCALL tscreate_(MPI_Comm *comm,TSProblemType *problemtype,TS *outts, int *__ierr )
{
  *__ierr = TSCreate((MPI_Comm)PetscToPointerComm( *comm ),*problemtype,outts);
}

void PETSC_STDCALL tsgetsnes_(TS *ts,SNES *snes, int *__ierr )
{
  *__ierr = TSGetSNES(*ts,snes);
}

void PETSC_STDCALL tsgetsles_(TS *ts,SLES *sles, int *__ierr )
{
  *__ierr = TSGetSLES(*ts,sles);
}

void PETSC_STDCALL tsgettype_(TS *ts,CHAR name,int *__ierr,int len)
{
  char *tname;

  *__ierr = TSGetType(*ts,(TSType *)&tname);
#if defined(PETSC_USES_CPTOFCD)
  {
    char *t = _fcdtocp(name); int len1 = _fcdlen(name);
    *__ierr = PetscStrncpy(t,tname,len1);
  }
#else
  *__ierr = PetscStrncpy(name,tname,len);
#endif
}

#if defined(PETSC_HAVE_PVODE)  && !defined(__cplusplus)
void PETSC_STDCALL tspvodegetiterations_(TS *ts,int *nonlin, int *lin, int *__ierr)
{
  if (FORTRANNULLINTEGER(nonlin)) nonlin = PETSC_NULL;
  if (FORTRANNULLINTEGER(lin))    lin    = PETSC_NULL;
  *__ierr = TSPVodeGetIterations(*ts,nonlin,lin);
}
#endif

void PETSC_STDCALL tsdestroy_(TS *ts, int *__ierr ){
  *__ierr = TSDestroy(*ts);
}

static int (*f7)(TS*,int*,double*,Vec*,void*,int*);
static int ourtsmonitor(TS ts,int i,double d,Vec v,void*ctx)
{
  int              ierr = 0;
  (*f7)(&ts,&i,&d,&v,ctx,&ierr);CHKERRQ(ierr);
  return 0;
}
void PETSC_STDCALL tssetmonitor_(TS *ts,int (*func)(TS*,int*,double*,Vec*,void*,int*),
                    void *mctx, int *__ierr ){
  f7 = func;
  *__ierr = TSSetMonitor(*ts,ourtsmonitor,mctx);
}

void PETSC_STDCALL tsgetoptionsprefix_(TS *ts, CHAR prefix,int *__ierr,int len)
{
  char *tname;

  *__ierr = TSGetOptionsPrefix(*ts,&tname);
#if defined(PETSC_USES_CPTOFCD)
  {
    char *t = _fcdtocp(prefix); int len1 = _fcdlen(prefix);
    *__ierr = PetscStrncpy(t,tname,len1);
  }
#else
  *__ierr = PetscStrncpy(prefix,tname,len);
#endif
}

EXTERN_C_END

