#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: zsnes.c,v 1.22 1999/03/01 17:38:41 bsmith Exp bsmith $";
#endif

#include "src/fortran/custom/zpetsc.h"
#include "snes.h"

#ifdef HAVE_FORTRAN_CAPS
#define snesregisterdestroy_         SNESREGISTERDESTROY
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
#define matcreatesnesfdmf_           MATCREATESNESFDMF
#define snessettype_                 SNESSETTYPE
#define snesgetconvergencehistory_   SNESGETCONVERGENCEHISTORY
#elif !defined(HAVE_FORTRAN_UNDERSCORE)
#define snesregisterdestroy_         snesregisterdestroy
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
#define matcreatesnesfdmf_           matcreatesnesfdmf
#define snessettype_                 snessettype
#define snesgetconvergencehistory_   snesgetconvergencehistory
#endif

EXTERN_C_BEGIN

void snesgetconvergencehistory_(SNES *snes,int *na,int *__ierr)
{
  *__ierr = SNESGetConvergenceHistory(*snes, PETSC_NULL,PETSC_NULL, na);
}

void snessettype_(SNES *snes,CHAR itmethod, int *__ierr,int len )
{
  char *t;

  FIXCHAR(itmethod,len,t);
  *__ierr = SNESSetType(*snes,t);
  FREECHAR(itmethod,t);
}

void snesappendoptionsprefix_(SNES *snes,CHAR prefix, int *__ierr,int len )
{
  char *t;

  FIXCHAR(prefix,len,t);
  *__ierr = SNESAppendOptionsPrefix(*snes,t);
  FREECHAR(prefix,t);
}

void matcreatesnewsfdmf_(SNES *snes,Vec *x,Mat *J, int *__ierr )
{
  *__ierr = MatCreateSNESFDMF(*snes,*x,J);
}

static int (*f7)(SNES*,int*,double*,void*,int*);
static int oursnesmonitor(SNES snes,int i,double d,void*ctx)
{
  int              ierr = 0;

  (*f7)(&snes,&i,&d,ctx,&ierr); CHKERRQ(ierr);
  return 0;
}
void snessetmonitor_(SNES *snes,int (*func)(SNES*,int*,double*,void*,int*),
                    void *mctx, int *__ierr )
{
  f7 = func;
  *__ierr = SNESSetMonitor(*snes,oursnesmonitor,mctx);
}

static int (*f8)(SNES*,double*,double*,double*,void*,int*);
static int oursnestest(SNES snes,double a,double d,double c,void*ctx)
{
  int              ierr = 0;

  (*f8)(&snes,&a,&d,&c,ctx,&ierr); CHKERRQ(ierr);
  return 0;
}

void snessetconvergencetest_(SNES *snes,
       int (*func)(SNES*,double*,double*,double*,void*,int*),
       void *cctx, int *__ierr )
{
  f8 = func;
  *__ierr = SNESSetConvergenceTest(*snes,oursnestest,cctx);
}

void snesgetsolution_(SNES *snes,Vec *x, int *__ierr )
{
  *__ierr = SNESGetSolution(*snes,x);
}

void snesgetsolutionupdate_(SNES *snes,Vec *x, int *__ierr )
{
  *__ierr = SNESGetSolutionUpdate(*snes,x);
}

void snesgetfunction_(SNES *snes,Vec *r, int *__ierr )
{
  *__ierr = SNESGetFunction(*snes,r);
}

void snesgetgradient_(SNES *snes,Vec *r, int *__ierr )
{
  *__ierr = SNESGetGradient(*snes,r);
}

void snesdestroy_(SNES *snes, int *__ierr )
{
  *__ierr = SNESDestroy(*snes);
}

void snesgetsles_(SNES *snes,SLES *sles, int *__ierr )
{
  *__ierr = SNESGetSLES(*snes,sles);
}

static int (*f6)(SNES *,Vec *,Mat *,Mat *,int*,void*,int*);
static int oursneshessianfunction(SNES snes,Vec x,Mat* mat,Mat* pmat,
                                  MatStructure* st,void *ctx)
{
  int              ierr = 0;

  (*f6)(&snes,&x,mat,pmat,(int*)st,ctx,&ierr); CHKERRQ(ierr);

  return 0;
}

void snessethessian_(SNES *snes,Mat *A,Mat *B,int (*func)(SNES*,Vec*,Mat*,Mat*,int*,void*,int*),
                     void *ctx, int *__ierr )
{
  f6 = func;
  *__ierr = SNESSetHessian(*snes,*A,*B,oursneshessianfunction,ctx);
}

static int (*f5)(SNES*,Vec*,Vec *,void*,int*);
static int oursnesgradientfunction(SNES snes,Vec x,Vec d,void *ctx)
{
  int ierr = 0;
  (*f5)(&snes,&x,&d,ctx,&ierr); CHKERRQ(ierr);
  return 0;
}

void snessetgradient_(SNES *snes,Vec *r,int (*func)(SNES*,Vec*,Vec*,void*,int*),void *ctx, int *__ierr ){
  f5 = func;
  *__ierr = SNESSetGradient(*snes,*r,oursnesgradientfunction,ctx);
}

static int (*f4)(SNES*,Vec*,double*,void*,int*);
static int oursnesminfunction(SNES snes,Vec x,double* d,void *ctx)
{
  int ierr = 0;
  (*f4)(&snes,&x,d,ctx,&ierr); CHKERRQ(ierr);
  return 0;
}

void snessetminimizationfunction_(SNES *snes,
          int (*func)(SNES*,Vec*,double*,void*,int*),void *ctx, int *__ierr ){
  f4 = func;
  *__ierr = SNESSetMinimizationFunction(*snes,oursnesminfunction,ctx);
}

static int (*f2)(SNES*,Vec*,Vec*,void*,int*);
static int oursnesfunction(SNES snes,Vec x,Vec f,void *ctx)
{
  int ierr = 0;
  (*f2)(&snes,&x,&f,ctx,&ierr); CHKERRQ(ierr);
  return 0;
}
void snessetfunction_(SNES *snes,Vec *r,int (*func)(SNES*,Vec*,Vec*,void*,int*),
                      void *ctx, int *__ierr ){
   f2 = func;
   *__ierr = SNESSetFunction(*snes,*r,oursnesfunction,ctx);
}
/* ---------------------------------------------------------*/

void snescreate_(MPI_Comm *comm,SNESProblemType *type,SNES *outsnes, int *__ierr ){

*__ierr = SNESCreate((MPI_Comm)PetscToPointerComm( *comm ),*type,outsnes);
}

/* ---------------------------------------------------------*/
/*
     snesdefaultcomputejacobian() and snesdefaultcomputejacobianwithcoloring()
  are special and get mapped directly to their C equivalent.
*/
extern void snesdefaultcomputejacobian_(SNES*,Vec*,Mat*,Mat*,MatStructure*,void*,int*);
extern void snesdefaultcomputejacobianwithcoloring_(SNES*,Vec*,Mat*,Mat*,MatStructure*,void*,int*);

static void (*f3)(SNES*,Vec*,Mat*,Mat*,MatStructure*,void*,int*);
static int oursnesjacobian(SNES snes,Vec x,Mat* m,Mat* p,MatStructure* type,
                          void*ctx)
{
  int              ierr = 0;
  (*f3)(&snes,&x,m,p,type,ctx,&ierr); CHKERRQ(ierr);
  return 0;
}

void snessetjacobian_(SNES *snes,Mat *A,Mat *B,void (*func)(SNES*,Vec*,Mat*,Mat*,
            MatStructure*,void*,int*),void *ctx, int *__ierr )
{
  if (func == snesdefaultcomputejacobian_) {
    ;
  } else if (func == snesdefaultcomputejacobian_) {
    ;
  } else {
    f3 = func;
    *__ierr = SNESSetJacobian(*snes,*A,*B,oursnesjacobian,ctx);
  }
}

/* -------------------------------------------------------------*/

void snesregisterdestroy_(int *__ierr)
{
  *__ierr = SNESRegisterDestroy();
}

void snesgettype_(SNES *snes,CHAR name,int *__ierr,int len)
{
  char *tname;

  *__ierr = SNESGetType(*snes,&tname);
#if defined(USES_CPTOFCD)
  {
    char *t = _fcdtocp(name); int len1 = _fcdlen(name);
    PetscStrncpy(t,tname,len1);
  }
#else
  PetscStrncpy(name,tname,len);
#endif
}

EXTERN_C_END


