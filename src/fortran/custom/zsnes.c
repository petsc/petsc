#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: zsnes.c,v 1.37 1999/10/01 21:23:14 bsmith Exp balay $";
#endif

#include "src/fortran/custom/zpetsc.h"
#include "snes.h"

#ifdef PETSC_HAVE_FORTRAN_CAPS
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
#define snesgetminimizationfunction_ SNESGETMINIMIZATIONFUNCTION
#define snesgetgradient_             SNESGETGRADIENT
#define snesdestroy_                 SNESDESTROY
#define snesgettype_                 SNESGETTYPE
#define snessetoptionsprefix_        SNESSETOPTIONSPREFIX 
#define snesappendoptionsprefix_     SNESAPPENDOPTIONSPREFIX 
#define matcreatesnesmf_             MATCREATESNESMF
#define snessettype_                 SNESSETTYPE
#define snesgetconvergencehistory_   SNESGETCONVERGENCEHISTORY
#define snesdefaultcomputejacobian_  SNESDEFAULTCOMPUTEJACOBIAN
#define snesdefaultcomputejacobiancolor_ SNESDEFAULTCOMPUTEJACOBIANCOLOR
#define matsnesmfsettype_                MATSNESMFSETTYPE
#define snesgetoptionsprefix_            SNESGETOPTIONSPREFIX
#define snesgetjacobian_                 SNESGETJACOBIAN
#define matsnesmfsetfunction_            MATSNESMFSETFUNCTION
#define snessetlinesearchparams_         SNESSETLINESEARCHPARAMS
#define snesgetlinesearchparams_         SNESGETLINESEARCHPARAMS
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define matsnesmfsetfunction_            matsnesmfsetfunction
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
#define snesgetminimizationfunction_ snesgetminimizationfunction
#define snesgetgradient_             snesgetgradient
#define snesgettype_                 snesgettype
#define snessetoptionsprefix_        snessetoptionsprefix 
#define snesappendoptionsprefix_     snesappendoptionsprefix
#define matcreatesnesmf_             matcreatesnesmf
#define snessettype_                 snessettype
#define snesgetconvergencehistory_   snesgetconvergencehistory
#define snesdefaultcomputejacobian_  snesdefaultcomputejacobian
#define snesdefaultcomputejacobiancolor_ snesdefaultcomputejacobiancolor
#define matsnesmfsettype_                matsnesmfsettype
#define snesgetoptionsprefix_            snesgetoptionsprefix
#define snesgetjacobian_                 snesgetjacobian
#define snessetlinesearchparams_         snessetlinesearchparams
#define snesgetlinesearchparams_         snesgetlinesearchparams
#endif

EXTERN_C_BEGIN

void PETSC_STDCALL snessetlinesearchparams_(SNES *snes, double *alpha, double *maxstep, double *steptol,int *__ierr)
{
  *__ierr = SNESSetLineSearchParams(*snes,*alpha,*maxstep,*steptol);
}

void PETSC_STDCALL snesgetlinesearchparams_(SNES *snes, double *alpha, double *maxstep, double *steptol,int *__ierr)
{
  if (FORTRANNULLDOUBLE(alpha)) alpha = PETSC_NULL;
  if (FORTRANNULLDOUBLE(maxstep)) maxstep = PETSC_NULL;
  if (FORTRANNULLDOUBLE(steptol)) steptol = PETSC_NULL;
  *__ierr = SNESGetLineSearchParams(*snes,alpha,maxstep,steptol);
}

void PETSC_STDCALL snesgetjacobian_(SNES *snes,Mat *A,Mat *B,void **ctx, int *__ierr )
{
  if (FORTRANNULLINTEGER(ctx)) ctx = PETSC_NULL;
  if (FORTRANNULLOBJECT(A))    A = PETSC_NULL;
  if (FORTRANNULLOBJECT(B))    B = PETSC_NULL;
  *__ierr = SNESGetJacobian(*snes,A,B,ctx);
}

void PETSC_STDCALL matsnesmfsettype_(Mat *mat,CHAR ftype PETSC_MIXED_LEN(len),
                                     int *__ierr PETSC_END_LEN(len) )
{
  char *t;
  FIXCHAR(ftype,len,t);
  *__ierr = MatSNESMFSetType(*mat,t);
  FREECHAR(ftype,t);
}

void PETSC_STDCALL snesgetconvergencehistory_(SNES *snes,int *na,int *__ierr)
{
  *__ierr = SNESGetConvergenceHistory(*snes, PETSC_NULL,PETSC_NULL, na);
}

void PETSC_STDCALL snessettype_(SNES *snes,CHAR type PETSC_MIXED_LEN(len),
                                int *__ierr PETSC_END_LEN(len) )
{
  char *t;

  FIXCHAR(type,len,t);
  *__ierr = SNESSetType(*snes,t);
  FREECHAR(type,t);
}

void PETSC_STDCALL snesappendoptionsprefix_(SNES *snes,CHAR prefix PETSC_MIXED_LEN(len),
                                            int *__ierr PETSC_END_LEN(len) )
{
  char *t;

  FIXCHAR(prefix,len,t);
  *__ierr = SNESAppendOptionsPrefix(*snes,t);
  FREECHAR(prefix,t);
}

void PETSC_STDCALL matcreatesnewsmf_(SNES *snes,Vec *x,Mat *J, int *__ierr )
{
  *__ierr = MatCreateSNESMF(*snes,*x,J);
}

static void (*f7)(SNES*,int*,double*,void*,int*);
static int oursnesmonitor(SNES snes,int i,double d,void*ctx)
{
  int              ierr = 0;

  (*f7)(&snes,&i,&d,ctx,&ierr);CHKERRQ(ierr);
  return 0;
}
void PETSC_STDCALL snessetmonitor_(SNES *snes,void (*func)(SNES*,int*,double*,void*,int*),
                    void *mctx, int (*mondestroy)(void *,int *),int *__ierr )
{
  f7 = func;
  *__ierr = SNESSetMonitor(*snes,oursnesmonitor,mctx,0);
}

static void (*f8)(SNES*,double*,double*,double*,SNESConvergedReason*,void*,int*);
static int oursnestest(SNES snes,double a,double d,double c,SNESConvergedReason*reason,void*ctx)
{
  int              ierr = 0;

  (*f8)(&snes,&a,&d,&c,reason,ctx,&ierr);CHKERRQ(ierr);
  return 0;
}

void PETSC_STDCALL snessetconvergencetest_(SNES *snes,
       void (*func)(SNES*,double*,double*,double*,SNESConvergedReason*,void*,int*),
       void *cctx, int *__ierr )
{
  f8 = func;
  *__ierr = SNESSetConvergenceTest(*snes,oursnestest,cctx);
}

void PETSC_STDCALL snesgetsolution_(SNES *snes,Vec *x, int *__ierr )
{
  *__ierr = SNESGetSolution(*snes,x);
}

void PETSC_STDCALL snesgetsolutionupdate_(SNES *snes,Vec *x, int *__ierr )
{
  *__ierr = SNESGetSolutionUpdate(*snes,x);
}

void PETSC_STDCALL snesgetfunction_(SNES *snes,Vec *r, void **ctx,int *__ierr )
{
  if (FORTRANNULLINTEGER(ctx)) ctx = PETSC_NULL;
  if (FORTRANNULLINTEGER(r))   r   = PETSC_NULL;
  *__ierr = SNESGetFunction(*snes,r,ctx);
}

void PETSC_STDCALL snesgetminimizationfunction_(SNES *snes,double *r, void **ctx,int *__ierr )
{
  if (FORTRANNULLINTEGER(ctx)) ctx = PETSC_NULL;
  if (FORTRANNULLDOUBLE(r))    r   = PETSC_NULL;
  *__ierr = SNESGetMinimizationFunction(*snes,r,ctx);
}

void PETSC_STDCALL snesgetgradient_(SNES *snes,Vec *r,void **ctx, int *__ierr )
{
  if (FORTRANNULLINTEGER(ctx)) ctx = PETSC_NULL;
  if (FORTRANNULLINTEGER(r))   r   = PETSC_NULL;
  *__ierr = SNESGetGradient(*snes,r,ctx);
}

void PETSC_STDCALL snesdestroy_(SNES *snes, int *__ierr )
{
  *__ierr = SNESDestroy(*snes);
}

void PETSC_STDCALL snesgetsles_(SNES *snes,SLES *sles, int *__ierr )
{
  *__ierr = SNESGetSLES(*snes,sles);
}

static void (*f6)(SNES *,Vec *,Mat *,Mat *,int*,void*,int*);
static int oursneshessianfunction(SNES snes,Vec x,Mat* mat,Mat* pmat,
                                  MatStructure* st,void *ctx)
{
  int              ierr = 0;

  (*f6)(&snes,&x,mat,pmat,(int*)st,ctx,&ierr);CHKERRQ(ierr);

  return 0;
}

void PETSC_STDCALL snessethessian_(SNES *snes,Mat *A,Mat *B,void (*func)(SNES*,Vec*,Mat*,Mat*,int*,void*,int*),
                     void *ctx, int *__ierr )
{
  f6 = func;
  *__ierr = SNESSetHessian(*snes,*A,*B,oursneshessianfunction,ctx);
}

static void (*f5)(SNES*,Vec*,Vec *,void*,int*);
static int oursnesgradientfunction(SNES snes,Vec x,Vec d,void *ctx)
{
  int ierr = 0;
  (*f5)(&snes,&x,&d,ctx,&ierr);CHKERRQ(ierr);
  return 0;
}

void PETSC_STDCALL snessetgradient_(SNES *snes,Vec *r,void (*func)(SNES*,Vec*,Vec*,void*,int*),void *ctx, int *__ierr ){
  f5 = func;
  *__ierr = SNESSetGradient(*snes,*r,oursnesgradientfunction,ctx);
}

static void (*f4)(SNES*,Vec*,double*,void*,int*);
static int oursnesminfunction(SNES snes,Vec x,double* d,void *ctx)
{
  int ierr = 0;
  (*f4)(&snes,&x,d,ctx,&ierr);CHKERRQ(ierr);
  return 0;
}

void PETSC_STDCALL snessetminimizationfunction_(SNES *snes,
          void (*func)(SNES*,Vec*,double*,void*,int*),void *ctx, int *__ierr ){
  f4 = func;
  *__ierr = SNESSetMinimizationFunction(*snes,oursnesminfunction,ctx);
}

/* ---------------------------------------------------------*/

static void (*f2)(SNES*,Vec*,Vec*,void*,int*);
static int oursnesfunction(SNES snes,Vec x,Vec f,void *ctx)
{
  int ierr = 0;
  (*f2)(&snes,&x,&f,ctx,&ierr);CHKERRQ(ierr);
  return 0;
}
void PETSC_STDCALL snessetfunction_(SNES *snes,Vec *r,void (*func)(SNES*,Vec*,Vec*,void*,int*),
                      void *ctx, int *__ierr ){
   f2 = func;
   *__ierr = SNESSetFunction(*snes,*r,oursnesfunction,ctx);
}

/* ---------------------------------------------------------*/

static void (*f11)(SNES*,Vec*,Vec*,void*,int*);
static int ourmatsnesmffunction(SNES snes,Vec x,Vec f,void *ctx)
{
  int ierr = 0;
  (*f11)(&snes,&x,&f,ctx,&ierr);CHKERRQ(ierr);
  return 0;
}
void PETSC_STDCALL matsnesmfsetfunction_(Mat *mat,Vec *r,void (*func)(SNES*,Vec*,Vec*,void*,int*),
                      void *ctx, int *__ierr ){
   f11 = func;
   *__ierr = MatSNESMFSetFunction(*mat,*r,ourmatsnesmffunction,ctx);
}
/* ---------------------------------------------------------*/

void PETSC_STDCALL snescreate_(MPI_Comm *comm,SNESProblemType *type,SNES *outsnes, int *__ierr ){

*__ierr = SNESCreate((MPI_Comm)PetscToPointerComm( *comm ),*type,outsnes);
}

/* ---------------------------------------------------------*/
/*
     snesdefaultcomputejacobian() and snesdefaultcomputejacobiancolor()
  are special and get mapped directly to their C equivalent.
*/
void PETSC_STDCALL snesdefaultcomputejacobian_(SNES *snes,Vec *x,Mat *m,Mat *p,MatStructure* type,
                                 void *ctx,int *__ierr)
{
  *__ierr = SNESDefaultComputeJacobian(*snes,*x,m,p,type,ctx);
}
void PETSC_STDCALL snesdefaultcomputejacobiancolor_(SNES *snes,Vec *x,Mat *m,Mat *p,
                                             MatStructure* type,void *ctx,int *__ierr)
{
  *__ierr = SNESDefaultComputeJacobianColor(*snes,*x,m,p,type,*(MatFDColoring*)ctx);
}

static void (*f3)(SNES*,Vec*,Mat*,Mat*,MatStructure*,void*,int*);
static int oursnesjacobian(SNES snes,Vec x,Mat* m,Mat* p,MatStructure* type,
                          void*ctx)
{
  int              ierr = 0;
  (*f3)(&snes,&x,m,p,type,ctx,&ierr);CHKERRQ(ierr);
  return 0;
}

void PETSC_STDCALL snessetjacobian_(SNES *snes,Mat *A,Mat *B,void (*func)(SNES*,Vec*,Mat*,Mat*,
            MatStructure*,void*,int*),void *ctx, int *__ierr )
{
  if ((void*)func == (void*)snesdefaultcomputejacobian_) {
    *__ierr = SNESSetJacobian(*snes,*A,*B,SNESDefaultComputeJacobian,ctx);
  } else if ((void*)func == (void*)snesdefaultcomputejacobiancolor_) {
    *__ierr = SNESSetJacobian(*snes,*A,*B,SNESDefaultComputeJacobianColor,*(MatFDColoring*)ctx);
  } else {
    f3 = func;
    *__ierr = SNESSetJacobian(*snes,*A,*B,oursnesjacobian,ctx);
  }
}

/* -------------------------------------------------------------*/

void PETSC_STDCALL snesregisterdestroy_(int *__ierr)
{
  *__ierr = SNESRegisterDestroy();
}

void PETSC_STDCALL snesgettype_(SNES *snes,CHAR name PETSC_MIXED_LEN(len),
                                int *__ierr PETSC_END_LEN(len) )
{
  char *tname;

  *__ierr = SNESGetType(*snes,&tname);
#if defined(USES_CPTOFCD)
  {
    char *t = _fcdtocp(name); int len1 = _fcdlen(name);
    *__ierr = PetscStrncpy(t,tname,len1);if (*__ierr) return;
  }
#else
  *__ierr = PetscStrncpy(name,tname,len);if (*__ierr) return;
#endif
}

void PETSC_STDCALL snesgetoptionsprefix_(SNES *snes, CHAR prefix PETSC_MIXED_LEN(len),
                                         int *__ierr PETSC_END_LEN(len) )
{
  char *tname;

  *__ierr = SNESGetOptionsPrefix(*snes,&tname);
#if defined(USES_CPTOFCD)
  {
    char *t = _fcdtocp(prefix); int len1 = _fcdlen(prefix);
    *__ierr = PetscStrncpy(t,tname,len1);if (*__ierr) return;
  }
#else
  *__ierr = PetscStrncpy(prefix,tname,len);if (*__ierr) return;
#endif
}

EXTERN_C_END


