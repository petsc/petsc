#include <petsc-private/fortranimpl.h>
#include <petscsnes.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define matmffdcomputejacobian_          MATMFFDCOMPUTEJACOBIAN
#define snessolve_                       SNESSOLVE
#define snesdefaultcomputejacobian_      SNESDEFAULTCOMPUTEJACOBIAN
#define snesdefaultcomputejacobiancolor_ SNESDEFAULTCOMPUTEJACOBIANCOLOR
#define snessetjacobian_                 SNESSETJACOBIAN
#define snesgetoptionsprefix_            SNESGETOPTIONSPREFIX
#define snesgettype_                     SNESGETTYPE
#define snessetfunction_                 SNESSETFUNCTION
#define snessetgs_                       SNESSETGS
#define snesgetfunction_                 SNESGETFUNCTION
#define snesgetgs_                       SNESGETGS
#define snessetconvergencetest_          SNESSETCONVERGENCETEST
#define snesdefaultconverged_            SNESDEFAULTCONVERGED
#define snesskipconverged_               SNESSKIPCONVERGED
#define snesview_                        SNESVIEW
#define snesgetconvergencehistory_       SNESGETCONVERGENCEHISTORY
#define snesgetjacobian_                 SNESGETJACOBIAN
#define snessettype_                     SNESSETTYPE
#define snesappendoptionsprefix_         SNESAPPENDOPTIONSPREFIX
#define snessetoptionsprefix_            SNESSETOPTIONSPREFIX
#define snesmonitordefault_              SNESMONITORDEFAULT
#define snesmonitorsolution_             SNESMONITORSOLUTION
#define snesmonitorlgresidualnorm_       SNESMONITORLGRESIDUALNORM
#define snesmonitorsolutionupdate_       SNESMONITORSOLUTIONUPDATE
#define snesmonitorset_                  SNESMONITORSET
#define snesgetsneslinesearch_           SNESGETSNESLINESEARCH
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define matmffdcomputejacobian_          matmffdcomputejacobian
#define snessolve_                       snessolve
#define snesdefaultcomputejacobian_      snesdefaultcomputejacobian
#define snesdefaultcomputejacobiancolor_ snesdefaultcomputejacobiancolor
#define snessetjacobian_                 snessetjacobian
#define snesgetoptionsprefix_            snesgetoptionsprefix
#define snesgettype_                     snesgettype
#define snessetfunction_                 snessetfunction
#define snessetgs_                       snessetgs
#define snesgetfunction_                 snesgetfunction
#define snesgetgs_                       snesgetgs
#define snessetconvergencetest_          snessetconvergencetest
#define snesdefaultconverged_            snesdefaultconverged
#define snesskipconverged_               snesskipconverged
#define snesview_                        snesview
#define snesgetjacobian_                 snesgetjacobian
#define snesgetconvergencehistory_       snesgetconvergencehistory
#define snessettype_                     snessettype
#define snesappendoptionsprefix_         snesappendoptionsprefix
#define snessetoptionsprefix_            snessetoptionsprefix
#define snesmonitorlgresidualnorm_       snesmonitorlgresidualnorm
#define snesmonitordefault_              snesmonitordefault
#define snesmonitorsolution_             snesmonitorsolution
#define snesmonitorsolutionupdate_       snesmonitorsolutionupdate
#define snesmonitorset_                  snesmonitorset
#define snesgetsneslinesearch_           snesgetsneslinesearch
#endif

static struct {
  PetscFortranCallbackId function;
  PetscFortranCallbackId test;
  PetscFortranCallbackId destroy;
  PetscFortranCallbackId jacobian;
  PetscFortranCallbackId monitor;
  PetscFortranCallbackId mondestroy;
  PetscFortranCallbackId gs;
} _cb;

static PetscErrorCode oursnesfunction(SNES snes,Vec x,Vec f,void *ctx)
{
  PetscObjectUseFortranCallback(snes,_cb.function,(SNES*,Vec*,Vec*,void*,PetscErrorCode*),(&snes,&x,&f,_ctx,&ierr));
  return 0;
}

static PetscErrorCode oursnestest(SNES snes,PetscInt it,PetscReal a,PetscReal d,PetscReal c,SNESConvergedReason*reason,void*ctx)
{
  PetscObjectUseFortranCallback(snes,_cb.test,(SNES*,PetscInt*,PetscReal*,PetscReal*,PetscReal*,SNESConvergedReason*,void*,PetscErrorCode*),(&snes,&it,&a,&d,&c,reason,_ctx,&ierr));
  return 0;
}

static PetscErrorCode ourdestroy(void*ctx)
{
  PetscObjectUseFortranCallback(ctx,_cb.destroy,(void*,PetscErrorCode*),(_ctx,&ierr));
  return 0;
}

static PetscErrorCode oursnesjacobian(SNES snes,Vec x,Mat* m,Mat* p,MatStructure* type,void*ctx)
{
  PetscObjectUseFortranCallback(snes,_cb.jacobian,(SNES*,Vec*,Mat*,Mat*,MatStructure*,void*,PetscErrorCode*),(&snes,&x,m,p,type,_ctx,&ierr));
  return 0;
}

static PetscErrorCode oursnesgs(SNES snes,Vec x,Vec b,void*ctx)
{
  PetscObjectUseFortranCallback(snes,_cb.gs,(SNES*,Vec*,Vec*,void*,PetscErrorCode*),(&snes,&x,&b,_ctx,&ierr));
  return 0;
}
static PetscErrorCode oursnesmonitor(SNES snes,PetscInt i,PetscReal d,void*ctx)
{
  PetscObjectUseFortranCallback(snes,_cb.monitor,(SNES*,PetscInt*,PetscReal*,void*,PetscErrorCode*),(&snes,&i,&d,_ctx,&ierr));
  return 0;
}
static PetscErrorCode ourmondestroy(void** ctx)
{
  SNES snes = (SNES)*ctx;
  PetscObjectUseFortranCallback(snes,_cb.mondestroy,(void*,PetscErrorCode*),(_ctx,&ierr));
  return 0;
}

EXTERN_C_BEGIN
/* ---------------------------------------------------------*/
/*
     snesdefaultcomputejacobian() and snesdefaultcomputejacobiancolor()
  These can be used directly from Fortran but are mostly so that
  Fortran SNESSetJacobian() will properly handle the defaults being passed in.

  functions, hence no STDCALL
*/
void matmffdcomputejacobian_(SNES *snes,Vec *x,Mat *m,Mat *p,MatStructure* type,void *ctx,PetscErrorCode *ierr)
{
  *ierr = MatMFFDComputeJacobian(*snes,*x,m,p,type,ctx);
}
void snesdefaultcomputejacobian_(SNES *snes,Vec *x,Mat *m,Mat *p,MatStructure* type,void *ctx,PetscErrorCode *ierr)
{
  *ierr = SNESDefaultComputeJacobian(*snes,*x,m,p,type,ctx);
}
void  snesdefaultcomputejacobiancolor_(SNES *snes,Vec *x,Mat *m,Mat *p,MatStructure* type,void *ctx,PetscErrorCode *ierr)
{
  *ierr = SNESDefaultComputeJacobianColor(*snes,*x,m,p,type,*(MatFDColoring*)ctx);
}

void PETSC_STDCALL snessetjacobian_(SNES *snes,Mat *A,Mat *B,void (PETSC_STDCALL *func)(SNES*,Vec*,Mat*,Mat*,
            MatStructure*,void*,PetscErrorCode*),void *ctx,PetscErrorCode *ierr)
{
  CHKFORTRANNULLOBJECT(ctx);
  CHKFORTRANNULLFUNCTION(func);
  if ((PetscVoidFunction)func == (PetscVoidFunction)snesdefaultcomputejacobian_) {
    *ierr = SNESSetJacobian(*snes,*A,*B,SNESDefaultComputeJacobian,ctx);
  } else if ((PetscVoidFunction)func == (PetscVoidFunction)snesdefaultcomputejacobiancolor_) {
    *ierr = SNESSetJacobian(*snes,*A,*B,SNESDefaultComputeJacobianColor,*(MatFDColoring*)ctx);
  } else if ((PetscVoidFunction)func == (PetscVoidFunction)matmffdcomputejacobian_) {
    *ierr = SNESSetJacobian(*snes,*A,*B,MatMFFDComputeJacobian,ctx);
  } else if (!func) {
    *ierr = SNESSetJacobian(*snes,*A,*B,0,ctx);
  } else {
    *ierr = PetscObjectSetFortranCallback((PetscObject)*snes,PETSC_FORTRAN_CALLBACK_CLASS,&_cb.jacobian,(PetscVoidFunction)func,ctx);
    if (!*ierr) *ierr = SNESSetJacobian(*snes,*A,*B,oursnesjacobian,PETSC_NULL);
  }
}
/* -------------------------------------------------------------*/

void PETSC_STDCALL   snessolve_(SNES *snes,Vec *b,Vec *x, int *__ierr )
{
  Vec B = *b,X = *x;
  if (FORTRANNULLOBJECT(b)) B = PETSC_NULL;
  if (FORTRANNULLOBJECT(x)) X = PETSC_NULL;
  *__ierr = SNESSolve(*snes,B,X);
}

void PETSC_STDCALL snesgetoptionsprefix_(SNES *snes,CHAR prefix PETSC_MIXED_LEN(len),PetscErrorCode *ierr PETSC_END_LEN(len))
{
  const char *tname;

  *ierr = SNESGetOptionsPrefix(*snes,&tname);
  *ierr = PetscStrncpy(prefix,tname,len);if (*ierr) return;
}

void PETSC_STDCALL snesgettype_(SNES *snes,CHAR name PETSC_MIXED_LEN(len), PetscErrorCode *ierr PETSC_END_LEN(len))
{
  const char *tname;

  *ierr = SNESGetType(*snes,&tname);
  *ierr = PetscStrncpy(name,tname,len);if (*ierr) return;
  FIXRETURNCHAR(PETSC_TRUE,name,len);
}

/* ---------------------------------------------------------*/

/*
   These are not usually called from Fortran but allow Fortran users
   to transparently set these monitors from .F code

   functions, hence no STDCALL
*/

void PETSC_STDCALL snessetfunction_(SNES *snes,Vec *r,void (PETSC_STDCALL *func)(SNES*,Vec*,Vec*,void*,PetscErrorCode*),void *ctx,PetscErrorCode *ierr)
{
  CHKFORTRANNULLOBJECT(ctx);
  *ierr = PetscObjectSetFortranCallback((PetscObject)*snes,PETSC_FORTRAN_CALLBACK_CLASS,&_cb.function,(PetscVoidFunction)func,ctx);
  if (!*ierr) *ierr = SNESSetFunction(*snes,*r,oursnesfunction,PETSC_NULL);
}


void PETSC_STDCALL snessetgs_(SNES *snes,void (PETSC_STDCALL *func)(SNES*,Vec*,Vec*,void*,PetscErrorCode*),void *ctx,PetscErrorCode *ierr)
{
  CHKFORTRANNULLOBJECT(ctx);
  *ierr = PetscObjectSetFortranCallback((PetscObject)*snes,PETSC_FORTRAN_CALLBACK_CLASS,&_cb.gs,(PetscVoidFunction)func,ctx);
  if (!*ierr) *ierr = SNESSetGS(*snes,oursnesgs,PETSC_NULL);
}
/* ---------------------------------------------------------*/

/* the func argument is ignored */
void PETSC_STDCALL snesgetfunction_(SNES *snes,Vec *r,void *func,void **ctx,PetscErrorCode *ierr)
{
  CHKFORTRANNULLINTEGER(ctx);
  CHKFORTRANNULLOBJECT(r);
  *ierr = SNESGetFunction(*snes,r,PETSC_NULL,PETSC_NULL); if (*ierr) return;
  *ierr = PetscObjectGetFortranCallback((PetscObject)*snes,PETSC_FORTRAN_CALLBACK_CLASS,_cb.function,PETSC_NULL,ctx);
}

void PETSC_STDCALL snesgetgs_(SNES *snes,void *func,void **ctx,PetscErrorCode *ierr)
{
  CHKFORTRANNULLINTEGER(ctx);
  *ierr = PetscObjectGetFortranCallback((PetscObject)*snes,PETSC_FORTRAN_CALLBACK_CLASS,_cb.gs,PETSC_NULL,ctx);
}

/*----------------------------------------------------------------------*/

void snesdefaultconverged_(SNES *snes,PetscInt *it,PetscReal *a,PetscReal *b,PetscReal *c,SNESConvergedReason *r, void *ct,PetscErrorCode *ierr)
{
  *ierr = SNESDefaultConverged(*snes,*it,*a,*b,*c,r,ct);
}

void snesskipconverged_(SNES *snes,PetscInt *it,PetscReal *a,PetscReal *b,PetscReal *c,SNESConvergedReason *r,
                                       void *ct,PetscErrorCode *ierr)
{
  *ierr = SNESSkipConverged(*snes,*it,*a,*b,*c,r,ct);
}

void PETSC_STDCALL snessetconvergencetest_(SNES *snes,void (PETSC_STDCALL *func)(SNES*,PetscInt*,PetscReal*,PetscReal*,PetscReal*,SNESConvergedReason*,void*,PetscErrorCode*), void *cctx,void (PETSC_STDCALL *destroy)(void*),PetscErrorCode *ierr)
{
  CHKFORTRANNULLOBJECT(cctx);
  CHKFORTRANNULLFUNCTION(destroy);

  if ((PetscVoidFunction)func == (PetscVoidFunction)snesdefaultconverged_) {
    *ierr = SNESSetConvergenceTest(*snes,SNESDefaultConverged,0,0);
  } else if ((PetscVoidFunction)func == (PetscVoidFunction)snesskipconverged_) {
    *ierr = SNESSetConvergenceTest(*snes,SNESSkipConverged,0,0);
  } else {
    *ierr = PetscObjectSetFortranCallback((PetscObject)*snes,PETSC_FORTRAN_CALLBACK_CLASS,&_cb.test,(PetscVoidFunction)func,cctx);
    if (*ierr) return;
    if (!destroy) {
      *ierr = SNESSetConvergenceTest(*snes,oursnestest,*snes,PETSC_NULL);
    } else {
      *ierr = PetscObjectSetFortranCallback((PetscObject)*snes,PETSC_FORTRAN_CALLBACK_CLASS,&_cb.destroy,(PetscVoidFunction)destroy,cctx);
      if (!*ierr) *ierr = SNESSetConvergenceTest(*snes,oursnestest,*snes,ourdestroy);
    }
  }
}
/*----------------------------------------------------------------------*/

void PETSC_STDCALL snesview_(SNES *snes,PetscViewer *viewer, PetscErrorCode *ierr)
{
  PetscViewer v;
  PetscPatchDefaultViewers_Fortran(viewer,v);
  *ierr = SNESView(*snes,v);
}

/*  func is currently ignored from Fortran */
void PETSC_STDCALL snesgetjacobian_(SNES *snes,Mat *A,Mat *B,int *func,void **ctx,PetscErrorCode *ierr)
{
  CHKFORTRANNULLINTEGER(ctx);
  CHKFORTRANNULLOBJECT(A);
  CHKFORTRANNULLOBJECT(B);
  *ierr = SNESGetJacobian(*snes,A,B,0,PETSC_NULL); if (*ierr) return;
  *ierr = PetscObjectGetFortranCallback((PetscObject)*snes,PETSC_FORTRAN_CALLBACK_CLASS,_cb.jacobian,PETSC_NULL,ctx);

}

void PETSC_STDCALL snesgetconvergencehistory_(SNES *snes,PetscInt *na,PetscErrorCode *ierr)
{
  *ierr = SNESGetConvergenceHistory(*snes,PETSC_NULL,PETSC_NULL,na);
}

void PETSC_STDCALL snessettype_(SNES *snes,CHAR type PETSC_MIXED_LEN(len),PetscErrorCode *ierr PETSC_END_LEN(len))
{
  char *t;

  FIXCHAR(type,len,t);
  *ierr = SNESSetType(*snes,t);
  FREECHAR(type,t);
}

void PETSC_STDCALL snesappendoptionsprefix_(SNES *snes,CHAR prefix PETSC_MIXED_LEN(len),PetscErrorCode *ierr PETSC_END_LEN(len))
{
  char *t;

  FIXCHAR(prefix,len,t);
  *ierr = SNESAppendOptionsPrefix(*snes,t);
  FREECHAR(prefix,t);
}

void PETSC_STDCALL snessetoptionsprefix_(SNES *snes,CHAR prefix PETSC_MIXED_LEN(len),PetscErrorCode *ierr PETSC_END_LEN(len))
{
  char *t;

  FIXCHAR(prefix,len,t);
  *ierr = SNESSetOptionsPrefix(*snes,t);
  FREECHAR(prefix,t);
}

/*----------------------------------------------------------------------*/
/* functions, hence no STDCALL */

void snesmonitorlgresidualnorm_(SNES *snes,PetscInt *its,PetscReal *fgnorm,void *dummy,PetscErrorCode *ierr)
{
  *ierr = SNESMonitorLGResidualNorm(*snes,*its,*fgnorm,dummy);
}

void snesmonitordefault_(SNES *snes,PetscInt *its,PetscReal *fgnorm,void *dummy,PetscErrorCode *ierr)
{
  *ierr = SNESMonitorDefault(*snes,*its,*fgnorm,dummy);
}

void snesmonitorsolution_(SNES *snes,PetscInt *its,PetscReal *fgnorm,void *dummy,PetscErrorCode *ierr)
{
  *ierr = SNESMonitorSolution(*snes,*its,*fgnorm,dummy);
}

void snesmonitorsolutionupdate_(SNES *snes,PetscInt *its,PetscReal *fgnorm,void *dummy,PetscErrorCode *ierr)
{
  *ierr = SNESMonitorSolutionUpdate(*snes,*its,*fgnorm,dummy);
}


void PETSC_STDCALL snesmonitorset_(SNES *snes,void (PETSC_STDCALL *func)(SNES*,PetscInt*,PetscReal*,void*,PetscErrorCode*),void *mctx,void (PETSC_STDCALL *mondestroy)(void*,PetscErrorCode*),PetscErrorCode *ierr)
{
  CHKFORTRANNULLOBJECT(mctx);
  if ((PetscVoidFunction)func == (PetscVoidFunction)snesmonitordefault_) {
    *ierr = SNESMonitorSet(*snes,SNESMonitorDefault,0,0);
  } else if ((PetscVoidFunction)func == (PetscVoidFunction)snesmonitorsolution_) {
    *ierr = SNESMonitorSet(*snes,SNESMonitorSolution,0,0);
  } else if ((PetscVoidFunction)func == (PetscVoidFunction)snesmonitorsolutionupdate_) {
    *ierr = SNESMonitorSet(*snes,SNESMonitorSolutionUpdate,0,0);
  } else if ((PetscVoidFunction)func == (PetscVoidFunction)snesmonitorlgresidualnorm_) {
    *ierr = SNESMonitorSet(*snes,SNESMonitorLGResidualNorm,0,0);
  } else {
    *ierr = PetscObjectSetFortranCallback((PetscObject)*snes,PETSC_FORTRAN_CALLBACK_CLASS,&_cb.monitor,(PetscVoidFunction)func,mctx);
    if (*ierr) return;
    if (FORTRANNULLFUNCTION(mondestroy)) {
      *ierr = SNESMonitorSet(*snes,oursnesmonitor,*snes,PETSC_NULL);
    } else {
      CHKFORTRANNULLFUNCTION(mondestroy);
      *ierr = PetscObjectSetFortranCallback((PetscObject)*snes,PETSC_FORTRAN_CALLBACK_CLASS,&_cb.mondestroy,(PetscVoidFunction)mondestroy,mctx);
      if (!*ierr) *ierr = SNESMonitorSet(*snes,oursnesmonitor,*snes,ourmondestroy);
    }
  }
}

void PETSC_STDCALL  snesgetsneslinesearch_(SNES *snes,SNESLineSearch *linesearch, int *__ierr) {
*__ierr = SNESGetSNESLineSearch(*snes, linesearch);
}

EXTERN_C_END
