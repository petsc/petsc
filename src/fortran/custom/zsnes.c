
#include "src/fortran/custom/zpetsc.h"
#include "petscsnes.h"
#include "petscda.h"

#ifdef PETSC_HAVE_FORTRAN_UNDERSCORE_UNDERSCORE
#define snesconverged_tr_                snesconverged_tr__
#define snesconverged_ls_                snesconverged_ls__
#endif

#ifdef PETSC_HAVE_FORTRAN_CAPS
#define dmmgsetsnes_                     DMMGSETSNES
#define matcreatedaad_                   MATCREATEDAAD
#define matregisterdaad_                 MATREGISTERDAAD
#define matdaadsetsnes_                  MATDAADSETSNES
#define snesdacomputejacobian_           SNESDACOMPUTEJACOBIAN
#define snesdacomputejacobianwithadifor_ SNESDACOMPUTEJACOBIANWITHADIFOR
#define snesdaformfunction_              SNESDAFORMFUNCTION          
#define snesconverged_tr_                SNESCONVERGED_TR
#define snesconverged_ls_                SNESCONVERGED_LS
#define snesgetconvergedreason_          SNESGETCONVERGEDREASON
#define snesdefaultmonitor_              SNESDEFAULTMONITOR
#define snesvecviewmonitor_              SNESVECVIEWMONITOR
#define sneslgmonitor_                   SNESLGMONITOR
#define snesvecviewupdatemonitor_        SNESVECVIEWUPDATEMONITOR
#define snesregisterdestroy_             SNESREGISTERDESTROY
#define snessetjacobian_                 SNESSETJACOBIAN
#define snescreate_                      SNESCREATE
#define snessetfunction_                 SNESSETFUNCTION
#define snesgetksp_                      SNESGETKSP
#define snessetmonitor_                  SNESSETMONITOR
#define snessetconvergencetest_          SNESSETCONVERGENCETEST
#define snesregisterdestroy_             SNESREGISTERDESTROY
#define snesgetsolution_                 SNESGETSOLUTION
#define snesgetsolutionupdate_           SNESGETSOLUTIONUPDATE
#define snesgetfunction_                 SNESGETFUNCTION
#define snesdestroy_                     SNESDESTROY
#define snesgettype_                     SNESGETTYPE
#define snessetoptionsprefix_            SNESSETOPTIONSPREFIX 
#define snesappendoptionsprefix_         SNESAPPENDOPTIONSPREFIX 
#define matcreatesnesmf_                 MATCREATESNESMF
#define matcreatemf_                     MATCREATEMF
#define snessettype_                     SNESSETTYPE
#define snesgetconvergencehistory_       SNESGETCONVERGENCEHISTORY
#define snesdefaultcomputejacobian_      SNESDEFAULTCOMPUTEJACOBIAN
#define snesdefaultcomputejacobiancolor_ SNESDEFAULTCOMPUTEJACOBIANCOLOR
#define matsnesmfsettype_                MATSNESMFSETTYPE
#define snesgetoptionsprefix_            SNESGETOPTIONSPREFIX
#define snesgetjacobian_                 SNESGETJACOBIAN
#define matsnesmfsetfunction_            MATSNESMFSETFUNCTION
#define sneslinesearchsetparams_         SNESLINESEARCHSETPARAMS
#define sneslinesearchgetparams_         SNESLINESEARCHGETPARAMS
#define sneslinesearchset_               SNESLINESEARCHSET
#define sneslinesearchsetpostcheck_      SNESLINESEARCHSETPOSTCHECK
#define sneslinesearchcubic_             SNESLINESEARCHCUBIC
#define sneslinesearchquadratic_         SNESLINESEARCHQUADRATIC
#define sneslinesearchno_                SNESLINESEARCHNO
#define sneslinesearchnonorms_           SNESLINESEARCHNONORMS
#define snesview_                        SNESVIEW
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define dmmgsetsnes_                     dmmgsetsnes
#define matcreatedaad_                   matcreatedaad
#define matregisterdaad_                 matregisterdaad
#define matdaadsetsnes_                  matdaadsetsnes
#define snesdacomputejacobian_           snesdacomputejacobian
#define snesdacomputejacobianwithadifor_ snesdacomputejacobianwithadifor
#define snesdaformfunction_              snesdaformfunction
#define sneslinesearchcubic_             sneslinesearchcubic     
#define sneslinesearchquadratic_         sneslinesearchquadratic    
#define sneslinesearchno_                sneslinesearchno    
#define sneslinesearchnonorms_           sneslinesearchnonorms    
#define sneslinesearchsetparams_         sneslinesearchsetparams
#define sneslinesearchgetparams_         sneslinesearchgetparams
#define sneslinesearchset_               sneslinesearchset
#define sneslinesearchsetpostcheck_      sneslinesearchsetpostcheck
#define snesconverged_tr_                snesconverged_tr
#define snesconverged_ls_                snesconverged_ls
#define snesgetconvergedreason_          snesgetconvergedreason
#define sneslgmonitor_                   sneslgmonitor
#define snesdefaultmonitor_              snesdefaultmonitor
#define snesvecviewmonitor_              snesvecviewmonitor
#define snesvecviewupdatemonitor_        snesvecviewupdatemonitor
#define matsnesmfsetfunction_            matsnesmfsetfunction
#define snesregisterdestroy_             snesregisterdestroy
#define snessetjacobian_                 snessetjacobian
#define snescreate_                      snescreate
#define snessetfunction_                 snessetfunction
#define snesgetksp_                      snesgetksp
#define snesdestroy_                     snesdestroy
#define snessetmonitor_                  snessetmonitor
#define snessetconvergencetest_          snessetconvergencetest
#define snesregisterdestroy_             snesregisterdestroy
#define snesgetsolution_                 snesgetsolution
#define snesgetsolutionupdate_           snesgetsolutionupdate
#define snesgetfunction_                 snesgetfunction
#define snesgettype_                     snesgettype
#define snessetoptionsprefix_            snessetoptionsprefix 
#define snesappendoptionsprefix_         snesappendoptionsprefix
#define matcreatesnesmf_                 matcreatesnesmf
#define matcreatemf_                     matcreatemf
#define snessettype_                     snessettype
#define snesgetconvergencehistory_       snesgetconvergencehistory
#define snesdefaultcomputejacobian_      snesdefaultcomputejacobian
#define snesdefaultcomputejacobiancolor_ snesdefaultcomputejacobiancolor
#define matsnesmfsettype_                matsnesmfsettype
#define snesgetoptionsprefix_            snesgetoptionsprefix
#define snesgetjacobian_                 snesgetjacobian
#define snesview_                        snesview
#endif

EXTERN_C_BEGIN
static void (PETSC_STDCALL *f7)(SNES*,PetscInt*,PetscReal*,void*,PetscErrorCode*);
static void (PETSC_STDCALL *f71)(void*,PetscErrorCode*);
static void (PETSC_STDCALL *f8)(SNES*,PetscReal*,PetscReal*,PetscReal*,SNESConvergedReason*,void*,PetscErrorCode*);
static void (PETSC_STDCALL *f2)(SNES*,Vec*,Vec*,void*,PetscErrorCode*);
static void (PETSC_STDCALL *f11)(SNES*,Vec*,Vec*,void*,PetscErrorCode*);
static void (PETSC_STDCALL *f3)(SNES*,Vec*,Mat*,Mat*,MatStructure*,void*,PetscErrorCode*);
static void (PETSC_STDCALL *f73)(SNES*,void *,Vec*,Vec*,Vec*,Vec*,Vec*,PetscReal*,PetscReal*,PetscReal*,PetscTruth*,PetscErrorCode*);
static void (PETSC_STDCALL *f74)(SNES*,Vec*,Vec*,Vec*,void*,PetscTruth*,PetscTruth*,PetscErrorCode*);
EXTERN_C_END

/* These are not extern C because they are passed into non-extern C user level functions */
PetscErrorCode OurSNESLineSearch(SNES snes,void *ctx,Vec x,Vec f,Vec g,Vec y,Vec w,PetscReal fnorm,PetscReal*ynorm,PetscReal*gnorm,PetscTruth *flag)
{
  PetscErrorCode ierr = 0;
  (*f73)(&snes,(void*)&ctx,&x,&f,&g,&y,&w,&fnorm,ynorm,gnorm,flag,&ierr);CHKERRQ(ierr);
  return 0;
}

PetscErrorCode OurSNESLineSearchPostCheck(SNES snes,Vec x,Vec y,Vec z,void *checkCtx,PetscTruth *flag1,PetscTruth *flag2)
{
  PetscErrorCode ierr = 0;
  (*f74)(&snes,&x,&y,&z,(void*)&checkCtx,flag1,flag2,&ierr);CHKERRQ(ierr);
  return 0;
}

static PetscErrorCode oursnesmonitor(SNES snes,PetscInt i,PetscReal d,void*ctx)
{
  PetscErrorCode ierr = 0;

  (*f7)(&snes,&i,&d,ctx,&ierr);CHKERRQ(ierr);
  return 0;
}
static PetscErrorCode ourmondestroy(void* ctx)
{
  PetscErrorCode ierr = 0;

  (*f71)(ctx,&ierr);CHKERRQ(ierr);
  return 0;
}
static PetscErrorCode oursnestest(SNES snes,PetscReal a,PetscReal d,PetscReal c,SNESConvergedReason*reason,void*ctx)
{
  PetscErrorCode ierr = 0;

  (*f8)(&snes,&a,&d,&c,reason,ctx,&ierr);CHKERRQ(ierr);
  return 0;
}
static PetscErrorCode oursnesfunction(SNES snes,Vec x,Vec f,void *ctx)
{
  PetscErrorCode ierr = 0;
  (*f2)(&snes,&x,&f,ctx,&ierr);CHKERRQ(ierr);
  return 0;
}
static PetscErrorCode ourmatsnesmffunction(SNES snes,Vec x,Vec f,void *ctx)
{
  PetscErrorCode ierr = 0;
  (*f11)(&snes,&x,&f,ctx,&ierr);CHKERRQ(ierr);
  return 0;
}
static PetscErrorCode oursnesjacobian(SNES snes,Vec x,Mat* m,Mat* p,MatStructure* type,void*ctx)
{
  PetscErrorCode ierr = 0;
  (*f3)(&snes,&x,m,p,type,ctx,&ierr);CHKERRQ(ierr);
  return 0;
}

EXTERN_C_BEGIN

#if defined(notused)
static PetscErrorCode ourrhs(SNES snes,Vec vec,Vec vec2,void*ctx)
{
  PetscErrorCode ierr = 0;
  DMMG *dmmg = (DMMG*)ctx;
  (*(PetscErrorCode (PETSC_STDCALL *)(SNES*,Vec*,Vec*,PetscErrorCode*))(((PetscObject)dmmg->dm)->fortran_func_pointers[0]))(&snes,&vec,&vec2,&ierr);
  return ierr;
}

static PetscErrorCode ourmat(DMMG dmmg,Mat mat)
{
  PetscErrorCode ierr = 0;
  (*(PetscErrorCode (PETSC_STDCALL *)(DMMG*,Vec*,PetscErrorCode*))(((PetscObject)dmmg->dm)->fortran_func_pointers[1]))(&dmmg,&vec,&ierr);
  return ierr;
}

void PETSC_STDCALL dmmgsetsnes_(DMMG **dmmg,PetscErrorCode (PETSC_STDCALL *rhs)(SNES*,Vec*,Vec*,PetscErrorCode*),PetscErrorCode (PETSC_STDCALL *mat)(DMMG*,Mat*,PetscErrorCode*),PetscErrorCode *ierr)
{
  PetscInt i;
  theirmat = mat;
  *ierr = DMMGSetSNES(*dmmg,ourrhs,ourmat,*dmmg);
  /*
    Save the fortran rhs function in the DM on each level; ourrhs() pulls it out when needed
  */
  for (i=0; i<(**dmmg)->nlevels; i++) {
    ((PetscObject)(*dmmg)[i]->dm)->fortran_func_pointers[0] = (FCNVOID)rhs;
    ((PetscObject)(*dmmg)[i]->dm)->fortran_func_pointers[1] = (FCNVOID)mat;
  }
}

#endif

#if defined (PETSC_HAVE_ADIC)
void PETSC_STDCALL matregisterdaad_(PetscErrorCode *ierr)
{
  *ierr = MatRegisterDAAD();
}

void PETSC_STDCALL matcreatedaad_(DA *da,Mat *mat,PetscErrorCode *ierr)
{
  *ierr = MatCreateDAAD(*da,mat);
}

void PETSC_STDCALL matdaadsetsnes_(Mat *mat,SNES *snes,PetscErrorCode *ierr)
{
  *ierr = MatDAADSetSNES(*mat,*snes);
}
#endif

void PETSC_STDCALL snesview_(SNES *snes,PetscViewer *viewer, PetscErrorCode *ierr)
{
  PetscViewer v;
  PetscPatchDefaultViewers_Fortran(viewer,v);
  *ierr = SNESView(*snes,v);
}

void PETSC_STDCALL snesgetconvergedreason_(SNES *snes,SNESConvergedReason *r,PetscErrorCode *ierr)
{
  *ierr = SNESGetConvergedReason(*snes,r);
}

void PETSC_STDCALL sneslinesearchsetparams_(SNES *snes,PetscReal *alpha,PetscReal *maxstep,PetscReal *steptol,PetscErrorCode *ierr)
{
  *ierr = SNESLineSearchSetParams(*snes,*alpha,*maxstep,*steptol);
}

void PETSC_STDCALL sneslinesearchgetparams_(SNES *snes,PetscReal *alpha,PetscReal *maxstep,PetscReal *steptol,PetscErrorCode *ierr)
{
  CHKFORTRANNULLREAL(alpha);
  CHKFORTRANNULLREAL(maxstep);
  CHKFORTRANNULLREAL(steptol);
  *ierr = SNESLineSearchGetParams(*snes,alpha,maxstep,steptol);
}

/*  func is currently ignored from Fortran */
void PETSC_STDCALL snesgetjacobian_(SNES *snes,Mat *A,Mat *B,void **ctx,int *func,PetscErrorCode *ierr)
{
  CHKFORTRANNULLINTEGER(ctx);
  CHKFORTRANNULLOBJECT(A);
  CHKFORTRANNULLOBJECT(B);
  *ierr = SNESGetJacobian(*snes,A,B,ctx,0);
}

void PETSC_STDCALL matsnesmfsettype_(Mat *mat,CHAR ftype PETSC_MIXED_LEN(len),
                                     PetscErrorCode *ierr PETSC_END_LEN(len))
{
  char *t;
  FIXCHAR(ftype,len,t);
  *ierr = MatSNESMFSetType(*mat,t);
  FREECHAR(ftype,t);
}

void PETSC_STDCALL snesgetconvergencehistory_(SNES *snes,PetscInt *na,PetscErrorCode *ierr)
{
  *ierr = SNESGetConvergenceHistory(*snes,PETSC_NULL,PETSC_NULL,na);
}

void PETSC_STDCALL snessettype_(SNES *snes,CHAR type PETSC_MIXED_LEN(len),
                                PetscErrorCode *ierr PETSC_END_LEN(len))
{
  char *t;

  FIXCHAR(type,len,t);
  *ierr = SNESSetType(*snes,t);
  FREECHAR(type,t);
}

void PETSC_STDCALL snesappendoptionsprefix_(SNES *snes,CHAR prefix PETSC_MIXED_LEN(len),
                                            PetscErrorCode *ierr PETSC_END_LEN(len))
{
  char *t;

  FIXCHAR(prefix,len,t);
  *ierr = SNESAppendOptionsPrefix(*snes,t);
  FREECHAR(prefix,t);
}

void PETSC_STDCALL matcreatesnesmf_(SNES *snes,Vec *x,Mat *J,PetscErrorCode *ierr)
{
  *ierr = MatCreateSNESMF(*snes,*x,J);
}

void PETSC_STDCALL matcreatemf_(Vec *x,Mat *J,PetscErrorCode *ierr)
{
  *ierr = MatCreateMF(*x,J);
}

/* functions, hence no STDCALL */

void sneslgmonitor_(SNES *snes,PetscInt *its,PetscReal *fgnorm,void *dummy,PetscErrorCode *ierr)
{
  *ierr = SNESLGMonitor(*snes,*its,*fgnorm,dummy);
}

void snesdefaultmonitor_(SNES *snes,PetscInt *its,PetscReal *fgnorm,void *dummy,PetscErrorCode *ierr)
{
  *ierr = SNESDefaultMonitor(*snes,*its,*fgnorm,dummy);
}

void snesvecviewmonitor_(SNES *snes,PetscInt *its,PetscReal *fgnorm,void *dummy,PetscErrorCode *ierr)
{
  *ierr = SNESVecViewMonitor(*snes,*its,*fgnorm,dummy);
}

void snesvecviewupdatemonitor_(SNES *snes,PetscInt *its,PetscReal *fgnorm,void *dummy,PetscErrorCode *ierr)
{
  *ierr = SNESVecViewUpdateMonitor(*snes,*its,*fgnorm,dummy);
}


void PETSC_STDCALL snessetmonitor_(SNES *snes,void (PETSC_STDCALL *func)(SNES*,PetscInt*,PetscReal*,void*,PetscErrorCode*),
                    void *mctx,void (PETSC_STDCALL *mondestroy)(void*,PetscErrorCode*),PetscErrorCode *ierr)
{
  CHKFORTRANNULLOBJECT(mctx);
  if ((FCNVOID)func == (FCNVOID)snesdefaultmonitor_) {
    *ierr = SNESSetMonitor(*snes,SNESDefaultMonitor,0,0);
  } else if ((FCNVOID)func == (FCNVOID)snesvecviewmonitor_) {
    *ierr = SNESSetMonitor(*snes,SNESVecViewMonitor,0,0);
  } else if ((FCNVOID)func == (FCNVOID)snesvecviewupdatemonitor_) {
    *ierr = SNESSetMonitor(*snes,SNESVecViewUpdateMonitor,0,0);
  } else if ((FCNVOID)func == (FCNVOID)sneslgmonitor_) {
    *ierr = SNESSetMonitor(*snes,SNESLGMonitor,0,0);
  } else {
    f7 = func;
    if (FORTRANNULLFUNCTION(mondestroy)){
      *ierr = SNESSetMonitor(*snes,oursnesmonitor,mctx,0);
    } else {
      f71 = mondestroy;
      *ierr = SNESSetMonitor(*snes,oursnesmonitor,mctx,ourmondestroy);
    }
  }
}

/* -----------------------------------------------------------------------------------------------------*/
void sneslinesearchcubic_(SNES *snes,void *lsctx,Vec *x,Vec *f,Vec *g,Vec *y,Vec *w,PetscReal*fnorm,
                                        PetscReal *ynorm,PetscReal *gnorm,PetscTruth *flag,PetscErrorCode *ierr)
{
  *ierr = SNESLineSearchCubic(*snes,lsctx,*x,*f,*g,*y,*w,*fnorm,ynorm,gnorm,flag);
}
void sneslinesearchquadratic_(SNES *snes,void *lsctx,Vec *x,Vec *f,Vec *g,Vec *y,Vec *w,PetscReal*fnorm,
                                        PetscReal *ynorm,PetscReal *gnorm,PetscTruth *flag,PetscErrorCode *ierr)
{
  *ierr = SNESLineSearchQuadratic(*snes,lsctx,*x,*f,*g,*y,*w,*fnorm,ynorm,gnorm,flag);
}
void sneslinesearchno_(SNES *snes,void *lsctx,Vec *x,Vec *f,Vec *g,Vec *y,Vec *w,PetscReal*fnorm,
                                        PetscReal *ynorm,PetscReal *gnorm,PetscTruth *flag,PetscErrorCode *ierr)
{
  *ierr = SNESLineSearchNo(*snes,lsctx,*x,*f,*g,*y,*w,*fnorm,ynorm,gnorm,flag);
}
void sneslinesearchnonorms_(SNES *snes,void *lsctx,Vec *x,Vec *f,Vec *g,Vec *y,Vec *w,PetscReal*fnorm,
                                        PetscReal *ynorm,PetscReal *gnorm,PetscTruth *flag,PetscErrorCode *ierr)
{
  *ierr = SNESLineSearchNoNorms(*snes,lsctx,*x,*f,*g,*y,*w,*fnorm,ynorm,gnorm,flag);
}


void PETSC_STDCALL sneslinesearchset_(SNES *snes,void (PETSC_STDCALL *f)(SNES*,void *,Vec*,Vec*,Vec*,Vec*,Vec*,PetscReal*,PetscReal*,PetscReal*,PetscTruth*,PetscErrorCode*),void *ctx,PetscErrorCode *ierr)
{
  if ((FCNVOID)f == (FCNVOID)sneslinesearchcubic_) {
    *ierr = SNESLineSearchSet(*snes,SNESLineSearchCubic,ctx);
  } else if ((FCNVOID)f == (FCNVOID)sneslinesearchquadratic_) {
    *ierr = SNESLineSearchSet(*snes,SNESLineSearchQuadratic,ctx);
  } else if ((FCNVOID)f == (FCNVOID)sneslinesearchno_) {
    *ierr = SNESLineSearchSet(*snes,SNESLineSearchNo,ctx);
  } else if ((FCNVOID)f == (FCNVOID)sneslinesearchnonorms_) {
    *ierr = SNESLineSearchSet(*snes,SNESLineSearchNoNorms,ctx);
  } else {
    f73 = f;
    *ierr = SNESLineSearchSet(*snes,OurSNESLineSearch,ctx);
  }
}


void PETSC_STDCALL sneslinesearchsetpostcheck_(SNES *snes,void (PETSC_STDCALL *f)(SNES*,Vec*,Vec *,Vec *,void *,PetscTruth*,PetscTruth*,PetscErrorCode*),void *ctx,PetscErrorCode *ierr)
{
  f74 = f;
  *ierr = SNESLineSearchSetPostCheck(*snes,OurSNESLineSearchPostCheck,ctx);
}  

/*----------------------------------------------------------------------*/

void snesconverged_tr_(SNES *snes,PetscReal *a,PetscReal *b,PetscReal *c,SNESConvergedReason *r,
                                       void *ct,PetscErrorCode *ierr)
{
  *ierr = SNESConverged_TR(*snes,*a,*b,*c,r,ct);
}

void snesconverged_ls_(SNES *snes,PetscReal *a,PetscReal *b,PetscReal *c,SNESConvergedReason *r,
                                       void *ct,PetscErrorCode *ierr)
{
  *ierr = SNESConverged_LS(*snes,*a,*b,*c,r,ct);
}


void PETSC_STDCALL snessetconvergencetest_(SNES *snes,
       void (PETSC_STDCALL *func)(SNES*,PetscReal*,PetscReal*,PetscReal*,SNESConvergedReason*,void*,PetscErrorCode*),
       void *cctx,PetscErrorCode *ierr)
{
  CHKFORTRANNULLOBJECT(cctx);
  if ((FCNVOID)func == (FCNVOID)snesconverged_ls_){
    *ierr = SNESSetConvergenceTest(*snes,SNESConverged_LS,0);
  } else if ((FCNVOID)func == (FCNVOID)snesconverged_tr_){
    *ierr = SNESSetConvergenceTest(*snes,SNESConverged_TR,0);
  } else {
    f8 = func;
    *ierr = SNESSetConvergenceTest(*snes,oursnestest,cctx);
  }
}

/*--------------------------------------------------------------------------------------------*/

void PETSC_STDCALL snesgetsolution_(SNES *snes,Vec *x,PetscErrorCode *ierr)
{
  *ierr = SNESGetSolution(*snes,x);
}

void PETSC_STDCALL snesgetsolutionupdate_(SNES *snes,Vec *x,PetscErrorCode *ierr)
{
  *ierr = SNESGetSolutionUpdate(*snes,x);
}

/* the func argument is ignored */
void PETSC_STDCALL snesgetfunction_(SNES *snes,Vec *r,void **ctx,void *func,PetscErrorCode *ierr)
{
  CHKFORTRANNULLINTEGER(ctx);
  CHKFORTRANNULLOBJECT(r);
  *ierr = SNESGetFunction(*snes,r,ctx,PETSC_NULL);
}

void PETSC_STDCALL snesdestroy_(SNES *snes,PetscErrorCode *ierr)
{
  *ierr = SNESDestroy(*snes);
}

void PETSC_STDCALL snesgetksp_(SNES *snes,KSP *ksp,PetscErrorCode *ierr)
{
  *ierr = SNESGetKSP(*snes,ksp);
}

/* ---------------------------------------------------------*/


/*
        These are not usually called from Fortran but allow Fortran users 
   to transparently set these monitors from .F code
   
   functions, hence no STDCALL
*/
void  snesdaformfunction_(SNES *snes,Vec *X, Vec *F,void *ptr,PetscErrorCode *ierr)
{
  *ierr = SNESDAFormFunction(*snes,*X,*F,ptr);
}


void PETSC_STDCALL snessetfunction_(SNES *snes,Vec *r,void (PETSC_STDCALL *func)(SNES*,Vec*,Vec*,void*,PetscErrorCode*),
                      void *ctx,PetscErrorCode *ierr)
{
  CHKFORTRANNULLOBJECT(ctx);
  f2 = func;
  if ((FCNVOID)func == (FCNVOID)snesdaformfunction_) {
    *ierr = SNESSetFunction(*snes,*r,SNESDAFormFunction,ctx);
  } else {
    *ierr = SNESSetFunction(*snes,*r,oursnesfunction,ctx);
  }
}

/* ---------------------------------------------------------*/

void PETSC_STDCALL matsnesmfsetfunction_(Mat *mat,Vec *r,void (PETSC_STDCALL *func)(SNES*,Vec*,Vec*,void*,PetscErrorCode*),
                      void *ctx,PetscErrorCode *ierr){
  f11 = func;
  CHKFORTRANNULLOBJECT(ctx);
  *ierr = MatSNESMFSetFunction(*mat,*r,ourmatsnesmffunction,ctx);
}
/* ---------------------------------------------------------*/

void PETSC_STDCALL snescreate_(MPI_Comm *comm,SNES *outsnes,PetscErrorCode *ierr){

*ierr = SNESCreate((MPI_Comm)PetscToPointerComm(*comm),outsnes);
}

/* ---------------------------------------------------------*/
/*
     snesdefaultcomputejacobian() and snesdefaultcomputejacobiancolor()
  These can be used directly from Fortran but are mostly so that 
  Fortran SNESSetJacobian() will properly handle the defaults being passed in.

  functions, hence no STDCALL
*/
void snesdefaultcomputejacobian_(SNES *snes,Vec *x,Mat *m,Mat *p,MatStructure* type,void *ctx,PetscErrorCode *ierr)
{
  *ierr = SNESDefaultComputeJacobian(*snes,*x,m,p,type,ctx);
}
void  snesdefaultcomputejacobiancolor_(SNES *snes,Vec *x,Mat *m,Mat *p,MatStructure* type,void *ctx,PetscErrorCode *ierr)
{
  *ierr = SNESDefaultComputeJacobianColor(*snes,*x,m,p,type,*(MatFDColoring*)ctx);
}

void  snesdacomputejacobianwithadifor_(SNES *snes,Vec *X,Mat *m,Mat *p,MatStructure* type,void *ctx,PetscErrorCode *ierr) 
{
  (*PetscErrorPrintf)("Cannot call this function from Fortran");
  *ierr = 1;
}

void  snesdacomputejacobian_(SNES *snes,Vec *X,Mat *m,Mat *p,MatStructure* type,void *ctx,PetscErrorCode *ierr) 
{
  (*PetscErrorPrintf)("Cannot call this function from Fortran");
  *ierr = 1;
}

void PETSC_STDCALL snessetjacobian_(SNES *snes,Mat *A,Mat *B,void (PETSC_STDCALL *func)(SNES*,Vec*,Mat*,Mat*,
            MatStructure*,void*,PetscErrorCode*),void *ctx,PetscErrorCode *ierr)
{
  CHKFORTRANNULLOBJECT(ctx);
  if ((FCNVOID)func == (FCNVOID)snesdefaultcomputejacobian_) {
    *ierr = SNESSetJacobian(*snes,*A,*B,SNESDefaultComputeJacobian,ctx);
  } else if ((FCNVOID)func == (FCNVOID)snesdefaultcomputejacobiancolor_) {
    *ierr = SNESSetJacobian(*snes,*A,*B,SNESDefaultComputeJacobianColor,*(MatFDColoring*)ctx);
  } else if ((FCNVOID)func == (FCNVOID)snesdacomputejacobianwithadifor_) {
    *ierr = SNESSetJacobian(*snes,*A,*B,SNESDAComputeJacobianWithAdifor,ctx);
  } else if ((FCNVOID)func == (FCNVOID)snesdacomputejacobian_) {
    *ierr = SNESSetJacobian(*snes,*A,*B,SNESDAComputeJacobian,ctx);
  } else {
    f3 = func;
    *ierr = SNESSetJacobian(*snes,*A,*B,oursnesjacobian,ctx);
  }
}

/* -------------------------------------------------------------*/

void PETSC_STDCALL snesregisterdestroy_(PetscErrorCode *ierr) 
{
  *ierr = SNESRegisterDestroy();
}

void PETSC_STDCALL snesgettype_(SNES *snes,CHAR name PETSC_MIXED_LEN(len),
                                PetscErrorCode *ierr PETSC_END_LEN(len))
{
  char *tname;

  *ierr = SNESGetType(*snes,&tname);
#if defined(PETSC_USES_CPTOFCD)
  {
    char *t = _fcdtocp(name); int len1 = _fcdlen(name);
    *ierr = PetscStrncpy(t,tname,len1);if (*ierr) return;
  }
#else
  *ierr = PetscStrncpy(name,tname,len);if (*ierr) return;
#endif
  FIXRETURNCHAR(name,len);
}

void PETSC_STDCALL snesgetoptionsprefix_(SNES *snes,CHAR prefix PETSC_MIXED_LEN(len),
                                         PetscErrorCode *ierr PETSC_END_LEN(len))
{
  char *tname;

  *ierr = SNESGetOptionsPrefix(*snes,&tname);
#if defined(PETSC_USES_CPTOFCD)
  {
    char *t = _fcdtocp(prefix); int len1 = _fcdlen(prefix);
    *ierr = PetscStrncpy(t,tname,len1);if (*ierr) return;
  }
#else
  *ierr = PetscStrncpy(prefix,tname,len);if (*ierr) return;
#endif
}

EXTERN_C_END


