#include <petsc-private/fortranimpl.h>
#include <petscts.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define tssetrhsfunction_                    TSSETRHSFUNCTION
#define tsgetrhsfunction_                    TSGETRHSFUNCTION
#define tssetrhsjacobian_                    TSSETRHSJACOBIAN
#define tsgetrhsjacobian_                    TSGETRHSJACOBIAN
#define tssetifunction_                      TSSETIFUNCTION
#define tsgetifunction_                      TSGETIFUNCTION
#define tssetijacobian_                      TSSETIJACOBIAN
#define tsgetijacobian_                      TSGETIJACOBIAN
#define tsview_                              TSVIEW
#define tssetoptionsprefix_                  TSSETOPTIONSPREFIX
#define tsgetoptionsprefix_                  TSGETOPTIONSPREFIX
#define tsappendoptionsprefix_               TSAPPENDOPTIONSPREFIX
#define tsmonitorset_                        TSMONITORSET
#define tscomputerhsfunctionlinear_          TSCOMPUTERHSFUNCTIONLINEAR
#define tscomputerhsjacobianconstant_        TSCOMPUTERHSJACOBIANCONSTANT
#define tscomputeifunctionlinear_            TSCOMPUTEIFUNCTIONLINEAR
#define tscomputeijacobianconstant_          TSCOMPUTEIJACOBIANCONSTANT
#define tsmonitordefault_                    TSMONITORDEFAULT
#define tssetprestep_                        TSSETPRESTEP
#define tssetpoststep_                       TSSETPOSTSTEP
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define tssetrhsfunction_                    tssetrhsfunction
#define tsgetrhsfunction_                    tsgetrhsfunction
#define tssetrhsjacobian_                    tssetrhsjacobian
#define tsgetrhsjacobian_                    tsgetrhsjacobian
#define tssetifunction_                      tssetifunction
#define tsgetifunction_                      tsgetifunction
#define tssetijacobian_                      tssetijacobian
#define tsgetijacobian_                      tsgetijacobian
#define tsview_                              tsview
#define tssetoptionsprefix_                  tssetoptionsprefix
#define tsgetoptionsprefix_                  tsgetoptionsprefix
#define tsappendoptionsprefix_               tsappendoptionsprefix
#define tsmonitorset_                        tsmonitorset
#define tscomputerhsfunctionlinear_          tscomputerhsfunctionlinear
#define tscomputerhsjacobianconstant_        tscomputerhsjacobianconstant
#define tscomputeifunctionlinear_            tscomputeifunctionlinear
#define tscomputeijacobianconstant_          tscomputeijacobianconstant
#define tsmonitordefault_                    tsmonitordefault
#define tssetprestep_                        tssetprestep
#define tssetpoststep_                       tssetpoststep
#endif

enum {OUR_PRESTEP = 0,
      OUR_POSTSTEP,
      OUR_RHSFUNCTION,
      OUR_IFUNCTION,
      OUR_RHSJACOBIAN,
      OUR_IJACOBIAN,
      OUR_MONITOR,
      OUR_MONITORDESTROY,
      OUR_MONITOR_CTX,   /* Casting from function pointer is invalid according to the standard. */
      OUR_COUNT};

static PetscErrorCode ourprestep(TS ts)
{
  PetscErrorCode ierr = 0;
  (*(void (PETSC_STDCALL *)(TS*,PetscErrorCode*))(((PetscObject)ts)->fortran_func_pointers[OUR_PRESTEP]))(&ts,&ierr);
  return 0;
}
static PetscErrorCode ourpoststep(TS ts)
{
  PetscErrorCode ierr = 0;
  (*(void (PETSC_STDCALL *)(TS*,PetscErrorCode*))(((PetscObject)ts)->fortran_func_pointers[OUR_POSTSTEP]))(&ts,&ierr);
  return 0;
}
static PetscErrorCode ourrhsfunction(TS ts,PetscReal d,Vec x,Vec f,void *ctx)
{
  PetscErrorCode ierr = 0;
  (*(void (PETSC_STDCALL *)(TS*,PetscReal*,Vec*,Vec*,void*,PetscErrorCode*))(((PetscObject)ts)->fortran_func_pointers[OUR_RHSFUNCTION]))(&ts,&d,&x,&f,ctx,&ierr);
  return 0;
}
static PetscErrorCode ourifunction(TS ts,PetscReal d,Vec x,Vec xdot,Vec f,void *ctx)
{
  PetscErrorCode ierr = 0;
  (*(void (PETSC_STDCALL *)(TS*,PetscReal*,Vec*,Vec*,Vec*,void*,PetscErrorCode*))(((PetscObject)ts)->fortran_func_pointers[OUR_IFUNCTION]))(&ts,&d,&x,&xdot,&f,ctx,&ierr);
  return 0;
}
static PetscErrorCode ourrhsjacobian(TS ts,PetscReal d,Vec x,Mat* m,Mat* p,MatStructure* type,void*ctx)
{
  PetscErrorCode ierr = 0;
  (*(void (PETSC_STDCALL *)(TS*,PetscReal*,Vec*,Mat*,Mat*,MatStructure*,void*,PetscErrorCode*))(((PetscObject)ts)->fortran_func_pointers[OUR_RHSJACOBIAN]))(&ts,&d,&x,m,p,type,ctx,&ierr);
  return 0;
}
static PetscErrorCode ourijacobian(TS ts,PetscReal d,Vec x,Vec xdot,PetscReal shift,Mat* m,Mat* p,MatStructure* type,void*ctx)
{
  PetscErrorCode ierr = 0;
  (*(void (PETSC_STDCALL *)(TS*,PetscReal*,Vec*,Vec*,PetscReal*,Mat*,Mat*,MatStructure*,void*,PetscErrorCode*))(((PetscObject)ts)->fortran_func_pointers[OUR_IJACOBIAN]))(&ts,&d,&x,&xdot,&shift,m,p,type,ctx,&ierr);
  return 0;
}

static PetscErrorCode ourmonitordestroy(void **ctx)
{
  PetscErrorCode ierr = 0;
  TS          ts = *(TS*)ctx;
  void        *mctx = (void*) ((PetscObject)ts)->fortran_func_pointers[OUR_MONITOR_CTX];
  (*(void (PETSC_STDCALL *)(void*,PetscErrorCode*))(((PetscObject)ts)->fortran_func_pointers[OUR_MONITORDESTROY]))(mctx,&ierr);
  return 0;
}

/*
   Note ctx is the same as ts so we need to get the Fortran context out of the TS
*/
static PetscErrorCode ourmonitor(TS ts,PetscInt i,PetscReal d,Vec v,void*ctx)
{
  PetscErrorCode ierr = 0;
  void           *mctx = (void*) ((PetscObject)ts)->fortran_func_pointers[OUR_MONITOR_CTX];
  (*(void (PETSC_STDCALL *)(TS*,PetscInt*,PetscReal*,Vec*,void*,PetscErrorCode*))(((PetscObject)ts)->fortran_func_pointers[OUR_MONITOR]))(&ts,&i,&d,&v,mctx,&ierr);
  return 0;
}

EXTERN_C_BEGIN

void PETSC_STDCALL tssetprestep_(TS *ts,PetscErrorCode (PETSC_STDCALL *f)(TS*,PetscErrorCode*),PetscErrorCode *ierr)
{
  PetscObjectAllocateFortranPointers(*ts,OUR_COUNT);
  ((PetscObject)*ts)->fortran_func_pointers[OUR_PRESTEP] = (PetscVoidFunction)f;
  *ierr = TSSetPreStep(*ts,ourprestep);
}

void PETSC_STDCALL tssetpoststep_(TS *ts,PetscErrorCode (PETSC_STDCALL *f)(TS*,PetscErrorCode*),PetscErrorCode *ierr)
{
  PetscObjectAllocateFortranPointers(*ts,OUR_COUNT);
  ((PetscObject)*ts)->fortran_func_pointers[OUR_POSTSTEP] = (PetscVoidFunction)f;
  *ierr = TSSetPreStep(*ts,ourpoststep);
}

void tscomputerhsfunctionlinear_(TS *ts,PetscReal *t,Vec *X,Vec *F,void *ctx,PetscErrorCode *ierr)
{
  *ierr = TSComputeRHSFunctionLinear(*ts,*t,*X,*F,ctx);
}
void PETSC_STDCALL tssetrhsfunction_(TS *ts,Vec *r,PetscErrorCode (PETSC_STDCALL *f)(TS*,PetscReal*,Vec*,Vec*,void*,PetscErrorCode*),void*fP,PetscErrorCode *ierr)
{
  Vec R;
  CHKFORTRANNULLOBJECT(r);
  CHKFORTRANNULLFUNCTION(f);
  CHKFORTRANNULLOBJECT(fP);
  R = r ? *r : (Vec)PETSC_NULL;
  if ((PetscVoidFunction)f == (PetscVoidFunction)tscomputerhsfunctionlinear_) {
    *ierr = TSSetRHSFunction(*ts,R,TSComputeRHSFunctionLinear,fP);
  } else {
    PetscObjectAllocateFortranPointers(*ts,OUR_COUNT);
    ((PetscObject)*ts)->fortran_func_pointers[OUR_RHSFUNCTION] = (PetscVoidFunction)f;
    *ierr = TSSetRHSFunction(*ts,R,ourrhsfunction,fP);
  }
}
void PETSC_STDCALL tsgetrhsfunction_(TS *ts,Vec *r,void *func,void **ctx,PetscErrorCode *ierr)
{
  CHKFORTRANNULLINTEGER(ctx);
  CHKFORTRANNULLOBJECT(r);
  *ierr = TSGetRHSFunction(*ts,r,PETSC_NULL,ctx);
}

void tscomputeifunctionlinear_(TS *ts,PetscReal *t,Vec *X,Vec *Xdot,Vec *F,void *ctx,PetscErrorCode *ierr)
{
  *ierr = TSComputeIFunctionLinear(*ts,*t,*X,*Xdot,*F,ctx);
}
void PETSC_STDCALL tssetifunction_(TS *ts,Vec *r,PetscErrorCode (PETSC_STDCALL *f)(TS*,PetscReal*,Vec*,Vec*,Vec*,void*,PetscErrorCode*),void*fP,PetscErrorCode *ierr)
{
  Vec R;
  CHKFORTRANNULLOBJECT(r);
  CHKFORTRANNULLFUNCTION(f);
  CHKFORTRANNULLOBJECT(fP);
  R = r ? *r : (Vec)PETSC_NULL;
  if ((PetscVoidFunction)f == (PetscVoidFunction)tscomputeifunctionlinear_) {
    *ierr = TSSetIFunction(*ts,R,TSComputeIFunctionLinear,fP);
  } else {
    PetscObjectAllocateFortranPointers(*ts,OUR_COUNT);
    ((PetscObject)*ts)->fortran_func_pointers[OUR_IFUNCTION] = (PetscVoidFunction)f;
    *ierr = TSSetIFunction(*ts,R,ourifunction,fP);
  }
}
void PETSC_STDCALL tsgetifunction_(TS *ts,Vec *r,void *func,void **ctx,PetscErrorCode *ierr)
{
  CHKFORTRANNULLINTEGER(ctx);
  CHKFORTRANNULLOBJECT(r);
  *ierr = TSGetIFunction(*ts,r,PETSC_NULL,ctx);
}

/* ---------------------------------------------------------*/
void tscomputerhsjacobianconstant_(TS *ts,PetscReal *t,Vec *X,Mat *A,Mat *B,MatStructure *flg,void *ctx,PetscErrorCode *ierr)
{
  *ierr = TSComputeRHSJacobianConstant(*ts,*t,*X,A,B,flg,ctx);
}
void PETSC_STDCALL tssetrhsjacobian_(TS *ts,Mat *A,Mat *B,void (PETSC_STDCALL *f)(TS*,PetscReal*,Vec*,Mat*,Mat*,MatStructure*,
               void*,PetscErrorCode*),void*fP,PetscErrorCode *ierr)
{
  PetscObjectAllocateFortranPointers(*ts,OUR_COUNT);
  if (FORTRANNULLFUNCTION(f)) {
    *ierr = TSSetRHSJacobian(*ts,*A,*B,PETSC_NULL,fP);
  } else if ((PetscVoidFunction)f == (PetscVoidFunction)tscomputerhsjacobianconstant_) {
    *ierr = TSSetRHSJacobian(*ts,*A,*B,TSComputeRHSJacobianConstant,fP);
  } else {
    ((PetscObject)*ts)->fortran_func_pointers[OUR_RHSJACOBIAN] = (PetscVoidFunction)f;
    *ierr = TSSetRHSJacobian(*ts,*A,*B,ourrhsjacobian,fP);
  }
}

void tscomputeijacobianconstant_(TS *ts,PetscReal *t,Vec *X,Vec *Xdot,PetscReal *shift,Mat *A,Mat *B,MatStructure *flg,void *ctx,PetscErrorCode *ierr)
{
  *ierr = TSComputeIJacobianConstant(*ts,*t,*X,*Xdot,*shift,A,B,flg,ctx);
}
void PETSC_STDCALL tssetijacobian_(TS *ts,Mat *A,Mat *B,void (PETSC_STDCALL *f)(TS*,PetscReal*,Vec*,Mat*,Mat*,MatStructure*,
               void*,PetscErrorCode*),void*fP,PetscErrorCode *ierr)
{
  PetscObjectAllocateFortranPointers(*ts,OUR_COUNT);
  if (FORTRANNULLFUNCTION(f)) {
    *ierr = TSSetIJacobian(*ts,*A,*B,PETSC_NULL,fP);
  } else if ((PetscVoidFunction)f == (PetscVoidFunction)tscomputeijacobianconstant_) {
    *ierr = TSSetIJacobian(*ts,*A,*B,TSComputeIJacobianConstant,fP);
  } else {
    ((PetscObject)*ts)->fortran_func_pointers[OUR_IJACOBIAN] = (PetscVoidFunction)f;
    *ierr = TSSetIJacobian(*ts,*A,*B,ourijacobian,fP);
  }
}
void PETSC_STDCALL tsgetijacobian_(TS *ts,Mat *J,Mat *M,int *func,void **ctx,PetscErrorCode *ierr)
{
  CHKFORTRANNULLINTEGER(ctx);
  CHKFORTRANNULLOBJECT(J);
  CHKFORTRANNULLOBJECT(M);
  *ierr = TSGetIJacobian(*ts,J,M,0,ctx);
}

/* ---------------------------------------------------------*/

extern void PETSC_STDCALL tsmonitordefault_(TS*,PetscInt*,PetscReal*,Vec*,void*,PetscErrorCode*);

void PETSC_STDCALL tsmonitorset_(TS *ts,void (PETSC_STDCALL *func)(TS*,PetscInt*,PetscReal*,Vec*,void*,PetscErrorCode*),void (*mctx)(void),void (PETSC_STDCALL *d)(void*,PetscErrorCode*),PetscErrorCode *ierr)
{
  PetscObjectAllocateFortranPointers(*ts,OUR_COUNT);
  if ((PetscVoidFunction)func == (PetscVoidFunction)tsmonitordefault_) {
    *ierr = TSMonitorSet(*ts,TSMonitorDefault,0,0);
  } else {
    ((PetscObject)*ts)->fortran_func_pointers[OUR_MONITOR]        = (PetscVoidFunction)func;
    ((PetscObject)*ts)->fortran_func_pointers[OUR_MONITORDESTROY] = (PetscVoidFunction)d;
    ((PetscObject)*ts)->fortran_func_pointers[OUR_MONITOR_CTX]    = (PetscVoidFunction)mctx;
    if (FORTRANNULLFUNCTION(d)) {
      *ierr = TSMonitorSet(*ts,ourmonitor,*ts,0);
    } else {
      *ierr = TSMonitorSet(*ts,ourmonitor,*ts,ourmonitordestroy);
    }
  }
}

/* ---------------------------------------------------------*/
/*  func is currently ignored from Fortran */
void PETSC_STDCALL tsgetrhsjacobian_(TS *ts,Mat *J,Mat *M,int *func,void **ctx,PetscErrorCode *ierr)
{
  *ierr = TSGetRHSJacobian(*ts,J,M,0,ctx);
}

void PETSC_STDCALL tsview_(TS *ts,PetscViewer *viewer, PetscErrorCode *ierr)
{
  PetscViewer v;
  PetscPatchDefaultViewers_Fortran(viewer,v);
  *ierr = TSView(*ts,v);
}

void PETSC_STDCALL tssetoptionsprefix_(TS *ts,CHAR prefix PETSC_MIXED_LEN(len),PetscErrorCode *ierr PETSC_END_LEN(len))
{
  char *t;
  FIXCHAR(prefix,len,t);
  *ierr = TSSetOptionsPrefix(*ts,t);
  FREECHAR(prefix,t);
}
void PETSC_STDCALL tsgetoptionsprefix_(TS *ts,CHAR prefix PETSC_MIXED_LEN(len),PetscErrorCode *ierr PETSC_END_LEN(len))
{
  const char *tname;

  *ierr = TSGetOptionsPrefix(*ts,&tname);
  *ierr = PetscStrncpy(prefix,tname,len);
}
void PETSC_STDCALL tsappendoptionsprefix_(TS *ts,CHAR prefix PETSC_MIXED_LEN(len),PetscErrorCode *ierr PETSC_END_LEN(len))
{
  char *t;
  FIXCHAR(prefix,len,t);
  *ierr = TSAppendOptionsPrefix(*ts,t);
  FREECHAR(prefix,t);
}


EXTERN_C_END
