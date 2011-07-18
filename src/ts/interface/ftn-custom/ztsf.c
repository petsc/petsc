#include <private/fortranimpl.h>
#include <petscts.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define tssetrhsfunction_                    TSSETRHSFUNCTION
#define tssetrhsjacobian_                    TSSETRHSJACOBIAN
#define tsgetrhsjacobian_                    TSGETRHSJACOBIAN
#define tsview_                              TSVIEW
#define tsgetoptionsprefix_                  TSGETOPTIONSPREFIX
#define tsmonitorset_                        TSMONITORSET
#define tscomputerhsfunctionlinear_          TSCOMPUTERHSFUNCTIONLINEAR
#define tscomputerhsjacobianconstant_        TSCOMPUTERHSJACOBIANCONSTANT
#define tsdefaultcomputejacobian_            TSDEFAULTCOMPUTEJACOBIAN
#define tsdefaultcomputejacobiancolor_       TSDEFAULTCOMPUTEJACOBIANCOLOR
#define tsmonitordefault_                    TSMONITORDEFAULT
#define tssetprestep_                        TSSETPRESTEP
#define tssetpoststep_                       TSSETPOSTSTEP
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define tssetrhsfunction_                    tssetrhsfunction
#define tssetrhsjacobian_                    tssetrhsjacobian
#define tsgetrhsjacobian_                    tsgetrhsjacobian
#define tsview_                              tsview
#define tsgetoptionsprefix_                  tsgetoptionsprefix
#define tsmonitorset_                        tsmonitorset
#define tscomputerhsfunctionlinear_          tscomputerhsfunctionlinear
#define tscomputerhsjacobianconstant_        tscomputerhsjacobianconstant
#define tsdefaultcomputejacobian_            tsdefaultcomputejacobian
#define tsdefaultcomputejacobiancolor_       tsdefaultcomputejacobiancolor
#define tsmonitordefault_                    tsmonitordefault
#define tssetprestep_                        tssetprestep
#define tssetpoststep_                       tssetpoststep
#endif

static PetscErrorCode ourprestep(TS ts)
{
  PetscErrorCode ierr = 0;
  (*(void (PETSC_STDCALL *)(TS*,PetscErrorCode*))(((PetscObject)ts)->fortran_func_pointers[8]))(&ts,&ierr);
  return 0;
}
static PetscErrorCode ourpoststep(TS ts)
{
  PetscErrorCode ierr = 0;
  (*(void (PETSC_STDCALL *)(TS*,PetscErrorCode*))(((PetscObject)ts)->fortran_func_pointers[9]))(&ts,&ierr);
  return 0;
}
static PetscErrorCode ourtsfunction(TS ts,PetscReal d,Vec x,Vec f,void *ctx)
{
  PetscErrorCode ierr = 0;
  (*(void (PETSC_STDCALL *)(TS*,PetscReal*,Vec*,Vec*,void*,PetscErrorCode*))(((PetscObject)ts)->fortran_func_pointers[1]))(&ts,&d,&x,&f,ctx,&ierr);
  return 0;
}
static PetscErrorCode ourtsjacobian(TS ts,PetscReal d,Vec x,Mat* m,Mat* p,MatStructure* type,void*ctx)
{
  PetscErrorCode ierr = 0;
  (*(void (PETSC_STDCALL *)(TS*,PetscReal*,Vec*,Mat*,Mat*,MatStructure*,void*,PetscErrorCode*))(((PetscObject)ts)->fortran_func_pointers[3]))(&ts,&d,&x,m,p,type,ctx,&ierr);
  return 0;
}

static PetscErrorCode ourmonitordestroy(void **ctx)
{
  PetscErrorCode ierr = 0;
  TS          ts = *(TS*)ctx;
  void        *mctx = (void*) ((PetscObject)ts)->fortran_func_pointers[6];
  (*(void (PETSC_STDCALL *)(void*,PetscErrorCode*))(((PetscObject)ts)->fortran_func_pointers[5]))(mctx,&ierr);
  return 0;
}

/*
   Note ctx is the same as ts so we need to get the Fortran context out of the TS
*/
static PetscErrorCode ourtsmonitor(TS ts,PetscInt i,PetscReal d,Vec v,void*ctx)
{
  PetscErrorCode ierr = 0;
  void           *mctx = (void*) ((PetscObject)ts)->fortran_func_pointers[6];
  (*(void (PETSC_STDCALL *)(TS*,PetscInt*,PetscReal*,Vec*,void*,PetscErrorCode*))(((PetscObject)ts)->fortran_func_pointers[4]))(&ts,&i,&d,&v,mctx,&ierr);
  return 0;
}

EXTERN_C_BEGIN

void PETSC_STDCALL tssetprestep_(TS *ts,PetscErrorCode (PETSC_STDCALL *f)(TS*,PetscErrorCode*),PetscErrorCode *ierr)
{
  PetscObjectAllocateFortranPointers(*ts,10);
  ((PetscObject)*ts)->fortran_func_pointers[8] = (PetscVoidFunction)f;
  *ierr = TSSetPreStep(*ts,ourprestep);
}

void PETSC_STDCALL tssetpoststep_(TS *ts,PetscErrorCode (PETSC_STDCALL *f)(TS*,PetscErrorCode*),PetscErrorCode *ierr)
{
  PetscObjectAllocateFortranPointers(*ts,10);
  ((PetscObject)*ts)->fortran_func_pointers[9] = (PetscVoidFunction)f;
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
  R = r ? *r : PETSC_NULL;
  if ((PetscVoidFunction)f == (PetscVoidFunction)tscomputerhsfunctionlinear_) {
    *ierr = TSSetRHSFunction(*ts,R,TSComputeRHSFunctionLinear,fP);
  } else {
    PetscObjectAllocateFortranPointers(*ts,10);
    ((PetscObject)*ts)->fortran_func_pointers[1] = (PetscVoidFunction)f;
    *ierr = TSSetRHSFunction(*ts,R,ourtsfunction,fP);
  }
}

/* ---------------------------------------------------------*/
extern void tsdefaultcomputejacobian_(TS*,PetscReal*,Vec*,Mat*,Mat*,MatStructure*,void*,PetscErrorCode*);
extern void tsdefaultcomputejacobiancolor_(TS*,PetscReal*,Vec*,Mat*,Mat*,MatStructure*,void*,PetscErrorCode*);
void tscomputerhsjacobianconstant_(TS *ts,PetscReal *t,Vec *X,Mat *A,Mat *B,MatStructure *flg,void *ctx,PetscErrorCode *ierr)
{
  *ierr = TSComputeRHSJacobianConstant(*ts,*t,*X,A,B,flg,ctx);
}
void PETSC_STDCALL tssetrhsjacobian_(TS *ts,Mat *A,Mat *B,void (PETSC_STDCALL *f)(TS*,PetscReal*,Vec*,Mat*,Mat*,MatStructure*,
               void*,PetscErrorCode*),void*fP,PetscErrorCode *ierr)
{
  PetscObjectAllocateFortranPointers(*ts,10);
  if (FORTRANNULLFUNCTION(f)) {
    *ierr = TSSetRHSJacobian(*ts,*A,*B,PETSC_NULL,fP);
  } else if ((PetscVoidFunction)f == (PetscVoidFunction)tscomputerhsjacobianconstant_) {
    *ierr = TSSetRHSJacobian(*ts,*A,*B,TSComputeRHSJacobianConstant,fP);
  } else if ((PetscVoidFunction)f == (PetscVoidFunction)tsdefaultcomputejacobian_) {
    *ierr = TSSetRHSJacobian(*ts,*A,*B,TSDefaultComputeJacobian,fP);
  } else if ((PetscVoidFunction)f == (PetscVoidFunction)tsdefaultcomputejacobiancolor_) {
    *ierr = TSSetRHSJacobian(*ts,*A,*B,TSDefaultComputeJacobianColor,*(MatFDColoring*)fP);
  } else {
  ((PetscObject)*ts)->fortran_func_pointers[3] = (PetscVoidFunction)f;
    *ierr = TSSetRHSJacobian(*ts,*A,*B,ourtsjacobian,fP);
  }
}

/* ---------------------------------------------------------*/

extern void PETSC_STDCALL tsmonitordefault_(TS*,PetscInt*,PetscReal*,Vec*,void*,PetscErrorCode*);

void PETSC_STDCALL tsmonitorset_(TS *ts,void (PETSC_STDCALL *func)(TS*,PetscInt*,PetscReal*,Vec*,void*,PetscErrorCode*),void (*mctx)(void),void (PETSC_STDCALL *d)(void*,PetscErrorCode*),PetscErrorCode *ierr)
{
  PetscObjectAllocateFortranPointers(*ts,10);
  if ((PetscVoidFunction)func == (PetscVoidFunction)tsmonitordefault_) {
    *ierr = TSMonitorSet(*ts,TSMonitorDefault,0,0);
  } else {
    ((PetscObject)*ts)->fortran_func_pointers[4] = (PetscVoidFunction)func;
    ((PetscObject)*ts)->fortran_func_pointers[5] = (PetscVoidFunction)d;
    ((PetscObject)*ts)->fortran_func_pointers[6] = (PetscVoidFunction)mctx;
    if (FORTRANNULLFUNCTION(d)) {
      *ierr = TSMonitorSet(*ts,ourtsmonitor,*ts,0);
    } else {
      *ierr = TSMonitorSet(*ts,ourtsmonitor,*ts,ourmonitordestroy);
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

void PETSC_STDCALL tsgetoptionsprefix_(TS *ts,CHAR prefix PETSC_MIXED_LEN(len),PetscErrorCode *ierr PETSC_END_LEN(len))
{
  const char *tname;

  *ierr = TSGetOptionsPrefix(*ts,&tname);
  *ierr = PetscStrncpy(prefix,tname,len);
}


EXTERN_C_END
