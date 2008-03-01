#include "private/fortranimpl.h"
#include "petscts.h"

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define tssetrhsfunction_                    TSSETRHSFUNCTION
#define tssetmatrices_                       TSSETMATRICES
#define tsgetmatrices_                       TSGETMATRICES
#define tssetrhsjacobian_                    TSSETRHSJACOBIAN
#define tsgetrhsjacobian_                    TSGETRHSJACOBIAN
#define tsview_                              TSVIEW
#define tsgetoptionsprefix_                  TSGETOPTIONSPREFIX
#define tsmonitorset_                        TSMONITORSET
#define tsdefaultcomputejacobian_            TSDEFAULTCOMPUTEJACOBIAN
#define tsdefaultcomputejacobiancolor_       TSDEFAULTCOMPUTEJACOBIANCOLOR
#define tsmonitordefault_                    TSMONITORDEFAULT
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define tssetrhsfunction_                    tssetrhsfunction
#define tssetmatrices_                       tssetmatrices
#define tsgetmatrices_                       tsgetmatrices
#define tssetrhsjacobian_                    tssetrhsjacobian
#define tsgetrhsjacobian_                    tsgetrhsjacobian
#define tsview_                              tsview
#define tsgetoptionsprefix_                  tsgetoptionsprefix
#define tsmonitorset_                        tsmonitorset
#define tsdefaultcomputejacobian_            tsdefaultcomputejacobian
#define tsdefaultcomputejacobiancolor_       tsdefaultcomputejacobiancolor
#define tsmonitordefault_                    tsmonitordefault
#endif

static PetscErrorCode ourtsfunction(TS ts,PetscReal d,Vec x,Vec f,void *ctx)
{
  PetscErrorCode ierr = 0;
  (*(void (PETSC_STDCALL *)(TS*,PetscReal*,Vec*,Vec*,void*,PetscErrorCode*))(((PetscObject)ts)->fortran_func_pointers[1]))(&ts,&d,&x,&f,ctx,&ierr);
  return 0;
}
static PetscErrorCode ourtsmatrix(TS ts,PetscReal d,Mat* m,Mat* p,MatStructure* type,void*ctx)
{
  PetscErrorCode ierr = 0;
  (*(void (PETSC_STDCALL *)(TS*,PetscReal*,Mat*,Mat*,MatStructure*,void*,PetscErrorCode*))(((PetscObject)ts)->fortran_func_pointers[2]))(&ts,&d,m,p,type,ctx,&ierr);
  return 0;
}
static PetscErrorCode ourtslhsmatrix(TS ts,PetscReal d,Mat* m,Mat* p,MatStructure* type,void*ctx)
{
  PetscErrorCode ierr = 0;
  (*(void (PETSC_STDCALL *)(TS*,PetscReal*,Mat*,Mat*,MatStructure*,void*,PetscErrorCode*))(((PetscObject)ts)->fortran_func_pointers[7]))(&ts,&d,m,p,type,ctx,&ierr);
  return 0;
}
static PetscErrorCode ourtsjacobian(TS ts,PetscReal d,Vec x,Mat* m,Mat* p,MatStructure* type,void*ctx)
{
  PetscErrorCode ierr = 0;
  (*(void (PETSC_STDCALL *)(TS*,PetscReal*,Vec*,Mat*,Mat*,MatStructure*,void*,PetscErrorCode*))(((PetscObject)ts)->fortran_func_pointers[3]))(&ts,&d,&x,m,p,type,ctx,&ierr);
  return 0;
}

static PetscErrorCode ourmonitordestroy(void *ctx)
{
  PetscErrorCode ierr = 0;
  TS          ts = (TS)ctx;
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

void PETSC_STDCALL tssetrhsfunction_(TS *ts,PetscErrorCode (PETSC_STDCALL *f)(TS*,PetscReal*,Vec*,Vec*,void*,PetscErrorCode*),void*fP,PetscErrorCode *ierr)
{
  PetscObjectAllocateFortranPointers(*ts,8);
  ((PetscObject)*ts)->fortran_func_pointers[1] = (PetscVoidFunction)f;
  *ierr = TSSetRHSFunction(*ts,ourtsfunction,fP);
}

void PETSC_STDCALL tssetmatrices_(TS *ts,Mat *Arhs,PetscErrorCode (PETSC_STDCALL *frhs)(TS*,PetscReal*,Mat*,Mat*,MatStructure*,
                                                   void*,PetscInt *),
                                         Mat *Alhs,PetscErrorCode (PETSC_STDCALL *flhs)(TS*,PetscReal*,Mat*,Mat*,MatStructure*,
                                                   void*,PetscInt *),
                                         MatStructure *flag,void*fP,PetscErrorCode *ierr)
{
  PetscObjectAllocateFortranPointers(*ts,8);
  if (FORTRANNULLFUNCTION(frhs) && FORTRANNULLFUNCTION(flhs)) {
    *ierr = TSSetMatrices(*ts,*Arhs,PETSC_NULL,*Alhs,PETSC_NULL,*flag,fP);
  } else if (FORTRANNULLFUNCTION(flhs)){
    ((PetscObject)*ts)->fortran_func_pointers[2] = (PetscVoidFunction)frhs;
    *ierr = TSSetMatrices(*ts,*Arhs,ourtsmatrix,*Alhs,PETSC_NULL,*flag,fP);
  } else if (FORTRANNULLFUNCTION(frhs)){
    ((PetscObject)*ts)->fortran_func_pointers[7] = (PetscVoidFunction)flhs;
    *ierr = TSSetMatrices(*ts,*Arhs,PETSC_NULL,*Alhs,ourtslhsmatrix,*flag,fP);
  } else {
    ((PetscObject)*ts)->fortran_func_pointers[2] = (PetscVoidFunction)frhs;
    ((PetscObject)*ts)->fortran_func_pointers[7] = (PetscVoidFunction)flhs;
    *ierr = TSSetMatrices(*ts,*Arhs,ourtsmatrix,*Alhs,ourtslhsmatrix,*flag,fP);
  }          
}

/* ---------------------------------------------------------*/
extern void tsdefaultcomputejacobian_(TS*,PetscReal*,Vec*,Mat*,Mat*,MatStructure*,void*,PetscErrorCode*);
extern void tsdefaultcomputejacobiancolor_(TS*,PetscReal*,Vec*,Mat*,Mat*,MatStructure*,void*,PetscErrorCode*);

void PETSC_STDCALL tssetrhsjacobian_(TS *ts,Mat *A,Mat *B,void (PETSC_STDCALL *f)(TS*,PetscReal*,Vec*,Mat*,Mat*,MatStructure*,
               void*,PetscErrorCode*),void*fP,PetscErrorCode *ierr)
{
  PetscObjectAllocateFortranPointers(*ts,8);
  if (FORTRANNULLFUNCTION(f)) {
    *ierr = TSSetRHSJacobian(*ts,*A,*B,PETSC_NULL,fP);
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
  PetscObjectAllocateFortranPointers(*ts,8);
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
void PETSC_STDCALL tsgetrhsjacobian_(TS *ts,Mat *J,Mat *M,void **ctx,PetscErrorCode *ierr)
{
  *ierr = TSGetRHSJacobian(*ts,J,M,ctx);
}

void PETSC_STDCALL tsgetmatrices_(TS *ts,Mat *Arhs,Mat *Alhs,void **ctx,PetscErrorCode *ierr)
{
  *ierr = TSGetMatrices(*ts,Arhs,Alhs,ctx);
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
