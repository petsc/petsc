
#include "zpetsc.h"
#include "petscts.h"

#ifdef PETSC_HAVE_FORTRAN_CAPS
#define tssetrhsfunction_                    TSSETRHSFUNCTION
#define tssetrhsmatrix_                      TSSETRHSMATRIX
#define tssetrhsjacobian_                    TSSETRHSJACOBIAN
#define tscreate_                            TSCREATE
#define tsgettype_                           TSGETTYPE
#define tssetmonitor_                        TSSETMONITOR
#define tssettype_                           TSSETTYPE
#define tssundialsgetiterations_             TSSUNDIALSGETITERATIONS
#define tsgetoptionsprefix_                  TSGETOPTIONSPREFIX
#define tsview_                              TSVIEW
#define tsgetrhsjacobian_                    TSGETRHSJACOBIAN
#define tsgetrhsmatrix_                      TSGETRHSMATRIX
#define tssetrhsboundaryconditions_          TSSETRHSBOUNDARYCONDITIONS
#define tsdefaultcomputejacobian_            TSDEFAULTCOMPUTEJACOBIAN
#define tsdefaultcomputejacobiancolor_       TSDEFAULTCOMPUTEJACOBIANCOLOR
#define tsdefaultmonitor_                    TSDEFAULTMONITOR
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define tssundialsgetiterations_             tssundialsgetiterations
#define tssetrhsfunction_                    tssetrhsfunction
#define tssetrhsmatrix_                      tssetrhsmatrix
#define tssetrhsjacobian_                    tssetrhsjacobian
#define tscreate_                            tscreate
#define tsgettype_                           tsgettype
#define tssetmonitor_                        tssetmonitor
#define tssettype_                           tssettype
#define tsgetoptionsprefix_                  tsgetoptionsprefix
#define tsview_                              tsview
#define tsgetrhsjacobian_                    tsgetrhsjacobian
#define tsgetrhsmatrix_                      tsgetrhsmatrix
#define tssetrhsboundaryconditions_          tssetrhsboundaryconditions
#define tsdefaultcomputejacobian_            tsdefaultcomputejacobian
#define tsdefaultcomputejacobiancolor_       tsdefaultcomputejacobiancolor
#define tsdefaultmonitor_                    tsdefaultmonitor
#endif


static PetscErrorCode ourtsbcfunction(TS ts,PetscReal d,Vec x,void *ctx)
{
  PetscErrorCode ierr = 0;
  (*(void (PETSC_STDCALL *)(TS*,PetscReal*,Vec*,void*,PetscErrorCode*))(((PetscObject)ts)->fortran_func_pointers[0]))(&ts,&d,&x,ctx,&ierr);
  return 0;
}

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

static PetscErrorCode ourtsjacobian(TS ts,PetscReal d,Vec x,Mat* m,Mat* p,MatStructure* type,void*ctx)
{
  PetscErrorCode ierr = 0;
  (*(void (PETSC_STDCALL *)(TS*,PetscReal*,Vec*,Mat*,Mat*,MatStructure*,void*,PetscErrorCode*))(((PetscObject)ts)->fortran_func_pointers[3]))(&ts,&d,&x,m,p,type,ctx,&ierr);
  return 0;
}

/*
   Note ctx is the same as ts so we need to get the Fortran context out of the TS
*/
static PetscErrorCode ourtsmonitor(TS ts,PetscInt i,PetscReal d,Vec v,void*ctx)
{
  PetscErrorCode ierr = 0;
  void       (*mctx)(void) = ((PetscObject)ts)->fortran_func_pointers[6];
  (*(void (PETSC_STDCALL *)(TS*,PetscInt*,PetscReal*,Vec*,FCNVOID,PetscErrorCode*))(((PetscObject)ts)->fortran_func_pointers[4]))(&ts,&i,&d,&v,mctx,&ierr);
  return 0;
}

static PetscErrorCode ourtsdestroy(void *ctx)
{
  PetscErrorCode ierr = 0;
  TS          ts = (TS)ctx;
  void        (*mctx)(void) = ((PetscObject)ts)->fortran_func_pointers[6];
  (*(void (PETSC_STDCALL *)(FCNVOID,PetscErrorCode*))(((PetscObject)ts)->fortran_func_pointers[5]))(mctx,&ierr);
  return 0;
}

EXTERN_C_BEGIN

void PETSC_STDCALL tssetrhsboundaryconditions_(TS *ts,PetscErrorCode (PETSC_STDCALL *f)(TS*,PetscReal*,Vec*,void*,PetscErrorCode*),void *ctx,PetscErrorCode *ierr)
{
  ((PetscObject)*ts)->fortran_func_pointers[0] = (FCNVOID)f;
  *ierr = TSSetRHSBoundaryConditions(*ts,ourtsbcfunction,ctx);
}

void PETSC_STDCALL tsgetrhsjacobian_(TS *ts,Mat *J,Mat *M,void **ctx,PetscErrorCode *ierr)
{
  *ierr = TSGetRHSJacobian(*ts,J,M,ctx);
}

void PETSC_STDCALL tsgetrhsmatrix_(TS *ts,Mat *J,Mat *M,void **ctx,PetscErrorCode *ierr)
{
  *ierr = TSGetRHSMatrix(*ts,J,M,ctx);
}

void PETSC_STDCALL tsview_(TS *ts,PetscViewer *viewer, PetscErrorCode *ierr)
{
  PetscViewer v;
  PetscPatchDefaultViewers_Fortran(viewer,v);
  *ierr = TSView(*ts,v);
}

void PETSC_STDCALL tssettype_(TS *ts,CHAR type PETSC_MIXED_LEN(len),PetscErrorCode *ierr PETSC_END_LEN(len))
{
  char *t;

  FIXCHAR(type,len,t);
  *ierr = TSSetType(*ts,t);
  FREECHAR(type,t);
}


void PETSC_STDCALL tssetrhsfunction_(TS *ts,PetscErrorCode (PETSC_STDCALL *f)(TS*,PetscReal*,Vec*,Vec*,void*,PetscErrorCode*),void*fP,PetscErrorCode *ierr)
{
  ((PetscObject)*ts)->fortran_func_pointers[1] = (FCNVOID)f;
  *ierr = TSSetRHSFunction(*ts,ourtsfunction,fP);
}


/* ---------------------------------------------------------*/

void PETSC_STDCALL tssetrhsmatrix_(TS *ts,Mat *A,Mat *B,PetscErrorCode (PETSC_STDCALL *f)(TS*,PetscReal*,Mat*,Mat*,MatStructure*,
                                                   void*,PetscInt *),void*fP,PetscErrorCode *ierr)
{
  if (FORTRANNULLFUNCTION(f)) {
    *ierr = TSSetRHSMatrix(*ts,*A,*B,PETSC_NULL,fP);
  } else {
    ((PetscObject)*ts)->fortran_func_pointers[2] = (FCNVOID)f;
    *ierr = TSSetRHSMatrix(*ts,*A,*B,ourtsmatrix,fP);
  }
}

/* ---------------------------------------------------------*/

extern void tsdefaultcomputejacobian_(TS*,PetscReal*,Vec*,Mat*,Mat*,MatStructure*,void*,PetscErrorCode*);
extern void tsdefaultcomputejacobiancolor_(TS*,PetscReal*,Vec*,Mat*,Mat*,MatStructure*,void*,PetscErrorCode*);

void PETSC_STDCALL tssetrhsjacobian_(TS *ts,Mat *A,Mat *B,void (PETSC_STDCALL *f)(TS*,PetscReal*,Vec*,Mat*,Mat*,MatStructure*,
               void*,PetscErrorCode*),void*fP,PetscErrorCode *ierr)
{
  if (FORTRANNULLFUNCTION(f)) {
    *ierr = TSSetRHSJacobian(*ts,*A,*B,PETSC_NULL,fP);
  } else if ((FCNVOID)f == (FCNVOID)tsdefaultcomputejacobian_) {
    *ierr = TSSetRHSJacobian(*ts,*A,*B,TSDefaultComputeJacobian,fP);
  } else if ((FCNVOID)f == (FCNVOID)tsdefaultcomputejacobiancolor_) {
    *ierr = TSSetRHSJacobian(*ts,*A,*B,TSDefaultComputeJacobianColor,*(MatFDColoring*)fP);
  } else {
  ((PetscObject)*ts)->fortran_func_pointers[3] = (FCNVOID)f;
    *ierr = TSSetRHSJacobian(*ts,*A,*B,ourtsjacobian,fP);
  }
}

void PETSC_STDCALL tscreate_(MPI_Comm *comm,TS *outts,PetscErrorCode *ierr)
{
  *ierr = TSCreate((MPI_Comm)PetscToPointerComm(*comm),outts);
  *ierr = PetscMalloc(7*sizeof(void*),&((PetscObject)*outts)->fortran_func_pointers);
}

void PETSC_STDCALL tsgettype_(TS *ts,CHAR name PETSC_MIXED_LEN(len),PetscErrorCode *ierr PETSC_END_LEN(len))
{
  char *tname;

  *ierr = TSGetType(*ts,(TSType *)&tname);
#if defined(PETSC_USES_CPTOFCD)
  {
    char *t = _fcdtocp(name); int len1 = _fcdlen(name);
    *ierr = PetscStrncpy(t,tname,len1);
  }
#else
  *ierr = PetscStrncpy(name,tname,len);
#endif
  FIXRETURNCHAR(name,len);
}

#if defined(PETSC_HAVE_SUNDIALS)
void PETSC_STDCALL tssundialsgetiterations_(TS *ts,PetscInt *nonlin,PetscInt *lin,PetscErrorCode *ierr)
{
  CHKFORTRANNULLINTEGER(nonlin);
  CHKFORTRANNULLINTEGER(lin);
  *ierr = TSSundialsGetIterations(*ts,nonlin,lin);
}
#endif

extern void PETSC_STDCALL tsdefaultmonitor_(TS*,PetscInt*,PetscReal*,Vec*,void*,PetscErrorCode*);

void PETSC_STDCALL tssetmonitor_(TS *ts,void (PETSC_STDCALL *func)(TS*,PetscInt*,PetscReal*,Vec*,void*,PetscErrorCode*),void (*mctx)(void),void (PETSC_STDCALL *d)(void*,PetscErrorCode*),PetscErrorCode *ierr)
{
  if ((FCNVOID)func == (FCNVOID)tsdefaultmonitor_) {
    *ierr = TSSetMonitor(*ts,TSDefaultMonitor,0,0);
  } else {
    ((PetscObject)*ts)->fortran_func_pointers[4] = (FCNVOID)func;
    ((PetscObject)*ts)->fortran_func_pointers[5] = (FCNVOID)d;
    ((PetscObject)*ts)->fortran_func_pointers[6] = (FCNVOID)mctx;
    if (FORTRANNULLFUNCTION(d)) {
      *ierr = TSSetMonitor(*ts,ourtsmonitor,*ts,0);
    } else {
      *ierr = TSSetMonitor(*ts,ourtsmonitor,*ts,ourtsdestroy);
    }
  }
}

void PETSC_STDCALL tsgetoptionsprefix_(TS *ts,CHAR prefix PETSC_MIXED_LEN(len),PetscErrorCode *ierr PETSC_END_LEN(len))
{
  const char *tname;

  *ierr = TSGetOptionsPrefix(*ts,&tname);
#if defined(PETSC_USES_CPTOFCD)
  {
    char *t = _fcdtocp(prefix); int len1 = _fcdlen(prefix);
    *ierr = PetscStrncpy(t,tname,len1);
  }
#else
  *ierr = PetscStrncpy(prefix,tname,len);
#endif
}


EXTERN_C_END

