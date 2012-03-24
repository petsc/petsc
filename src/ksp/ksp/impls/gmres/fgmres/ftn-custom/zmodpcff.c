#include <petsc-private/fortranimpl.h>
#include <petscksp.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define kspfgmressetmodifypc_      KSPFGMRESSETMODIFYPC
#define kspfgmresmodifypcnochange_ KSPFGMRESMODIFYPCNOCHANGE
#define kspfgmresmodifypcksp_      KSPFGMRESMODIFYPCKSP
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define kspfgmressetmodifypc_      kspfgmressetmodifypc
#define kspfgmresmodifypcnochange_ kspfgmresmodifypcnochange
#define kspfgmresmodifypcksp_      kspfgmresmodifypcksp
#endif

static PetscErrorCode ourmodify(KSP ksp,PetscInt i,PetscInt i2,PetscReal d,void* ctx)
{
  PetscErrorCode ierr = 0;
  (*(void (PETSC_STDCALL *)(KSP*,PetscInt*,PetscInt*,PetscReal*,void *,PetscErrorCode*))(((PetscObject)ksp)->fortran_func_pointers[0]))(&ksp,&i,&i2,&d,(void*)((PetscObject)ksp)->fortran_func_pointers[2],&ierr);CHKERRQ(ierr);
  return 0;
}

static PetscErrorCode ourmoddestroy(void* ctx)
{
  PetscErrorCode ierr = 0;
  KSP            ksp = (KSP) ctx;
  (*(void (PETSC_STDCALL *)(void*,PetscErrorCode*))(((PetscObject)ksp)->fortran_func_pointers[1]))((void*)((PetscObject)ksp)->fortran_func_pointers[2],&ierr);CHKERRQ(ierr);
  return 0;
}

EXTERN_C_BEGIN
extern void PETSC_STDCALL kspfgmresmodifypcnochange_(KSP*,PetscInt*,PetscInt*,PetscReal*,void*,PetscErrorCode*);
extern void PETSC_STDCALL kspfgmresmodifypcksp_(KSP*,PetscInt*,PetscInt*,PetscReal*,void*,PetscErrorCode*);

void PETSC_STDCALL kspfgmressetmodifypc_(KSP *ksp,void (PETSC_STDCALL *fcn)(KSP*,PetscInt*,PetscInt*,PetscReal*,void*,PetscErrorCode*),void* ctx,void (PETSC_STDCALL *d)(void*,PetscErrorCode*),PetscErrorCode *ierr)
{
  PetscObjectAllocateFortranPointers(*ksp,3);
  if ((PetscVoidFunction)fcn == (PetscVoidFunction)kspfgmresmodifypcksp_) {
    *ierr = KSPFGMRESSetModifyPC(*ksp,KSPFGMRESModifyPCKSP,0,0);
  } else if ((PetscVoidFunction)fcn == (PetscVoidFunction)kspfgmresmodifypcnochange_) {
    *ierr = KSPFGMRESSetModifyPC(*ksp,KSPFGMRESModifyPCNoChange,0,0);
  } else {
    ((PetscObject)*ksp)->fortran_func_pointers[0] = (PetscVoidFunction)fcn;
    ((PetscObject)*ksp)->fortran_func_pointers[2] = (PetscVoidFunction)ctx;
    if (FORTRANNULLFUNCTION(d)) {
      *ierr = KSPFGMRESSetModifyPC(*ksp,ourmodify,*ksp,0);
    } else {
    ((PetscObject)*ksp)->fortran_func_pointers[1] = (PetscVoidFunction)d;
      *ierr = KSPFGMRESSetModifyPC(*ksp,ourmodify,*ksp,ourmoddestroy);
    }
  }
}

EXTERN_C_END
