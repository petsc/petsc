#include "zpetsc.h"
#include "petscksp.h"

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define kspfgmressetmodifypc_      KSPFGMRESSETMODIFYPC
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define kspfgmressetmodifypc_      kspfgmressetmodifypc
#endif

EXTERN_C_BEGIN
static void (PETSC_STDCALL *f109)(KSP*,PetscInt*,PetscInt*,PetscReal*,void*,PetscErrorCode*);
static void (PETSC_STDCALL *f210)(void*,PetscErrorCode*);
EXTERN_C_END

static PetscErrorCode ourmodify(KSP ksp,PetscInt i,PetscInt i2,PetscReal d,void* ctx)
{
  PetscErrorCode ierr = 0;
  (*f109)(&ksp,&i,&i2,&d,ctx,&ierr);CHKERRQ(ierr);
  return 0;
}

static PetscErrorCode ourmoddestroy(void* ctx)
{
  PetscErrorCode ierr = 0;
  (*f210)(ctx,&ierr);CHKERRQ(ierr);
  return 0;
}

EXTERN_C_BEGIN
extern void PETSC_STDCALL kspfgmresmodifypcnochange_(KSP*,PetscInt*,PetscInt*,PetscReal*,void*,PetscErrorCode*);
extern void PETSC_STDCALL kspfgmresmodifypcksp_(KSP*,PetscInt*,PetscInt*,PetscReal*,void*,PetscErrorCode*);

void PETSC_STDCALL kspfgmressetmodifypc_(KSP *ksp,void (PETSC_STDCALL *fcn)(KSP*,PetscInt*,PetscInt*,PetscReal*,void*,PetscErrorCode*),void* ctx,void (PETSC_STDCALL *d)(void*,PetscErrorCode*),PetscErrorCode *ierr)
{
  if ((FCNVOID)fcn == (FCNVOID)kspfgmresmodifypcksp_) {
    *ierr = KSPFGMRESSetModifyPC(*ksp,KSPFGMRESModifyPCKSP,0,0);
  } else if ((FCNVOID)fcn == (FCNVOID)kspfgmresmodifypcnochange_) {
    *ierr = KSPFGMRESSetModifyPC(*ksp,KSPFGMRESModifyPCNoChange,0,0);
  } else {
    f109 = fcn;
    if (FORTRANNULLFUNCTION(d)) {
      *ierr = KSPFGMRESSetModifyPC(*ksp,ourmodify,ctx,0);
    } else {
      f210 = d;
      *ierr = KSPFGMRESSetModifyPC(*ksp,ourmodify,ctx,ourmoddestroy);
    }
  }
}

EXTERN_C_END
