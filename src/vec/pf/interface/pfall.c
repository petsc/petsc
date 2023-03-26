
#include <petscpf.h> /*I   "petscpf.h"   I*/
#include <../src/vec/pf/pfimpl.h>

PETSC_EXTERN PetscErrorCode PFCreate_Constant(PF, void *);
PETSC_EXTERN PetscErrorCode PFCreate_Quick(PF, void *);
PETSC_EXTERN PetscErrorCode PFCreate_Identity(PF, void *);
#if defined(PETSC_HAVE_POPEN) && defined(PETSC_USE_SHARED_LIBRARIES) && defined(PETSC_HAVE_DYNAMIC_LIBRARIES)
PETSC_EXTERN PetscErrorCode PFCreate_String(PF, void *);
#endif
#if defined(PETSC_HAVE_MATLAB)
PETSC_EXTERN PetscErrorCode PFCreate_Matlab(PF, void *);
#endif

/*@C
   PFRegisterAll - Registers all of the preconditioners in the PF package.

   Not Collective

   Level: advanced

.seealso: `PFRegister()`, `PFRegisterDestroy()`
@*/
PetscErrorCode PFRegisterAll(void)
{
  PetscFunctionBegin;
  if (PFRegisterAllCalled) PetscFunctionReturn(PETSC_SUCCESS);
  PFRegisterAllCalled = PETSC_TRUE;

  PetscCall(PFRegister(PFCONSTANT, PFCreate_Constant));
  PetscCall(PFRegister(PFQUICK, PFCreate_Quick));
  PetscCall(PFRegister(PFIDENTITY, PFCreate_Identity));
#if defined(PETSC_HAVE_POPEN) && defined(PETSC_USE_SHARED_LIBRARIES) && defined(PETSC_HAVE_DYNAMIC_LIBRARIES)
  PetscCall(PFRegister(PFSTRING, PFCreate_String));
#endif
#if defined(PETSC_HAVE_MATLAB)
  PetscCall(PFRegister(PFMATLAB, PFCreate_Matlab));
#endif
  PetscFunctionReturn(PETSC_SUCCESS);
}
