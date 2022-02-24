
#include <petscpf.h>          /*I   "petscpf.h"   I*/
#include <../src/vec/pf/pfimpl.h>

PETSC_EXTERN PetscErrorCode PFCreate_Constant(PF,void*);
PETSC_EXTERN PetscErrorCode PFCreate_String(PF,void*);
PETSC_EXTERN PetscErrorCode PFCreate_Quick(PF,void*);
PETSC_EXTERN PetscErrorCode PFCreate_Identity(PF,void*);
#if defined(PETSC_HAVE_MATLAB_ENGINE)
PETSC_EXTERN PetscErrorCode PFCreate_Matlab(PF,void*);
#endif

/*@C
   PFRegisterAll - Registers all of the preconditioners in the PF package.

   Not Collective

   Level: advanced

.seealso: PFRegister(), PFRegisterDestroy()
@*/
PetscErrorCode  PFRegisterAll(void)
{
  PetscFunctionBegin;
  if (PFRegisterAllCalled) PetscFunctionReturn(0);
  PFRegisterAllCalled = PETSC_TRUE;

  CHKERRQ(PFRegister(PFCONSTANT,         PFCreate_Constant));
  CHKERRQ(PFRegister(PFSTRING,           PFCreate_String));
  CHKERRQ(PFRegister(PFQUICK,            PFCreate_Quick));
  CHKERRQ(PFRegister(PFIDENTITY,         PFCreate_Identity));
#if defined(PETSC_HAVE_MATLAB_ENGINE)
  CHKERRQ(PFRegister(PFMATLAB,           PFCreate_Matlab));
#endif
  PetscFunctionReturn(0);
}
