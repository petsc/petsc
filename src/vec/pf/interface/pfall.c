
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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (PFRegisterAllCalled) PetscFunctionReturn(0);
  PFRegisterAllCalled = PETSC_TRUE;

  ierr = PFRegister(PFCONSTANT,         PFCreate_Constant);CHKERRQ(ierr);
  ierr = PFRegister(PFSTRING,           PFCreate_String);CHKERRQ(ierr);
  ierr = PFRegister(PFQUICK,            PFCreate_Quick);CHKERRQ(ierr);
  ierr = PFRegister(PFIDENTITY,         PFCreate_Identity);CHKERRQ(ierr);
#if defined(PETSC_HAVE_MATLAB_ENGINE)
  ierr = PFRegister(PFMATLAB,           PFCreate_Matlab);CHKERRQ(ierr);
#endif
  PetscFunctionReturn(0);
}


