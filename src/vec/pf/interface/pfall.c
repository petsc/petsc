
#include <petscpf.h>          /*I   "petscpf.h"   I*/

PETSC_EXTERN PetscErrorCode PFCreate_Constant(PF,void*);
PETSC_EXTERN PetscErrorCode PFCreate_String(PF,void*);
PETSC_EXTERN PetscErrorCode PFCreate_Quick(PF,void*);
PETSC_EXTERN PetscErrorCode PFCreate_Identity(PF,void*);
#if defined(PETSC_HAVE_MATLAB_ENGINE)
PETSC_EXTERN PetscErrorCode PFCreate_Matlab(PF,void*);
#endif

#undef __FUNCT__
#define __FUNCT__ "PFRegisterAll"
/*@C
   PFRegisterAll - Registers all of the preconditioners in the PF package.

   Not Collective

   Input Parameter:
.  path - the library where the routines are to be found (optional)

   Level: advanced

.keywords: PF, register, all

.seealso: PFRegister(), PFRegisterDestroy()
@*/
PetscErrorCode  PFRegisterAll(const char path[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PFRegisterAllCalled = PETSC_TRUE;

  ierr = PFRegister(PFCONSTANT         ,path,"PFCreate_Constant",PFCreate_Constant);CHKERRQ(ierr);
  ierr = PFRegister(PFSTRING           ,path,"PFCreate_String",PFCreate_String);CHKERRQ(ierr);
  ierr = PFRegister(PFQUICK            ,path,"PFCreate_Quick",PFCreate_Quick);CHKERRQ(ierr);
  ierr = PFRegister(PFIDENTITY         ,path,"PFCreate_Identity",PFCreate_Identity);CHKERRQ(ierr);
#if defined(PETSC_HAVE_MATLAB_ENGINE)
  ierr = PFRegister(PFMATLAB           ,path,"PFCreate_Matlab",PFCreate_Matlab);CHKERRQ(ierr);
#endif
  PetscFunctionReturn(0);
}


