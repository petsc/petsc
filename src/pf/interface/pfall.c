/*$Id: pfall.c,v 1.14 2001/08/06 21:19:08 bsmith Exp $*/

#include "petscpf.h"          /*I   "petscpf.h"   I*/

EXTERN_C_BEGIN
EXTERN int PFCreate_Constant(PF,void*);
EXTERN int PFCreate_String(PF,void*);
EXTERN int PFCreate_Quick(PF,void*);
EXTERN int PFCreate_Identity(PF,void*);
#if defined(PETSC_HAVE_MATLAB) && !defined(PETSC_USE_COMPLEX) && !defined(PETSC_USE_SINGLE)
EXTERN int PFCreate_Matlab(PF,void*);
#endif
EXTERN_C_END

#undef __FUNCT__  
#define __FUNCT__ "PFRegisterAll"
/*@C
   PFRegisterAll - Registers all of the preconditioners in the PF package.

   Not Collective

   Input Parameter:
.  path - the library where the routines are to be found (optional)

   Level: advanced

.keywords: PF, register, all

.seealso: PFRegisterDynamic(), PFRegisterDestroy()
@*/
int PFRegisterAll(const char path[])
{
  int ierr;

  PetscFunctionBegin;
  PFRegisterAllCalled = PETSC_TRUE;

  ierr = PFRegisterDynamic(PFCONSTANT         ,path,"PFCreate_Constant",PFCreate_Constant);CHKERRQ(ierr);
  ierr = PFRegisterDynamic(PFSTRING           ,path,"PFCreate_String",PFCreate_String);CHKERRQ(ierr);
  ierr = PFRegisterDynamic(PFQUICK            ,path,"PFCreate_Quick",PFCreate_Quick);CHKERRQ(ierr);
  ierr = PFRegisterDynamic(PFIDENTITY         ,path,"PFCreate_Identity",PFCreate_Identity);CHKERRQ(ierr);
#if defined(PETSC_HAVE_MATLAB) && !defined(PETSC_USE_COMPLEX) && !defined(PETSC_USE_SINGLE)
  ierr = PFRegisterDynamic(PFMATLAB           ,path,"PFCreate_Matlab",PFCreate_Matlab);CHKERRQ(ierr);
#endif
  PetscFunctionReturn(0);
}


