/*$Id: pfall.c,v 1.1 2000/01/25 00:49:51 bsmith Exp bsmith $*/

#include "pf.h"          /*I   "pf.h"   I*/

EXTERN_C_BEGIN
extern int PFCreate_Constant(PF,void*);
EXTERN_C_END

#undef __FUNC__  
#define __FUNC__ "PFRegisterAll"
/*@C
   PFRegisterAll - Registers all of the preconditioners in the PF package.

   Not Collective

   Input Parameter:
.  path - the library where the routines are to be found (optional)

   Level: advanced

.keywords: PF, register, all

.seealso: PFRegisterDynamic(), PFRegisterDestroy()
@*/
int PFRegisterAll(char *path)
{
  int ierr;

  PetscFunctionBegin;
  PFRegisterAllCalled = PETSC_TRUE;

  ierr = PFRegisterDynamic(PFCONSTANT         ,path,"PFCreate_Constant",PFCreate_Constant);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


