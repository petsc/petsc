/*$Id: pfall.c,v 1.5 2000/04/12 04:26:58 bsmith Exp balay $*/

#include "petscpf.h"          /*I   "petscpf.h"   I*/

EXTERN_C_BEGIN
extern int PFCreate_Constant(PF,void*);
extern int PFCreate_String(PF,void*);
extern int PFCreate_Quick(PF,void*);
EXTERN_C_END

#undef __FUNC__  
#define __FUNC__ /*<a name=""></a>*/"PFRegisterAll"
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
  ierr = PFRegisterDynamic(PFSTRING           ,path,"PFCreate_String",PFCreate_String);CHKERRQ(ierr);
  ierr = PFRegisterDynamic(PFQUICK            ,path,"PFCreate_Quick",PFCreate_Quick);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


