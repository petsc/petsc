
#include "src/snes/snesimpl.h"     /*I  "petscsnes.h"  I*/

EXTERN_C_BEGIN
EXTERN int SNESCreate_LS(SNES);
EXTERN int SNESCreate_TR(SNES);
EXTERN int SNESCreate_Test(SNES);
EXTERN_C_END
  
/*
      This is used by SNESSetType() to make sure that at least one 
    SNESRegisterAll() is called. In general, if there is more than one
    DLL then SNESRegisterAll() may be called several times.
*/
extern PetscTruth SNESRegisterAllCalled;

#undef __FUNCT__  
#define __FUNCT__ "SNESRegisterAll"
/*@C
   SNESRegisterAll - Registers all of the nonlinear solver methods in the SNES package.

   Not Collective

   Level: advanced

.keywords: SNES, register, all

.seealso:  SNESRegisterDestroy()
@*/
int SNESRegisterAll(const char path[])
{
  int ierr;

  PetscFunctionBegin;
  SNESRegisterAllCalled = PETSC_TRUE;

  ierr = SNESRegisterDynamic("ls",   path,"SNESCreate_LS",SNESCreate_LS);CHKERRQ(ierr);
  ierr = SNESRegisterDynamic("tr",   path,"SNESCreate_TR",SNESCreate_TR);CHKERRQ(ierr);
  ierr = SNESRegisterDynamic("test", path,"SNESCreate_Test", SNESCreate_Test);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

