#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: snesregi.c,v 1.26 1998/04/21 19:37:00 curfman Exp curfman $";
#endif

#include "src/snes/snesimpl.h"     /*I  "snes.h"  I*/
extern int SNESCreate_EQ_LS(SNES);
extern int SNESCreate_EQ_TR(SNES);
extern int SNESCreate_UM_TR(SNES);
extern int SNESCreate_UM_LS(SNES);
extern int SNESCreate_Test(SNES);
  
/*
      This is used by SNESSetType() to make sure that at least one 
    SNESRegisterAll() is called. In general, if there is more than one
    DLL then SNESRegisterAll() may be called several times.
*/
extern int SNESRegisterAllCalled;

#undef __FUNC__  
#define __FUNC__ "SNESRegisterAll"
/*@C
  SNESRegisterAll - Registers all of the nonlinear solver methods in the SNES package.

  Not Collective

.keywords: SNES, register, all

.seealso:  SNESRegisterDestroy()
@*/
int SNESRegisterAll(char *path)
{
  int ierr;

  PetscFunctionBegin;
  SNESRegisterAllCalled = 1;

  ierr = SNESRegister("ls",   path,"SNESCreate_EQ_LS",SNESCreate_EQ_LS);CHKERRQ(ierr);
  ierr = SNESRegister("tr",   path,"SNESCreate_EQ_TR",SNESCreate_EQ_TR);CHKERRQ(ierr);
  ierr = SNESRegister("test", path,"SNESCreate_Test", SNESCreate_Test);CHKERRQ(ierr);
  ierr = SNESRegister("umtr", path,"SNESCreate_UM_TR",SNESCreate_UM_TR);CHKERRQ(ierr);
  ierr = SNESRegister("umls", path,"SNESCreate_UM_LS",SNESCreate_UM_LS);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

