#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: mfregis.c,v 1.2 1998/11/12 03:43:23 bsmith Exp bsmith $";
#endif

#include "src/snes/mf/snesmfj.h"   /*I  "snes.h"   I*/

EXTERN_C_BEGIN
extern int MatSNESFDMFCreate_Default(MatSNESFDMFCtx);
extern int MatSNESFDMFCreate_WP(MatSNESFDMFCtx);
EXTERN_C_END

#undef __FUNC__  
#define __FUNC__ "MatSNESFDMFRegisterAll"
/*@C
  MatSNESFDMFRegisterAll - Registers all of the compute-h in the MatSNESFDMF package.

  Not Collective

  Level: developer

.keywords: MatSNESFDMF, register, all

.seealso:  MatSNESFDMFRegisterDestroy(), MatSNESFDMFRegister(), MatSNESFDMFCreate(), 
           MatSNESFDMFSetType()
@*/
int MatSNESFDMFRegisterAll(char *path)
{
  int ierr;

  PetscFunctionBegin;
  MatSNESFDMFRegisterAllCalled = 1;

  ierr = MatSNESFDMFRegister("default",path,"MatSNESFDMFCreate_Default",MatSNESFDMFCreate_Default);CHKERRQ(ierr);
  ierr = MatSNESFDMFRegister("wp",path,"MatSNESFDMFCreate_WP",MatSNESFDMFCreate_WP);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

