/*$Id: mfregis.c,v 1.14 2001/03/23 23:24:10 balay Exp $*/

#include "src/snes/mf/snesmfj.h"   /*I  "petscsnes.h"   I*/

EXTERN_C_BEGIN
EXTERN int MatSNESMFCreate_Default(MatSNESMFCtx);
EXTERN int MatSNESMFCreate_WP(MatSNESMFCtx);
EXTERN_C_END

#undef __FUNCT__  
#define __FUNCT__ "MatSNESMFRegisterAll"
/*@C
  MatSNESMFRegisterAll - Registers all of the compute-h in the MatSNESMF package.

  Not Collective

  Level: developer

.keywords: MatSNESMF, register, all

.seealso:  MatSNESMFRegisterDestroy(), MatSNESMFRegisterDynamic), MatSNESMFCreate(), 
           MatSNESMFSetType()
@*/
int MatSNESMFRegisterAll(const char *path)
{
  int ierr;

  PetscFunctionBegin;
  MatSNESMFRegisterAllCalled = PETSC_TRUE;

  ierr = MatSNESMFRegisterDynamic(MATSNESMF_DEFAULT,path,"MatSNESMFCreate_Default",MatSNESMFCreate_Default);CHKERRQ(ierr);
  ierr = MatSNESMFRegisterDynamic(MATSNESMF_WP,path,"MatSNESMFCreate_WP",MatSNESMFCreate_WP);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

