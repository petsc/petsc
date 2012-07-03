#include <petsc-private/linesearchimpl.h>     /*I  "petscsnes.h"  I*/

EXTERN_C_BEGIN
extern PetscErrorCode  SNESLineSearchCreate_Basic(SNESLineSearch);
extern PetscErrorCode  SNESLineSearchCreate_L2(SNESLineSearch);
extern PetscErrorCode  SNESLineSearchCreate_CP(SNESLineSearch);
extern PetscErrorCode  SNESLineSearchCreate_BT(SNESLineSearch);
extern PetscErrorCode  SNESLineSearchCreate_Shell(SNESLineSearch);
EXTERN_C_END

/*
extern PetscErrorCode  SNESLineSearchCreate_Cubic(SNESLineSearch);
extern PetscErrorCode  SNESLineSearchCreate_Quadratic(SNESLineSearch);
extern PetscErrorCode  SNESLineSearchCreate_BasicNoNorms(SNESLineSearch);
extern PetscErrorCode  SNESLineSearchCreate_QuadraticSecant(SNESLineSearch);
extern PetscErrorCode  SNESLineSearchCreate_CriticalSecant(SNESLineSearch);
 */


#undef __FUNCT__
#define __FUNCT__ "SNESLineSearchRegisterAll"
/*@C
   SNESLineSearchRegisterAll - Registers all of the nonlinear solver methods in the SNESLineSearch package.

   Not Collective

   Level: advanced

.keywords: SNESLineSearch, register, all

.seealso:  SNESLineSearchRegisterDestroy()
@*/
PetscErrorCode SNESLineSearchRegisterAll(const char path[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  SNESLineSearchRegisterAllCalled = PETSC_TRUE;
  /*
  ierr = SNESLineSearchRegisterDynamic(LINESEARCHCUBIC,             path,"SNESLineSearchCreate_Cubic",             SNESLineSearchCreate_Cubic);CHKERRQ(ierr);

  ierr = SNESLineSearchRegisterDynamic(LINESEARCHQUADRATIC,         path,"SNESLineSearchCreate_Quadratic",         SNESLineSearchCreate_Quadratic);CHKERRQ(ierr);
  ierr = SNESLineSearchRegisterDynamic(LINESEARCHCRITICALSECANT,    path,"SNESLineSearchCreate_CriticalSecant",    SNESLineSearchCreate_CriticalSecant);CHKERRQ(ierr);

   */
  ierr = SNESLineSearchRegisterDynamic(SNESLINESEARCHSHELL,             path,"SNESLineSearchCreate_Shell",             SNESLineSearchCreate_Shell);CHKERRQ(ierr);
  ierr = SNESLineSearchRegisterDynamic(SNESLINESEARCHBASIC,             path,"SNESLineSearchCreate_Basic",             SNESLineSearchCreate_Basic);CHKERRQ(ierr);
  ierr = SNESLineSearchRegisterDynamic(SNESLINESEARCHL2,                path,"SNESLineSearchCreate_L2",                SNESLineSearchCreate_L2);CHKERRQ(ierr);
  ierr = SNESLineSearchRegisterDynamic(SNESLINESEARCHBT,                path,"SNESLineSearchCreate_BT",                SNESLineSearchCreate_BT);CHKERRQ(ierr);
  ierr = SNESLineSearchRegisterDynamic(SNESLINESEARCHCP,                path,"SNESLineSearchCreate_CP",                SNESLineSearchCreate_CP);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SNESLineSearchRegisterDestroy"
PetscErrorCode  SNESLineSearchRegisterDestroy(void)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFListDestroy(&SNESLineSearchList);CHKERRQ(ierr);
  SNESLineSearchRegisterAllCalled = PETSC_FALSE;
  PetscFunctionReturn(0);
}
