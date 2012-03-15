#include <private/linesearchimpl.h>     /*I  "petscsnes.h"  I*/

EXTERN_C_BEGIN
extern PetscErrorCode  PetscLineSearchCreate_Basic(PetscLineSearch);
extern PetscErrorCode  PetscLineSearchCreate_L2(PetscLineSearch);
extern PetscErrorCode  PetscLineSearchCreate_CP(PetscLineSearch);
extern PetscErrorCode  PetscLineSearchCreate_BT(PetscLineSearch);
extern PetscErrorCode  PetscLineSearchCreate_Shell(PetscLineSearch);
EXTERN_C_END

/*
extern PetscErrorCode  PetscLineSearchCreate_Cubic(PetscLineSearch);
extern PetscErrorCode  PetscLineSearchCreate_Quadratic(PetscLineSearch);
extern PetscErrorCode  PetscLineSearchCreate_BasicNoNorms(PetscLineSearch);
extern PetscErrorCode  PetscLineSearchCreate_QuadraticSecant(PetscLineSearch);
extern PetscErrorCode  PetscLineSearchCreate_CriticalSecant(PetscLineSearch);
 */


#undef __FUNCT__
#define __FUNCT__ "PetscLineSearchRegisterAll"
/*@C
   PetscLineSearchRegisterAll - Registers all of the nonlinear solver methods in the PetscLineSearch package.

   Not Collective

   Level: advanced

.keywords: PetscLineSearch, register, all

.seealso:  PetscLineSearchRegisterDestroy()
@*/
PetscErrorCode PetscLineSearchRegisterAll(const char path[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscLineSearchRegisterAllCalled = PETSC_TRUE;
  /*
  ierr = PetscLineSearchRegisterDynamic(LINESEARCHCUBIC,             path,"PetscLineSearchCreate_Cubic",             PetscLineSearchCreate_Cubic);CHKERRQ(ierr);

  ierr = PetscLineSearchRegisterDynamic(LINESEARCHQUADRATIC,         path,"PetscLineSearchCreate_Quadratic",         PetscLineSearchCreate_Quadratic);CHKERRQ(ierr);
  ierr = PetscLineSearchRegisterDynamic(LINESEARCHCRITICALSECANT,    path,"PetscLineSearchCreate_CriticalSecant",    PetscLineSearchCreate_CriticalSecant);CHKERRQ(ierr);

   */
  ierr = PetscLineSearchRegisterDynamic(PETSCLINESEARCHSHELL,             path,"PetscLineSearchCreate_Shell",             PetscLineSearchCreate_Shell);CHKERRQ(ierr);
  ierr = PetscLineSearchRegisterDynamic(PETSCLINESEARCHBASIC,             path,"PetscLineSearchCreate_Basic",             PetscLineSearchCreate_Basic);CHKERRQ(ierr);
  ierr = PetscLineSearchRegisterDynamic(PETSCLINESEARCHL2,                path,"PetscLineSearchCreate_L2",                PetscLineSearchCreate_L2);CHKERRQ(ierr);
  ierr = PetscLineSearchRegisterDynamic(PETSCLINESEARCHBT,                path,"PetscLineSearchCreate_BT",                PetscLineSearchCreate_BT);CHKERRQ(ierr);
  ierr = PetscLineSearchRegisterDynamic(PETSCLINESEARCHCP,                path,"PetscLineSearchCreate_CP",                PetscLineSearchCreate_CP);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscLineSearchRegisterDestroy"
PetscErrorCode  PetscLineSearchRegisterDestroy(void)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFListDestroy(&PetscLineSearchList);CHKERRQ(ierr);
  PetscLineSearchRegisterAllCalled = PETSC_FALSE;
  PetscFunctionReturn(0);
}
