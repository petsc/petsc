#include <private/linesearchimpl.h>     /*I  "petsclinesearch.h"  I*/

EXTERN_C_BEGIN
extern PetscErrorCode  LineSearchCreate_Basic(LineSearch);
extern PetscErrorCode  LineSearchCreate_L2(LineSearch);
extern PetscErrorCode  LineSearchCreate_CP(LineSearch);
extern PetscErrorCode  LineSearchCreate_Shell(LineSearch);
EXTERN_C_END

/*
extern PetscErrorCode  LineSearchCreate_Cubic(LineSearch);
extern PetscErrorCode  LineSearchCreate_Quadratic(LineSearch);
extern PetscErrorCode  LineSearchCreate_BasicNoNorms(LineSearch);
extern PetscErrorCode  LineSearchCreate_QuadraticSecant(LineSearch);
extern PetscErrorCode  LineSearchCreate_CriticalSecant(LineSearch);
 */


#undef __FUNCT__
#define __FUNCT__ "LineSearchRegisterAll"
/*@C
   LineSearchRegisterAll - Registers all of the nonlinear solver methods in the LineSearch package.

   Not Collective

   Level: advanced

.keywords: LineSearch, register, all

.seealso:  LineSearchRegisterDestroy()
@*/
PetscErrorCode LineSearchRegisterAll(const char path[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  LineSearchRegisterAllCalled = PETSC_TRUE;
  /*
  ierr = LineSearchRegisterDynamic(LINESEARCHCUBIC,             path,"LineSearchCreate_Cubic",             LineSearchCreate_Cubic);CHKERRQ(ierr);

  ierr = LineSearchRegisterDynamic(LINESEARCHQUADRATIC,         path,"LineSearchCreate_Quadratic",         LineSearchCreate_Quadratic);CHKERRQ(ierr);
  ierr = LineSearchRegisterDynamic(LINESEARCHCRITICALSECANT,    path,"LineSearchCreate_CriticalSecant",    LineSearchCreate_CriticalSecant);CHKERRQ(ierr);

   */
  ierr = LineSearchRegisterDynamic(LINESEARCHSHELL,             path,"LineSearchCreate_Shell",             LineSearchCreate_Shell);CHKERRQ(ierr);
  ierr = LineSearchRegisterDynamic(LINESEARCHBASIC,             path,"LineSearchCreate_Basic",             LineSearchCreate_Basic);CHKERRQ(ierr);
  ierr = LineSearchRegisterDynamic(LINESEARCHL2,                path,"LineSearchCreate_L2",                LineSearchCreate_L2);CHKERRQ(ierr);
  ierr = LineSearchRegisterDynamic(LINESEARCHCP,                path,"LineSearchCreate_CP",                LineSearchCreate_CP);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "LineSearchRegisterDestroy"
PetscErrorCode  LineSearchRegisterDestroy(void)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFListDestroy(&LineSearchList);CHKERRQ(ierr);
  LineSearchRegisterAllCalled = PETSC_FALSE;
  PetscFunctionReturn(0);
}
