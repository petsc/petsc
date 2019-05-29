#include <petsc/private/vecimpl.h>     /*I  "petscvec.h"  I*/

PETSC_EXTERN PetscErrorCode VecTaggerCreate_Absolute(VecTagger);
PETSC_EXTERN PetscErrorCode VecTaggerCreate_Relative(VecTagger);
PETSC_EXTERN PetscErrorCode VecTaggerCreate_CDF(VecTagger);
PETSC_EXTERN PetscErrorCode VecTaggerCreate_Or(VecTagger);
PETSC_EXTERN PetscErrorCode VecTaggerCreate_And(VecTagger);

PetscFunctionList VecTaggerList;

/*@C
   VecTaggerRegisterAll - Registers all the VecTagger communication implementations

   Not Collective

   Level: advanced

.seealso:  VecTaggerRegisterDestroy()
@*/
PetscErrorCode  VecTaggerRegisterAll(void)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (VecTaggerRegisterAllCalled) PetscFunctionReturn(0);
  VecTaggerRegisterAllCalled = PETSC_TRUE;
  ierr = VecTaggerRegister(VECTAGGERABSOLUTE, VecTaggerCreate_Absolute);CHKERRQ(ierr);
  ierr = VecTaggerRegister(VECTAGGERRELATIVE, VecTaggerCreate_Relative);CHKERRQ(ierr);
  ierr = VecTaggerRegister(VECTAGGERCDF,      VecTaggerCreate_CDF);CHKERRQ(ierr);
  ierr = VecTaggerRegister(VECTAGGEROR,       VecTaggerCreate_Or);CHKERRQ(ierr);
  ierr = VecTaggerRegister(VECTAGGERAND,      VecTaggerCreate_And);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
  VecTaggerRegister  - Adds an implementation of the VecTagger communication protocol.

   Not collective

   Input Parameters:
+  name_impl - name of a new user-defined implementation
-  routine_create - routine to create method context

   Notes:
   VecTaggerRegister() may be called multiple times to add several user-defined implementations.

   Sample usage:
.vb
   VecTaggerRegister("my_impl",MyImplCreate);
.ve

   Then, this implementation can be chosen with the procedural interface via
$     VecTaggerSetType(tagger,"my_impl")
   or at runtime via the option
$     -snes_type my_solver

   Level: advanced

.seealso: VecTaggerRegisterAll(), VecTaggerRegisterDestroy()
@*/
PetscErrorCode  VecTaggerRegister(const char sname[],PetscErrorCode (*function)(VecTagger))
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFunctionListAdd(&VecTaggerList,sname,function);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

