#include <petsc/private/vecimpl.h> /*I  "petscvec.h"  I*/

PETSC_EXTERN PetscErrorCode VecTaggerCreate_Absolute(VecTagger);
PETSC_EXTERN PetscErrorCode VecTaggerCreate_Relative(VecTagger);
PETSC_EXTERN PetscErrorCode VecTaggerCreate_CDF(VecTagger);
PETSC_EXTERN PetscErrorCode VecTaggerCreate_Or(VecTagger);
PETSC_EXTERN PetscErrorCode VecTaggerCreate_And(VecTagger);

PetscFunctionList VecTaggerList;

/*@C
   VecTaggerRegisterAll - Registers all the `VecTagger` communication implementations

   Not Collective

   Level: advanced

.seealso: `VecTaggerRegisterDestroy()`
@*/
PetscErrorCode VecTaggerRegisterAll(void)
{
  PetscFunctionBegin;
  if (VecTaggerRegisterAllCalled) PetscFunctionReturn(PETSC_SUCCESS);
  VecTaggerRegisterAllCalled = PETSC_TRUE;
  PetscCall(VecTaggerRegister(VECTAGGERABSOLUTE, VecTaggerCreate_Absolute));
  PetscCall(VecTaggerRegister(VECTAGGERRELATIVE, VecTaggerCreate_Relative));
  PetscCall(VecTaggerRegister(VECTAGGERCDF, VecTaggerCreate_CDF));
  PetscCall(VecTaggerRegister(VECTAGGEROR, VecTaggerCreate_Or));
  PetscCall(VecTaggerRegister(VECTAGGERAND, VecTaggerCreate_And));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  VecTaggerRegister  - Adds an implementation of the `VecTagger` communication protocol.

   Not Collective

   Input Parameters:
+  sname - name of a new user-defined implementation
-  functions - routine to create method context

   Level: advanced

   Notes:
   `VecTaggerRegister()` may be called multiple times to add several user-defined implementations.

   Sample usage:
.vb
   VecTaggerRegister("my_impl",MyImplCreate);
.ve

   Then, this implementation can be chosen with the procedural interface via
$     VecTaggerSetType(tagger,"my_impl")
   or at runtime via the option
$     -snes_type my_solver

.seealso: `VecTaggerRegisterAll()`, `VecTaggerRegisterDestroy()`
@*/
PetscErrorCode VecTaggerRegister(const char sname[], PetscErrorCode (*function)(VecTagger))
{
  PetscFunctionBegin;
  PetscCall(PetscFunctionListAdd(&VecTaggerList, sname, function));
  PetscFunctionReturn(PETSC_SUCCESS);
}
