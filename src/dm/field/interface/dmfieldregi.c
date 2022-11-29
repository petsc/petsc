#include <petsc/private/dmfieldimpl.h> /*I  "petscdmfield.h"  I*/

PETSC_EXTERN PetscErrorCode DMFieldCreate_DA(DMField);
PETSC_EXTERN PetscErrorCode DMFieldCreate_DS(DMField);
PETSC_EXTERN PetscErrorCode DMFieldCreate_Shell(DMField);

PetscFunctionList DMFieldList;

/*@C
   DMFieldRegisterAll - Registers all the `DMField` implementations

   Not Collective

   Level: advanced

.seealso: `DMField`, `DMFieldRegisterDestroy()`
@*/
PetscErrorCode DMFieldRegisterAll(void)
{
  PetscFunctionBegin;
  if (DMFieldRegisterAllCalled) PetscFunctionReturn(0);
  DMFieldRegisterAllCalled = PETSC_TRUE;
  PetscCall(DMFieldRegister(DMFIELDDA, DMFieldCreate_DA));
  PetscCall(DMFieldRegister(DMFIELDDS, DMFieldCreate_DS));
  PetscCall(DMFieldRegister(DMFIELDSHELL, DMFieldCreate_Shell));
  PetscFunctionReturn(0);
}

/*@C
  DMFieldRegister  - Adds an implementation of the `DMField` object.

   Not collective

   Input Parameters:
+  name_impl - name of a new user-defined implementation
-  routine_create - routine to create method context

   Sample usage:
.vb
   DMFieldRegister("my_impl",MyImplCreate);
.ve

   Then, this implementation can be chosen with the procedural interface via
$     DMFieldSetType(tagger,"my_impl")

   Level: advanced

   Note:
   `DMFieldRegister()` may be called multiple times to add several user-defined implementations.

.seealso: `DMField`, `DMFieldRegisterAll()`, `DMFieldRegisterDestroy()`
@*/
PetscErrorCode DMFieldRegister(const char sname[], PetscErrorCode (*function)(DMField))
{
  PetscFunctionBegin;
  PetscCall(PetscFunctionListAdd(&DMFieldList, sname, function));
  PetscFunctionReturn(0);
}
