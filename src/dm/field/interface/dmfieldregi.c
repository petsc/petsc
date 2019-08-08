#include <petsc/private/dmfieldimpl.h>     /*I  "petscdmfield.h"  I*/

PETSC_EXTERN PetscErrorCode DMFieldCreate_DA(DMField);
PETSC_EXTERN PetscErrorCode DMFieldCreate_DS(DMField);
PETSC_EXTERN PetscErrorCode DMFieldCreate_Shell(DMField);

PetscFunctionList DMFieldList;

/*@C
   DMFieldRegisterAll - Registers all the DMField implementations

   Not Collective

   Level: advanced

.seealso:  DMFieldRegisterDestroy()
@*/
PetscErrorCode  DMFieldRegisterAll(void)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (DMFieldRegisterAllCalled) PetscFunctionReturn(0);
  DMFieldRegisterAllCalled = PETSC_TRUE;
  ierr = DMFieldRegister(DMFIELDDA,    DMFieldCreate_DA);CHKERRQ(ierr);
  ierr = DMFieldRegister(DMFIELDDS,    DMFieldCreate_DS);CHKERRQ(ierr);
  ierr = DMFieldRegister(DMFIELDSHELL, DMFieldCreate_Shell);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
  DMFieldRegister  - Adds an implementation of the DMField object.

   Not collective

   Input Parameters:
+  name_impl - name of a new user-defined implementation
-  routine_create - routine to create method context

   Notes:
   DMFieldRegister() may be called multiple times to add several user-defined implementations.

   Sample usage:
.vb
   DMFieldRegister("my_impl",MyImplCreate);
.ve

   Then, this implementation can be chosen with the procedural interface via
$     DMFieldSetType(tagger,"my_impl")

   Level: advanced

.seealso: DMFieldRegisterAll(), DMFieldRegisterDestroy()
@*/
PetscErrorCode  DMFieldRegister(const char sname[],PetscErrorCode (*function)(DMField))
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFunctionListAdd(&DMFieldList,sname,function);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

