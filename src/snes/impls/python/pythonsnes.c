#include <petsc/private/snesimpl.h> /*I "petscsnes.h" I*/

/*@C
   SNESPythonSetType - Initialize a `SNES` object implemented in Python.

   Collective

   Input Parameters:
+  snes - the nonlinear solver (`SNES`) context.
-  pyname - full dotted Python name [package].module[.{class|function}]

   Options Database Key:
.  -snes_python_type <pyname> - python class

   Level: intermediate

.seealso: `SNESCreate()`, `SNESSetType()`, `SNESPYTHON`, `PetscPythonInitialize()`, `SNESPythonGetType()`
@*/
PetscErrorCode SNESPythonSetType(SNES snes, const char pyname[])
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes, SNES_CLASSID, 1);
  PetscValidCharPointer(pyname, 2);
  PetscTryMethod(snes, "SNESPythonSetType_C", (SNES, const char[]), (snes, pyname));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
   SNESPythonGetType - Get the type of a `SNES` object implemented in Python.

   Not Collective

   Input Parameter:
.  snes - the nonlinear solver (`SNES`) context.

   Output Parameter:
.  pyname - full dotted Python name [package].module[.{class|function}]

   Level: intermediate

.seealso: `SNESCreate()`, `SNESSetType()`, `SNESPYTHON`, `PetscPythonInitialize()`, `SNESPythonSetType()`
@*/
PetscErrorCode SNESPythonGetType(SNES snes, const char *pyname[])
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes, SNES_CLASSID, 1);
  PetscValidPointer(pyname, 2);
  PetscUseMethod(snes, "SNESPythonGetType_C", (SNES, const char *[]), (snes, pyname));
  PetscFunctionReturn(PETSC_SUCCESS);
}
