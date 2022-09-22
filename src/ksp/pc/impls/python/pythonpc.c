#include <petsc/private/pcimpl.h> /*I "petscpc.h" I*/

/*@C
   PCPythonSetType - Initialize a `PC` object implemented in Python, a `PCPYTHON`.

   Collective on pc

   Input Parameters:
+  pc - the preconditioner (`PC`) context.
-  pyname - full dotted Python name [package].module[.{class|function}]

   Options Database Key:
.  -pc_python_type <pyname> - python class

   Level: intermediate

.seealso: `PC`, `PCSHELL`, `PCCreate()`, `PCSetType()`, `PCPYTHON`, `PetscPythonInitialize()`
@*/
PetscErrorCode PCPythonSetType(PC pc, const char pyname[])
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc, PC_CLASSID, 1);
  PetscValidCharPointer(pyname, 2);
  PetscTryMethod(pc, "PCPythonSetType_C", (PC, const char[]), (pc, pyname));
  PetscFunctionReturn(0);
}

/*@C
   PCPythonGetType - Get the type of a `PC` object implemented in Python, a `PCPYTHON`.

   Not collective

   Input Parameter:
.  pc - the preconditioner (`PC`) context.

   Output Parameter:
.  pyname - full dotted Python name [package].module[.{class|function}]

   Level: intermediate

.seealso: `PC`, `PCSHELL`, `PCCreate()`, `PCSetType()`, `PCPYTHON`, `PetscPythonInitialize()`, `PCPythonSetType()`
@*/
PetscErrorCode PCPythonGetType(PC pc, const char *pyname[])
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc, PC_CLASSID, 1);
  PetscValidPointer(pyname, 2);
  PetscUseMethod(pc, "PCPythonGetType_C", (PC, const char *[]), (pc, pyname));
  PetscFunctionReturn(0);
}
