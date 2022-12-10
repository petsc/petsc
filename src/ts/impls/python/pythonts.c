#include <petsc/private/tsimpl.h> /*I "petscts.h" I*/

/*@C
   TSPythonSetType - Initialize a `TS` object implemented in Python.

   Collective

   Input Parameters:
+  ts - the `TS` context
-  pyname - full dotted Python name [package].module[.{class|function}]

   Options Database Key:
.  -ts_python_type <pyname> - python class

   Level: intermediate

.seealso: [](chapter_ts), `TSCreate()`, `TSSetType()`, `TSPYTHON`, `PetscPythonInitialize()`
@*/
PetscErrorCode TSPythonSetType(TS ts, const char pyname[])
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts, TS_CLASSID, 1);
  PetscValidCharPointer(pyname, 2);
  PetscTryMethod(ts, "TSPythonSetType_C", (TS, const char[]), (ts, pyname));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
   TSPythonGetType - Get the type of a `TS` object implemented in Python.

   Not Collective

   Input Parameter:
.  ts - the `TS` context

   Output Parameter:
.  pyname - full dotted Python name [package].module[.{class|function}]

   Level: intermediate

.seealso: [](chapter_ts), `TSCreate()`, `TSSetType()`, `TSPYTHON`, `PetscPythonInitialize()`, `TSPythonSetType()`
@*/
PetscErrorCode TSPythonGetType(TS ts, const char *pyname[])
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts, TS_CLASSID, 1);
  PetscValidPointer(pyname, 2);
  PetscUseMethod(ts, "TSPythonGetType_C", (TS, const char *[]), (ts, pyname));
  PetscFunctionReturn(PETSC_SUCCESS);
}
