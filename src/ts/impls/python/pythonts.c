#include <petsc/private/tsimpl.h>          /*I "petscts.h" I*/

/*@C
   TSPythonSetType - Initialize a TS object implemented in Python.

   Collective on TS

   Input Parameter:
+  ts - the nonlinear solver (TS) context.
-  pyname - full dotted Python name [package].module[.{class|function}]

   Options Database Key:
.  -ts_python_type <pyname>

   Level: intermediate

.seealso: TSCreate(), TSSetType(), TSPYTHON, PetscPythonInitialize()
@*/
PetscErrorCode  TSPythonSetType(TS ts,const char pyname[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  PetscValidCharPointer(pyname,2);
  ierr = PetscTryMethod(ts,"TSPythonSetType_C",(TS, const char[]),(ts,pyname));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
