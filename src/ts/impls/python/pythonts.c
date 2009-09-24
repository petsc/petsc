#include "private/tsimpl.h"          /*I "petscts.h" I*/

#undef __FUNCT__
#define __FUNCT__ "TSPythonSetType"
/*@C
   TSPythonSetType - Initalize a TS object implemented in Python.

   Collective on TS

   Input Parameter:
+  ts - the nonlinear solver (TS) context.
-  pyname - full dotted Python name [package].module[.{class|function}]

   Options Database Key:
.  -ts_python_type <pyname>

   Level: intermediate

.keywords: TS, Python

.seealso: TSCreate(), TSSetType(), TSPYTHON, PetscPythonInitialize()
@*/
PetscErrorCode PETSCTS_DLLEXPORT TSPythonSetType(TS ts,const char pyname[])
{
  PetscErrorCode (*f)(TS, const char[]) = 0;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_COOKIE,1);
  PetscValidCharPointer(pyname,2);
  ierr = PetscObjectQueryFunction((PetscObject)ts,"TSPythonSetType_C",
				  (PetscVoidFunction*)&f);CHKERRQ(ierr);
  if (f) {ierr = (*f)(ts,pyname);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}
