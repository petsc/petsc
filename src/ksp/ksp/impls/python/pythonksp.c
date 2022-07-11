#include <petsc/private/kspimpl.h>          /*I "petscksp.h" I*/

/*@C
   KSPPythonSetType - Initialize a KSP object implemented in Python.

   Collective on ksp

   Input Parameters:
+  ksp - the linear solver (KSP) context.
-  pyname - full dotted Python name [package].module[.{class|function}]

   Options Database Key:
.  -ksp_python_type <pyname> - python class

   Level: intermediate

.seealso: `KSPCreate()`, `KSPSetType()`, `KSPPYTHON`, `PetscPythonInitialize()`
@*/
PetscErrorCode  KSPPythonSetType(KSP ksp,const char pyname[])
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_CLASSID,1);
  PetscValidCharPointer(pyname,2);
  PetscTryMethod(ksp,"KSPPythonSetType_C",(KSP, const char[]),(ksp,pyname));
  PetscFunctionReturn(0);
}

/*@C
   KSPPythonGetType - Get the type of a KSP object implemented in Python.

   Not collective

   Input Parameter:
.  ksp - the linear solver (KSP) context.

   Output Parameter:
.  pyname - full dotted Python name [package].module[.{class|function}]

   Level: intermediate

.seealso: `KSPCreate()`, `KSPSetType()`, `KSPPYTHON`, `PetscPythonInitialize()`, `KSPPythonSetType()`
@*/
PetscErrorCode  KSPPythonGetType(KSP ksp,const char *pyname[])
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_CLASSID,1);
  PetscValidPointer(pyname,2);
  PetscUseMethod(ksp,"KSPPythonGetType_C",(KSP, const char*[]),(ksp,pyname));
  PetscFunctionReturn(0);
}
