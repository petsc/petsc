#include <petsc/private/kspimpl.h>          /*I "petscksp.h" I*/

/*@C
   KSPPythonSetType - Initalize a KSP object implemented in Python.

   Collective on ksp

   Input Parameter:
+  ksp - the linear solver (KSP) context.
-  pyname - full dotted Python name [package].module[.{class|function}]

   Options Database Key:
.  -ksp_python_type <pyname>

   Level: intermediate

.seealso: KSPCreate(), KSPSetType(), KSPPYTHON, PetscPythonInitialize()
@*/
PetscErrorCode  KSPPythonSetType(KSP ksp,const char pyname[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_CLASSID,1);
  PetscValidCharPointer(pyname,2);
  ierr = PetscTryMethod(ksp,"KSPPythonSetType_C",(KSP, const char[]),(ksp,pyname));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

