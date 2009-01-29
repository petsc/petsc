#include "private/kspimpl.h"          /*I "petscksp.h" I*/

#undef __FUNCT__
#define __FUNCT__ "KSPPythonSetType"
/*@C
   KSPPythonSetType - Initalize a KSP object implemented in Python.

   Collective on KSP

   Input Parameter:
+  ksp - the linear solver (KSP) context.
-  pyname - full dotted Python name [package].module[.{class|function}]

   Options Database Key:
.  -ksp_python_type <pyname>

   Level: intermediate

.keywords: KSP, Python

.seealso: KSPCreate(), KSPSetType(), KSPPYTHON, PetscPythonInitialize()
@*/
PetscErrorCode PETSCKSP_DLLEXPORT KSPPythonSetType(KSP ksp,const char pyname[])
{
  PetscErrorCode (*f)(KSP, const char[]) = 0;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_COOKIE,1);
  PetscValidCharPointer(pyname,2);
  ierr = PetscObjectQueryFunction((PetscObject)ksp,"KSPPythonSetType_C",(PetscVoidFunction*)&f);CHKERRQ(ierr);
  if (f) {ierr = (*f)(ksp,pyname);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}
