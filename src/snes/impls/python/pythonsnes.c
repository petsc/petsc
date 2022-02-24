#include <petsc/private/snesimpl.h>          /*I "petscsnes.h" I*/

/*@C
   SNESPythonSetType - Initialize a SNES object implemented in Python.

   Collective on SNES

   Input Parameters:
+  snes - the nonlinear solver (SNES) context.
-  pyname - full dotted Python name [package].module[.{class|function}]

   Options Database Key:
.  -snes_python_type <pyname> - python class

   Level: intermediate

.seealso: SNESCreate(), SNESSetType(), SNESPYTHON, PetscPythonInitialize()
@*/
PetscErrorCode  SNESPythonSetType(SNES snes,const char pyname[])
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_CLASSID,1);
  PetscValidCharPointer(pyname,2);
  CHKERRQ(PetscTryMethod(snes,"SNESPythonSetType_C",(SNES, const char[]),(snes,pyname)));
  PetscFunctionReturn(0);
}
