#include <petsc/private/pcimpl.h>          /*I "petscpc.h" I*/

/*@C
   PCPythonSetType - Initialize a PC object implemented in Python.

   Collective on PC

   Input Parameters:
+  pc - the preconditioner (PC) context.
-  pyname - full dotted Python name [package].module[.{class|function}]

   Options Database Key:
.  -pc_python_type <pyname> - python class

   Level: intermediate

.seealso: PCCreate(), PCSetType(), PCPYTHON, PetscPythonInitialize()
@*/
PetscErrorCode  PCPythonSetType(PC pc,const char pyname[])
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  PetscValidCharPointer(pyname,2);
  CHKERRQ(PetscTryMethod(pc,"PCPythonSetType_C",(PC, const char[]),(pc,pyname)));
  PetscFunctionReturn(0);
}
