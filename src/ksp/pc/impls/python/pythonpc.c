#include <petsc/private/pcimpl.h>          /*I "petscpc.h" I*/

/*@C
   PCPythonSetType - Initalize a PC object implemented in Python.

   Collective on PC

   Input Parameter:
+  pc - the preconditioner (PC) context.
-  pyname - full dotted Python name [package].module[.{class|function}]

   Options Database Key:
.  -pc_python_type <pyname>

   Level: intermediate

.seealso: PCCreate(), PCSetType(), PCPYTHON, PetscPythonInitialize()
@*/
PetscErrorCode  PCPythonSetType(PC pc,const char pyname[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  PetscValidCharPointer(pyname,2);
  ierr = PetscTryMethod(pc,"PCPythonSetType_C",(PC, const char[]),(pc,pyname));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
