#include "private/pcimpl.h"          /*I "petscpc.h" I*/

#undef __FUNCT__
#define __FUNCT__ "PCPythonSetType"
/*@C
   PCPythonSetType - Initalize a PC object implemented in Python.

   Collective on PC

   Input Parameter:
+  pc - the preconditioner (PC) context.
-  pyname - full dotted Python name [package].module[.{class|function}]

   Options Database Key:
.  -pc_python_type <pyname>

   Level: intermediate

.keywords: PC, Python

.seealso: PCCreate(), PCSetType(), PCPYTHON, PetscPythonInitialize()
@*/
PetscErrorCode PETSCKSP_DLLEXPORT PCPythonSetType(PC pc,const char pyname[])
{
  PetscErrorCode (*f)(PC, const char[]) = 0;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_COOKIE,1);
  PetscValidCharPointer(pyname,2);
  ierr = PetscObjectQueryFunction((PetscObject)pc,"PCPythonSetType_C",(PetscVoidFunction*)&f);CHKERRQ(ierr);
  if (f) {ierr = (*f)(pc,pyname);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}
