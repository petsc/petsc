#include "private/snesimpl.h"          /*I "petscsnes.h" I*/

#undef __FUNCT__
#define __FUNCT__ "SNESPythonSetType"
/*@C
   SNESPythonSetType - Initalize a SNES object implemented in Python.

   Collective on SNES

   Input Parameter:
+  snes - the nonlinear solver (SNES) context.
-  pyname - full dotted Python name [package].module[.{class|function}]

   Options Database Key:
.  -snes_python_type <pyname>

   Level: intermediate

.keywords: SNES, Python

.seealso: SNESCreate(), SNESSetType(), SNESPYTHON, PetscPythonInitialize()
@*/
PetscErrorCode PETSCSNES_DLLEXPORT SNESPythonSetType(SNES snes,const char pyname[])
{
  PetscErrorCode (*f)(SNES, const char[]) = 0;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_COOKIE,1);
  PetscValidCharPointer(pyname,2);
  ierr = PetscObjectQueryFunction((PetscObject)snes,"SNESPythonSetType_C",
				  (PetscVoidFunction*)&f);CHKERRQ(ierr);
  if (f) {ierr = (*f)(snes,pyname);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}
