#include <petsc/private/taoimpl.h>          /*I "petsctao.h" I*/

/*@C
   TaoPythonSetType - Initialize a Tao object implemented in Python.

   Collective on tao

   Input Parameters:
+  tao - the optimation solver (Tao) context.
-  pyname - full dotted Python name [package].module[.{class|function}]

   Options Database Key:
.  -tao_python_type <pyname> - python class

   Level: intermediate

.seealso: TaoCreate(), TaoSetType(), TAOPYTHON, PetscPythonInitialize()
@*/
PetscErrorCode TaoPythonSetType(Tao tao, const char pyname[])
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao,TAO_CLASSID,1);
  PetscValidCharPointer(pyname,2);
  PetscTryMethod(tao,"TaoPythonSetType_C",(Tao,const char[]),(tao,pyname));
  PetscFunctionReturn(0);
}
