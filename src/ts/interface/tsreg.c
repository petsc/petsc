#include <petsc/private/tsimpl.h>      /*I "petscts.h"  I*/

PetscFunctionList TSList              = NULL;
PetscBool         TSRegisterAllCalled = PETSC_FALSE;

/*@C
  TSSetType - Sets the method to be used as the timestepping solver.

  Collective on TS

  Input Parameters:
+ ts   - The TS context
- type - A known method

  Options Database Command:
. -ts_type <type> - Sets the method; use -help for a list of available methods (for instance, euler)

   Notes:
   See "petsc/include/petscts.h" for available methods (for instance)
+  TSEULER - Euler
.  TSSUNDIALS - SUNDIALS interface
.  TSBEULER - Backward Euler
-  TSPSEUDO - Pseudo-timestepping

   Normally, it is best to use the TSSetFromOptions() command and
   then set the TS type from the options database rather than by using
   this routine.  Using the options database provides the user with
   maximum flexibility in evaluating the many different solvers.
   The TSSetType() routine is provided for those situations where it
   is necessary to set the timestepping solver independently of the
   command line or options database.  This might be the case, for example,
   when the choice of solver changes during the execution of the
   program, and the user's application is taking responsibility for
   choosing the appropriate method.  In other words, this routine is
   not for beginners.

   Level: intermediate

.seealso: `TS`, `TSSolve()`, `TSCreate()`, `TSSetFromOptions()`, `TSDestroy()`, `TSType`

@*/
PetscErrorCode  TSSetType(TS ts,TSType type)
{
  PetscErrorCode (*r)(TS);
  PetscBool      match;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts, TS_CLASSID,1);
  PetscValidCharPointer(type,2);
  PetscCall(PetscObjectTypeCompare((PetscObject) ts, type, &match));
  if (match) PetscFunctionReturn(0);

  PetscCall(PetscFunctionListFind(TSList,type,&r));
  PetscCheck(r,PETSC_COMM_SELF,PETSC_ERR_ARG_UNKNOWN_TYPE, "Unknown TS type: %s", type);
  if (ts->ops->destroy) PetscCall((*(ts)->ops->destroy)(ts));
  PetscCall(PetscMemzero(ts->ops,sizeof(*ts->ops)));
  ts->usessnes           = PETSC_FALSE;
  ts->default_adapt_type = TSADAPTNONE;

  ts->setupcalled = PETSC_FALSE;

  PetscCall(PetscObjectChangeTypeName((PetscObject)ts, type));
  PetscCall((*r)(ts));
  PetscFunctionReturn(0);
}

/*@C
  TSGetType - Gets the TS method type (as a string).

  Not Collective

  Input Parameter:
. ts - The TS

  Output Parameter:
. type - The name of TS method

  Level: intermediate

.seealso `TSSetType()`
@*/
PetscErrorCode  TSGetType(TS ts, TSType *type)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  PetscValidPointer(type,2);
  *type = ((PetscObject)ts)->type_name;
  PetscFunctionReturn(0);
}

/*--------------------------------------------------------------------------------------------------------------------*/

/*@C
  TSRegister - Adds a creation method to the TS package.

  Not Collective

  Input Parameters:
+ name        - The name of a new user-defined creation routine
- create_func - The creation routine itself

  Notes:
  TSRegister() may be called multiple times to add several user-defined tses.

  Sample usage:
.vb
  TSRegister("my_ts",  MyTSCreate);
.ve

  Then, your ts type can be chosen with the procedural interface via
.vb
    TS ts;
    TSCreate(MPI_Comm, &ts);
    TSSetType(ts, "my_ts")
.ve
  or at runtime via the option
.vb
    -ts_type my_ts
.ve

  Level: advanced

.seealso: `TSRegisterAll()`, `TSRegisterDestroy()`
@*/
PetscErrorCode  TSRegister(const char sname[], PetscErrorCode (*function)(TS))
{
  PetscFunctionBegin;
  PetscCall(TSInitializePackage());
  PetscCall(PetscFunctionListAdd(&TSList,sname,function));
  PetscFunctionReturn(0);
}
