/*$Id: tsreg.c,v 1.71 2001/08/06 21:18:08 bsmith Exp $*/

#include "src/ts/tsimpl.h"      /*I "petscts.h"  I*/

PetscFList TSList              = PETSC_NULL;
PetscTruth TSRegisterAllCalled = PETSC_FALSE;

#undef __FUNCT__  
#define __FUNCT__ "TSSetType"
/*@C
  TSSetType - Sets the method for the timestepping solver.  

  Collective on TS

  Input Parameters:
+ ts   - The TS context
- type - A known method

  Options Database Command:
. -ts_type <type> - Sets the method; use -help for a list of available methods (for instance, euler)

   Notes:
   See "petsc/include/petscts.h" for available methods (for instance)
+  TS_EULER - Euler
.  TS_PVODE - PVODE interface
.  TS_BEULER - Backward Euler
-  TS_PSEUDO - Pseudo-timestepping

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

.keywords: TS, set, type
.seealso TSSetSerializeType()
@*/
int TSSetType(TS ts, TSType type)
{
  int      (*r)(TS);
  PetscTruth match;
  int        ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts, TS_COOKIE);
  ierr = PetscTypeCompare((PetscObject) ts, type, &match);                                                CHKERRQ(ierr);
  if (match == PETSC_TRUE) PetscFunctionReturn(0);

  /* Get the function pointers for the method requested */
  if (TSRegisterAllCalled == PETSC_FALSE) {
    ierr = TSRegisterAll(PETSC_NULL);                                                                     CHKERRQ(ierr);
  }
  ierr = PetscFListFind(ts->comm, TSList, type, (void (**)(void)) &r);                                    CHKERRQ(ierr);
  if (!r) SETERRQ1(PETSC_ERR_ARG_OUTOFRANGE, "Unknown TS type: %s", type);

  if (ts->sles != PETSC_NULL) {
    ierr = SLESDestroy(ts->sles);                                                                         CHKERRQ(ierr);
    ts->sles = PETSC_NULL;
  }
  if (ts->snes != PETSC_NULL) {
    ierr = SNESDestroy(ts->snes);                                                                         CHKERRQ(ierr);
    ts->snes = PETSC_NULL;
  }
  if (ts->ops->destroy != PETSC_NULL) {
    ierr = (*(ts)->ops->destroy)(ts);                                                                     CHKERRQ(ierr);
  }
  ierr = (*r)(ts);                                                                                        CHKERRQ(ierr);

  ierr = PetscObjectChangeTypeName((PetscObject)ts, type);                                                CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "TSGetType"
/*@C
  TSGetType - Gets the TS method type (as a string).

  Not Collective

  Input Parameter:
. ts - The TS

  Output Parameter:
. type - The name of TS method

  Level: intermediate

.keywords: TS, timestepper, get, type, name
.seealso TSSetType()
@*/
int TSGetType(TS ts, TSType *type)
{
  int ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts, TS_COOKIE);
  PetscValidPointer(type);
  if (TSRegisterAllCalled == PETSC_FALSE) {
    ierr = TSRegisterAll(PETSC_NULL);                                                                     CHKERRQ(ierr);
  }
  *type = ts->type_name;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "TSSetSerializeType"
/*@C
  TSSetSerializeType - Sets the serialization method for the ts.

  Collective on TS

  Input Parameters:
+ ts     - The TS context
- method - A known method

  Options Database Command:
. -ts_serialize_type <method> - Sets the method; use -help for a list
                                of available methods (for instance, gbeuler_binary)

  Notes:
  See "petsc/include/ts.h" for available methods (for instance)
. GTS_SER_BEULER_BINARY - Grid Backwards Euler TS to binary file

  Normally, it is best to use the TSSetFromOptions() command and
  then set the TS type from the options database rather than by using
  this routine.  Using the options database provides the user with
  maximum flexibility in evaluating the many different solvers.
  The TSSetSerializeType() routine is provided for those situations
  where it is necessary to set the application ordering independently of the
  command line or options database.  This might be the case, for example,
  when the choice of solver changes during the execution of the
  program, and the user's application is taking responsibility for
  choosing the appropriate method.  In other words, this routine is
  not for beginners.

  Level: intermediate

.keywords: TS, set, type, serialization
.seealso TSSetType()
@*/
int TSSetSerializeType(TS ts, TSSerializeType method)
{
  int      (*r)(MPI_Comm, TS *, PetscViewer, PetscTruth);
  PetscTruth match;
  int        ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts, TS_COOKIE);
  ierr = PetscSerializeCompare((PetscObject) ts, method, &match);                                         CHKERRQ(ierr);
  if (match == PETSC_TRUE) PetscFunctionReturn(0);

  /* Get the function pointers for the method requested but do not call */
  if (TSSerializeRegisterAllCalled == PETSC_FALSE) {
    ierr = TSSerializeRegisterAll(PETSC_NULL);                                                            CHKERRQ(ierr);
  }
  ierr = PetscFListFind(ts->comm, TSSerializeList, method, (void (**)(void)) &r);                         CHKERRQ(ierr);
  if (!r) SETERRQ1(PETSC_ERR_ARG_WRONG, "Unknown ts serialization type: %s", method);

  ierr = PetscObjectChangeSerializeName((PetscObject) ts, method);                                        CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSGetSerializeType"
/*@C
  TSGetSerializeType - Gets the TS serialization method (as a string).

  Not collective

  Input Parameter:
. ts   - The ts

  Output Parameter:
. type - The name of TS serialization method

  Level: intermediate

.keywords: TS, get, serialize, type, name
.seealso TSSetType()
@*/
int TSGetSerializeType(TS ts, TSSerializeType *type)
{
  int ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts, TS_COOKIE);
  PetscValidPointer(type);
  if (TSSerializeRegisterAllCalled == PETSC_FALSE) {
    ierr = TSSerializeRegisterAll(PETSC_NULL);                                                            CHKERRQ(ierr);
  }
  *type = ts->serialize_name;
  PetscFunctionReturn(0);
}

/*--------------------------------------------------------------------------------------------------------------------*/
/*@C
  TSRegister - Adds a creation method to the TS package.

  Synopsis:

  TSRegister(char *name, char *path, char *func_name, int (*create_func)(TS))

  Not Collective

  Input Parameters:
+ name        - The name of a new user-defined creation routine
. path        - The path (either absolute or relative) of the library containing this routine
. func_name   - The name of the creation routine
- create_func - The creation routine itself

  Notes:
  TSRegister() may be called multiple times to add several user-defined tses.

  If dynamic libraries are used, then the fourth input argument (create_func) is ignored.

  Sample usage:
.vb
  TSRegisterDynamic("my_ts", "/home/username/my_lib/lib/libO/solaris/libmy.a", "MyTSCreate", MyTSCreate);
.ve

  Then, your ts type can be chosen with the procedural interface via
.vb
    TSCreate(MPI_Comm, TS *);
    TSSetType(vec, "my_ts")
.ve
  or at runtime via the option
.vb
    -ts_type my_ts
.ve

  Note: $PETSC_ARCH and $BOPT occuring in pathname will be replaced with appropriate values.

  Level: advanced

.keywords: TS, register
.seealso: TSRegisterAll(), TSRegisterDestroy()
@*/
#undef __FUNCT__  
#define __FUNCT__ "TSRegister"
int TSRegister(const char sname[], const char path[], const char name[], int (*function)(TS))
{
  char fullname[256];
  int  ierr;

  PetscFunctionBegin;
  ierr = PetscStrcpy(fullname, path);                                                                     CHKERRQ(ierr);
  ierr = PetscStrcat(fullname, ":");                                                                      CHKERRQ(ierr);
  ierr = PetscStrcat(fullname, name);                                                                     CHKERRQ(ierr);
  ierr = PetscFListAdd(&TSList, sname, fullname, (void (*)(void)) function);                              CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
  TSSerializeRegister - Adds a serialization method to the ts package.

  Synopsis:

  TSSerializeRegister(char *name, char *path, char *func_name,
                        int (*serialize_func)(MPI_Comm, TS *, PetscViewer, PetscTruth))

  Not Collective

  Input Parameters:
+ name           - The name of a new user-defined serialization routine
. path           - The path (either absolute or relative) of the library containing this routine
. func_name      - The name of the serialization routine
- serialize_func - The serialization routine itself

  Notes:
  TSSerializeRegister() may be called multiple times to add several user-defined serializers.

  If dynamic libraries are used, then the fourth input argument (serialize_func) is ignored.

  Sample usage:
.vb
  TSSerializeRegisterDynamic("my_store", "/home/username/my_lib/lib/libO/solaris/libmy.a", "MyStoreFunc", MyStoreFunc);
.ve

  Then, your serialization can be chosen with the procedural interface via
.vb
    TSSetSerializeType(ts, "my_store")
.ve
  or at runtime via the option
.vb
    -ts_serialize_type my_store
.ve

  Note: $PETSC_ARCH and $BOPT occuring in pathname will be replaced with appropriate values.

  Level: advanced

.keywords: ts, register
.seealso: TSSerializeRegisterAll(), TSSerializeRegisterDestroy()
M*/
#undef __FUNCT__  
#define __FUNCT__ "TSSerializeRegister"
int TSSerializeRegister(const char sname[], const char path[], const char name[],
                          int (*function)(MPI_Comm, TS *, PetscViewer, PetscTruth))
{
  char fullname[256];
  int  ierr;

  PetscFunctionBegin;
  ierr = PetscStrcpy(fullname, path);                                                                     CHKERRQ(ierr);
  ierr = PetscStrcat(fullname, ":");                                                                      CHKERRQ(ierr);
  ierr = PetscStrcat(fullname, name);                                                                     CHKERRQ(ierr);
  ierr = PetscFListAdd(&TSSerializeList, sname, fullname, (void (*)(void)) function);                     CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*-------------------------------------------------------------------------------------------------------------------*/
#undef __FUNCT__  
#define __FUNCT__ "TSRegisterDestroy"
/*@C
   TSRegisterDestroy - Frees the list of timestepping routines that were registered by TSREgister().

   Not Collective

   Level: advanced

.keywords: TS, timestepper, register, destroy
.seealso: TSRegister(), TSRegisterAll(), TSSerializeRegisterDestroy()
@*/
int TSRegisterDestroy(void)
{
  int ierr;

  PetscFunctionBegin;
  if (TSList != PETSC_NULL) {
    ierr = PetscFListDestroy(&TSList);                                                                    CHKERRQ(ierr);
    TSList = PETSC_NULL;
  }
  TSRegisterAllCalled = PETSC_FALSE;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "TSSerializeRegisterDestroy"
/*@C
  TSSerializeRegisterDestroy - Frees the list of serialization routines for
  timesteppers that were registered by FListAdd().

  Not collective

  Level: advanced

.keywords: ts, serialization, register, destroy
.seealso: TSSerializeRegisterAll(), TSRegisterDestroy()
@*/
int TSSerializeRegisterDestroy()
{
  int ierr;

  PetscFunctionBegin;
  if (TSSerializeList != PETSC_NULL) {
    ierr = PetscFListDestroy(&TSSerializeList);                                                           CHKERRQ(ierr);
    TSSerializeList = PETSC_NULL;
  }
  TSSerializeRegisterAllCalled = PETSC_FALSE;
  PetscFunctionReturn(0);
}
