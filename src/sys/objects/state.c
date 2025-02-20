/*
     Provides utility routines for manulating any type of PETSc object.
*/
#include <petsc/private/petscimpl.h> /*I   "petscsys.h"    I*/

/*@
  PetscObjectStateGet - Gets the state of any `PetscObject`,
  regardless of the type.

  Not Collective

  Input Parameter:
. obj - any PETSc object, for example a `Vec`, `Mat` or `KSP`. This must be
         cast with a (`PetscObject`), for example,
         `PetscObjectStateGet`((`PetscObject`)mat,&state);

  Output Parameter:
. state - the object state

  Level: advanced

  Note:
  Object state is an integer which gets increased every time
  the object is changed. By saving and later querying the object state
  one can determine whether information about the object is still current.
  Currently, state is maintained for `Vec` and `Mat` objects.

.seealso: `PetscObjectStateIncrease()`, `PetscObjectStateSet()`
@*/
PetscErrorCode PetscObjectStateGet(PetscObject obj, PetscObjectState *state)
{
  PetscFunctionBegin;
  PetscValidHeader(obj, 1);
  PetscAssertPointer(state, 2);
  *state = obj->state;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscObjectStateSet - Sets the state of any `PetscObject`,
  regardless of the type.

  Logically Collective

  Input Parameters:
+ obj   - any PETSc object, for example a `Vec`, `Mat` or `KSP`. This must be
         cast with a (`PetscObject`), for example,
         `PetscObjectStateSet`((`PetscObject`)mat,state);
- state - the object state

  Level: advanced

  Note:
  This function should be used with extreme caution. There is
  essentially only one use for it: if the user calls `Mat`(`Vec`)GetRow(Array),
  which increases the state, but does not alter the data, then this
  routine can be used to reset the state.  Such a reset must be collective.

.seealso: `PetscObjectStateGet()`, `PetscObjectStateIncrease()`
@*/
PetscErrorCode PetscObjectStateSet(PetscObject obj, PetscObjectState state)
{
  PetscFunctionBegin;
  PetscValidHeader(obj, 1);
  obj->state = state;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscInt PetscObjectComposedDataMax = 10;

/*@C
  PetscObjectComposedDataRegister - Get an available id for composing data with a `PetscObject`

  Not Collective

  Output Parameter:
. id - an identifier under which data can be stored

  Level: developer

  Notes:
  You must keep this value (for example in a global variable) in order to attach the data to an object or access in an object.

  `PetscObjectCompose()` and  `PetscObjectQuery()` provide a way to attach any data to an object

.seealso: `PetscObjectComposedDataSetInt()`, `PetscObjectComposedDataSetReal()`, `PetscObjectComposedDataGetReal()`, `PetscObjectComposedDataSetIntstar()`,
          `PetscObjectComposedDataGetInt()`, `PetscObject`,
          `PetscObjectCompose()`, `PetscObjectQuery()`, `PetscObjectComposedDataSetRealstar()`, `PetscObjectComposedDataGetScalarstar()`,
          `PetscObjectComposedDataSetScalarstar()`
@*/
PetscErrorCode PetscObjectComposedDataRegister(PetscInt *id)
{
  static PetscInt globalcurrentstate = 0;

  PetscFunctionBegin;
  PetscAssertPointer(id, 1);
  *id = globalcurrentstate++;
  if (globalcurrentstate > PetscObjectComposedDataMax) PetscObjectComposedDataMax += 10;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscObjectComposedDataIncrease_(PetscInt *id_max, char **composed, PetscObjectState **composed_state, size_t obj_size)
{
  // must use char here since PetscCalloc2() and PetscMemcpy() use sizeof(**ptr), so if
  // composed is void ** (to match PetscObjectComposedDataStarIncrease_()) that would expand to
  // sizeof(void) which is illegal.
  char             *ar = *composed;
  PetscObjectState *ir = *composed_state;
  const PetscInt    n = *id_max, new_n = PetscObjectComposedDataMax;
  char             *new_ar;
  PetscObjectState *new_ir;

  PetscFunctionBegin;
  PetscAssert(n >= 0, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Number of composed data ids: %" PetscInt_FMT " < 0", n);
  PetscCall(PetscCalloc2(new_n * obj_size, &new_ar, new_n, &new_ir));
  PetscCall(PetscMemcpy(new_ar, ar, n * obj_size));
  PetscCall(PetscArraycpy(new_ir, ir, n));
  PetscCall(PetscFree2(ar, ir));
  *id_max         = new_n;
  *composed       = new_ar;
  *composed_state = new_ir;
  PetscFunctionReturn(PETSC_SUCCESS);
}

#define PetscObjectComposedDataIncrease(id_max, composed, composed_state) PetscObjectComposedDataIncrease_(id_max, (char **)(composed), composed_state, sizeof(**(composed)))

static PetscErrorCode PetscObjectComposedDataStarIncrease_(PetscInt *id_max, void ***composed, PetscObjectState **composed_state, size_t obj_size)
{
  void                  **ar = *composed;
  const PetscObjectState *ir = *composed_state;
  const PetscInt          n = *id_max, new_n = PetscObjectComposedDataMax;
  void                  **new_ar;
  PetscObjectState       *new_ir;

  PetscFunctionBegin;
  PetscAssert(n >= 0, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Number of composed star data ids: %" PetscInt_FMT " < 0", n);
  PetscCall(PetscCalloc2(new_n, &new_ar, new_n, &new_ir));
  PetscCall(PetscMemcpy(new_ar, ar, n * obj_size));
  PetscCall(PetscArraycpy(new_ir, ir, n));
  PetscCall(PetscFree2(ar, ir));
  *id_max         = new_n;
  *composed       = new_ar;
  *composed_state = new_ir;
  PetscFunctionReturn(PETSC_SUCCESS);
}

#define PetscObjectComposedDataStarIncrease(id_max, composed, composed_state) PetscObjectComposedDataStarIncrease_(id_max, (void ***)(composed), composed_state, sizeof(**(composed)))

PetscErrorCode PetscObjectComposedDataIncreaseInt(PetscObject obj)
{
  PetscFunctionBegin;
  PetscCall(PetscObjectComposedDataIncrease(&obj->int_idmax, &obj->intcomposeddata, &obj->intcomposedstate));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PetscObjectComposedDataIncreaseIntstar(PetscObject obj)
{
  PetscFunctionBegin;
  PetscCall(PetscObjectComposedDataStarIncrease(&obj->intstar_idmax, &obj->intstarcomposeddata, &obj->intstarcomposedstate));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PetscObjectComposedDataIncreaseReal(PetscObject obj)
{
  PetscFunctionBegin;
  PetscCall(PetscObjectComposedDataIncrease(&obj->real_idmax, &obj->realcomposeddata, &obj->realcomposedstate));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PetscObjectComposedDataIncreaseRealstar(PetscObject obj)
{
  PetscFunctionBegin;
  PetscCall(PetscObjectComposedDataStarIncrease(&obj->realstar_idmax, &obj->realstarcomposeddata, &obj->realstarcomposedstate));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PetscObjectComposedDataIncreaseScalar(PetscObject obj)
{
  PetscFunctionBegin;
#if PetscDefined(USE_COMPLEX)
  PetscCall(PetscObjectComposedDataIncrease(&obj->scalar_idmax, &obj->scalarcomposeddata, &obj->scalarcomposedstate));
#else
  PetscCall(PetscObjectComposedDataIncreaseReal(obj));
#endif
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PetscObjectComposedDataIncreaseScalarstar(PetscObject obj)
{
  PetscFunctionBegin;
#if PetscDefined(USE_COMPLEX)
  PetscCall(PetscObjectComposedDataStarIncrease(&obj->scalarstar_idmax, &obj->scalarstarcomposeddata, &obj->scalarstarcomposedstate));
#else
  PetscCall(PetscObjectComposedDataIncreaseRealstar(obj));
#endif
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscObjectGetId - get a unique object ID for the `PetscObject`

  Not Collective

  Input Parameter:
. obj - object

  Output Parameter:
. id - integer ID

  Level: developer

  Note:
  The object ID may be different on different processes, but object IDs are never reused so local equality implies global equality.

.seealso: `PetscObjectStateGet()`, `PetscObjectCompareId()`
@*/
PetscErrorCode PetscObjectGetId(PetscObject obj, PetscObjectId *id)
{
  PetscFunctionBegin;
  PetscValidHeader(obj, 1);
  PetscAssertPointer(id, 2);
  *id = obj->id;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscObjectCompareId - compares the objects ID with a given id

  Not Collective

  Input Parameters:
+ obj - object
- id  - integer ID

  Output Parameter:
. eq - the ids are equal

  Level: developer

  Note:
  The object ID may be different on different processes, but object IDs are never reused so
  local equality implies global equality.

.seealso: `PetscObjectStateGet()`, `PetscObjectGetId()`
@*/
PetscErrorCode PetscObjectCompareId(PetscObject obj, PetscObjectId id, PetscBool *eq)
{
  PetscObjectId oid;

  PetscFunctionBegin;
  PetscValidHeader(obj, 1);
  PetscAssertPointer(eq, 3);
  PetscCall(PetscObjectGetId(obj, &oid));
  *eq = (id == oid) ? PETSC_TRUE : PETSC_FALSE;
  PetscFunctionReturn(PETSC_SUCCESS);
}
