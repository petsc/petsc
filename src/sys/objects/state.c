
/*
     Provides utility routines for manulating any type of PETSc object.
*/
#include <petsc/private/petscimpl.h>  /*I   "petscsys.h"    I*/

/*@C
   PetscObjectStateGet - Gets the state of any PetscObject,
   regardless of the type.

   Not Collective

   Input Parameter:
.  obj - any PETSc object, for example a Vec, Mat or KSP. This must be
         cast with a (PetscObject), for example,
         PetscObjectStateGet((PetscObject)mat,&state);

   Output Parameter:
.  state - the object state

   Notes:
    object state is an integer which gets increased every time
   the object is changed. By saving and later querying the object state
   one can determine whether information about the object is still current.
   Currently, state is maintained for Vec and Mat objects.

   Level: advanced

   seealso: PetscObjectStateIncrease(), PetscObjectStateSet()

@*/
PetscErrorCode PetscObjectStateGet(PetscObject obj,PetscObjectState *state)
{
  PetscFunctionBegin;
  PetscValidHeader(obj,1);
  PetscValidIntPointer(state,2);
  *state = obj->state;
  PetscFunctionReturn(0);
}

/*@C
   PetscObjectStateSet - Sets the state of any PetscObject,
   regardless of the type.

   Logically Collective

   Input Parameters:
+  obj - any PETSc object, for example a Vec, Mat or KSP. This must be
         cast with a (PetscObject), for example,
         PetscObjectStateSet((PetscObject)mat,state);
-  state - the object state

   Notes:
    This function should be used with extreme caution. There is
   essentially only one use for it: if the user calls Mat(Vec)GetRow(Array),
   which increases the state, but does not alter the data, then this
   routine can be used to reset the state.  Such a reset must be collective.

   Level: advanced

   seealso: PetscObjectStateGet(),PetscObjectStateIncrease()

@*/
PetscErrorCode PetscObjectStateSet(PetscObject obj,PetscObjectState state)
{
  PetscFunctionBegin;
  PetscValidHeader(obj,1);
  obj->state = state;
  PetscFunctionReturn(0);
}

PetscInt PetscObjectComposedDataMax = 10;

/*@C
   PetscObjectComposedDataRegister - Get an available id for composed data

   Not Collective

   Output parameter:
.  id - an identifier under which data can be stored

   Level: developer

   Notes:
    You must keep this value (for example in a global variable) in order to attach the data to an object or
          access in an object.

   seealso: PetscObjectComposedDataSetInt()

@*/
PetscErrorCode PetscObjectComposedDataRegister(PetscInt *id)
{
  static PetscInt globalcurrentstate = 0;

  PetscFunctionBegin;
  *id = globalcurrentstate++;
  if (globalcurrentstate > PetscObjectComposedDataMax) PetscObjectComposedDataMax += 10;
  PetscFunctionReturn(0);
}

PetscErrorCode PetscObjectComposedDataIncreaseInt(PetscObject obj)
{
  PetscInt         *ar = obj->intcomposeddata,*new_ar,n = obj->int_idmax,new_n;
  PetscObjectState *ir = obj->intcomposedstate,*new_ir;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  new_n = PetscObjectComposedDataMax;
  ierr  = PetscCalloc2(new_n,&new_ar,new_n,&new_ir);CHKERRQ(ierr);
  ierr  = PetscMemcpy(new_ar,ar,n*sizeof(PetscInt));CHKERRQ(ierr);
  ierr  = PetscMemcpy(new_ir,ir,n*sizeof(PetscObjectState));CHKERRQ(ierr);
  ierr  = PetscFree2(ar,ir);CHKERRQ(ierr);
  obj->int_idmax       = new_n;
  obj->intcomposeddata = new_ar; obj->intcomposedstate = new_ir;
  PetscFunctionReturn(0);
}

PetscErrorCode PetscObjectComposedDataIncreaseIntstar(PetscObject obj)
{
  PetscInt         **ar = obj->intstarcomposeddata,**new_ar,n = obj->intstar_idmax,new_n;
  PetscObjectState *ir  = obj->intstarcomposedstate,*new_ir;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  new_n = PetscObjectComposedDataMax;
  ierr  = PetscCalloc2(new_n,&new_ar,new_n,&new_ir);CHKERRQ(ierr);
  ierr  = PetscMemcpy(new_ar,ar,n*sizeof(PetscInt*));CHKERRQ(ierr);
  ierr  = PetscMemcpy(new_ir,ir,n*sizeof(PetscObjectState));CHKERRQ(ierr);
  ierr  = PetscFree2(ar,ir);CHKERRQ(ierr);
  obj->intstar_idmax        = new_n;
  obj->intstarcomposeddata  = new_ar;
  obj->intstarcomposedstate = new_ir;
  PetscFunctionReturn(0);
}

PetscErrorCode PetscObjectComposedDataIncreaseReal(PetscObject obj)
{
  PetscReal        *ar = obj->realcomposeddata,*new_ar;
  PetscObjectState *ir = obj->realcomposedstate,*new_ir;
  PetscInt         n   = obj->real_idmax,new_n;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  new_n = PetscObjectComposedDataMax;
  ierr  = PetscCalloc2(new_n,&new_ar,new_n,&new_ir);CHKERRQ(ierr);
  ierr  = PetscMemcpy(new_ar,ar,n*sizeof(PetscReal));CHKERRQ(ierr);
  ierr  = PetscMemcpy(new_ir,ir,n*sizeof(PetscObjectState));CHKERRQ(ierr);
  ierr  = PetscFree2(ar,ir);CHKERRQ(ierr);
  obj->real_idmax       = new_n;
  obj->realcomposeddata = new_ar; obj->realcomposedstate = new_ir;
  PetscFunctionReturn(0);
}

PetscErrorCode PetscObjectComposedDataIncreaseRealstar(PetscObject obj)
{
  PetscReal        **ar = obj->realstarcomposeddata,**new_ar;
  PetscObjectState *ir  = obj->realstarcomposedstate,*new_ir;
  PetscInt         n    = obj->realstar_idmax,new_n;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  new_n = PetscObjectComposedDataMax;
  ierr  = PetscCalloc2(new_n,&new_ar,new_n,&new_ir);CHKERRQ(ierr);
  ierr  = PetscMemcpy(new_ar,ar,n*sizeof(PetscReal*));CHKERRQ(ierr);
  ierr  = PetscMemcpy(new_ir,ir,n*sizeof(PetscObjectState));CHKERRQ(ierr);
  ierr  = PetscFree2(ar,ir);CHKERRQ(ierr);
  obj->realstar_idmax       = new_n;
  obj->realstarcomposeddata = new_ar; obj->realstarcomposedstate = new_ir;
  PetscFunctionReturn(0);
}

PetscErrorCode PetscObjectComposedDataIncreaseScalar(PetscObject obj)
{
  PetscScalar      *ar = obj->scalarcomposeddata,*new_ar;
  PetscObjectState *ir = obj->scalarcomposedstate,*new_ir;
  PetscInt         n   = obj->scalar_idmax,new_n;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  new_n = PetscObjectComposedDataMax;
  ierr  = PetscCalloc2(new_n,&new_ar,new_n,&new_ir);CHKERRQ(ierr);
  ierr  = PetscMemcpy(new_ar,ar,n*sizeof(PetscScalar));CHKERRQ(ierr);
  ierr  = PetscMemcpy(new_ir,ir,n*sizeof(PetscObjectState));CHKERRQ(ierr);
  ierr  = PetscFree2(ar,ir);CHKERRQ(ierr);
  obj->scalar_idmax       = new_n;
  obj->scalarcomposeddata = new_ar; obj->scalarcomposedstate = new_ir;
  PetscFunctionReturn(0);
}

PetscErrorCode PetscObjectComposedDataIncreaseScalarstar(PetscObject obj)
{
  PetscScalar      **ar = obj->scalarstarcomposeddata,**new_ar;
  PetscObjectState *ir  = obj->scalarstarcomposedstate,*new_ir;
  PetscInt         n    = obj->scalarstar_idmax,new_n;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  new_n = PetscObjectComposedDataMax;
  ierr  = PetscCalloc2(new_n,&new_ar,new_n,&new_ir);CHKERRQ(ierr);
  ierr  = PetscMemcpy(new_ar,ar,n*sizeof(PetscScalar*));CHKERRQ(ierr);
  ierr  = PetscMemcpy(new_ir,ir,n*sizeof(PetscObjectState));CHKERRQ(ierr);
  ierr  = PetscFree2(ar,ir);CHKERRQ(ierr);
  obj->scalarstar_idmax       = new_n;
  obj->scalarstarcomposeddata = new_ar; obj->scalarstarcomposedstate = new_ir;
  PetscFunctionReturn(0);
}

/*@
   PetscObjectGetId - get unique object ID

   Not Collective

   Input Arguments:
.  obj - object

   Output Arguments:
.  id - integer ID

   Level: developer

   Notes:
   The object ID may be different on different processes, but object IDs are never reused so local equality implies global equality.

.seealso: PetscObjectStateGet(), PetscObjectCompareId()
@*/
PetscErrorCode PetscObjectGetId(PetscObject obj,PetscObjectId *id)
{
  PetscFunctionBegin;
  *id = obj->id;
  PetscFunctionReturn(0);
}

/*@
   PetscObjectCompareId - compares the objects ID with a given id

   Not Collective

   Input Arguments:
+  obj - object
-  id - integer ID

   Output Argument;
.  eq - the ids are equal

   Level: developer

   Notes:
   The object ID may be different on different processes, but object IDs are never reused so local equality implies global equality.

.seealso: PetscObjectStateGet(), PetscObjectGetId()
@*/
PetscErrorCode PetscObjectCompareId(PetscObject obj,PetscObjectId id,PetscBool *eq)
{
  PetscFunctionBegin;
  *eq = (id == obj->id) ? PETSC_TRUE : PETSC_FALSE;
  PetscFunctionReturn(0);
}
