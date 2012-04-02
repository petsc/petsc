
/*
     Provides utility routines for manulating any type of PETSc object.
*/
#include <petscsys.h>  /*I   "petscsys.h"    I*/

#undef __FUNCT__  
#define __FUNCT__ "PetscObjectStateQuery"
/*@C
   PetscObjectStateQuery - Gets the state of any PetscObject, 
   regardless of the type.

   Not Collective

   Input Parameter:
.  obj - any PETSc object, for example a Vec, Mat or KSP. This must be
         cast with a (PetscObject), for example, 
         PetscObjectStateQuery((PetscObject)mat,&state);

   Output Parameter:
.  state - the object state

   Notes: object state is an integer which gets increased every time
   the object is changed. By saving and later querying the object state
   one can determine whether information about the object is still current.
   Currently, state is maintained for Vec and Mat objects.

   Level: advanced

   seealso: PetscObjectStateIncrease(), PetscObjectSetState()

   Concepts: state

@*/
PetscErrorCode  PetscObjectStateQuery(PetscObject obj,PetscInt *state)
{
  PetscFunctionBegin;
  PetscValidHeader(obj,1);
  PetscValidIntPointer(state,2);
  *state = obj->state;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscObjectSetState"
/*@C
   PetscObjectSetState - Sets the state of any PetscObject, 
   regardless of the type.

   Not Collective

   Input Parameter:
+  obj - any PETSc object, for example a Vec, Mat or KSP. This must be
         cast with a (PetscObject), for example, 
         PetscObjectSetState((PetscObject)mat,state);
-  state - the object state

   Notes: This function should be used with extreme caution. There is 
   essentially only one use for it: if the user calls Mat(Vec)GetRow(Array),
   which increases the state, but does not alter the data, then this 
   routine can be used to reset the state.

   Level: advanced

   seealso: PetscObjectStateQuery(),PetscObjectStateIncrease()

   Concepts: state

@*/
PetscErrorCode  PetscObjectSetState(PetscObject obj,PetscInt state)
{
  PetscFunctionBegin;
  PetscValidHeader(obj,1);
  obj->state = state;
  PetscFunctionReturn(0);
}

PetscInt  PetscObjectComposedDataMax = 10;

#undef __FUNCT__  
#define __FUNCT__ "PetscObjectComposedDataRegister"
/*@C
   PetscObjectComposedDataRegister - Get an available id for 
   composed data

   Not Collective

   Output parameter:
.  id - an identifier under which data can be stored

   Level: developer

   seealso: PetscObjectComposedDataSetInt()

@*/
PetscErrorCode  PetscObjectComposedDataRegister(PetscInt *id)
{
  static PetscInt globalcurrentstate = 0;

  PetscFunctionBegin;
  *id = globalcurrentstate++;
  if (globalcurrentstate > PetscObjectComposedDataMax) PetscObjectComposedDataMax += 10;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscObjectComposedDataIncreaseInt"
PetscErrorCode  PetscObjectComposedDataIncreaseInt(PetscObject obj)
{
  PetscInt       *ar = obj->intcomposeddata,*new_ar;
  PetscInt       *ir = obj->intcomposedstate,*new_ir,n = obj->int_idmax,new_n,i;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  new_n = PetscObjectComposedDataMax;
  ierr = PetscMalloc(new_n*sizeof(PetscInt),&new_ar);CHKERRQ(ierr);
  ierr = PetscMemzero(new_ar,new_n*sizeof(PetscInt));CHKERRQ(ierr);
  ierr = PetscMalloc(new_n*sizeof(PetscInt),&new_ir);CHKERRQ(ierr);
  ierr = PetscMemzero(new_ir,new_n*sizeof(PetscInt));CHKERRQ(ierr);
  if (n) {
    for (i=0; i<n; i++) {
      new_ar[i] = ar[i]; new_ir[i] = ir[i];
    }
    ierr = PetscFree(ar);CHKERRQ(ierr);
    ierr = PetscFree(ir);CHKERRQ(ierr);
  }
  obj->int_idmax = new_n;
  obj->intcomposeddata = new_ar; obj->intcomposedstate = new_ir;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscObjectComposedDataIncreaseIntstar"
PetscErrorCode  PetscObjectComposedDataIncreaseIntstar(PetscObject obj)
{
  PetscInt       **ar = obj->intstarcomposeddata,**new_ar;
  PetscInt       *ir = obj->intstarcomposedstate,*new_ir,n = obj->intstar_idmax,new_n,i;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  new_n = PetscObjectComposedDataMax;
  ierr = PetscMalloc(new_n*sizeof(PetscInt*),&new_ar);CHKERRQ(ierr);
  ierr = PetscMemzero(new_ar,new_n*sizeof(PetscInt*));CHKERRQ(ierr);
  ierr = PetscMalloc(new_n*sizeof(PetscInt),&new_ir);CHKERRQ(ierr);
  ierr = PetscMemzero(new_ir,new_n*sizeof(PetscInt));CHKERRQ(ierr);
  if (n) {
    for (i=0; i<n; i++) {
      new_ar[i] = ar[i]; new_ir[i] = ir[i];
    }
    ierr = PetscFree(ar);CHKERRQ(ierr);
    ierr = PetscFree(ir);CHKERRQ(ierr);
  }
  obj->intstar_idmax = new_n;
  obj->intstarcomposeddata = new_ar; obj->intstarcomposedstate = new_ir;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscObjectComposedDataIncreaseReal"
PetscErrorCode  PetscObjectComposedDataIncreaseReal(PetscObject obj)
{
  PetscReal      *ar = obj->realcomposeddata,*new_ar;
  PetscInt       *ir = obj->realcomposedstate,*new_ir,n = obj->real_idmax,new_n,i;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  new_n = PetscObjectComposedDataMax;
  ierr = PetscMalloc(new_n*sizeof(PetscReal),&new_ar);CHKERRQ(ierr);
  ierr = PetscMemzero(new_ar,new_n*sizeof(PetscReal));CHKERRQ(ierr);
  ierr = PetscMalloc(new_n*sizeof(PetscInt),&new_ir);CHKERRQ(ierr);
  ierr = PetscMemzero(new_ir,new_n*sizeof(PetscInt));CHKERRQ(ierr);
  if (n) {
    for (i=0; i<n; i++) {
      new_ar[i] = ar[i]; new_ir[i] = ir[i];
    }
    ierr = PetscFree(ar);CHKERRQ(ierr);
    ierr = PetscFree(ir);CHKERRQ(ierr);
  }
  obj->real_idmax = new_n;
  obj->realcomposeddata = new_ar; obj->realcomposedstate = new_ir;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscObjectComposedDataIncreaseRealstar"
PetscErrorCode  PetscObjectComposedDataIncreaseRealstar(PetscObject obj)
{
  PetscReal      **ar = obj->realstarcomposeddata,**new_ar;
  PetscInt       *ir = obj->realstarcomposedstate,*new_ir,n = obj->realstar_idmax,new_n,i;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  new_n = PetscObjectComposedDataMax;
  ierr = PetscMalloc(new_n*sizeof(PetscReal*),&new_ar);CHKERRQ(ierr);
  ierr = PetscMemzero(new_ar,new_n*sizeof(PetscReal*));CHKERRQ(ierr);
  ierr = PetscMalloc(new_n*sizeof(PetscInt),&new_ir);CHKERRQ(ierr);
  ierr = PetscMemzero(new_ir,new_n*sizeof(PetscInt));CHKERRQ(ierr);
  if (n) {
    for (i=0; i<n; i++) {
      new_ar[i] = ar[i]; new_ir[i] = ir[i];
    }
    ierr = PetscFree(ar);CHKERRQ(ierr);
    ierr = PetscFree(ir);CHKERRQ(ierr);
  }
  obj->realstar_idmax = new_n;
  obj->realstarcomposeddata = new_ar; obj->realstarcomposedstate = new_ir;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscObjectComposedDataIncreaseScalar"
PetscErrorCode  PetscObjectComposedDataIncreaseScalar(PetscObject obj)
{
  PetscScalar    *ar = obj->scalarcomposeddata,*new_ar;
  PetscInt       *ir = obj->scalarcomposedstate,*new_ir,n = obj->scalar_idmax,new_n,i;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  new_n = PetscObjectComposedDataMax;
  ierr = PetscMalloc(new_n*sizeof(PetscScalar),&new_ar);CHKERRQ(ierr);
  ierr = PetscMemzero(new_ar,new_n*sizeof(PetscScalar));CHKERRQ(ierr);
  ierr = PetscMalloc(new_n*sizeof(PetscInt),&new_ir);CHKERRQ(ierr);
  ierr = PetscMemzero(new_ir,new_n*sizeof(PetscInt));CHKERRQ(ierr);
  if (n) {
    for (i=0; i<n; i++) {
      new_ar[i] = ar[i]; new_ir[i] = ir[i];
    }
    ierr = PetscFree(ar);CHKERRQ(ierr);
    ierr = PetscFree(ir);CHKERRQ(ierr);
  }
  obj->scalar_idmax = new_n;
  obj->scalarcomposeddata = new_ar; obj->scalarcomposedstate = new_ir;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscObjectComposedDataIncreaseScalarStar"
PetscErrorCode  PetscObjectComposedDataIncreaseScalarstar(PetscObject obj)
{
  PetscScalar    **ar = obj->scalarstarcomposeddata,**new_ar;
  PetscInt       *ir = obj->scalarstarcomposedstate,*new_ir,n = obj->scalarstar_idmax,new_n,i;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  new_n = PetscObjectComposedDataMax;
  ierr = PetscMalloc(new_n*sizeof(PetscScalar*),&new_ar);CHKERRQ(ierr);
  ierr = PetscMemzero(new_ar,new_n*sizeof(PetscScalar*));CHKERRQ(ierr);
  ierr = PetscMalloc(new_n*sizeof(PetscInt),&new_ir);CHKERRQ(ierr);
  ierr = PetscMemzero(new_ir,new_n*sizeof(PetscInt));CHKERRQ(ierr);
  if (n) {
    for (i=0; i<n; i++) {
      new_ar[i] = ar[i]; new_ir[i] = ir[i];
    }
    ierr = PetscFree(ar);CHKERRQ(ierr);
    ierr = PetscFree(ir);CHKERRQ(ierr);
  }
  obj->scalarstar_idmax = new_n;
  obj->scalarstarcomposeddata = new_ar; obj->scalarstarcomposedstate = new_ir;
  PetscFunctionReturn(0);
}

