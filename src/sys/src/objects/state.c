/*$Id: gcomm.c,v 1.25 2001/03/23 23:20:38 balay Exp $*/
/*
     Provides utility routines for manulating any type of PETSc object.
*/
#include "petsc.h"  /*I   "petsc.h"    I*/

#undef __FUNCT__  
#define __FUNCT__ "PetscObjectGetState"
/*@C
   PetscObjectGetState - Gets the state of any PetscObject, 
   regardless of the type.

   Not Collective

   Input Parameter:
.  obj - any PETSc object, for example a Vec, Mat or KSP. This must be
         cast with a (PetscObject), for example, 
         PetscObjectGetState((PetscObject)mat,&state);

   Output Parameter:
.  state - the object state

   Notes: object state is an integer which gets increased every time
   the object is changed. By saving and later querying the object state
   one can determine whether information about the object is still current.
   Currently, state is maintained for Vec and Mat objects.

   Level: advanced

   seealso: PetscObjectIncreaseState

   Concepts: state

@*/
int PetscObjectGetState(PetscObject obj,int *state)
{
  PetscFunctionBegin;
  if (!obj) SETERRQ(PETSC_ERR_ARG_CORRUPT,"Null object");
  *state = obj->state;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscObjectIncreaseState"
/*@C
   PetscObjectIncreaseState - Increases the state of any PetscObject, 
   regardless of the type.

   Not Collective

   Input Parameter:
.  obj - any PETSc object, for example a Vec, Mat or KSP. This must be
         cast with a (PetscObject), for example, 
         PetscObjectIncreaseState((PetscObject)mat);

   Notes: object state is an integer which gets increased every time
   the object is changed. By saving and later querying the object state
   one can determine whether information about the object is still current.
   Currently, state is maintained for Vec and Mat objects.

   This routine is mostly for internal use by PETSc; a developer need only
   call it after explicit access to an object's internals. Routines such
   as VecSet or MatScale already call this routine.

   Level: developer

   seealso: PetscObjectGetState

   Concepts: state

@*/
int PetscObjectIncreaseState(PetscObject obj)
{
  PetscFunctionBegin;
  if (!obj) SETERRQ(PETSC_ERR_ARG_CORRUPT,"Null object");
  obj->state++;
  PetscFunctionReturn(0);
}

int globalcurrentstate = 0, globalmaxstate = 10;
int PetscRegisterComposedData(int *id)
{
  PetscFunctionBegin;
  *id = globalcurrentstate++;
  if (globalcurrentstate > globalmaxstate) globalmaxstate += 10;
  PetscFunctionReturn(0);
}

int PetscObjectIncreaseIntComposeData(PetscObject obj)
{
  int *ar = obj->intcomposeddata,*new_ar;
  int *ir = obj->intcomposedstate,*new_ir,
    n = obj->int_idmax,new_n,i,ierr;

  PetscFunctionBegin;
  new_n = globalmaxstate;
  ierr = PetscMalloc(new_n*sizeof(int),&new_ar); CHKERRQ(ierr);
  ierr = PetscMemzero(new_ar,new_n*sizeof(int)); CHKERRQ(ierr);
  ierr = PetscMalloc(new_n*sizeof(int),&new_ir); CHKERRQ(ierr);
  ierr = PetscMemzero(new_ir,new_n*sizeof(int)); CHKERRQ(ierr);
  if (n) {
    for (i=0; i<n; i++) {
      new_ar[i] = ar[i]; new_ir[i] = ir[i];
    }
    ierr = PetscFree(ar); CHKERRQ(ierr);
    ierr = PetscFree(ir); CHKERRQ(ierr);
  }
  obj->int_idmax = new_n;
  obj->intcomposeddata = new_ar; obj->intcomposedstate = new_ir;
  PetscFunctionReturn(0);
}

int PetscObjectIncreaseRealComposeData(PetscObject obj)
{
  PetscReal *ar = obj->realcomposeddata,*new_ar;
  int *ir = obj->realcomposedstate,*new_ir,
    n = obj->real_idmax,new_n,i,ierr;

  PetscFunctionBegin;
  new_n = globalmaxstate;
  ierr = PetscMalloc(new_n*sizeof(PetscReal),&new_ar); CHKERRQ(ierr);
  ierr = PetscMemzero(new_ar,new_n*sizeof(PetscReal)); CHKERRQ(ierr);
  ierr = PetscMalloc(new_n*sizeof(int),&new_ir); CHKERRQ(ierr);
  ierr = PetscMemzero(new_ir,new_n*sizeof(int)); CHKERRQ(ierr);
  if (n) {
    for (i=0; i<n; i++) {
      new_ar[i] = ar[i]; new_ir[i] = ir[i];
    }
    ierr = PetscFree(ar); CHKERRQ(ierr);
    ierr = PetscFree(ir); CHKERRQ(ierr);
  }
  obj->real_idmax = new_n;
  obj->realcomposeddata = new_ar; obj->realcomposedstate = new_ir;
  PetscFunctionReturn(0);
}

int PetscObjectIncreaseScalarComposeData(PetscObject obj)
{
  PetscScalar *ar = obj->scalarcomposeddata,*new_ar;
  int *ir = obj->scalarcomposedstate,*new_ir,
    n = obj->scalar_idmax,new_n,i,ierr;

  PetscFunctionBegin;
  new_n = globalmaxstate;
  ierr = PetscMalloc(new_n*sizeof(PetscScalar),&new_ar); CHKERRQ(ierr);
  ierr = PetscMemzero(new_ar,new_n*sizeof(PetscScalar)); CHKERRQ(ierr);
  ierr = PetscMalloc(new_n*sizeof(int),&new_ir); CHKERRQ(ierr);
  ierr = PetscMemzero(new_ir,new_n*sizeof(int)); CHKERRQ(ierr);
  if (n) {
    for (i=0; i<n; i++) {
      new_ar[i] = ar[i]; new_ir[i] = ir[i];
    }
    ierr = PetscFree(ar); CHKERRQ(ierr);
    ierr = PetscFree(ir); CHKERRQ(ierr);
  }
  obj->scalar_idmax = new_n;
  obj->scalarcomposeddata = new_ar; obj->scalarcomposedstate = new_ir;
  PetscFunctionReturn(0);
}

