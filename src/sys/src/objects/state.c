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

/*@C
   PetscObjectSetComposedData - Attach a data item to an object, together
   with the current state.

   Not collective

   Input parameters:
+  obj    - any PETSc object, for example a Vec, Mat or KSP. This must be
            cast with a (PetscObject), for example, 
            PetscObjectGetState((PetscObject)mat,&state);
.  s      - the label under which the item is stored
.  type   - the data type; currently only PETSC_SCALAR and PETSC_REAL
            are supported.
-  result - the data item

   Notes: this routine is one of two that can be used to attach
   data item to an object, which can be retrieved and reused if the
   object has not been changed in the mean time. Presumably the data item
   is some expensive to compute quantity.

   Level: advanced

   seealso: PetscObjectGetComposedData

   Concepts: state

@*/
#undef __FUNCT__
#define __FUNCT__ "PetscObjectSetComposedData"
int PetscObjectSetComposedData(PetscObject obj,char *s,PetscDataType type,void *result)
{
  MPI_Comm comm; char *t;
  PetscObjectContainer container; void *store; int *istore,state,l,ierr;

  PetscFunctionBegin;
  if (!obj) SETERRQ(PETSC_ERR_ARG_CORRUPT,"Null object");

  ierr = PetscObjectGetComm(obj,&comm); CHKERRQ(ierr);
  ierr = PetscObjectGetState(obj,&state); CHKERRQ(ierr);

  /*
   * Make a container that stores the current state under label
   * <property>_state
   */
  ierr = PetscObjectContainerCreate(comm,&container); CHKERRQ(ierr);
  ierr = PetscMalloc(sizeof(int),&istore); CHKERRQ(ierr);
  istore[0] = state;
  ierr = PetscObjectContainerSetPointer
    (container,(void*)istore); CHKERRQ(ierr);
  ierr = PetscStrlen(s,&l); CHKERRQ(ierr);
  ierr = PetscMalloc((l+6)*sizeof(char),&t); CHKERRQ(ierr);
  sprintf(t,"%s_state",s);
  ierr = PetscObjectCompose
    (obj,t,(PetscObject)container); CHKERRQ(ierr);

  /*
   * Make a container that stores the user object
   */
  ierr = PetscObjectContainerCreate(comm,&container); CHKERRQ(ierr);
  {
    int size;
    switch (type) {
    case PETSC_SCALAR : size = sizeof(PetscScalar); break;
#if PETSC_SCALAR != PETSC_REAL
    case PETSC_REAL : size = sizeof(PetscReal); break;
#endif
    default : SETERRQ1(1,"Can not attach objects of type %d",(int)type);
    }
    ierr = PetscMalloc(size,&store); CHKERRQ(ierr);
    /*printf("storing data for reuse: %e\n",*(PetscReal*)result);*/
    ierr = PetscMemcpy(store,result,size); CHKERRQ(ierr);
  }
  ierr = PetscObjectContainerSetPointer
    (container,(void*)store); CHKERRQ(ierr);
  ierr = PetscObjectCompose
    (obj,s,(PetscObject)container); CHKERRQ(ierr);

  ierr = PetscFree(t); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
   PetscObjectGetComposedData - Attach a data item from an object, provided
   the current state of the object is the state under which the item was stored.

   Not collective

   Input parameters:
+  obj    - any PETSc object, for example a Vec, Mat or KSP. This must be
            cast with a (PetscObject), for example, 
            PetscObjectGetState((PetscObject)mat,&state);
.  s      - the label under which the item is stored
-  type   - the data type; currently only PETSC_SCALAR and PETSC_REAL
            are supported.

   Output parameters:
+  result - the data item
-  flg    - PETSC_TRUE is the item was present and its stored state
            equals the current state of the obj.

   Notes: this routine is one of two that can be used to attach
   data item to an object, which can be retrieved and reused if the
   object has not been changed in the mean time. Presumably the data item
   is some expensive to compute quantity.

   Level: advanced

   seealso: PetscObjectSetComposedData

   Concepts: state

@*/
#undef __FUNCT__
#define __FUNCT__ "PetscObjectGetComposedData"
int PetscObjectGetComposedData
(PetscObject obj,char *s,PetscDataType type,void *result,PetscTruth *flg)
{
  PetscObjectContainer container; void *store; int *istore,s1,l,ierr;
  char *t;

  PetscFunctionBegin;
  if (!obj) SETERRQ(PETSC_ERR_ARG_CORRUPT,"Null object");

  /*
   * Generate the label under which the state of the user object
   * was stored; test if anything is stored under this label
   */
  ierr = PetscStrlen(s,&l); CHKERRQ(ierr);
  ierr = PetscMalloc((l+6)*sizeof(char),&t); CHKERRQ(ierr);
  sprintf(t,"%s_state",s);
  ierr = PetscObjectQuery(obj,t,(PetscObject*)&container); CHKERRQ(ierr);
  *flg = PETSC_FALSE;
  if (container) {
    /*
     * If the state label exists, get its content and compare to current state
     */
    ierr = PetscObjectGetState(obj,&s1); CHKERRQ(ierr);
    ierr = PetscObjectContainerGetPointer
      (container,(void**)&istore); CHKERRQ(ierr);
    if (istore && s1==istore[0]) {
      /*
       * If the data is still valid, retrieve it
       */
      ierr = PetscObjectQuery(obj,s,(PetscObject*)&container); CHKERRQ(ierr);
      ierr = PetscObjectContainerGetPointer
	(container,(void**)&store); CHKERRQ(ierr);
      {
	int size;
	switch (type) {
	case PETSC_SCALAR : size = sizeof(PetscScalar); break;
#if PETSC_SCALAR != PETSC_REAL
	case PETSC_REAL : size = sizeof(PetscReal); break;
#endif
	default : SETERRQ1(1,"Cannot retrieve data of type %d",(int)type);
	}
	ierr = PetscMemcpy(result,store,size); CHKERRQ(ierr);
      }
      /*printf("reused scalar <%s>=%e\n",s,*(double*)result);*/
      *flg = PETSC_TRUE;
      PetscFunctionReturn(0);
    }
  }
  ierr = PetscFree(t); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

