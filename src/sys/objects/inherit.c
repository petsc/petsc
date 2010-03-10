#define PETSC_DLL
/*
     Provides utility routines for manipulating any type of PETSc object.
*/
#include "petscsys.h"  /*I   "petscsys.h"    I*/

EXTERN PetscErrorCode PetscObjectGetComm_Petsc(PetscObject,MPI_Comm *);
EXTERN PetscErrorCode PetscObjectCompose_Petsc(PetscObject,const char[],PetscObject);
EXTERN PetscErrorCode PetscObjectQuery_Petsc(PetscObject,const char[],PetscObject *);
EXTERN PetscErrorCode PetscObjectComposeFunction_Petsc(PetscObject,const char[],const char[],void (*)(void));
EXTERN PetscErrorCode PetscObjectQueryFunction_Petsc(PetscObject,const char[],void (**)(void));

#undef __FUNCT__  
#define __FUNCT__ "PetscHeaderCreate_Private"
/*
   PetscHeaderCreate_Private - Creates a base PETSc object header and fills
   in the default values.  Called by the macro PetscHeaderCreate().
*/
PetscErrorCode PETSC_DLLEXPORT PetscHeaderCreate_Private(PetscObject h,PetscCookie cookie,PetscInt type,const char class_name[],MPI_Comm comm,
                                         PetscErrorCode (*des)(PetscObject),PetscErrorCode (*vie)(PetscObject,PetscViewer))
{
  static PetscInt idcnt = 1;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  h->cookie                 = cookie;
  h->type                   = type;
  h->class_name             = (char*)class_name;
  h->prefix                 = 0;
  h->refct                  = 1;
  h->amem                   = -1;
  h->id                     = idcnt++;
  h->parentid               = 0;
  h->qlist                  = 0;
  h->olist                  = 0;
  h->bops->destroy          = des;
  h->bops->view             = vie;
  h->bops->getcomm          = PetscObjectGetComm_Petsc;
  h->bops->compose          = PetscObjectCompose_Petsc;
  h->bops->query            = PetscObjectQuery_Petsc;
  h->bops->composefunction  = PetscObjectComposeFunction_Petsc;
  h->bops->queryfunction    = PetscObjectQueryFunction_Petsc;
  ierr = PetscCommDuplicate(comm,&h->comm,&h->tag);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

extern PetscTruth     PetscMemoryCollectMaximumUsage;
extern PetscLogDouble PetscMemoryMaximumUsage;

#undef __FUNCT__  
#define __FUNCT__ "PetscHeaderDestroy_Private"
/*
    PetscHeaderDestroy_Private - Destroys a base PETSc object header. Called by 
    the macro PetscHeaderDestroy().
*/
PetscErrorCode PETSC_DLLEXPORT PetscHeaderDestroy_Private(PetscObject h)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (PetscMemoryCollectMaximumUsage) {
    PetscLogDouble usage;
    ierr = PetscMemoryGetCurrentUsage(&usage);CHKERRQ(ierr);
    if (usage > PetscMemoryMaximumUsage) PetscMemoryMaximumUsage = usage;
  }
  /* first destroy things that could execute arbitrary code */
  if (h->python_destroy) {
    void           *python_context          = h->python_context;
    PetscErrorCode (*python_destroy)(void*) = h->python_destroy;
    h->python_context = 0;
    h->python_destroy = 0;
    ierr = (*python_destroy)(python_context);CHKERRQ(ierr);
  }
  ierr = PetscOListDestroy(h->olist);CHKERRQ(ierr);
  ierr = PetscCommDestroy(&h->comm);CHKERRQ(ierr);
  /* next destroy other things */
  h->cookie = PETSCFREEDHEADER;
  ierr = PetscFree(h->bops);CHKERRQ(ierr);
  ierr = PetscFListDestroy(&h->qlist);CHKERRQ(ierr);
  ierr = PetscStrfree(h->type_name);CHKERRQ(ierr);
  ierr = PetscStrfree(h->name);CHKERRQ(ierr);
  ierr = PetscStrfree(h->prefix);CHKERRQ(ierr);
  ierr = PetscFree(h->fortran_func_pointers);CHKERRQ(ierr);
  ierr = PetscFree(h->intcomposeddata);CHKERRQ(ierr);
  ierr = PetscFree(h->intcomposedstate);CHKERRQ(ierr);
  ierr = PetscFree(h->realcomposeddata);CHKERRQ(ierr);
  ierr = PetscFree(h->realcomposedstate);CHKERRQ(ierr);
  ierr = PetscFree(h->scalarcomposeddata);CHKERRQ(ierr);
  ierr = PetscFree(h->scalarcomposedstate);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscObjectReference"
/*@C
   PetscObjectReference - Indicates to any PetscObject that it is being
   referenced by another PetscObject. This increases the reference
   count for that object by one.

   Collective on PetscObject

   Input Parameter:
.  obj - the PETSc object. This must be cast with (PetscObject), for example, 
         PetscObjectReference((PetscObject)mat);

   Level: advanced

.seealso: PetscObjectCompose(), PetscObjectDereference()
@*/
PetscErrorCode PETSC_DLLEXPORT PetscObjectReference(PetscObject obj)
{
  PetscFunctionBegin;
  PetscValidHeader(obj,1);
  obj->refct++;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscObjectGetReference"
/*@C
   PetscObjectGetReference - Gets the current reference count for 
   any PETSc object.

   Not Collective

   Input Parameter:
.  obj - the PETSc object; this must be cast with (PetscObject), for example, 
         PetscObjectGetReference((PetscObject)mat,&cnt);

   Output Parameter:
.  cnt - the reference count

   Level: advanced

.seealso: PetscObjectCompose(), PetscObjectDereference(), PetscObjectReference()
@*/
PetscErrorCode PETSC_DLLEXPORT PetscObjectGetReference(PetscObject obj,PetscInt *cnt)
{
  PetscFunctionBegin;
  PetscValidHeader(obj,1);
  PetscValidIntPointer(cnt,2);
  *cnt = obj->refct;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscObjectDereference"
/*@
   PetscObjectDereference - Indicates to any PetscObject that it is being
   referenced by one less PetscObject. This decreases the reference
   count for that object by one.

   Collective on PetscObject

   Input Parameter:
.  obj - the PETSc object; this must be cast with (PetscObject), for example, 
         PetscObjectDereference((PetscObject)mat);

   Level: advanced

.seealso: PetscObjectCompose(), PetscObjectReference()
@*/
PetscErrorCode PETSC_DLLEXPORT PetscObjectDereference(PetscObject obj)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeader(obj,1);
  if (obj->bops->destroy) {
    ierr = (*obj->bops->destroy)(obj);CHKERRQ(ierr);
  } else if (!--obj->refct) {
    SETERRQ(PETSC_ERR_SUP,"This PETSc object does not have a generic destroy routine");
  }
  PetscFunctionReturn(0);
}

/* ----------------------------------------------------------------------- */
/*
     The following routines are the versions private to the PETSc object
     data structures.
*/
#undef __FUNCT__  
#define __FUNCT__ "PetscObjectGetComm_Petsc"
PetscErrorCode PetscObjectGetComm_Petsc(PetscObject obj,MPI_Comm *comm)
{
  PetscFunctionBegin;
  *comm = obj->comm;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscObjectCompose_Petsc"
PetscErrorCode PetscObjectCompose_Petsc(PetscObject obj,const char name[],PetscObject ptr)
{
  PetscErrorCode ierr;
  char           *tname;

  PetscFunctionBegin;
  if (ptr) {
    ierr = PetscOListReverseFind(ptr->olist,obj,&tname);CHKERRQ(ierr);
    if (tname){
      SETERRQ(PETSC_ERR_ARG_INCOMP,"An object cannot be composed with an object that was compose with it");
    }
  }
  ierr = PetscOListAdd(&obj->olist,name,ptr);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscObjectQuery_Petsc"
PetscErrorCode PetscObjectQuery_Petsc(PetscObject obj,const char name[],PetscObject *ptr)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscOListFind(obj->olist,name,ptr);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscObjectComposeFunction_Petsc"
PetscErrorCode PetscObjectComposeFunction_Petsc(PetscObject obj,const char name[],const char fname[],void (*ptr)(void))
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFListAdd(&obj->qlist,name,fname,ptr);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscObjectQueryFunction_Petsc"
PetscErrorCode PetscObjectQueryFunction_Petsc(PetscObject obj,const char name[],void (**ptr)(void))
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFListFind(obj->qlist,obj->comm,name,ptr);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
        These are the versions that are usable to any CCA compliant objects
*/
#undef __FUNCT__  
#define __FUNCT__ "PetscObjectCompose"
/*@C
   PetscObjectCompose - Associates another PETSc object with a given PETSc object. 
                       
   Not Collective

   Input Parameters:
+  obj - the PETSc object; this must be cast with (PetscObject), for example, 
         PetscObjectCompose((PetscObject)mat,...);
.  name - name associated with the child object 
-  ptr - the other PETSc object to associate with the PETSc object; this must also be 
         cast with (PetscObject)

   Level: advanced

   Notes:
   The second objects reference count is automatically increased by one when it is
   composed.

   Replaces any previous object that had the same name.

   If ptr is null and name has previously been composed using an object, then that
   entry is removed from the obj.

   PetscObjectCompose() can be used with any PETSc object (such as
   Mat, Vec, KSP, SNES, etc.) or any user-provided object.  See 
   PetscContainerCreate() for info on how to create an object from a 
   user-provided pointer that may then be composed with PETSc objects.
   
   Concepts: objects^composing
   Concepts: composing objects

.seealso: PetscObjectQuery(), PetscContainerCreate()
@*/
PetscErrorCode PETSC_DLLEXPORT PetscObjectCompose(PetscObject obj,const char name[],PetscObject ptr)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeader(obj,1);
  PetscValidCharPointer(name,2);
  if (ptr) PetscValidHeader(ptr,3);
  ierr = (*obj->bops->compose)(obj,name,ptr);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}



#undef __FUNCT__  
#define __FUNCT__ "PetscObjectQuery"
/*@C
   PetscObjectQuery  - Gets a PETSc object associated with a given object.
                       
   Not Collective

   Input Parameters:
+  obj - the PETSc object
         Thus must be cast with a (PetscObject), for example, 
         PetscObjectCompose((PetscObject)mat,...);
.  name - name associated with child object 
-  ptr - the other PETSc object associated with the PETSc object, this must also be 
         cast with (PetscObject)

   Level: advanced

   Concepts: objects^composing
   Concepts: composing objects
   Concepts: objects^querying
   Concepts: querying objects

.seealso: PetscObjectQuery()
@*/
PetscErrorCode PETSC_DLLEXPORT PetscObjectQuery(PetscObject obj,const char name[],PetscObject *ptr)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeader(obj,1);
  PetscValidCharPointer(name,2);
  PetscValidPointer(ptr,3);
  ierr = (*obj->bops->query)(obj,name,ptr);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscObjectComposeFunction"
PetscErrorCode PETSC_DLLEXPORT PetscObjectComposeFunction(PetscObject obj,const char name[],const char fname[],void (*ptr)(void))
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeader(obj,1);
  PetscValidCharPointer(name,2);
  PetscValidCharPointer(fname,2);
  ierr = (*obj->bops->composefunction)(obj,name,fname,ptr);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscObjectQueryFunction"
/*@C
   PetscObjectQueryFunction - Gets a function associated with a given object.
                       
   Collective on PetscObject

   Input Parameters:
+  obj - the PETSc object; this must be cast with (PetscObject), for example, 
         PetscObjectQueryFunction((PetscObject)ksp,...);
-  name - name associated with the child function

   Output Parameter:
.  ptr - function pointer

   Level: advanced

   Concepts: objects^composing functions
   Concepts: composing functions
   Concepts: functions^querying
   Concepts: objects^querying
   Concepts: querying objects

.seealso: PetscObjectComposeFunctionDynamic()
@*/
PetscErrorCode PETSC_DLLEXPORT PetscObjectQueryFunction(PetscObject obj,const char name[],void (**ptr)(void))
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeader(obj,1);
  PetscValidCharPointer(name,2);
  ierr = (*obj->bops->queryfunction)(obj,name,ptr);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

struct _p_PetscContainer {
  PETSCHEADER(int);
  void   *ptr;
  PetscErrorCode (*userdestroy)(void*);
};

#undef __FUNCT__  
#define __FUNCT__ "PetscContainerGetPointer"
/*@C
   PetscContainerGetPointer - Gets the pointer value contained in the container.

   Collective on PetscContainer

   Input Parameter:
.  obj - the object created with PetscContainerCreate()

   Output Parameter:
.  ptr - the pointer value

   Level: advanced

.seealso: PetscContainerCreate(), PetscContainerDestroy(), 
          PetscContainerSetPointer()
@*/
PetscErrorCode PETSC_DLLEXPORT PetscContainerGetPointer(PetscContainer obj,void **ptr)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(obj,PETSC_CONTAINER_COOKIE,1);
  PetscValidPointer(ptr,2);
  *ptr = obj->ptr;
  PetscFunctionReturn(0);
}


#undef __FUNCT__  
#define __FUNCT__ "PetscContainerSetPointer"
/*@C
   PetscContainerSetPointer - Sets the pointer value contained in the container.

   Collective on PetscContainer

   Input Parameters:
+  obj - the object created with PetscContainerCreate()
-  ptr - the pointer value

   Level: advanced

.seealso: PetscContainerCreate(), PetscContainerDestroy(), 
          PetscContainerGetPointer()
@*/
PetscErrorCode PETSC_DLLEXPORT PetscContainerSetPointer(PetscContainer obj,void *ptr)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(obj,PETSC_CONTAINER_COOKIE,1);
  if (ptr) PetscValidPointer(ptr,2);
  obj->ptr = ptr;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscContainerDestroy"
/*@C
   PetscContainerDestroy - Destroys a PETSc container object.

   Collective on PetscContainer

   Input Parameter:
.  obj - an object that was created with PetscContainerCreate()

   Level: advanced

.seealso: PetscContainerCreate()
@*/
PetscErrorCode PETSC_DLLEXPORT PetscContainerDestroy(PetscContainer obj)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(obj,PETSC_CONTAINER_COOKIE,1);
  if (--((PetscObject)obj)->refct > 0) PetscFunctionReturn(0);
  if (obj->userdestroy) (*obj->userdestroy)(obj->ptr);
  ierr = PetscHeaderDestroy(obj);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscContainerSetUserDestroy"
/*@C
   PetscContainerSetUserDestroy - Sets name of the user destroy function.

   Collective on PetscContainer

   Input Parameter:
+  obj - an object that was created with PetscContainerCreate()
-  des - name of the user destroy function

   Level: advanced

.seealso: PetscContainerDestroy()
@*/
PetscErrorCode PETSC_DLLEXPORT PetscContainerSetUserDestroy(PetscContainer obj, PetscErrorCode (*des)(void*))
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(obj,PETSC_CONTAINER_COOKIE,1);
  obj->userdestroy = des;
  PetscFunctionReturn(0);
}

PetscCookie PETSC_DLLEXPORT PETSC_CONTAINER_COOKIE;

#undef __FUNCT__  
#define __FUNCT__ "PetscContainerCreate"
/*@C
   PetscContainerCreate - Creates a PETSc object that has room to hold
   a single pointer. This allows one to attach any type of data (accessible
   through a pointer) with the PetscObjectCompose() function to a PetscObject.
   The data item itself is attached by a call to PetscContainerSetPointer.

   Collective on MPI_Comm

   Input Parameters:
.  comm - MPI communicator that shares the object

   Output Parameters:
.  container - the container created

   Level: advanced

.seealso: PetscContainerDestroy(), PetscContainerSetPointer(), PetscContainerGetPointer()
@*/
PetscErrorCode PETSC_DLLEXPORT PetscContainerCreate(MPI_Comm comm,PetscContainer *container)
{
  PetscErrorCode ierr;
  PetscContainer contain;

  PetscFunctionBegin;
  PetscValidPointer(container,2);
  ierr = PetscHeaderCreate(contain,_p_PetscContainer,PetscInt,PETSC_CONTAINER_COOKIE,0,"PetscContainer",comm,PetscContainerDestroy,0);CHKERRQ(ierr);
  *container = contain;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscObjectSetFromOptions"
/*@
   PetscObjectSetFromOptions - Sets generic parameters from user options.

   Collective on obj

   Input Parameter:
.  obj - the PetscObjcet

   Options Database Keys:

   Notes:
   We have no generic options at present, so this does nothing

   Level: beginner

.keywords: set, options, database
.seealso: PetscObjectSetOptionsPrefix(), PetscObjectGetOptionsPrefix()
@*/
PetscErrorCode PETSC_DLLEXPORT PetscObjectSetFromOptions(PetscObject obj)
{
  PetscFunctionBegin;
  PetscValidHeader(obj,1);
  PetscFunctionReturn(0); 
}

#undef __FUNCT__  
#define __FUNCT__ "PetscObjectSetUp"
/*@
   PetscObjectSetUp - Sets up the internal data structures for the later use.

   Collective on PetscObject

   Input Parameters:
.  obj - the PetscObject

   Notes:
   This does nothing at present.

   Level: advanced

.keywords: setup
.seealso: PetscObjectDestroy()
@*/
PetscErrorCode PETSC_DLLEXPORT PetscObjectSetUp(PetscObject obj)
{
  PetscFunctionBegin;
  PetscValidHeader(obj,1);
  PetscFunctionReturn(0);
}
