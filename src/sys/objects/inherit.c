
/*
     Provides utility routines for manipulating any type of PETSc object.
*/
#include <petscsys.h>  /*I   "petscsys.h"    I*/

PetscObject *PetscObjects = 0;
PetscInt    PetscObjectsCounts = 0, PetscObjectsMaxCounts = 0;

extern PetscErrorCode PetscObjectGetComm_Petsc(PetscObject,MPI_Comm *);
extern PetscErrorCode PetscObjectCompose_Petsc(PetscObject,const char[],PetscObject);
extern PetscErrorCode PetscObjectQuery_Petsc(PetscObject,const char[],PetscObject *);
extern PetscErrorCode PetscObjectComposeFunction_Petsc(PetscObject,const char[],const char[],void (*)(void));
extern PetscErrorCode PetscObjectQueryFunction_Petsc(PetscObject,const char[],void (**)(void));

#undef __FUNCT__  
#define __FUNCT__ "PetscHeaderCreate_Private"
/*
   PetscHeaderCreate_Private - Creates a base PETSc object header and fills
   in the default values.  Called by the macro PetscHeaderCreate().
*/
PetscErrorCode  PetscHeaderCreate_Private(PetscObject h,PetscClassId classid,PetscInt type,const char class_name[],const char descr[],const char mansec[],
                                          MPI_Comm comm,PetscErrorCode (*des)(PetscObject*),PetscErrorCode (*vie)(PetscObject,PetscViewer))
{
  static PetscInt idcnt = 1;
  PetscErrorCode  ierr;
  PetscObject     *newPetscObjects;
  PetscInt         newPetscObjectsMaxCounts,i;

  PetscFunctionBegin;
  h->classid                = classid;
  h->type                   = type;
  h->class_name             = (char*)class_name;
  h->description            = (char*)descr;
  h->mansec                 = (char*)mansec;
  h->prefix                 = 0;
  h->refct                  = 1;
  h->amem                   = -1;
  h->id                     = idcnt++;
  h->parentid               = 0;
  h->qlist                  = 0;
  h->olist                  = 0;
  h->precision              = (PetscPrecision) sizeof(PetscScalar);
  h->bops->destroy          = des;
  h->bops->view             = vie;
  h->bops->getcomm          = PetscObjectGetComm_Petsc;
  h->bops->compose          = PetscObjectCompose_Petsc;
  h->bops->query            = PetscObjectQuery_Petsc;
  h->bops->composefunction  = PetscObjectComposeFunction_Petsc;
  h->bops->queryfunction    = PetscObjectQueryFunction_Petsc;
  ierr = PetscCommDuplicate(comm,&h->comm,&h->tag);CHKERRQ(ierr);

  /* Keep a record of object created */
  PetscObjectsCounts++;
  for (i=0; i<PetscObjectsMaxCounts; i++) {
    if (!PetscObjects[i]) {
      PetscObjects[i] = h;
      PetscFunctionReturn(0);
    }
  }
  /* Need to increase the space for storing PETSc objects */
  if (!PetscObjectsMaxCounts) newPetscObjectsMaxCounts = 100;
  else                        newPetscObjectsMaxCounts = 2*PetscObjectsMaxCounts;
  ierr = PetscMalloc(newPetscObjectsMaxCounts*sizeof(PetscObject),&newPetscObjects);CHKERRQ(ierr);
  ierr = PetscMemcpy(newPetscObjects,PetscObjects,PetscObjectsMaxCounts*sizeof(PetscObject));CHKERRQ(ierr);
  ierr = PetscMemzero(newPetscObjects+PetscObjectsMaxCounts,(newPetscObjectsMaxCounts - PetscObjectsMaxCounts)*sizeof(PetscObject));CHKERRQ(ierr);
  ierr = PetscFree(PetscObjects);CHKERRQ(ierr);
  PetscObjects                        = newPetscObjects;
  PetscObjects[PetscObjectsMaxCounts] = h;
  PetscObjectsMaxCounts               = newPetscObjectsMaxCounts;

  PetscFunctionReturn(0);
}

extern PetscBool      PetscMemoryCollectMaximumUsage;
extern PetscLogDouble PetscMemoryMaximumUsage;

#undef __FUNCT__  
#define __FUNCT__ "PetscHeaderDestroy_Private"
/*
    PetscHeaderDestroy_Private - Destroys a base PETSc object header. Called by 
    the macro PetscHeaderDestroy().
*/
PetscErrorCode  PetscHeaderDestroy_Private(PetscObject h)
{
  PetscErrorCode ierr;
  PetscInt       i;

  PetscFunctionBegin;
  PetscValidHeader(h,1);
#if defined(PETSC_HAVE_AMS)
  if (PetscAMSPublishAll) {
    ierr = PetscObjectUnPublish((PetscObject)h);CHKERRQ(ierr);
  }
#endif
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
  ierr = PetscOListDestroy(&h->olist);CHKERRQ(ierr);
  ierr = PetscCommDestroy(&h->comm);CHKERRQ(ierr);
  /* next destroy other things */
  h->classid = PETSCFREEDHEADER;
  ierr = PetscFree(h->bops);CHKERRQ(ierr);
  ierr = PetscFListDestroy(&h->qlist);CHKERRQ(ierr);
  ierr = PetscFree(h->type_name);CHKERRQ(ierr);
  ierr = PetscFree(h->name);CHKERRQ(ierr);
  ierr = PetscFree(h->prefix);CHKERRQ(ierr);
  ierr = PetscFree(h->fortran_func_pointers);CHKERRQ(ierr);

  /* Record object removal from list of all objects */
  for (i=0; i<PetscObjectsMaxCounts; i++) {
    if (PetscObjects[i] == h) {
      PetscObjects[i] = 0;
      PetscObjectsCounts--;
      break;
    }
  }
  if (!PetscObjectsCounts) {
    ierr = PetscFree(PetscObjects);CHKERRQ(ierr);
    PetscObjectsMaxCounts = 0;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscObjectCopyFortranFunctionPointers"
/*@C
   PetscObjectCopyFortranFunctionPointers - Copy function pointers to another object

   Logically Collective on PetscObject

   Input Parameter:
+  src - source object
-  dest - destination object

   Level: developer

   Note:
   Both objects must have the same class.
@*/
PetscErrorCode PetscObjectCopyFortranFunctionPointers(PetscObject src,PetscObject dest)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeader(src,1);
  PetscValidHeader(dest,2);
  if (src->classid != dest->classid) SETERRQ(src->comm,PETSC_ERR_ARG_INCOMP,"Objects must be of the same class");

  ierr = PetscFree(dest->fortran_func_pointers);CHKERRQ(ierr);
  ierr = PetscMalloc(src->num_fortran_func_pointers*sizeof(void(*)(void)),&dest->fortran_func_pointers);CHKERRQ(ierr);
  ierr = PetscMemcpy(dest->fortran_func_pointers,src->fortran_func_pointers,src->num_fortran_func_pointers*sizeof(void(*)(void)));CHKERRQ(ierr);
  dest->num_fortran_func_pointers = src->num_fortran_func_pointers;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscObjectsView"
/*@C
   PetscObjectsView - Prints the currently existing objects.

   Logically Collective on PetscViewer

   Input Parameter:
.  viewer - must be an PETSCVIEWERASCII viewer

   Level: advanced

   Concepts: options database^printing

@*/
PetscErrorCode  PetscObjectsView(PetscViewer viewer) 
{
  PetscErrorCode ierr;
  PetscInt       i;
  PetscBool      isascii;
  PetscObject    h;

  PetscFunctionBegin;
  if (!viewer) viewer = PETSC_VIEWER_STDOUT_WORLD;
  ierr = PetscTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&isascii);CHKERRQ(ierr);
  if (!isascii) SETERRQ(((PetscObject)viewer)->comm,PETSC_ERR_SUP,"Only supports ASCII viewer");

  for (i=0; i<PetscObjectsMaxCounts; i++) {
    if ((h = PetscObjects[i])) {
      ierr = PetscObjectName(h);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer,"%s %s %s\n",h->class_name,h->type_name,h->name);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscObjectsGetObject"
/*@C
   PetscObjectsGetObject - Get a pointer to a named object

   Not collective

   Input Parameter:
.  name - the name of an object

   Output Parameter:
.   obj - the object or null if there is no object

   Level: advanced

   Concepts: options database^printing

@*/
PetscErrorCode  PetscObjectsGetObject(const char* name,PetscObject *obj,char **classname) 
{
  PetscErrorCode ierr;
  PetscInt       i;
  PetscObject    h;
  PetscBool      flg;

  PetscFunctionBegin;
  *obj = PETSC_NULL;
  for (i=0; i<PetscObjectsMaxCounts; i++) {
    if ((h = PetscObjects[i])) {
      ierr = PetscObjectName(h);CHKERRQ(ierr);
      ierr = PetscStrcmp(h->name,name,&flg);CHKERRQ(ierr);
      if (flg) {
        *obj = h;
        if (classname) *classname = h->class_name;
        PetscFunctionReturn(0);
      }
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscObjectsGetObjectMatlab"
char* PetscObjectsGetObjectMatlab(const char* name,PetscObject *obj) 
{
  PetscErrorCode ierr;
  PetscInt       i;
  PetscObject    h;
  PetscBool      flg;

  PetscFunctionBegin;
  *obj = PETSC_NULL;
  for (i=0; i<PetscObjectsMaxCounts; i++) {
    if ((h = PetscObjects[i])) {
      ierr = PetscObjectName(h);if (ierr) PetscFunctionReturn(0);
      ierr = PetscStrcmp(h->name,name,&flg);if (ierr) PetscFunctionReturn(0);
      if (flg) {
        *obj = h;
        PetscFunctionReturn(h->class_name);
      }
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscObjectAddOptionsHandler"
/*@C
    PetscObjectAddOptionsHandler - Adds an additional function to check for options when XXXSetFromOptions() is called.

    Not Collective

    Input Parameter:
+   obj - the PETSc object
.   handle - function that checks for options
.   destroy - function to destroy context if provided
-   ctx - optional context for check function

    Level: developer


.seealso: KSPSetFromOptions(), PCSetFromOptions(), SNESSetFromOptions(), PetscObjectProcessOptionsHandlers(), PetscObjectDestroyOptionsHandlers()

@*/
PetscErrorCode  PetscObjectAddOptionsHandler(PetscObject obj,PetscErrorCode (*handle)(PetscObject,void*),PetscErrorCode (*destroy)(PetscObject,void*),void *ctx)
{
  PetscFunctionBegin;
  PetscValidHeader(obj,1);
  if (obj->noptionhandler >= PETSC_MAX_OPTIONS_HANDLER) SETERRQ(obj->comm,PETSC_ERR_ARG_OUTOFRANGE,"To many options handlers added");
  obj->optionhandler[obj->noptionhandler]   = handle;
  obj->optiondestroy[obj->noptionhandler]   = destroy;
  obj->optionctx[obj->noptionhandler++]     = ctx;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscObjectProcessOptionsHandlers"
/*@C
    PetscObjectProcessOptionsHandlers - Calls all the options handler attached to an object

    Not Collective

    Input Parameter:
.   obj - the PETSc object

    Level: developer


.seealso: KSPSetFromOptions(), PCSetFromOptions(), SNESSetFromOptions(), PetscObjectAddOptionsHandler(), PetscObjectDestroyOptionsHandlers()

@*/
PetscErrorCode  PetscObjectProcessOptionsHandlers(PetscObject obj)
{
  PetscInt       i;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeader(obj,1);
  for (i=0; i<obj->noptionhandler; i++) {
    ierr = (*obj->optionhandler[i])(obj,obj->optionctx[i]);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscObjectDestroyOptionsHandlers"
/*@C
    PetscObjectDestroyOptionsHandlers - Destroys all the option handlers attached to an objeft

    Not Collective

    Input Parameter:
.   obj - the PETSc object

    Level: developer


.seealso: KSPSetFromOptions(), PCSetFromOptions(), SNESSetFromOptions(), PetscObjectAddOptionsHandler(), PetscObjectProcessOptionsHandlers()

@*/
PetscErrorCode  PetscObjectDestroyOptionsHandlers(PetscObject obj)
{
  PetscInt       i;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeader(obj,1);
  for (i=0; i<obj->noptionhandler; i++) {
    ierr = (*obj->optiondestroy[i])(obj,obj->optionctx[i]);CHKERRQ(ierr);
  }
  obj->noptionhandler = 0;
  PetscFunctionReturn(0);
}


#undef __FUNCT__  
#define __FUNCT__ "PetscObjectReference"
/*@
   PetscObjectReference - Indicates to any PetscObject that it is being
   referenced by another PetscObject. This increases the reference
   count for that object by one.

   Logically Collective on PetscObject

   Input Parameter:
.  obj - the PETSc object. This must be cast with (PetscObject), for example, 
         PetscObjectReference((PetscObject)mat);

   Level: advanced

.seealso: PetscObjectCompose(), PetscObjectDereference()
@*/
PetscErrorCode  PetscObjectReference(PetscObject obj)
{
  PetscFunctionBegin;
  PetscValidHeader(obj,1);
  obj->refct++;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscObjectGetReference"
/*@
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
PetscErrorCode  PetscObjectGetReference(PetscObject obj,PetscInt *cnt)
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

   Collective on PetscObject if reference reaches 0 otherwise Logically Collective

   Input Parameter:
.  obj - the PETSc object; this must be cast with (PetscObject), for example, 
         PetscObjectDereference((PetscObject)mat);

   Notes: PetscObjectDestroy(PetscObject *obj)  sets the obj pointer to null after the call, this routine does not.

   Level: advanced

.seealso: PetscObjectCompose(), PetscObjectReference()
@*/
PetscErrorCode  PetscObjectDereference(PetscObject obj)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeader(obj,1);
  if (obj->bops->destroy) {
    ierr = (*obj->bops->destroy)(&obj);CHKERRQ(ierr);
  } else if (!--obj->refct) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"This PETSc object does not have a generic destroy routine");
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
  PetscValidHeader(obj,1);
  *comm = obj->comm;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscObjectRemoveReference"
PetscErrorCode PetscObjectRemoveReference(PetscObject obj,const char name[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeader(obj,1);
  ierr = PetscOListRemoveReference(&obj->olist,name);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscObjectCompose_Petsc"
PetscErrorCode PetscObjectCompose_Petsc(PetscObject obj,const char name[],PetscObject ptr)
{
  PetscErrorCode ierr;
  char           *tname;
  PetscBool      skipreference;

  PetscFunctionBegin;
  if (ptr) {
    ierr = PetscOListReverseFind(ptr->olist,obj,&tname,&skipreference);CHKERRQ(ierr);
    if (tname && !skipreference) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"An object cannot be composed with an object that was composed with it");
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
  PetscValidHeader(obj,1);
  ierr = PetscOListFind(obj->olist,name,ptr);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscObjectComposeFunction_Petsc"
PetscErrorCode PetscObjectComposeFunction_Petsc(PetscObject obj,const char name[],const char fname[],void (*ptr)(void))
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeader(obj,1);
  ierr = PetscFListAdd(&obj->qlist,name,fname,ptr);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscObjectQueryFunction_Petsc"
PetscErrorCode PetscObjectQueryFunction_Petsc(PetscObject obj,const char name[],void (**ptr)(void))
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeader(obj,1);
  ierr = PetscFListFind(obj->qlist,obj->comm,name,PETSC_FALSE,ptr);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

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
PetscErrorCode  PetscObjectCompose(PetscObject obj,const char name[],PetscObject ptr)
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
#define __FUNCT__ "PetscObjectSetPrecision"
/*@C
   PetscObjectSetPrecision - sets the precision used within a given object.
                       
   Collective on the PetscObject

   Input Parameters:
+  obj - the PETSc object; this must be cast with (PetscObject), for example, 
         PetscObjectCompose((PetscObject)mat,...);
-  precision - the precision

   Level: advanced

.seealso: PetscObjectQuery(), PetscContainerCreate()
@*/
PetscErrorCode  PetscObjectSetPrecision(PetscObject obj,PetscPrecision precision)
{
  PetscFunctionBegin;
  PetscValidHeader(obj,1);
  obj->precision = precision;
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
-  ptr - the other PETSc object associated with the PETSc object, this must be 
         cast with (PetscObject *)

   Level: advanced

   The reference count of neither object is increased in this call

   Concepts: objects^composing
   Concepts: composing objects
   Concepts: objects^querying
   Concepts: querying objects

.seealso: PetscObjectCompose()
@*/
PetscErrorCode  PetscObjectQuery(PetscObject obj,const char name[],PetscObject *ptr)
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
PetscErrorCode  PetscObjectComposeFunction(PetscObject obj,const char name[],const char fname[],void (*ptr)(void))
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeader(obj,1);
  PetscValidCharPointer(name,2);
  ierr = (*obj->bops->composefunction)(obj,name,fname,ptr);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscObjectQueryFunction"
/*@C
   PetscObjectQueryFunction - Gets a function associated with a given object.
                       
   Logically Collective on PetscObject

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
PetscErrorCode  PetscObjectQueryFunction(PetscObject obj,const char name[],void (**ptr)(void))
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

   Not Collective

   Input Parameter:
.  obj - the object created with PetscContainerCreate()

   Output Parameter:
.  ptr - the pointer value

   Level: advanced

.seealso: PetscContainerCreate(), PetscContainerDestroy(), 
          PetscContainerSetPointer()
@*/
PetscErrorCode  PetscContainerGetPointer(PetscContainer obj,void **ptr)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(obj,PETSC_CONTAINER_CLASSID,1);
  PetscValidPointer(ptr,2);
  *ptr = obj->ptr;
  PetscFunctionReturn(0);
}


#undef __FUNCT__  
#define __FUNCT__ "PetscContainerSetPointer"
/*@C
   PetscContainerSetPointer - Sets the pointer value contained in the container.

   Logically Collective on PetscContainer

   Input Parameters:
+  obj - the object created with PetscContainerCreate()
-  ptr - the pointer value

   Level: advanced

.seealso: PetscContainerCreate(), PetscContainerDestroy(), 
          PetscContainerGetPointer()
@*/
PetscErrorCode  PetscContainerSetPointer(PetscContainer obj,void *ptr)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(obj,PETSC_CONTAINER_CLASSID,1);
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

.seealso: PetscContainerCreate(), PetscContainerSetUserDestroy()
@*/
PetscErrorCode  PetscContainerDestroy(PetscContainer *obj)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  if (!*obj) PetscFunctionReturn(0);
  PetscValidHeaderSpecific(*obj,PETSC_CONTAINER_CLASSID,1);
  if (--((PetscObject)(*obj))->refct > 0) {*obj = 0; PetscFunctionReturn(0);}
  if ((*obj)->userdestroy) (*(*obj)->userdestroy)((*obj)->ptr);
  ierr = PetscHeaderDestroy(obj);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscContainerSetUserDestroy"
/*@C
   PetscContainerSetUserDestroy - Sets name of the user destroy function.

   Logically Collective on PetscContainer

   Input Parameter:
+  obj - an object that was created with PetscContainerCreate()
-  des - name of the user destroy function

   Level: advanced

.seealso: PetscContainerDestroy()
@*/
PetscErrorCode  PetscContainerSetUserDestroy(PetscContainer obj, PetscErrorCode (*des)(void*))
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(obj,PETSC_CONTAINER_CLASSID,1);
  obj->userdestroy = des;
  PetscFunctionReturn(0);
}

PetscClassId  PETSC_CONTAINER_CLASSID;

#undef __FUNCT__  
#define __FUNCT__ "PetscContainerCreate"
/*@C
   PetscContainerCreate - Creates a PETSc object that has room to hold
   a single pointer. This allows one to attach any type of data (accessible
   through a pointer) with the PetscObjectCompose() function to a PetscObject.
   The data item itself is attached by a call to PetscContainerSetPointer().

   Collective on MPI_Comm

   Input Parameters:
.  comm - MPI communicator that shares the object

   Output Parameters:
.  container - the container created

   Level: advanced

.seealso: PetscContainerDestroy(), PetscContainerSetPointer(), PetscContainerGetPointer()
@*/
PetscErrorCode  PetscContainerCreate(MPI_Comm comm,PetscContainer *container)
{
  PetscErrorCode ierr;
  PetscContainer contain;

  PetscFunctionBegin;
  PetscValidPointer(container,2);
  ierr = PetscHeaderCreate(contain,_p_PetscContainer,PetscInt,PETSC_CONTAINER_CLASSID,0,"PetscContainer","Container","Sys",comm,PetscContainerDestroy,0);CHKERRQ(ierr);
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
PetscErrorCode  PetscObjectSetFromOptions(PetscObject obj)
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
PetscErrorCode  PetscObjectSetUp(PetscObject obj)
{
  PetscFunctionBegin;
  PetscValidHeader(obj,1);
  PetscFunctionReturn(0);
}
