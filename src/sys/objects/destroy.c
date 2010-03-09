#define PETSC_DLL
/*
     Provides utility routines for manulating any type of PETSc object.
*/
#include "petscsys.h"  /*I   "petscsys.h"    I*/

typedef struct _p_GenericObject* GenericObject;

struct _p_GenericObject {
  PETSCHEADER(int);
};

PetscErrorCode PetscObjectDestroy_GenericObject(GenericObject obj)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeader(obj,1);
  if (--((PetscObject)obj)->refct > 0) PetscFunctionReturn(0);
  ierr = PetscHeaderDestroy(obj);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode PetscComposedQuantitiesDestroy(PetscObject obj)
{
  PetscErrorCode ierr; int i;
  PetscFunctionBegin;
  if (obj->intstar_idmax>0) {
    for (i=0; i<obj->intstar_idmax; i++) {
      if (obj->intstarcomposeddata[i]) {
	ierr = PetscFree(obj->intstarcomposeddata[i]);CHKERRQ(ierr);
	obj->intstarcomposeddata[i] = 0;
      }
    }
    ierr = PetscFree(obj->intstarcomposeddata);CHKERRQ(ierr);
    ierr = PetscFree(obj->intstarcomposedstate);CHKERRQ(ierr);
  }
  if (obj->realstar_idmax>0) {
    for (i=0; i<obj->realstar_idmax; i++) {
      if (obj->realstarcomposeddata[i]) {
	ierr = PetscFree(obj->realstarcomposeddata[i]);CHKERRQ(ierr);
	obj->realstarcomposeddata[i] = 0;
      }
    }
    ierr = PetscFree(obj->realstarcomposeddata);CHKERRQ(ierr);
    ierr = PetscFree(obj->realstarcomposedstate);CHKERRQ(ierr);
  }
  if (obj->scalarstar_idmax>0) {
    for (i=0; i<obj->scalarstar_idmax; i++) {
      if (obj->scalarstarcomposeddata[i]) {
	ierr = PetscFree(obj->scalarstarcomposeddata[i]);CHKERRQ(ierr);
	obj->scalarstarcomposeddata[i] = 0;
      }
    }
    ierr = PetscFree(obj->scalarstarcomposeddata);CHKERRQ(ierr);
    ierr = PetscFree(obj->scalarstarcomposedstate);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscObjectCreate"
/*@C
   PetscObjectCreate - Creates a PetscObject

   Collective on PetscObject

   Input Parameter:
.  comm - An MPI communicator

   Output Parameter:
.  obj - The object

   Level: developer

   Notes: This is a template intended as a starting point to cut and paste with PetscObjectDestroy_GenericObject()
          to make new object classes.

    Concepts: destroying object
    Concepts: freeing object
    Concepts: deleting object

@*/
PetscErrorCode PETSC_DLLEXPORT PetscObjectCreate(MPI_Comm comm, PetscObject *obj)
{
  GenericObject  o;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidPointer(obj,2);
#if !defined(PETSC_USE_DYNAMIC_LIBRARIES)
  ierr = PetscInitializePackage(PETSC_NULL);CHKERRQ(ierr);
#endif
  ierr = PetscHeaderCreate(o,_p_GenericObject,-1,PETSC_OBJECT_COOKIE,0,"PetscObject",comm,PetscObjectDestroy_GenericObject,0);CHKERRQ(ierr);
  /* records not yet defined in PetscObject 
  o->data        = 0;
  o->setupcalled = 0;
  */
  *obj = (PetscObject)o;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscObjectCreateGeneric"
/*@C
   PetscObjectCreateGeneric - Creates a PetscObject

   Collective on PetscObject

   Input Parameter:
+  comm - An MPI communicator
.  cookie - The class cookie
-  name - The class name

   Output Parameter:
.  obj - The object

   Level: developer

   Notes: This is a template intended as a starting point to cut and paste with PetscObjectDestroy_GenericObject()
          to make new object classes.

    Concepts: destroying object
    Concepts: freeing object
    Concepts: deleting object

@*/
PetscErrorCode PETSC_DLLEXPORT PetscObjectCreateGeneric(MPI_Comm comm, PetscCookie cookie, const char name[], PetscObject *obj)
{
  GenericObject  o;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidPointer(obj,2);
#if !defined(PETSC_USE_DYNAMIC_LIBRARIES)
  ierr = PetscInitializePackage(PETSC_NULL);CHKERRQ(ierr);
#endif
  ierr = PetscHeaderCreate(o,_p_GenericObject,-1,cookie,0,name,comm,PetscObjectDestroy_GenericObject,0);CHKERRQ(ierr);
  /* records not yet defined in PetscObject 
  o->data        = 0;
  o->setupcalled = 0;
  */
  *obj = (PetscObject)o;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscObjectDestroy"
/*@
   PetscObjectDestroy - Destroys any PetscObject, regardless of the type. 

   Collective on PetscObject

   Input Parameter:
.  obj - any PETSc object, for example a Vec, Mat or KSP.
         This must be cast with a (PetscObject), for example, 
         PetscObjectDestroy((PetscObject)mat);

   Level: beginner

    Concepts: destroying object
    Concepts: freeing object
    Concepts: deleting object

@*/
PetscErrorCode PETSC_DLLEXPORT PetscObjectDestroy(PetscObject obj)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeader(obj,1);
  if (obj->bops->destroy) {
    ierr = (*obj->bops->destroy)(obj);CHKERRQ(ierr);
  } else {
    SETERRQ1(PETSC_ERR_PLIB,"This PETSc object of class %s does not have a generic destroy routine",obj->class_name);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscObjectView" 
/*@C
   PetscObjectView - Views any PetscObject, regardless of the type. 

   Collective on PetscObject

   Input Parameters:
+  obj - any PETSc object, for example a Vec, Mat or KSP.
         This must be cast with a (PetscObject), for example, 
         PetscObjectView((PetscObject)mat,viewer);
-  viewer - any PETSc viewer

   Level: intermediate

@*/
PetscErrorCode PETSC_DLLEXPORT PetscObjectView(PetscObject obj,PetscViewer viewer)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeader(obj,1);
  if (!viewer) {
    ierr = PetscViewerASCIIGetStdout(obj->comm,&viewer);CHKERRQ(ierr);
  }
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_COOKIE,2);

  if (obj->bops->view) {
    ierr = (*obj->bops->view)(obj,viewer);CHKERRQ(ierr);
  } else {
    SETERRQ(PETSC_ERR_SUP,"This PETSc object does not have a generic viewer routine");
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscTypeCompare"
/*@C
   PetscTypeCompare - Determines whether a PETSc object is of a particular type.

   Not Collective

   Input Parameters:
+  obj - any PETSc object, for example a Vec, Mat or KSP.
         This must be cast with a (PetscObject), for example, 
         PetscObjectDestroy((PetscObject)mat);
-  type_name - string containing a type name

   Output Parameter:
.  same - PETSC_TRUE if they are the same, else PETSC_FALSE
  
   Level: intermediate

.seealso: VecGetType(), KSPGetType(), PCGetType(), SNESGetType()

   Concepts: comparing^object types
   Concepts: types^comparing
   Concepts: object type^comparing

@*/
PetscErrorCode PETSC_DLLEXPORT PetscTypeCompare(PetscObject obj,const char type_name[],PetscTruth *same)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!obj) {
    *same = PETSC_FALSE;
  } else if (!type_name && !obj->type_name) {
    *same = PETSC_TRUE;
  } else if (!type_name || !obj->type_name) {
    *same = PETSC_FALSE;
  } else {
    PetscValidHeader(obj,1);
    PetscValidCharPointer(type_name,2);
    PetscValidPointer(same,3);
    ierr = PetscStrcmp((char*)(obj->type_name),type_name,same);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#define MAXREGDESOBJS 256
static int         PetscObjectRegisterDestroy_Count = 0;
static PetscObject PetscObjectRegisterDestroy_Objects[MAXREGDESOBJS];

#undef __FUNCT__  
#define __FUNCT__ "PetscObjectRegisterDestroy"
/*@C
   PetscObjectRegisterDestroy - Registers a PETSc object to be destroyed when
     PetscFinalize() is called.

   Collective on PetscObject

   Input Parameter:
.  obj - any PETSc object, for example a Vec, Mat or KSP.
         This must be cast with a (PetscObject), for example, 
         PetscObjectRegisterDestroy((PetscObject)mat);

   Level: developer

   Notes:
      This is used by, for example, PETSC_VIEWER_XXX_() routines to free the viewer
    when PETSc ends.

.seealso: PetscObjectRegisterDestroyAll()
@*/
PetscErrorCode PETSC_DLLEXPORT PetscObjectRegisterDestroy(PetscObject obj)
{
  PetscFunctionBegin;
  PetscValidHeader(obj,1);
  if (PetscObjectRegisterDestroy_Count < MAXREGDESOBJS) {
    PetscObjectRegisterDestroy_Objects[PetscObjectRegisterDestroy_Count++] = obj;
  } else {
    SETERRQ1(PETSC_ERR_PLIB,"No more room in array, limit %d \n recompile src/sys/objects/destroy.c with larger value for MAXREGDESOBJS\n",MAXREGDESOBJS);
    
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscObjectRegisterDestroyAll"
/*@C
   PetscObjectRegisterDestroyAll - Frees all the PETSc objects that have been registered
     with PetscObjectRegisterDestroy(). Called by PetscFinalize()

   Collective on individual PetscObjects

   Level: developer

.seealso: PetscObjectRegisterDestroy()
@*/
PetscErrorCode PETSC_DLLEXPORT PetscObjectRegisterDestroyAll(void)
{
  PetscErrorCode ierr;
  int i;

  PetscFunctionBegin;
  for (i=0; i<PetscObjectRegisterDestroy_Count; i++) {
    ierr = PetscObjectDestroy(PetscObjectRegisterDestroy_Objects[i]);CHKERRQ(ierr);
  }
  PetscObjectRegisterDestroy_Count = 0;
  PetscFunctionReturn(0);
}


#define MAXREGFIN 256
static int         PetscRegisterFinalize_Count = 0;
static PetscErrorCode ((*PetscRegisterFinalize_Functions[MAXREGFIN])(void));

#undef __FUNCT__  
#define __FUNCT__ "PetscRegisterFinalize"
/*@C
   PetscRegisterFinalize - Registers a function that is to be called in PetscFinalize()

   Not Collective

   Input Parameter:
.  PetscErrorCode (*fun)(void) - 

   Level: developer

   Notes:
      This is used by, for example, DMInitializePackage() to have DMFinalizePackage() called

.seealso: PetscRegisterFinalizeAll()
@*/
PetscErrorCode PETSC_DLLEXPORT PetscRegisterFinalize(PetscErrorCode (*f)(void))
{
  PetscFunctionBegin;

  if (PetscRegisterFinalize_Count < MAXREGFIN) {
    PetscRegisterFinalize_Functions[PetscRegisterFinalize_Count++] = f;
  } else {
    SETERRQ1(PETSC_ERR_PLIB,"No more room in array, limit %d \n recompile src/sys/objects/destroy.c with larger value for MAXREGFIN\n",MAXREGFIN);
    
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetsRegisterFinalizeAll"
/*@C
   PetscRegisterFinalizeAll - Runs all the finalize functions set with PetscRegisterFinalize()

   Not Collective unless registered functions are collective

   Level: developer

.seealso: PetscRegisterFinalize()
@*/
PetscErrorCode PETSC_DLLEXPORT PetscRegisterFinalizeAll(void)
{
  PetscErrorCode ierr;
  int i;

  PetscFunctionBegin;
  for (i=0; i<PetscRegisterFinalize_Count; i++) {
    ierr = (*PetscRegisterFinalize_Functions[i])();CHKERRQ(ierr);
  }
  PetscRegisterFinalize_Count = 0;
  PetscFunctionReturn(0);
}
