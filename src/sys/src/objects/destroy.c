#define PETSC_DLL
/*
     Provides utility routines for manulating any type of PETSc object.
*/
#include "petsc.h"  /*I   "petsc.h"    I*/

struct _p_Object {
  PETSCHEADER(int);
};

#undef __FUNCT__  
#define __FUNCT__ "PetscObjectCreate"
/*@C
   PetscObjectCreate - Creates a PetscObject

   Collective on PetscObject

   Input Parameter:
.  comm - An MPI communicator

   Output Parameter:
.  obj - The object

   Level: beginner

    Concepts: destroying object
    Concepts: freeing object
    Concepts: deleting object

@*/
PetscErrorCode PETSC_DLLEXPORT PetscObjectCreate(MPI_Comm comm, PetscObject *obj)
{
  PetscObject    o;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidPointer(obj,2);

  ierr = PetscHeaderCreate(o,_p_Object,PetscInt,-1,0,"PetscObject",comm,PetscObjectDestroy,0);CHKERRQ(ierr);
  /* records not yet defined in PetscObject 
  o->data        = 0;
  o->setupcalled = 0;
  */
  *obj = o;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscObjectDestroy"
/*@C
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
  if (!viewer) viewer = PETSC_VIEWER_STDOUT_(obj->comm);
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

static int         PetscObjectRegisterDestroy_Count = 0;
static PetscObject PetscObjectRegisterDestroy_Objects[256];

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
  PetscObjectRegisterDestroy_Objects[PetscObjectRegisterDestroy_Count++] = obj;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscObjectRegisterDestroyAll"
/*@C
   PetscObjectRegisterDestroyAll - Frees all the PETSc objects that have been registered
     with PetscObjectRegisterDestroy(). Called by PetscFinalize()
     PetscFinalize() is called.

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
  PetscFunctionReturn(0);
}


