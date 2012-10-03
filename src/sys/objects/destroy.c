
/*
     Provides utility routines for manulating any type of PETSc object.
*/
#include <petscsys.h>  /*I   "petscsys.h"    I*/


#undef __FUNCT__
#define __FUNCT__ "PetscComposedQuantitiesDestroy"
PetscErrorCode PetscComposedQuantitiesDestroy(PetscObject obj)
{
  PetscErrorCode ierr;
  PetscInt       i;

  PetscFunctionBegin;
  if (obj->intstar_idmax>0) {
    for (i=0; i<obj->intstar_idmax; i++) {
      ierr = PetscFree(obj->intstarcomposeddata[i]);CHKERRQ(ierr);
    }
    ierr = PetscFree(obj->intstarcomposeddata);CHKERRQ(ierr);
    ierr = PetscFree(obj->intstarcomposedstate);CHKERRQ(ierr);
  }
  if (obj->realstar_idmax>0) {
    for (i=0; i<obj->realstar_idmax; i++) {
      ierr = PetscFree(obj->realstarcomposeddata[i]);CHKERRQ(ierr);
    }
    ierr = PetscFree(obj->realstarcomposeddata);CHKERRQ(ierr);
    ierr = PetscFree(obj->realstarcomposedstate);CHKERRQ(ierr);
  }
  if (obj->scalarstar_idmax>0) {
    for (i=0; i<obj->scalarstar_idmax; i++) {
      ierr = PetscFree(obj->scalarstarcomposeddata[i]);CHKERRQ(ierr);
    }
    ierr = PetscFree(obj->scalarstarcomposeddata);CHKERRQ(ierr);
    ierr = PetscFree(obj->scalarstarcomposedstate);CHKERRQ(ierr);
  }
  ierr = PetscFree(obj->intcomposeddata);CHKERRQ(ierr);
  ierr = PetscFree(obj->intcomposedstate);CHKERRQ(ierr);
  ierr = PetscFree(obj->realcomposeddata);CHKERRQ(ierr);
  ierr = PetscFree(obj->realcomposedstate);CHKERRQ(ierr);
  ierr = PetscFree(obj->scalarcomposeddata);CHKERRQ(ierr);
  ierr = PetscFree(obj->scalarcomposedstate);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscObjectDestroy"
/*@
   PetscObjectDestroy - Destroys any PetscObject, regardless of the type.

   Collective on PetscObject

   Input Parameter:
.  obj - any PETSc object, for example a Vec, Mat or KSP.
         This must be cast with a (PetscObject*), for example,
         PetscObjectDestroy((PetscObject*)&mat);

   Level: beginner

    Concepts: destroying object
    Concepts: freeing object
    Concepts: deleting object

@*/
PetscErrorCode  PetscObjectDestroy(PetscObject *obj)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!*obj) PetscFunctionReturn(0);
  PetscValidHeader(*obj,1);
  if (*obj && (*obj)->bops->destroy) {
    ierr = (*(*obj)->bops->destroy)(obj);CHKERRQ(ierr);
  } else if (*obj) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_PLIB,"This PETSc object of class %s does not have a generic destroy routine",(*obj)->class_name);
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
PetscErrorCode  PetscObjectView(PetscObject obj,PetscViewer viewer)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeader(obj,1);
  if (!viewer) {
    ierr = PetscViewerASCIIGetStdout(obj->comm,&viewer);CHKERRQ(ierr);
  }
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_CLASSID,2);

  if (obj->bops->view) {
    ierr = (*obj->bops->view)(obj,viewer);CHKERRQ(ierr);
  } else {
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"This PETSc object does not have a generic viewer routine");
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscObjectTypeCompare"
/*@C
   PetscObjectTypeCompare - Determines whether a PETSc object is of a particular type.

   Not Collective

   Input Parameters:
+  obj - any PETSc object, for example a Vec, Mat or KSP.
         This must be cast with a (PetscObject), for example,
         PetscObjectTypeCompare((PetscObject)mat);
-  type_name - string containing a type name

   Output Parameter:
.  same - PETSC_TRUE if they are the same, else PETSC_FALSE

   Level: intermediate

.seealso: VecGetType(), KSPGetType(), PCGetType(), SNESGetType()

   Concepts: comparing^object types
   Concepts: types^comparing
   Concepts: object type^comparpeing

@*/
PetscErrorCode  PetscObjectTypeCompare(PetscObject obj,const char type_name[],PetscBool  *same)
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

#undef __FUNCT__
#define __FUNCT__ "PetscObjectTypeCompareAny"
/*@C
   PetscObjectTypeCompareAny - Determines whether a PETSc object is of any of a list of types.

   Not Collective

   Input Parameters:
+  obj - any PETSc object, for example a Vec, Mat or KSP.
         This must be cast with a (PetscObject), for example, PetscObjectTypeCompareAny((PetscObject)mat,...);
-  type_name - string containing a type name, pass the empty string "" to terminate the list

   Output Parameter:
.  match - PETSC_TRUE if the type of obj matches any in the list, else PETSC_FALSE

   Level: intermediate

.seealso: VecGetType(), KSPGetType(), PCGetType(), SNESGetType(), PetscObjectTypeCompare()

   Concepts: comparing^object types
   Concepts: types^comparing
   Concepts: object type^comparing

@*/
PetscErrorCode PetscObjectTypeCompareAny(PetscObject obj,PetscBool *match,const char type_name[],...)
{
  PetscErrorCode ierr;
  va_list Argp;

  PetscFunctionBegin;
  *match = PETSC_FALSE;
  va_start(Argp,type_name);
  while (type_name && type_name[0]) {
    PetscBool found;
    ierr = PetscObjectTypeCompare(obj,type_name,&found);CHKERRQ(ierr);
    if (found) {
      *match = PETSC_TRUE;
      break;
    }
    type_name = va_arg(Argp,const char*);
  }
  va_end(Argp);
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

   Logically Collective on PetscObject

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
PetscErrorCode  PetscObjectRegisterDestroy(PetscObject obj)
{
  PetscFunctionBegin;
  PetscValidHeader(obj,1);
  if (PetscObjectRegisterDestroy_Count < MAXREGDESOBJS) {
    PetscObjectRegisterDestroy_Objects[PetscObjectRegisterDestroy_Count++] = obj;
  } else SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_PLIB,"No more room in array, limit %d \n recompile src/sys/objects/destroy.c with larger value for MAXREGDESOBJS\n",MAXREGDESOBJS);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscObjectRegisterDestroyAll"
/*@C
   PetscObjectRegisterDestroyAll - Frees all the PETSc objects that have been registered
     with PetscObjectRegisterDestroy(). Called by PetscFinalize()

   Logically Collective on individual PetscObjects

   Level: developer

.seealso: PetscObjectRegisterDestroy()
@*/
PetscErrorCode  PetscObjectRegisterDestroyAll(void)
{
  PetscErrorCode ierr;
  PetscInt       i;

  PetscFunctionBegin;
  for (i=0; i<PetscObjectRegisterDestroy_Count; i++) {
    ierr = PetscObjectDestroy(&PetscObjectRegisterDestroy_Objects[i]);CHKERRQ(ierr);
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
PetscErrorCode  PetscRegisterFinalize(PetscErrorCode (*f)(void))
{
  PetscInt i;

  PetscFunctionBegin;
  for (i=0; i<PetscRegisterFinalize_Count; i++) {
    if (f == PetscRegisterFinalize_Functions[i]) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Called twice with same function to register");
  }
  if (PetscRegisterFinalize_Count < MAXREGFIN) {
    PetscRegisterFinalize_Functions[PetscRegisterFinalize_Count++] = f;
  } else SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_PLIB,"No more room in array, limit %d \n recompile src/sys/objects/destroy.c with larger value for MAXREGFIN\n",MAXREGFIN);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscRegisterFinalizeAll"
/*@C
   PetscRegisterFinalizeAll - Runs all the finalize functions set with PetscRegisterFinalize()

   Not Collective unless registered functions are collective

   Level: developer

.seealso: PetscRegisterFinalize()
@*/
PetscErrorCode  PetscRegisterFinalizeAll(void)
{
  PetscErrorCode ierr;
  PetscInt       i;

  PetscFunctionBegin;
  for (i=0; i<PetscRegisterFinalize_Count; i++) {
    ierr = (*PetscRegisterFinalize_Functions[i])();CHKERRQ(ierr);
  }
  PetscRegisterFinalize_Count = 0;
  PetscFunctionReturn(0);
}
