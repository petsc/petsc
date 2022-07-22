
/*
     Provides utility routines for manulating any type of PETSc object.
*/
#include <petsc/private/petscimpl.h>  /*I   "petscsys.h"    I*/
#include <petscviewer.h>

PetscErrorCode PetscComposedQuantitiesDestroy(PetscObject obj)
{
  PetscInt i;

  PetscFunctionBegin;
  if (obj->intstar_idmax>0) {
    for (i=0; i<obj->intstar_idmax; i++) PetscCall(PetscFree(obj->intstarcomposeddata[i]));
    PetscCall(PetscFree2(obj->intstarcomposeddata,obj->intstarcomposedstate));
  }
  if (obj->realstar_idmax>0) {
    for (i=0; i<obj->realstar_idmax; i++) PetscCall(PetscFree(obj->realstarcomposeddata[i]));
    PetscCall(PetscFree2(obj->realstarcomposeddata,obj->realstarcomposedstate));
  }
  if (obj->scalarstar_idmax>0) {
    for (i=0; i<obj->scalarstar_idmax; i++) PetscCall(PetscFree(obj->scalarstarcomposeddata[i]));
    PetscCall(PetscFree2(obj->scalarstarcomposeddata,obj->scalarstarcomposedstate));
  }
  PetscCall(PetscFree2(obj->intcomposeddata,obj->intcomposedstate));
  PetscCall(PetscFree2(obj->realcomposeddata,obj->realcomposedstate));
  PetscCall(PetscFree2(obj->scalarcomposeddata,obj->scalarcomposedstate));
  PetscFunctionReturn(0);
}

/*@
   PetscObjectDestroy - Destroys any PetscObject, regardless of the type.

   Collective on PetscObject

   Input Parameter:
.  obj - any PETSc object, for example a Vec, Mat or KSP.
         This must be cast with a (PetscObject*), for example,
         PetscObjectDestroy((PetscObject*)&mat);

   Level: beginner

@*/
PetscErrorCode  PetscObjectDestroy(PetscObject *obj)
{
  PetscFunctionBegin;
  if (!obj || !*obj) PetscFunctionReturn(0);
  PetscValidHeader(*obj,1);
  PetscCheck((*obj)->bops->destroy,PETSC_COMM_SELF,PETSC_ERR_PLIB,"This PETSc object of class %s does not have a generic destroy routine",(*obj)->class_name);
  PetscCall((*(*obj)->bops->destroy)(obj));
  PetscFunctionReturn(0);
}

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
  PetscFunctionBegin;
  PetscValidHeader(obj,1);
  PetscCheck(obj->bops->view,PETSC_COMM_SELF,PETSC_ERR_SUP,"This PETSc object does not have a generic viewer routine");
  if (!viewer) PetscCall(PetscViewerASCIIGetStdout(obj->comm,&viewer));
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_CLASSID,2);

  PetscCall((*obj->bops->view)(obj,viewer));
  PetscFunctionReturn(0);
}

/*@C
  PetscObjectViewFromOptions - Processes command line options to determine if/how a PetscObject is to be viewed.

  Collective on PetscObject

  Input Parameters:
+ obj   - the object
. bobj  - optional other object that provides prefix (if NULL then the prefix in obj is used)
- optionname - option string that is used to activate viewing

  Level: intermediate

@*/
PetscErrorCode PetscObjectViewFromOptions(PetscObject obj,PetscObject bobj,const char optionname[])
{
  PetscViewer       viewer;
  PetscBool         flg;
  static PetscBool  incall = PETSC_FALSE;
  PetscViewerFormat format;
  const char        *prefix;

  PetscFunctionBegin;
  if (incall) PetscFunctionReturn(0);
  incall = PETSC_TRUE;
  prefix = bobj ? bobj->prefix : obj->prefix;
  PetscCall(PetscOptionsGetViewer(PetscObjectComm((PetscObject)obj),obj->options,prefix,optionname,&viewer,&format,&flg));
  if (flg) {
    PetscCall(PetscViewerPushFormat(viewer,format));
    PetscCall(PetscObjectView(obj,viewer));
    PetscCall(PetscViewerFlush(viewer));
    PetscCall(PetscViewerPopFormat(viewer));
    PetscCall(PetscViewerDestroy(&viewer));
  }
  incall = PETSC_FALSE;
  PetscFunctionReturn(0);
}

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

.seealso: `VecGetType()`, `KSPGetType()`, `PCGetType()`, `SNESGetType()`, `PetscObjectBaseTypeCompare()`, `PetscObjectTypeCompareAny()`, `PetscObjectBaseTypeCompareAny()`

@*/
PetscErrorCode  PetscObjectTypeCompare(PetscObject obj,const char type_name[],PetscBool  *same)
{
  PetscFunctionBegin;
  PetscValidBoolPointer(same,3);
  if (!obj) *same = PETSC_FALSE;
  else if (!type_name && !obj->type_name) *same = PETSC_TRUE;
  else if (!type_name || !obj->type_name) *same = PETSC_FALSE;
  else {
    PetscValidHeader(obj,1);
    PetscValidCharPointer(type_name,2);
    PetscCall(PetscStrcmp((char*)(obj->type_name),type_name,same));
  }
  PetscFunctionReturn(0);
}

/*@C
   PetscObjectBaseTypeCompare - Determines whether a PetscObject is of a given base type. For example the base type of MATSEQAIJPERM is MATSEQAIJ

   Not Collective

   Input Parameters:
+  mat - the matrix
-  type_name - string containing a type name

   Output Parameter:
.  same - PETSC_TRUE if it is of the same base type

   Level: intermediate

.seealso: `PetscObjectTypeCompare()`, `PetscObjectTypeCompareAny()`, `PetscObjectBaseTypeCompareAny()`

@*/
PetscErrorCode  PetscObjectBaseTypeCompare(PetscObject obj,const char type_name[],PetscBool  *same)
{
  PetscFunctionBegin;
  PetscValidBoolPointer(same,3);
  if (!obj) *same = PETSC_FALSE;
  else if (!type_name && !obj->type_name) *same = PETSC_TRUE;
  else if (!type_name || !obj->type_name) *same = PETSC_FALSE;
  else {
    PetscValidHeader(obj,1);
    PetscValidCharPointer(type_name,2);
    PetscCall(PetscStrbeginswith((char*)(obj->type_name),type_name,same));
  }
  PetscFunctionReturn(0);
}

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

.seealso: `VecGetType()`, `KSPGetType()`, `PCGetType()`, `SNESGetType()`, `PetscObjectTypeCompare()`, `PetscObjectBaseTypeCompare()`

@*/
PetscErrorCode PetscObjectTypeCompareAny(PetscObject obj,PetscBool *match,const char type_name[],...)
{
  va_list Argp;

  PetscFunctionBegin;
  PetscValidBoolPointer(match,2);
  *match = PETSC_FALSE;
  if (!obj) PetscFunctionReturn(0);
  va_start(Argp,type_name);
  while (type_name && type_name[0]) {
    PetscBool found;
    PetscCall(PetscObjectTypeCompare(obj,type_name,&found));
    if (found) {
      *match = PETSC_TRUE;
      break;
    }
    type_name = va_arg(Argp,const char*);
  }
  va_end(Argp);
  PetscFunctionReturn(0);
}

/*@C
   PetscObjectBaseTypeCompareAny - Determines whether a PETSc object has the base type of any of a list of types.

   Not Collective

   Input Parameters:
+  obj - any PETSc object, for example a Vec, Mat or KSP.
         This must be cast with a (PetscObject), for example, PetscObjectBaseTypeCompareAny((PetscObject)mat,...);
-  type_name - string containing a type name, pass the empty string "" to terminate the list

   Output Parameter:
.  match - PETSC_TRUE if the type of obj matches any in the list, else PETSC_FALSE

   Level: intermediate

.seealso: `VecGetType()`, `KSPGetType()`, `PCGetType()`, `SNESGetType()`, `PetscObjectTypeCompare()`, `PetscObjectBaseTypeCompare()`, `PetscObjectTypeCompareAny()`

@*/
PetscErrorCode PetscObjectBaseTypeCompareAny(PetscObject obj,PetscBool *match,const char type_name[],...)
{
  va_list Argp;

  PetscFunctionBegin;
  PetscValidBoolPointer(match,2);
  *match = PETSC_FALSE;
  va_start(Argp,type_name);
  while (type_name && type_name[0]) {
    PetscBool found;
    PetscCall(PetscObjectBaseTypeCompare(obj,type_name,&found));
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

.seealso: `PetscObjectRegisterDestroyAll()`
@*/
PetscErrorCode  PetscObjectRegisterDestroy(PetscObject obj)
{
  PetscFunctionBegin;
  PetscValidHeader(obj,1);
  PetscCheck(PetscObjectRegisterDestroy_Count < MAXREGDESOBJS,PETSC_COMM_SELF,PETSC_ERR_PLIB,"No more room in array, limit %d \n recompile %s with larger value for " PetscStringize_(MAXREGDESOBJS),MAXREGDESOBJS,__FILE__);
  PetscObjectRegisterDestroy_Objects[PetscObjectRegisterDestroy_Count++] = obj;
  PetscFunctionReturn(0);
}

/*@C
   PetscObjectRegisterDestroyAll - Frees all the PETSc objects that have been registered
     with PetscObjectRegisterDestroy(). Called by PetscFinalize()

   Logically Collective on individual PetscObjects

   Level: developer

.seealso: `PetscObjectRegisterDestroy()`
@*/
PetscErrorCode  PetscObjectRegisterDestroyAll(void)
{
  PetscFunctionBegin;
  for (PetscInt i=0; i<PetscObjectRegisterDestroy_Count; i++) PetscCall(PetscObjectDestroy(&PetscObjectRegisterDestroy_Objects[i]));
  PetscObjectRegisterDestroy_Count = 0;
  PetscFunctionReturn(0);
}

#define MAXREGFIN 256
static int PetscRegisterFinalize_Count = 0;
static PetscErrorCode (*PetscRegisterFinalize_Functions[MAXREGFIN])(void);

/*@C
   PetscRegisterFinalize - Registers a function that is to be called in PetscFinalize()

   Not Collective

   Input Parameter:
.  PetscErrorCode (*fun)(void) -

   Level: developer

   Notes:
      This is used by, for example, DMInitializePackage() to have DMFinalizePackage() called

.seealso: `PetscRegisterFinalizeAll()`
@*/
PetscErrorCode  PetscRegisterFinalize(PetscErrorCode (*f)(void))
{
  PetscFunctionBegin;
  for (PetscInt i=0; i<PetscRegisterFinalize_Count; i++) {
    if (f == PetscRegisterFinalize_Functions[i]) PetscFunctionReturn(0);
  }
  PetscCheck(PetscRegisterFinalize_Count < MAXREGFIN,PETSC_COMM_SELF,PETSC_ERR_PLIB,"No more room in array, limit %d \n recompile %s with larger value for " PetscStringize_(MAXREGFIN),MAXREGFIN,__FILE__);
  PetscRegisterFinalize_Functions[PetscRegisterFinalize_Count++] = f;
  PetscFunctionReturn(0);
}

/*@C
   PetscRegisterFinalizeAll - Runs all the finalize functions set with PetscRegisterFinalize()

   Not Collective unless registered functions are collective

   Level: developer

.seealso: `PetscRegisterFinalize()`
@*/
PetscErrorCode  PetscRegisterFinalizeAll(void)
{
  PetscFunctionBegin;
  for (PetscInt i=0; i<PetscRegisterFinalize_Count; i++) PetscCall((*PetscRegisterFinalize_Functions[i])());
  PetscRegisterFinalize_Count = 0;
  PetscFunctionReturn(0);
}
