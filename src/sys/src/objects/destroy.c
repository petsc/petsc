/*$Id: destroy.c,v 1.51 2000/04/09 04:34:38 bsmith Exp bsmith $*/
/*
     Provides utility routines for manulating any type of PETSc object.
*/
#include "petsc.h"  /*I   "petsc.h"    I*/

#undef __FUNC__  
#define __FUNC__ /*<a name=""></a>*/"PetscObjectDestroy"
/*@C
   PetscObjectDestroy - Destroys any PetscObject, regardless of the type. 

   Collective on PetscObject

   Input Parameter:
.  obj - any PETSc object, for example a Vec, Mat or KSP.
         This must be cast with a (PetscObject), for example, 
         PetscObjectDestroy((PetscObject)mat);

   Level: intermediate

.keywords: object, destroy
@*/
int PetscObjectDestroy(PetscObject obj)
{
  int ierr;

  PetscFunctionBegin;
  PetscValidHeader(obj);

  if (obj->bops->destroy) {
    ierr = (*obj->bops->destroy)(obj);CHKERRQ(ierr);
  } else {
    SETERRQ(PETSC_ERR_SUP,0,"This PETSc object does not have a generic destroy routine");
  }
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ /*<a name=""></a>*/"PetscObjectView" 
/*@C
   PetscObjectView - Views any PetscObject, regardless of the type. 

   Collective on PetscObject

   Input Parameters:
+  obj - any PETSc object, for example a Vec, Mat or KSP.
         This must be cast with a (PetscObject), for example, 
         PetscObjectView((PetscObject)mat,viewer);
-  viewer - any PETSc viewer

   Level: intermediate

.keywords: object, view
@*/
int PetscObjectView(PetscObject obj,Viewer viewer)
{
  int ierr;

  PetscFunctionBegin;
  PetscValidHeader(obj);
  if (!viewer) viewer = VIEWER_STDOUT_(obj->comm);
  PetscValidHeaderSpecific(viewer,VIEWER_COOKIE);

  if (obj->bops->view) {
    ierr = (*obj->bops->view)(obj,viewer);CHKERRQ(ierr);
  } else {
    SETERRQ(PETSC_ERR_SUP,0,"This PETSc object does not have a generic viewer routine");
  }
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ /*<a name=""></a>*/"PetscTypeCompare"
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

.keywords: comparing types
@*/
int PetscTypeCompare(PetscObject obj,char *type_name,PetscTruth *same)
{
  int ierr;

  PetscFunctionBegin;
  PetscValidHeader(obj);
  PetscValidCharPointer(type_name);
  ierr = PetscStrcmp((char*)(obj->type_name),type_name,same);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}




