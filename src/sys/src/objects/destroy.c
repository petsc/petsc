#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: destroy.c,v 1.42 1999/03/17 23:21:46 bsmith Exp balay $";
#endif
/*
     Provides utility routines for manulating any type of PETSc object.
*/
#include "petsc.h"  /*I   "petsc.h"    I*/

#undef __FUNC__  
#define __FUNC__ "PetscObjectDestroy"
/*@C
   PetscObjectDestroy - Destroys any PetscObject, regardless of the type. 

   Collective on PetscObject

   Input Parameter:
.  obj - any PETSc object, for example a Vec, Mat or KSP.
         Thus must be cast with a (PetscObject), for example, 
         PetscObjectDestroy((PetscObject) mat);

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
#define __FUNC__ "PetscObjectView" 
/*@C
   PetscObjectView - Views any PetscObject, regardless of the type. 

   Collective on PetscObject, unless Viewer is sequential (e.g., VIEWER_STDOUT_SELF)

   Input Parameters:
+  obj - any PETSc object, for example a Vec, Mat or KSP.
         Thus must be cast with a (PetscObject), for example, 
         PetscObjectView((PetscObject) mat,viewer);
-  viewer - any PETSc viewer

   Level: intermediate

.keywords: object, view
@*/
int PetscObjectView(PetscObject obj,Viewer viewer)
{
  int ierr;

  PetscFunctionBegin;
  PetscValidHeader(obj);

  if (obj->bops->view) {
    ierr = (*obj->bops->view)(obj,viewer);CHKERRQ(ierr);
  } else {
    SETERRQ(PETSC_ERR_SUP,0,"This PETSc object does not have a generic viewer routine");
  }
  PetscFunctionReturn(0);
}





