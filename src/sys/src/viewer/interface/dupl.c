/*$Id: dupl.c,v 1.7 2000/05/05 22:13:21 balay Exp bsmith $*/

#include "src/sys/src/viewer/viewerimpl.h"  /*I "petscviewer.h" I*/

#undef __FUNC__  
#define __FUNC__ /*<a name="ViewerGetSingleton"></a>*/"ViewerGetSingleton" 
/*@
   ViewerGetSingleton - Creates a new viewer (same type as the old)
    that lives on a single processor (with MPI_comm PETSC_COMM_SELF)

    Collective on Viewer

   Input Parameter:
.  viewer - the viewer to be duplicated

   Output Parameter:
.  outviewer - new viewer

   Level: advanced

   Notes: Call ViewerRestoreSingleton() to return this viewer, NOT ViewerDestroy()

     This is most commonly used to view a sequential object that is part of a 
    parallel object. For example block Jacobi PC view could use this to obtain a
    viewer that is used with the sequential SLES on one block of the preconditioner.

.keywords: Viewer, duplication

.seealso: ViewerSocketOpen(), ViewerASCIIOpen(), ViewerDrawOpen(), ViewerRestoreSingleton()
@*/
int ViewerGetSingleton(Viewer viewer,Viewer *outviewer)
{
  int ierr,size;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(viewer,VIEWER_COOKIE);
  PetscValidPointer(outviewer);
  ierr = MPI_Comm_size(viewer->comm,&size);CHKERRQ(ierr);
  if (size == 1) {
    *outviewer = viewer;
    ierr = PetscObjectReference((PetscObject)viewer);CHKERRQ(ierr);
  } else if (viewer->ops->getsingleton) {
    ierr = (*viewer->ops->getsingleton)(viewer,outviewer);CHKERRQ(ierr);
  } else {
    SETERRQ1(1,1,"Cannot get singleton viewer for type %s",viewer->type_name);
  }
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ /*<a name="ViewerRestoreSingleton"></a>*/"ViewerRestoreSingleton" 
/*@
   ViewerRestoreSingleton - Restores a new viewer obtained with ViewerGetSingleton().

    Collective on Viewer

   Input Parameters:
+  viewer - the viewer to be duplicated
-  outviewer - new viewer

   Level: advanced

   Notes: Call ViewerGetSingleton() to get this viewer, NOT ViewerCreate()

.keywords: Viewer, duplication

.seealso: ViewerSocketOpen(), ViewerASCIIOpen(), ViewerDrawOpen(), ViewerGetSingleton()
@*/
int ViewerRestoreSingleton(Viewer viewer,Viewer *outviewer)
{
  int ierr,size;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(viewer,VIEWER_COOKIE);
  PetscValidPointer(outviewer);
  ierr = MPI_Comm_size(viewer->comm,&size);CHKERRQ(ierr);
  if (size == 1) {
    ierr = PetscObjectDereference((PetscObject)viewer);CHKERRQ(ierr);
    *outviewer = 0;
  } else if (viewer->ops->restoresingleton) {
    ierr = (*viewer->ops->restoresingleton)(viewer,outviewer);CHKERRQ(ierr);
  } 
  PetscFunctionReturn(0);
}

