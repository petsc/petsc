#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: dupl.c,v 1.1 1999/10/16 19:52:50 bsmith Exp bsmith $";
#endif

#include "src/sys/src/viewer/viewerimpl.h"  /*I "viewer.h" I*/

#undef __FUNC__  
#define __FUNC__ "ViewerGetSingleton"
/*@
   ViewerGetSingleton - Creates a new viewer (same type as the old)
    that lives on a single processor (with MPI_comm PETSC_COMM_SELF)

   Not Collective

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
  ierr = MPI_Comm_size(viewer->comm,&size); CHKERRQ(ierr);
  if (size == 1) {
    *outviewer = viewer;
  } else if (viewer->ops->getsingleton) {
    ierr = (*viewer->ops->getsingleton)(viewer,outviewer);CHKERRQ(ierr);
  } else {
    SETERRQ1(1,1,"Cannot get singleton viewer for type %s",viewer->type_name);
  }
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "ViewerRestoreSingleton"
/*@
   ViewerRestoreSingleton - Restores a new viewer obtained with ViewerGetSingleton().

   Not Collective

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
  ierr = MPI_Comm_size(viewer->comm,&size); CHKERRQ(ierr);
  if (size == 1) {
    *outviewer = 0;
  } else if (viewer->ops->restoresingleton) {
    ierr = (*viewer->ops->restoresingleton)(viewer,outviewer);CHKERRQ(ierr);
  } 
  PetscFunctionReturn(0);
}

