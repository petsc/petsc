/*$Id: view.c,v 1.44 2001/04/10 19:34:10 bsmith Exp $*/

#include "src/sys/src/viewer/viewerimpl.h"  /*I "petsc.h" I*/  

int PETSC_VIEWER_COOKIE = 0;

#undef __FUNCT__  
#define __FUNCT__ "PetscViewerDestroy" 
/*@C
   PetscViewerDestroy - Destroys a PetscViewer.

   Collective on PetscViewer

   Input Parameters:
.  viewer - the PetscViewer to be destroyed.

   Level: beginner

.seealso: PetscViewerSocketOpen(), PetscViewerASCIIOpen(), PetscViewerCreate(), PetscViewerDrawOpen()

@*/
int PetscViewerDestroy(PetscViewer viewer)
{
  int         ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_COOKIE);
  if (--viewer->refct > 0) PetscFunctionReturn(0);

  /* if memory was published with AMS then destroy it */
  ierr = PetscObjectDepublish(viewer);CHKERRQ(ierr);

  if (viewer->ops->destroy) {
    ierr = (*viewer->ops->destroy)(viewer);CHKERRQ(ierr);
  }
  PetscLogObjectDestroy((PetscObject)viewer);
  PetscHeaderDestroy((PetscObject)viewer);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscViewerGetType" 
/*@C
   PetscViewerGetType - Returns the type of a PetscViewer.

   Not Collective

   Input Parameter:
.   viewer - the PetscViewer

   Output Parameter:
.  type - PetscViewer type (see below)

   Available Types Include:
.  PETSC_VIEWER_SOCKET - Socket PetscViewer
.  PETSC_VIEWER_ASCII - ASCII PetscViewer
.  PETSC_VIEWER_BINARY - binary file PetscViewer
.  PETSC_VIEWER_STRING - string PetscViewer
.  PETSC_VIEWER_DRAW - drawing PetscViewer

   Level: intermediate

   Note:
   See include/petscviewer.h for a complete list of PetscViewers.

   PetscViewerType is actually a string

.seealso: PetscViewerCreate(), PetscViewerSetType()

@*/
int PetscViewerGetType(PetscViewer viewer,PetscViewerType *type)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_COOKIE);
  *type = (PetscViewerType) viewer->type_name;
  PetscFunctionReturn(0);
}






