#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: destroy.c,v 1.37 1998/03/12 23:16:41 bsmith Exp bsmith $";
#endif
/*
     Provides utility routines for manulating any type of PETSc object.
*/
#include "petsc.h"  /*I   "petsc.h"    I*/

#undef __FUNC__  
#define __FUNC__ "PetscObjectDestroy"
/*@C
   PetscObjectDestroy - Destroys any PetscObject, regardless of the type. 

   Input Parameters:
.  obj - any PETSc object, for example a Vec, Mat or KSP.
         Thus must be cast with a (PetscObject), for example, 
         PetscObjectDestroy((PetscObject) mat);

.keywords: object, destroy
@*/
int PetscObjectDestroy(PetscObject obj)
{
  int ierr;

  PetscFunctionBegin;
  PetscValidHeader(obj);

  if (obj->bops->destroy) {
    ierr = (*obj->bops->destroy)(obj); CHKERRQ(ierr);
  } else {
    SETERRQ(PETSC_ERR_SUP,0,"This PETSc object does not have a generic destroy routine");
  }
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "PetscObjectView" 
/*@C
   PetscObjectView - Views any PetscObject, regardless of the type. 

   Input Parameters:
.  obj - any PETSc object, for example a Vec, Mat or KSP.
         Thus must be cast with a (PetscObject), for example, 
         PetscObjectView((PetscObject) mat,viewer);
.  viewer - any PETSc viewer

.keywords: object, view
@*/
int PetscObjectView(PetscObject obj,Viewer viewer)
{
  int ierr;

  PetscFunctionBegin;
  PetscValidHeader(obj);

  if (obj->viewpublic) {
    ierr = (*obj->viewpublic)(obj,viewer); CHKERRQ(ierr);
  } else {
    SETERRQ(PETSC_ERR_SUP,0,"This PETSc object does not have a generic viewer routine");
  }
  PetscFunctionReturn(0);
}





