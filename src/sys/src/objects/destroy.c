#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: destroy.c,v 1.33 1997/07/09 20:51:14 balay Exp bsmith $";
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

.keywords: object, destroy
@*/
int PetscObjectDestroy(PetscObject obj)
{
  int ierr;

  PetscValidHeader(obj);

  if (obj->destroypublic) {
    ierr = (*obj->destroypublic)(obj); CHKERRQ(ierr);
  } else {
    SETERRQ(1,0,"This PETSc object does not have a generic destroy routine");
  }
  return 0;
}

#undef __FUNC__  
#define __FUNC__ "PetscObjectView" 
/*@C
   PetscObjectView - Views any PetscObject, regardless of the type. 

   Input Parameters:
.  obj - any PETSc object, for example a Vec, Mat or KSP.
.  viewer - any PETSc viewer

.keywords: object, view
@*/
int PetscObjectView(PetscObject obj,Viewer viewer)
{
  int ierr;

  PetscValidHeader(obj);

  if (obj->viewpublic) {
    ierr = (*obj->viewpublic)(obj,viewer); CHKERRQ(ierr);
  } else {
    SETERRQ(1,0,"This PETSc object does not have a generic viewer routine");
  }
  return 0;
}
