#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: inherit.c,v 1.16 1997/08/22 15:11:48 bsmith Exp curfman $";
#endif
/*
     Provides utility routines for manulating any type of PETSc object.
*/
#include "petsc.h"  /*I   "petsc.h"    I*/


#undef __FUNC__  
#define __FUNC__ "PetscHeaderCreate_Private"
/*
    Creates a base PETSc object header and fills in the default values.
   Called by the macro PetscHeaderCreate()
*/
int PetscHeaderCreate_Private(PetscObject h,int cookie,int type,MPI_Comm comm,int (*des)(PetscObject),
                              int (*vie)(PetscObject,Viewer))
{
  h->cookie        = cookie;
  h->type          = type;
  h->prefix        = 0;
  h->refct         = 1;
  h->destroypublic = des;
  h->viewpublic    = vie;
  PetscCommDup_Private(comm,&h->comm,&h->tag);
  return 0;
}

#undef __FUNC__  
#define __FUNC__ "PetscHeaderDestroy_Private"
/*
    Destroys a base PETSc object header. Called by macro PetscHeaderDestroy.
*/
int PetscHeaderDestroy_Private(PetscObject h)
{
  int ierr;

  PetscCommFree_Private(&h->comm);
  h->cookie = PETSCFREEDHEADER;
  if (h->prefix) PetscFree(h->prefix);
  if (h->child) {
    ierr = (*h->childdestroy)(h->child); CHKERRQ(ierr);
  }
  if (h->fortran_func_pointers) {
    PetscFree(h->fortran_func_pointers);
  }
  PetscFree(h);
  return 0;
}

#undef __FUNC__  
#define __FUNC__ "PetscObjectInherit_DefaultCopy"
/*
    The default copy simply copies the pointer and adds one to the 
  reference counter.

*/
static int PetscObjectInherit_DefaultCopy(void *in, void **out)
{
  PetscObject obj = (PetscObject) in;

  obj->refct++;
  *out = in;
  return 0;
}

#undef __FUNC__  
#define __FUNC__ "PetscObjectInherit_DefaultDestroy"
/*
    The default destroy treats it as a PETSc object and calls 
  its destroy routine.
*/
static int PetscObjectInherit_DefaultDestroy(void *in)
{
  int         ierr;
  PetscObject obj = (PetscObject) in;

  ierr = (*obj->destroypublic)(obj); CHKERRQ(ierr);
  return 0;
}

#undef __FUNC__  
#define __FUNC__ "PetscObjectReference"
/*@C
   PetscObjectReference - Indicates to any PetscObject that it is being
   referenced by another PetscObject. This increases the reference
   count for that object by one.

   Input Parameter:
.  obj - the PETSc object

.seealso: PetscObjectInherit()
@*/
int PetscObjectReference(PetscObject obj)
{
  PetscValidHeader(obj);
  obj->refct++;
  return 0;
}

#undef __FUNC__  
#define __FUNC__ "PetscObjectInherit"
/*@C
   PetscObjectInherit - Associates another object with a given PETSc object. 
                        This is to provide a limited support for inheritance.

   Input Parameters:
.  obj - the PETSc object
.  ptr - the other object to associate with the PETSc object
.  copy - a function used to copy the other object when the PETSc object 
          is copied, or PETSC_NULL to indicate the pointer is copied.
.  destroy - a function to call to destroy the object or PETSC_NULL to 
             call the standard destroy on the PETSc object.

   Notes:
   When ptr is a PetscObject one should almost always use PETSC_NULL as the 
   third and fourth argument.
   
   PetscObjectInherit() can be used with any PETSc object such at
   Mat, Vec, KSP, SNES, etc, or any user provided object. 

   Current limitation: 
   Each object can have only one child - we may extend this eventually.

.keywords: object, inherit

.seealso: PetscObjectGetChild()
@*/
int PetscObjectInherit(PetscObject obj,void *ptr, int (*copy)(void *,void **),int (*destroy)(void*))
{
/*
  if (obj->child) 
    SETERRQ(1,0,"Child already set;object can have only 1 child");
*/
  if (copy == PETSC_NULL)    copy = PetscObjectInherit_DefaultCopy;
  if (destroy == PETSC_NULL) destroy = PetscObjectInherit_DefaultDestroy;
  obj->child        = ptr;
  obj->childcopy    = copy;
  obj->childdestroy = destroy;
  return 0;
}

#undef __FUNC__  
#define __FUNC__ "PetscObjectGetChild"
/*@C
   PetscObjectGetChild - Gets the child of any PetscObject.

   Input Parameter:
.  obj - any PETSc object, for example a Vec, Mat or KSP.

   Output Parameter:
.  type - the child, if it has been set (otherwise PETSC_NULL)

.keywords: object, get, child

.seealso: PetscObjectInherit()
@*/
int PetscObjectGetChild(PetscObject obj,void **child)
{
  PetscValidHeader(obj);

  *child = obj->child;
  return 0;
}

