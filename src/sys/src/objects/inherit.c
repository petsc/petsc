
#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: inherit.c,v 1.21 1997/10/19 03:23:45 bsmith Exp bsmith $";
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
  PetscFunctionBegin;
  h->cookie        = cookie;
  h->type          = type;
  h->prefix        = 0;
  h->refct         = 1;
  h->destroypublic = des;
  h->viewpublic    = vie;
  PetscCommDup_Private(comm,&h->comm,&h->tag);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "PetscHeaderDestroy_Private"
/*
    Destroys a base PETSc object header. Called by macro PetscHeaderDestroy.
*/
int PetscHeaderDestroy_Private(PetscObject h)
{
  int ierr;

  PetscFunctionBegin;
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
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "PetscObjectCompose_DefaultCopy"
/*
    The default copy simply copies the pointer and adds one to the 
  reference counter.

*/
static int PetscObjectCompose_DefaultCopy(void *in, void **out)
{
  PetscObject obj = (PetscObject) in;

  PetscFunctionBegin;
  obj->refct++;
  *out = in;
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "PetscObjectCompose_DefaultDestroy"
/*
    The default destroy treats it as a PETSc object and calls 
  its destroy routine.
*/
static int PetscObjectCompose_DefaultDestroy(void *in)
{
  int         ierr;
  PetscObject obj = (PetscObject) in;

  PetscFunctionBegin;
  ierr = (*obj->destroypublic)(obj); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "PetscObjectReference"
/*@C
   PetscObjectReference - Indicates to any PetscObject that it is being
   referenced by another PetscObject. This increases the reference
   count for that object by one.

   Input Parameter:
.  obj - the PETSc object

.seealso: PetscObjectCompose(), PetscObjectDereference()

@*/
int PetscObjectReference(PetscObject obj)
{
  PetscFunctionBegin;
  PetscValidHeader(obj);
  obj->refct++;
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "PetscObjectDereference"
/*@
   PetscObjectDereference - Indicates to any PetscObject that it is being
   referenced by one less PetscObject. This decreases the reference
   count for that object by one.

   Input Parameter:
.  obj - the PETSc object

.seealso: PetscObjectCompose(), PetscObjectReference()

@*/
int PetscObjectDereference(PetscObject obj)
{
  int ierr;

  PetscFunctionBegin;
  PetscValidHeader(obj);
  if (obj->destroypublic) {
    ierr = (*obj->destroypublic)(obj); CHKERRQ(ierr);
  } else if (--obj->refct == 0) {
    SETERRQ(1,0,"This PETSc object does not have a generic destroy routine");
  }
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "PetscObjectCompose"
/*@C
   PetscObjectCompose - Associates another object with a given PETSc object. 
                        This is to provide a limited support for composition.

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
   
   PetscObjectCompose() can be used with any PETSc object such at
   Mat, Vec, KSP, SNES, etc, or any user provided object. 

   Current limitation: 
   Each object can have only one child - we may extend this eventually.

.keywords: object, composition

.seealso: PetscObjectGetChild()
@*/
int PetscObjectCompose(PetscObject obj,void *ptr, int (*copy)(void *,void **),int (*destroy)(void*))
{
  PetscFunctionBegin;
  if (obj->child) {
    PLogInfo(obj,"Child already set; releasing old child");
    PetscObjectDereference((PetscObject)obj->child);
  }
  if (copy == PETSC_NULL)    copy = PetscObjectCompose_DefaultCopy;
  if (destroy == PETSC_NULL) destroy = PetscObjectCompose_DefaultDestroy;
  obj->child        = ptr;
  obj->childcopy    = copy;
  obj->childdestroy = destroy;
  PetscFunctionReturn(0);
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

.seealso: PetscObjectCompose()
@*/
int PetscObjectGetChild(PetscObject obj,void **child)
{
  PetscFunctionBegin;
  PetscValidHeader(obj);

  *child = obj->child;
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "PetscDataTypeToMPIDataType"
int PetscDataTypeToMPIDataType(PetscDataType ptype,MPI_Datatype* mtype)
{
  PetscFunctionBegin;
  if (ptype == PETSC_INT) {
    *mtype = MPI_INT;
  } else if (ptype == PETSC_DOUBLE) {
    *mtype = MPI_DOUBLE;
  } else if (ptype == PETSC_SCALAR) {
    *mtype = MPIU_SCALAR;
#if defined(USE_PETSC_COMPLEX)
  } else if (ptype == PETSC_COMPLEX) {
    *mtype = MPIU_COMPLEX;
#endif
  } else if (ptype == PETSC_CHAR) {
    *mtype = MPI_CHAR;
  } else {
    SETERRQ(1,1,"Unknown PETSc datatype");
  }
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "PetscDataTypeGetSize"
int PetscDataTypeGetSize(PetscDataType ptype,int *size)
{
  PetscFunctionBegin;
  if (ptype == PETSC_INT) {
    *size = PETSC_INT_SIZE;
  } else if (ptype == PETSC_DOUBLE) {
    *size = PETSC_DOUBLE_SIZE;
  } else if (ptype == PETSC_SCALAR) {
    *size = PETSC_SCALAR_SIZE;
#if defined(USE_PETSC_COMPLEX)
  } else if (ptype == PETSC_COMPLEX) {
    *size = PETSC_COMPLEX_SIZE;
#endif
  } else if (ptype == PETSC_CHAR) {
    *size = PETSC_CHAR_SIZE;
  } else {
    SETERRQ(1,1,"Unknown PETSc datatype");
  }
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "PetscDataTypeGetName"
int PetscDataTypeGetName(PetscDataType ptype,char **name)
{
  PetscFunctionBegin;
  if (ptype == PETSC_INT) {
    *name = "int";
  } else if (ptype == PETSC_DOUBLE) {
    *name = "double";
  } else if (ptype == PETSC_SCALAR) {
    *name = "Scalar";
#if defined(USE_PETSC_COMPLEX)
  } else if (ptype == PETSC_COMPLEX) {
    *name = "complex";
#endif
  } else if (ptype == PETSC_CHAR) {
    *name = "char";
  } else {
    SETERRQ(1,1,"Unknown PETSc datatype");
  }
  PetscFunctionReturn(0);
}

