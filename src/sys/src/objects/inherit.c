
#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: inherit.c,v 1.28 1998/03/20 22:47:23 bsmith Exp bsmith $";
#endif
/*
     Provides utility routines for manulating any type of PETSc object.
*/
#include "petsc.h"  /*I   "petsc.h"    I*/


extern int PetscObjectCompose_Petsc(PetscObject,char *,PetscObject);
extern int PetscObjectQuery_Petsc(PetscObject,char *,PetscObject *);
extern int PetscObjectComposeFunction_Petsc(PetscObject,char *,char *,void *);
extern int PetscObjectQueryFunction_Petsc(PetscObject,char *,void **);

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
  h->cookie                 = cookie;
  h->type                   = type;
  h->prefix                 = 0;
  h->refct                  = 1;
  h->bops->destroy          = des;
  h->bops->view             = vie;
  h->bops->compose          = PetscObjectCompose_Petsc;
  h->bops->query            = PetscObjectQuery_Petsc;
  h->bops->composefunction  = PetscObjectComposeFunction_Petsc;
  h->bops->queryfunction    = PetscObjectQueryFunction_Petsc;
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
  PetscFree(h->bops);
  PetscFree(h->ops);
  ierr = OListDestroy(&h->olist);CHKERRQ(ierr);
  ierr = DLRegisterDestroy(h->qlist); CHKERRQ(ierr);
  if (h->type_name) PetscFree(h->type_name);
  h->cookie = PETSCFREEDHEADER;
  if (h->prefix) PetscFree(h->prefix);
  if (h->fortran_func_pointers) {
    PetscFree(h->fortran_func_pointers);
  }
  PetscFree(h);
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
         Thus must be cast with a (PetscObject), for example, 
         PetscObjectReference((PetscObject) mat);

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
#define __FUNC__ "PetscObjectGetReference"
/*@C
   PetscObjectGetReference - Gets the current reference count for 
      any PETSc object.

   Input Parameter:
.  obj - the PETSc object
         Thus must be cast with a (PetscObject), for example, 
         PetscObjectGetReference((PetscObject) mat,&cnt);

  Output Parameter:
.  cnt - the reference count

.seealso: PetscObjectCompose(), PetscObjectDereference(), PetscObjectReference()

@*/
int PetscObjectGetReference(PetscObject obj,int *cnt)
{
  PetscFunctionBegin;
  PetscValidHeader(obj);
  *cnt = obj->refct;
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
         Thus must be cast with a (PetscObject), for example, 
         PetscObjectDereference((PetscObject) mat);

.seealso: PetscObjectCompose(), PetscObjectReference()

@*/
int PetscObjectDereference(PetscObject obj)
{
  int ierr;

  PetscFunctionBegin;
  PetscValidHeader(obj);
  if (obj->bops->destroy) {
    ierr = (*obj->bops->destroy)(obj); CHKERRQ(ierr);
  } else if (--obj->refct == 0) {
    SETERRQ(PETSC_ERR_SUP,0,"This PETSc object does not have a generic destroy routine");
  }
  PetscFunctionReturn(0);
}

/*
       These are the versions private to the PETSc object data structures
*/
#undef __FUNC__  
#define __FUNC__ "PetscObjectCompose_Petsc"
int PetscObjectCompose_Petsc(PetscObject obj,char *name,PetscObject ptr)
{
  int ierr;

  PetscFunctionBegin;
  ierr = OListAdd(&obj->olist,name,ptr); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "PetscObjectQuery_Petsc"
int PetscObjectQuery_Petsc(PetscObject obj,char *name,PetscObject *ptr)
{
  int ierr;

  PetscFunctionBegin;
  ierr = OListFind(obj->olist,name,ptr); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "PetscObjectComposeFunction_Petsc"
int PetscObjectComposeFunction_Petsc(PetscObject obj,char *name,char *fname,void *ptr)
{
  int ierr;

  PetscFunctionBegin;
  ierr = DLRegister(&obj->qlist,name,fname,(int (*)(void *))ptr);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "PetscObjectQueryFunction_Petsc"
int PetscObjectQueryFunction_Petsc(PetscObject obj,char *name,void **ptr)
{
  int ierr;

  PetscFunctionBegin;
  ierr = DLRegisterFind(obj->comm,obj->qlist,name,( int(**)(void *)) ptr);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
        These are the versions that are usable to any CCA compliant objects
*/
#undef __FUNC__  
#define __FUNC__ "PetscObjectCompose"
/*@C
   PetscObjectCompose - Associates another PETSc object with a given PETSc object. 
                       
   Input Parameters:
.  obj - the PETSc object
         Thus must be cast with a (PetscObject), for example, 
         PetscObjectCompose((PetscObject) mat,...);
.  name - name associated with child object 
.  ptr - the other PETSc object to associate with the PETSc object, this must also be 
         cast with (PetscObject)

   Notes:
   The second objects reference count is automatically increased by one when it is
   composed.
   
   PetscObjectCompose() can be used with any PETSc object such at
   Mat, Vec, KSP, SNES, etc, or any user provided object. 

.keywords: object, composition

.seealso: PetscObjectQuery()
@*/
int PetscObjectCompose(PetscObject obj,char *name,PetscObject ptr)
{
  int ierr;

  PetscFunctionBegin;
  ierr = (*obj->bops->compose)(obj,name,ptr);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "PetscObjectQuery"
/*@C
   PetscObjectQuery  - Gets a PETSc object associated with a given object.
                       
   Input Parameters:
.  obj - the PETSc object
         Thus must be cast with a (PetscObject), for example, 
         PetscObjectCompose((PetscObject) mat,...);
.  name - name associated with child object 
.  ptr - the other PETSc object associated with the PETSc object, this must also be 
         cast with (PetscObject)

.keywords: object, composition

.seealso: PetscObjectQuery()
@*/
int PetscObjectQuery(PetscObject obj,char *name,PetscObject *ptr)
{
  int ierr;

  PetscFunctionBegin;
  ierr = (*obj->bops->query)(obj,name,ptr);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


/*MC
   PetscObjectComposeFunction - Associates a function with a given PETSc object. 
                       
   Input Parameters:
.  obj - the PETSc object
         Thus must be cast with a (PetscObject), for example, 
         PetscObjectCompose((PetscObject) mat,...);
.  name - name associated with child object 
.  fname - name of the function
.  ptr - function pointer (or PETSC_NULL if using dynamic libraries)

   
   PetscObjectCompose() can be used with any PETSc object such at
   Mat, Vec, KSP, SNES, etc, or any user provided object. 

.keywords: object, composition

.seealso: PetscObjectQueryFunction()
*/
#undef __FUNC__  
#define __FUNC__ "PetscObjectComposeFunction_Private"
int PetscObjectComposeFunction_Private(PetscObject obj,char *name,char *fname,void *ptr)
{
  int ierr;

  PetscFunctionBegin;
  ierr = (*obj->bops->composefunction)(obj,name,fname,ptr);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "PetscObjectQueryFunction"
/*@C
   PetscObjectQueryFunction  - Gets a function associated with a given object.
                       
   Input Parameters:
.  obj - the PETSc object
         Thus must be cast with a (PetscObject), for example, 
         PetscObjectCompose((PetscObject) mat,...);
.  name - name associated with childfunction

   Output Parameter:
.   ptr - function pointer

.keywords: object, composition

.seealso: PetscObjectComposeFunction()
@*/
int PetscObjectQueryFunction(PetscObject obj,char *name,void **ptr)
{
  int ierr;

  PetscFunctionBegin;
  ierr = (*obj->bops->queryfunction)(obj,name,ptr);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------------------*/

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
  } else if (ptype == PETSC_LOGICAL) {
    *mtype = MPI_BYTE;
  } else {
    SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,1,"Unknown PETSc datatype");
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
  } else if (ptype == PETSC_LOGICAL) {
    *size = PETSC_LOGICAL_SIZE;
  } else {
    SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,1,"Unknown PETSc datatype");
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
  } else if (ptype == PETSC_LOGICAL) {
    *name = "logical";
  } else {
    SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,1,"Unknown PETSc datatype");
  }
  PetscFunctionReturn(0);
}

