#define PETSC_DLL
/*
         Provides a general mechanism to maintain a linked list of PETSc objects.
     This is used to allow PETSc objects to carry a list of "composed" objects
*/
#include "petscsys.h"

struct _n_PetscOList {
    char        name[256];
    PetscObject obj;
    PetscOList  next;
};

#undef __FUNCT__  
#define __FUNCT__ "PetscOListAdd"
/*@C
     PetscOListAdd - Adds a new object to an PetscOList

    Input Parameters:
+     fl - the object list
.     name - the name to use for the object
-     obj - the object to attach

    Level: developer

       Notes: Replaces item if it is already in list. Removes item if you pass in a PETSC_NULL object.    
 
        Use PetscOListFind() or PetscOListReverseFind() to get the object back

.seealso: PetscOListDestroy(), PetscOListFind(), PetscOListDuplicate(), PetscOListReverseFind(), PetscOListDuplicate()

@*/
PetscErrorCode PETSC_DLLEXPORT PetscOListAdd(PetscOList *fl,const char name[],PetscObject obj)
{
  PetscOList     olist,nlist,prev;
  PetscErrorCode ierr;
  PetscTruth     match;

  PetscFunctionBegin;

  if (!obj) { /* this means remove from list if it is there */
    nlist = *fl; prev = 0;
    while (nlist) {
      ierr = PetscStrcmp(name,nlist->name,&match);CHKERRQ(ierr);
      if (match) {  /* found it already in the list */
        ierr = PetscObjectDereference(nlist->obj);CHKERRQ(ierr);
        if (prev) prev->next = nlist->next;
        else if (nlist->next) {
          *fl = nlist->next;
        } else {
          *fl = 0;
        }
        ierr = PetscFree(nlist);CHKERRQ(ierr);
        PetscFunctionReturn(0);
      }
      prev  = nlist;
      nlist = nlist->next;
    }
    PetscFunctionReturn(0); /* did not find it to remove */
  }
  /* look for it already in list */
  nlist = *fl;
  while (nlist) {
    ierr = PetscStrcmp(name,nlist->name,&match);CHKERRQ(ierr);
    if (match) {  /* found it in the list */
      ierr = PetscObjectReference(obj);CHKERRQ(ierr);
      ierr = PetscObjectDereference(nlist->obj);CHKERRQ(ierr);
      nlist->obj = obj;
      PetscFunctionReturn(0);
    }
    nlist = nlist->next;
  }

  /* add it to list, because it was not already there */

  ierr        = PetscNew(struct _n_PetscOList,&olist);CHKERRQ(ierr);
  olist->next = 0;
  olist->obj  = obj;
  ierr = PetscObjectReference(obj);CHKERRQ(ierr);
  ierr = PetscStrcpy(olist->name,name);CHKERRQ(ierr);

  if (!*fl) {
    *fl = olist;
  } else { /* go to end of list */
    nlist = *fl;
    while (nlist->next) {
      nlist = nlist->next;
    }
    nlist->next = olist;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscOListDestroy"
/*@C
    PetscOListDestroy - Destroy a list of objects

    Input Parameter:
.   fl   - pointer to list

    Level: developer

.seealso: PetscOListAdd(), PetscOListFind(), PetscOListDuplicate(), PetscOListReverseFind(), PetscOListDuplicate()

@*/
PetscErrorCode PETSC_DLLEXPORT PetscOListDestroy(PetscOList fl)
{
  PetscOList     tmp;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  while (fl) {
    tmp   = fl->next;
    ierr  = PetscObjectDereference(fl->obj);CHKERRQ(ierr);
    ierr  = PetscFree(fl);CHKERRQ(ierr);
    fl    = tmp;
  }
  PetscFunctionReturn(0);
}


#undef __FUNCT__  
#define __FUNCT__ "PetscOListFind"
/*@C
    PetscOListFind - givn a name, find the matching object

    Input Parameters:
+   fl   - pointer to list
-   name - name string

    Output Parameters:
.   ob - the PETSc object

    Level: developer

    Notes:
    The name must have been registered with the PetscOListAdd() before calling this 
    routine.

.seealso: PetscOListDestroy(), PetscOListAdd(), PetscOListDuplicate(), PetscOListReverseFind(), PetscOListDuplicate()

@*/
PetscErrorCode PETSC_DLLEXPORT PetscOListFind(PetscOList fl,const char name[],PetscObject *obj)
{
  PetscErrorCode ierr;
  PetscTruth     match;

  PetscFunctionBegin;

  *obj = 0;
  while (fl) {
    ierr = PetscStrcmp(name,fl->name,&match);CHKERRQ(ierr);
    if (match) {
      *obj = fl->obj;
      break;
    }
    fl = fl->next;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscOListReverseFind"
/*@C
    PetscOListReverseFind - given a object, find the matching name if it exists

    Input Parameters:
+   fl   - pointer to list
-   ob - the PETSc object

    Output Parameters:
.   name - name string

    Level: developer

    Notes:
    The name must have been registered with the PetscOListAdd() before calling this 
    routine.

.seealso: PetscOListDestroy(), PetscOListAdd(), PetscOListDuplicate(), PetscOListFind(), PetscOListDuplicate()

@*/
PetscErrorCode PETSC_DLLEXPORT PetscOListReverseFind(PetscOList fl,PetscObject obj,char **name)
{
  PetscFunctionBegin;

  *name = 0;
  while (fl) {
    if (fl->obj == obj) {
      *name = fl->name;
      break;
    }
    fl = fl->next;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscOListDuplicate"
/*@C
    PetscOListDuplicate - Creates a new list from a give object list.

    Input Parameters:
.   fl   - pointer to list

    Output Parameters:
.   nl - the new list (should point to 0 to start, otherwise appends)

    Level: developer

.seealso: PetscOListDestroy(), PetscOListAdd(), PetscOListReverseFind(), PetscOListFind(), PetscOListDuplicate()

@*/
PetscErrorCode PETSC_DLLEXPORT PetscOListDuplicate(PetscOList fl,PetscOList *nl)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  while (fl) {
    ierr = PetscOListAdd(nl,fl->name,fl->obj);CHKERRQ(ierr);
    fl = fl->next;
  }
  PetscFunctionReturn(0);
}





