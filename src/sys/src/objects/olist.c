
#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: olist.c,v 1.11 1999/06/30 23:49:38 balay Exp balay $";
#endif

/*
         Provides a general mechanism to maintain a linked list of PETSc objects.
     This is used to allow PETSc objects to carry a list of "composed" objects
*/
#include "petsc.h"
#include "sys.h"

struct _OList {
    char        name[128];
    PetscObject obj;
    OList       next;
};

#undef __FUNC__  
#define __FUNC__ "OListAdd"
/*

       Notes: Replaces item if it is already in list. Removes item if you pass in a 
              PETSC_NULL object.    

.seealso: OListDestroy()
*/
int OListAdd(OList *fl,const char name[],PetscObject obj )
{
  OList olist,nlist,prev;
  int   ierr;

  PetscFunctionBegin;

  if (!obj) { /* this means remove from list if it is there */
    nlist = *fl; prev = 0;
    while (nlist) {
      if (!PetscStrcmp(name,nlist->name)) {  /* found it already in the list */
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
    PetscFunctionReturn(0); /* didn't find it to remove */
  }
  /* look for it already in list */
  nlist = *fl;
  while (nlist) {
    if (!PetscStrcmp(name,nlist->name)) {  /* found it in the list */
      ierr = PetscObjectDereference(nlist->obj);CHKERRQ(ierr);
      ierr = PetscObjectReference(obj);CHKERRQ(ierr);
      nlist->obj = obj;
      PetscFunctionReturn(0);
    }
    nlist = nlist->next;
  }

  /* add it to list, because it was not already there */

  olist       = PetscNew(struct _OList);CHKPTRQ(olist);
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

#undef __FUNC__  
#define __FUNC__ "OListDestroy"
/*
    OListDestroy - Destroy a list of objects

    Input Parameter:
.   fl   - pointer to list
*/
int OListDestroy(OList *fl )
{
  OList   tmp, entry = *fl;
  int     ierr;

  PetscFunctionBegin;
  while (entry) {
    tmp = entry->next;
    ierr = PetscObjectDereference(entry->obj);CHKERRQ(ierr);
    ierr = PetscFree(entry);CHKERRQ(ierr);
    entry = tmp;
  }
  PetscFunctionReturn(0);
}


#undef __FUNC__  
#define __FUNC__ "OListFind"
/*
    OListFind - givn a name, find the matching object

    Input Parameters:
+   fl   - pointer to list
-   name - name string

    Output Parameters:
.   ob - the PETSc object

    Notes:
    The name must have been registered with the OListAdd() before calling this 
    routine.

.seealso: OListReverseFind()

*/
int OListFind(OList fl, const char name[], PetscObject *obj)
{
  PetscFunctionBegin;

  *obj = 0;
  while (fl) {
    if (!PetscStrcmp(name,fl->name)) {
      *obj = fl->obj;
      break;
    }
    fl = fl->next;
  }
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "OListReverseFind"
/*
    OListReverseFind - given a object, find the matching name if it exists

    Input Parameters:
+   fl   - pointer to list
-   ob - the PETSc object

    Output Parameters:
.   name - name string

    Notes:
    The name must have been registered with the OListAdd() before calling this 
    routine.

.seealso: OListFind()

*/
int OListReverseFind(OList fl, PetscObject obj, char **name)
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


#undef __FUNC__  
#define __FUNC__ "OListDuplicate"
/*
    OListDuplicate - Creates a new list from a give object list.

    Input Parameters:
.   fl   - pointer to list

    Output Parameters:
.   nl - the new list (should point to 0 to start, otherwise appends)


*/
int OListDuplicate(OList fl, OList *nl)
{
  int ierr;

  PetscFunctionBegin;
  while (fl) {
    ierr = OListAdd(nl,fl->name,fl->obj);CHKERRQ(ierr);
    fl = fl->next;
  }
  PetscFunctionReturn(0);
}





