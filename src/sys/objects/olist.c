
/*
         Provides a general mechanism to maintain a linked list of PETSc objects.
     This is used to allow PETSc objects to carry a list of "composed" objects
*/
#include <petsc/private/petscimpl.h>

struct _n_PetscObjectList {
  char            name[256];
  PetscBool       skipdereference; /* when the PetscObjectList is destroyed do not call PetscObjectDereference() on this object */
  PetscObject     obj;
  PetscObjectList next;
};

/*@C
     PetscObjectListRemoveReference - Calls `PetscObjectDereference()` on an object in the list immediately but keeps a pointer to the object in the list.

    Input Parameters:
+     fl - the object list
-     name - the name to use for the object

    Level: developer

    Notes:
    Use `PetscObjectListAdd`(`PetscObjectList`,const char name[],NULL) to truly remove the object from the list

    Use this routine ONLY if you know that the object referenced will remain in existence until the pointing object is destroyed

    Developer Note:
    This is to handle some cases that otherwise would result in having circular references so reference counts never got to zero

.seealso: `PetscObjectListDestroy()`, `PetscObjectListFind()`, `PetscObjectListDuplicate()`, `PetscObjectListReverseFind()`, `PetscObjectListDuplicate()`, `PetscObjectListAdd()`
@*/
PetscErrorCode PetscObjectListRemoveReference(PetscObjectList *fl, const char name[])
{
  PetscObjectList nlist;
  PetscBool       match;

  PetscFunctionBegin;
  PetscValidPointer(fl, 1);
  PetscValidCharPointer(name, 2);
  nlist = *fl;
  while (nlist) {
    PetscCall(PetscStrcmp(name, nlist->name, &match));
    if (match) { /* found it in the list */
      if (!nlist->skipdereference) PetscCall(PetscObjectDereference(nlist->obj));
      nlist->skipdereference = PETSC_TRUE;
      PetscFunctionReturn(0);
    }
    nlist = nlist->next;
  }
  PetscFunctionReturn(0);
}

/*@C
     PetscObjectListAdd - Adds a new object to an `PetscObjectList`

    Input Parameters:
+     fl - the object list
.     name - the name to use for the object
-     obj - the object to attach

    Level: developer

    Notes:
    Replaces item if it is already in list. Removes item if you pass in a NULL object.

    Use `PetscObjectListFind()` or `PetscObjectListReverseFind()` to get the object back

.seealso: `PetscObjectListDestroy()`, `PetscObjectListFind()`, `PetscObjectListDuplicate()`, `PetscObjectListReverseFind()`, `PetscObjectListDuplicate()`
@*/
PetscErrorCode PetscObjectListAdd(PetscObjectList *fl, const char name[], PetscObject obj)
{
  PetscObjectList olist, nlist, prev;
  PetscBool       match;

  PetscFunctionBegin;
  PetscValidPointer(fl, 1);
  if (!obj) { /* this means remove from list if it is there */
    nlist = *fl;
    prev  = NULL;
    while (nlist) {
      PetscCall(PetscStrcmp(name, nlist->name, &match));
      if (match) { /* found it already in the list */
        /* Remove it first to prevent circular derefs */
        if (prev) prev->next = nlist->next;
        else if (nlist->next) *fl = nlist->next;
        else *fl = NULL;
        if (!nlist->skipdereference) PetscCall(PetscObjectDereference(nlist->obj));
        PetscCall(PetscFree(nlist));
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
    PetscCall(PetscStrcmp(name, nlist->name, &match));
    if (match) { /* found it in the list */
      PetscCall(PetscObjectReference(obj));
      if (!nlist->skipdereference) PetscCall(PetscObjectDereference(nlist->obj));
      nlist->skipdereference = PETSC_FALSE;
      nlist->obj             = obj;
      PetscFunctionReturn(0);
    }
    nlist = nlist->next;
  }

  /* add it to list, because it was not already there */
  PetscCall(PetscNew(&olist));
  olist->next = NULL;
  olist->obj  = obj;

  PetscCall(PetscObjectReference(obj));
  PetscCall(PetscStrcpy(olist->name, name));

  if (!*fl) *fl = olist;
  else { /* go to end of list */ nlist = *fl;
    while (nlist->next) nlist = nlist->next;
    nlist->next = olist;
  }
  PetscFunctionReturn(0);
}

/*@C
    PetscObjectListDestroy - Destroy a list of objects

    Input Parameter:
.   ifl   - pointer to list

    Level: developer

.seealso: `PetscObjectListAdd()`, `PetscObjectListFind()`, `PetscObjectListDuplicate()`, `PetscObjectListReverseFind()`, `PetscObjectListDuplicate()`
@*/
PetscErrorCode PetscObjectListDestroy(PetscObjectList *ifl)
{
  PetscObjectList tmp, fl;

  PetscFunctionBegin;
  PetscValidPointer(ifl, 1);
  fl = *ifl;
  while (fl) {
    tmp = fl->next;
    if (!fl->skipdereference) PetscCall(PetscObjectDereference(fl->obj));
    PetscCall(PetscFree(fl));
    fl = tmp;
  }
  *ifl = NULL;
  PetscFunctionReturn(0);
}

/*@C
    PetscObjectListFind - givn a name, find the matching object

    Input Parameters:
+   fl   - pointer to list
-   name - name string

    Output Parameters:
.   obj - the PETSc object

    Level: developer

    Notes:
    The name must have been registered with the `PetscObjectListAdd()` before calling this routine.

    The reference count of the object is not increased

.seealso: `PetscObjectListDestroy()`, `PetscObjectListAdd()`, `PetscObjectListDuplicate()`, `PetscObjectListReverseFind()`, `PetscObjectListDuplicate()`
@*/
PetscErrorCode PetscObjectListFind(PetscObjectList fl, const char name[], PetscObject *obj)
{
  PetscFunctionBegin;
  PetscValidPointer(obj, 3);
  *obj = NULL;
  while (fl) {
    PetscBool match;
    PetscCall(PetscStrcmp(name, fl->name, &match));
    if (match) {
      *obj = fl->obj;
      break;
    }
    fl = fl->next;
  }
  PetscFunctionReturn(0);
}

/*@C
    PetscObjectListReverseFind - given a object, find the matching name if it exists

    Input Parameters:
+   fl   - pointer to list
-   obj - the PETSc object

    Output Parameters:
+  name - name string
-  skipdereference - if the object is in list but does not have the increased reference count for a circular dependency

    Level: developer

    Notes:
    The name must have been registered with the `PetscObjectListAdd()` before calling this routine.

    The reference count of the object is not increased

.seealso: `PetscObjectListDestroy()`, `PetscObjectListAdd()`, `PetscObjectListDuplicate()`, `PetscObjectListFind()`, `PetscObjectListDuplicate()`
@*/
PetscErrorCode PetscObjectListReverseFind(PetscObjectList fl, PetscObject obj, char **name, PetscBool *skipdereference)
{
  PetscFunctionBegin;
  PetscValidPointer(name, 3);
  if (skipdereference) PetscValidBoolPointer(skipdereference, 4);
  *name = NULL;
  while (fl) {
    if (fl->obj == obj) {
      *name = fl->name;
      if (skipdereference) *skipdereference = fl->skipdereference;
      break;
    }
    fl = fl->next;
  }
  PetscFunctionReturn(0);
}

/*@C
    PetscObjectListDuplicate - Creates a new list from a given object list.

    Input Parameters:
.   fl   - pointer to list

    Output Parameters:
.   nl - the new list (should point to 0 to start, otherwise appends)

    Level: developer

.seealso: `PetscObjectListDestroy()`, `PetscObjectListAdd()`, `PetscObjectListReverseFind()`, `PetscObjectListFind()`, `PetscObjectListDuplicate()`
@*/
PetscErrorCode PetscObjectListDuplicate(PetscObjectList fl, PetscObjectList *nl)
{
  PetscFunctionBegin;
  PetscValidPointer(nl, 2);
  while (fl) {
    PetscCall(PetscObjectListAdd(nl, fl->name, fl->obj));
    fl = fl->next;
  }
  PetscFunctionReturn(0);
}
