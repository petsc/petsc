/*
         Provides a general mechanism to maintain a linked list of PETSc objects.
     This is used to allow PETSc objects to carry a list of "composed" objects
*/
#include <petsc/private/petscimpl.h>

/*@C
  PetscObjectListRemoveReference - Calls `PetscObjectDereference()` on an object in the list immediately but keeps a pointer to the object in the list.

  No Fortran Support

  Input Parameters:
+ fl   - the object list
- name - the name to use for the object

  Level: developer

  Notes:
  Use `PetscObjectListAdd`(`PetscObjectList`,const char name[],NULL) to truly remove the object from the list

  Use this routine ONLY if you know that the object referenced will remain in existence until the pointing object is destroyed

  Developer Notes:
  This is to handle some cases that otherwise would result in having circular references so reference counts never got to zero

.seealso: `PetscObjectListDestroy()`,`PetscObjectListFind()`,`PetscObjectListDuplicate()`,`PetscObjectListReverseFind()`,
`PetscObject`, `PetscObjectListAdd()`
@*/
PetscErrorCode PetscObjectListRemoveReference(PetscObjectList *fl, const char name[])
{
  PetscObjectList nlist;
  PetscBool       match;

  PetscFunctionBegin;
  PetscAssertPointer(fl, 1);
  PetscAssertPointer(name, 2);
  nlist = *fl;
  while (nlist) {
    PetscCall(PetscStrcmp(name, nlist->name, &match));
    if (match) { /* found it in the list */
      if (!nlist->skipdereference) PetscCall(PetscObjectDereference(nlist->obj));
      nlist->skipdereference = PETSC_TRUE;
      PetscFunctionReturn(PETSC_SUCCESS);
    }
    nlist = nlist->next;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PetscObjectListAdd - Adds a new object to an `PetscObjectList`

  No Fortran Support

  Input Parameters:
+ fl   - the object list
. name - the name to use for the object
- obj  - the object to attach

  Level: developer

  Notes:
  Replaces item if it is already in list. Removes item if you pass in a `NULL` object.

  Use `PetscObjectListFind()` or `PetscObjectListReverseFind()` to get the object back

.seealso: `PetscObjectListDestroy()`,`PetscObjectListFind()`,`PetscObjectListDuplicate()`,`PetscObjectListReverseFind()`, `PetscObject`, `PetscObjectList`
@*/
PetscErrorCode PetscObjectListAdd(PetscObjectList *fl, const char name[], PetscObject obj)
{
  PetscObjectList olist, nlist, prev;
  PetscBool       match;

  PetscFunctionBegin;
  PetscAssertPointer(fl, 1);
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
        PetscFunctionReturn(PETSC_SUCCESS);
      }
      prev  = nlist;
      nlist = nlist->next;
    }
    PetscFunctionReturn(PETSC_SUCCESS); /* did not find it to remove */
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
      PetscFunctionReturn(PETSC_SUCCESS);
    }
    nlist = nlist->next;
  }

  /* add it to list, because it was not already there */
  PetscCall(PetscNew(&olist));
  olist->next = NULL;
  olist->obj  = obj;

  PetscCall(PetscObjectReference(obj));
  PetscCall(PetscStrncpy(olist->name, name, sizeof(olist->name)));

  if (!*fl) *fl = olist;
  else { /* go to end of list */ nlist = *fl;
    while (nlist->next) nlist = nlist->next;
    nlist->next = olist;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PetscObjectListDestroy - Destroy a list of objects

  No Fortran Support

  Input Parameter:
. ifl - pointer to list

  Level: developer

.seealso: `PetscObjectList`, `PetscObject`, `PetscObjectListAdd()`, `PetscObjectListFind()`, `PetscObjectListDuplicate()`,
          `PetscObjectListReverseFind()`
@*/
PetscErrorCode PetscObjectListDestroy(PetscObjectList *ifl)
{
  PetscObjectList tmp, fl;

  PetscFunctionBegin;
  PetscAssertPointer(ifl, 1);
  fl = *ifl;
  while (fl) {
    tmp = fl->next;
    if (!fl->skipdereference) PetscCall(PetscObjectDereference(fl->obj));
    PetscCall(PetscFree(fl));
    fl = tmp;
  }
  *ifl = NULL;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PetscObjectListFind - given a name, find the matching object in a list

  No Fortran Support

  Input Parameters:
+ fl   - pointer to list
- name - name string

  Output Parameter:
. obj - the PETSc object

  Level: developer

  Notes:
  The name must have been registered with the `PetscObjectListAdd()` before calling this routine.

  The reference count of the object is not increased

.seealso: `PetscObjectListDestroy()`,`PetscObjectListAdd()`,`PetscObjectListDuplicate()`,`PetscObjectListReverseFind()`, `PetscObjectList`
@*/
PetscErrorCode PetscObjectListFind(PetscObjectList fl, const char name[], PetscObject *obj)
{
  PetscFunctionBegin;
  PetscAssertPointer(obj, 3);
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
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PetscObjectListReverseFind - given a object, find the matching name if it exists

  No Fortran Support

  Input Parameters:
+ fl  - pointer to list
- obj - the PETSc object

  Output Parameters:
+ name            - name string
- skipdereference - if the object is in list but does not have the increased reference count for a circular dependency

  Level: developer

  Notes:
  The name must have been registered with the `PetscObjectListAdd()` before calling this routine.

  The reference count of the object is not increased

.seealso: `PetscObjectListDestroy()`,`PetscObjectListAdd()`,`PetscObjectListDuplicate()`,`PetscObjectListFind()`, `PetscObjectList`
@*/
PetscErrorCode PetscObjectListReverseFind(PetscObjectList fl, PetscObject obj, const char *name[], PetscBool *skipdereference)
{
  PetscFunctionBegin;
  PetscAssertPointer(name, 3);
  if (skipdereference) PetscAssertPointer(skipdereference, 4);
  *name = NULL;
  while (fl) {
    if (fl->obj == obj) {
      *name = fl->name;
      if (skipdereference) *skipdereference = fl->skipdereference;
      break;
    }
    fl = fl->next;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PetscObjectListDuplicate - Creates a new list from a given object list.

  No Fortran Support

  Input Parameter:
. fl - pointer to list

  Output Parameter:
. nl - the new list (should point to `NULL` to start, otherwise appends)

  Level: developer

.seealso: `PetscObjectListDestroy()`, `PetscObjectListAdd()`, `PetscObjectListReverseFind()`,
`PetscObjectListFind()`, `PetscObjectList`
@*/
PetscErrorCode PetscObjectListDuplicate(PetscObjectList fl, PetscObjectList *nl)
{
  PetscFunctionBegin;
  PetscAssertPointer(nl, 2);
  while (fl) {
    PetscCall(PetscObjectListAdd(nl, fl->name, fl->obj));
    fl = fl->next;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}
