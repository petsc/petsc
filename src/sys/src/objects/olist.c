
#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: olist.c,v 1.1 1998/03/30 20:06:40 bsmith Exp bsmith $";
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
#define __FUNC__ "OListCreate"
/*
    

.seealso: OListDestroy()
*/
int OListAdd(OList *fl,char *name,PetscObject obj )
{
  OList olist,nlist;

  PetscFunctionBegin;
  olist       = PetscNew(struct _OList);CHKPTRQ(olist);
  olist->next = 0;
  olist->obj  = obj;
  PetscObjectReference(obj);
  PetscStrcpy(olist->name,name);

  if (!*fl) {
    *fl = olist;
  } else { /* go to end of list */
    nlist = *fl;
    while (nlist->next) {nlist = nlist->next;}
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

  while (entry) {
    tmp = entry->next;
    PetscObjectDereference(entry->obj);
    PetscFree(entry);
    entry = tmp;
  }
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "OListFind"
/*
    OListFind - givn a name, find the matching object

    Input Parameters:
.   fl   - pointer to list
.   name - name string

    The id or name must have been registered with the OListAdd() before calling this 
    routine.
*/
int OListFind(OList fl, char *name, PetscObject *obj)
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
#define __FUNC__ "OListDuplicate"
/*
    OListDuplicate - Creates a new list from a give olist.

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
    ierr = OListAdd(nl,fl->name,fl->obj); CHKERRQ(ierr);
    fl = fl->next;
  }
  PetscFunctionReturn(0);
}





