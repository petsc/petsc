
#include "petscsys.h"

int DICT_COOKIE = 0;

typedef struct _p_DictNode {
  char               *key;
  int                 intData;
  double              realData;
  void               *data;
  struct _p_DictNode *next;
} DictNode;

struct _p_Dict {
  PETSCHEADER(int)
  DictNode *dictHead;
};

#undef __FUNCT__  
#define __FUNCT__ "ParameterDictCreate"
/*@C
  ParameterDictCreate - Creates an empty parameter dictionary.

  Collective on MPI_Comm
 
  Input Parameter:
. comm - The MPI communicator to use 

  Output Parameter:
. dict - The ParameterDict object

  Level: beginner

.keywords: ParameterDict, create
.seealso: ParameterDictDestroy(), ParameterDictSetObject(), ParameterDictGetObject()
@*/ 
int ParameterDictCreate(MPI_Comm comm, ParameterDict *dict)
{
  ParameterDict d;

  PetscFunctionBegin;
  PetscValidPointer(dict,2);
  PetscHeaderCreate(d, _p_Dict, int, DICT_COOKIE, 0, "Dict", comm, ParameterDictDestroy, PETSC_NULL);
  PetscLogObjectCreate(d);
  PetscLogObjectMemory(d, sizeof(struct _p_Dict));

  d->dictHead = PETSC_NULL;
  *dict       = d;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "ParameterDictDestroy"
/*@C
  ParameterDictDestroy - Destroys a parameter dictionary object.

  Not Collective

  Input Parameter:
. dict - The ParameterDict

  Level: beginner

.keywords: ParameterDict, destroy
.seealso: ParameterDictCreate(), ParameterDictSetObject(), ParameterDictGetObject()
@*/
int ParameterDictDestroy(ParameterDict dict)
{
  DictNode *node, *next;
  int       ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dict, DICT_COOKIE,1);
  if (--dict->refct > 0)
    PetscFunctionReturn(0);
  next = dict->dictHead;
  while(next != PETSC_NULL) {
    node = next;
    next = node->next;
    ierr = PetscFree(node->key);CHKERRQ(ierr);
    ierr = PetscFree(node);CHKERRQ(ierr);
  }
  PetscLogObjectDestroy(dict);
  PetscHeaderDestroy(dict);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "ParameterDictSetInteger"
/*@C
  ParameterDictSetInteger - Adds an integer argument to the dictionary

  Collective on ParameterDict

  Input Parameters:
+ dict - The ParameterDict
. key  - The argument name
- data - The integer to store

  Level: beginner

.keywords: ParameterDict, set, integer
.seealso: ParameterDictSetDouble(), ParameterDictSetObject(), ParameterDictGetInteger()
@*/
int ParameterDictSetInteger(ParameterDict dict, const char key[], int data)
{
  DictNode *node;
  int       ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dict, DICT_COOKIE,1);
  PetscValidCharPointer(key,2);
  ierr = PetscMalloc(sizeof(DictNode), &node);CHKERRQ(ierr);
  node->intData  = data;
  node->realData = 0.0;
  node->data     = PETSC_NULL;
  ierr = PetscStrallocpy(key, &node->key);CHKERRQ(ierr);
  /* Push node onto the linked list */
  node->next     = dict->dictHead;
  dict->dictHead = node;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "ParameterDictSetDouble"
/*@C
  ParameterDictSetDouble - Adds a real argument to the dictionary

  Collective on ParameterDict

  Input Parameters:
+ dict - The ParameterDict
. key  - The argument name
- data - The double to store

  Level: beginner

.keywords: ParameterDict, set, double
.seealso: ParameterDictSetInteger(), ParameterDictSetObject(), ParameterDictGetDouble()
@*/
int ParameterDictSetDouble(ParameterDict dict, const char key[], double data)
{
  DictNode *node;
  int       ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dict, DICT_COOKIE,1);
  PetscValidCharPointer(key,2);
  ierr = PetscMalloc(sizeof(DictNode), &node);CHKERRQ(ierr);
  node->intData  = 0;
  node->realData = data;
  node->data     = PETSC_NULL;
  ierr = PetscStrallocpy(key, &node->key);CHKERRQ(ierr);
  /* Push node onto the linked list */
  node->next     = dict->dictHead;
  dict->dictHead = node;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "ParameterDictSetObject"
/*@C
  ParameterDictSetObject - Adds an object argument to the dictionary

  Collective on ParameterDict

  Input Parameters:
+ dict - The ParameterDict
. key  - The argument name
- data - The object to store

  Level: beginner

.keywords: ParameterDict, set, object
.seealso: ParameterDictSetInteger(), ParameterDictSetDouble(), ParameterDictGetObject()
@*/
int ParameterDictSetObject(ParameterDict dict, const char key[], void *data)
{
  DictNode *node;
  int       ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dict, DICT_COOKIE,1);
  PetscValidCharPointer(key,2);
  ierr = PetscMalloc(sizeof(DictNode), &node);CHKERRQ(ierr);
  node->intData  = 0;
  node->realData = 0.0;
  node->data     = data;
  ierr = PetscStrallocpy(key, &node->key);CHKERRQ(ierr);
  /* Push node onto the linked list */
  node->next     = dict->dictHead;
  dict->dictHead = node;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "ParameterDictRemove"
/*@C
  ParameterDictRemove - Removes an argument from the dictionary

  Collective on ParameterDict

  Input Parameters:
+ dict - The ParameterDict
- key  - The argument name

  Level: beginner

.keywords: ParameterDict, remove
.seealso: ParameterDictSetObject(), ParameterDictGetObject()
@*/
int ParameterDictRemove(ParameterDict dict, const char key[])
{
  DictNode  *node, *prev;
  PetscTruth found;
  int        ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dict, DICT_COOKIE,1);
  PetscValidCharPointer(key,2);
  found = PETSC_FALSE;
  node  = dict->dictHead;
  prev  = node;
  /* Check the head separately */
  ierr = PetscStrcmp(key, node->key, &found);CHKERRQ(ierr);
  if (found == PETSC_TRUE) {
    dict->dictHead = node->next;
    ierr = PetscFree(node->key);CHKERRQ(ierr);
    ierr = PetscFree(node);CHKERRQ(ierr);
  }
  /* Check the rest */
  while((node != PETSC_NULL) && (found == PETSC_FALSE)) {
    ierr = PetscStrcmp(key, node->key, &found);CHKERRQ(ierr);
    if (found == PETSC_TRUE) {
      prev->next = node->next;
      ierr = PetscFree(node->key);CHKERRQ(ierr);
      ierr = PetscFree(node);CHKERRQ(ierr);
    }
    prev = node;
    node = node->next;
  }
  if (found == PETSC_FALSE) SETERRQ(PETSC_ERR_ARG_WRONG, "Key not found in dictionary");
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "ParameterDictGetInteger"
/*@C
  ParameterDictGetInteger - Gets an integer argument from the dictionary

  Not collective

  Input Parameters:
+ dict - The ParameterDict
- key  - The argument name

  Output Parameter:
. data - The integer

  Level: beginner

.keywords: ParameterDict, get, integer
.seealso: ParameterDictGetDouble(), ParameterDictGetObject(), ParameterDictSetInteger()
@*/
int ParameterDictGetInteger(ParameterDict dict, const char key[], int *data)
{
  DictNode  *node;
  PetscTruth found;
  int        ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dict, DICT_COOKIE,1);
  PetscValidCharPointer(key,2);
  PetscValidIntPointer(data,3);
  found = PETSC_FALSE;
  node  = dict->dictHead;
  /* Check the rest */
  while((node != PETSC_NULL) && (found == PETSC_FALSE)) {
    ierr = PetscStrcmp(key, node->key, &found);CHKERRQ(ierr);
    if (found == PETSC_TRUE) {
      *data = node->intData;
    }
    node = node->next;
  }
  if (found == PETSC_FALSE) SETERRQ(PETSC_ERR_ARG_WRONG, "Key not found in dictionary");
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "ParameterDictGetDouble"
/*@C
  ParameterDictGetDouble - Gets a real argument from the dictionary

  Not collective

  Input Parameters:
+ dict - The ParameterDict
- key  - The argument name

  Output Parameter:
. data - The double

  Level: beginner

.keywords: ParameterDict, get, double
.seealso: ParameterDictGetInteger(), ParameterDictGetObject(), ParameterDictSetDouble()
@*/
int ParameterDictGetDouble(ParameterDict dict, const char key[], double *data)
{
  DictNode  *node;
  PetscTruth found;
  int        ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dict, DICT_COOKIE,1);
  PetscValidCharPointer(key,2);
  PetscValidDoublePointer(data,3);
  found = PETSC_FALSE;
  node  = dict->dictHead;
  /* Check the rest */
  while((node != PETSC_NULL) && (found == PETSC_FALSE)) {
    ierr = PetscStrcmp(key, node->key, &found);CHKERRQ(ierr);
    if (found == PETSC_TRUE) {
      *data = node->realData;
    }
    node = node->next;
  }
  if (found == PETSC_FALSE) SETERRQ(PETSC_ERR_ARG_WRONG, "Key not found in dictionary");
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "ParameterDictGetObject"
/*@C
  ParameterDictGetObject - Gets an object argument from the dictionary

  Not collective

  Input Parameters:
+ dict - The ParameterDict
- key  - The argument name

  Output Parameter:
. data - The object

  Level: beginner

.keywords: ParameterDict, get, object
.seealso: ParameterDictGetInteger(), ParameterDictGetDouble(), ParameterDictSetObject()
@*/
int ParameterDictGetObject(ParameterDict dict, const char key[], void **data)
{
  DictNode  *node;
  PetscTruth found;
  int        ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dict, DICT_COOKIE,1);
  PetscValidCharPointer(key,2);
  PetscValidPointer(data,3);
  found = PETSC_FALSE;
  node  = dict->dictHead;
  /* Check the rest */
  while((node != PETSC_NULL) && (found == PETSC_FALSE)) {
    ierr = PetscStrcmp(key, node->key, &found);CHKERRQ(ierr);
    if (found == PETSC_TRUE) {
      *data = node->data;
    }
    node = node->next;
  }
  if (found == PETSC_FALSE) SETERRQ(PETSC_ERR_ARG_WRONG, "Key not found in dictionary");
  PetscFunctionReturn(0);
}
