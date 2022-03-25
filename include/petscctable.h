#ifndef PETSCCTABLE_H
#define PETSCCTABLE_H
#include <petscsys.h>

struct _n_PetscTable {
  PetscInt *keytable;
  PetscInt *table;
  PetscInt count;
  PetscInt tablesize;
  PetscInt head;
  PetscInt maxkey;   /* largest key allowed */
};

typedef struct _n_PetscTable* PetscTable;
typedef PetscInt* PetscTablePosition;

static inline unsigned long PetscHash(PetscTable ta,unsigned long x)
{
  return(x%(unsigned long)ta->tablesize);
}

static inline unsigned long PetscHashStep(PetscTable ta,unsigned long x)
{
  return(1+(x%(unsigned long)(ta->tablesize-1)));
}

PETSC_EXTERN PetscErrorCode PetscTableCreate(const PetscInt,PetscInt,PetscTable*);
PETSC_EXTERN PetscErrorCode PetscTableCreateCopy(const PetscTable,PetscTable*);
PETSC_EXTERN PetscErrorCode PetscTableDestroy(PetscTable*);
PETSC_EXTERN PetscErrorCode PetscTableGetCount(const PetscTable,PetscInt*);
PETSC_EXTERN PetscErrorCode PetscTableIsEmpty(const PetscTable,PetscInt*);
PETSC_EXTERN PetscErrorCode PetscTableAddExpand(PetscTable,PetscInt,PetscInt,InsertMode);
PETSC_EXTERN PetscErrorCode PetscTableAddCountExpand(PetscTable,PetscInt);
PETSC_EXTERN PetscErrorCode PetscTableGetHeadPosition(PetscTable,PetscTablePosition*);
PETSC_EXTERN PetscErrorCode PetscTableGetNext(PetscTable,PetscTablePosition*,PetscInt*,PetscInt*);
PETSC_EXTERN PetscErrorCode PetscTableRemoveAll(PetscTable);

static inline PetscErrorCode PetscTableAdd(PetscTable ta,PetscInt key,PetscInt data,InsertMode imode)
{
  PetscInt i,hash   = (PetscInt)PetscHash(ta,(unsigned long)key);
  PetscInt hashstep = (PetscInt)PetscHashStep(ta,(unsigned long)key);

  PetscFunctionBegin;
  PetscCheck(key > 0,PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"key (value %" PetscInt_FMT ") <= 0",key);
  PetscCheck(key <= ta->maxkey,PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"key %" PetscInt_FMT " is greater than largest key allowed %" PetscInt_FMT,key,ta->maxkey);
  PetscCheck(data,PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Null data");

  for (i=0; i<ta->tablesize; i++) {
    if (ta->keytable[hash] == key) {
      switch (imode) {
      case INSERT_VALUES:
        ta->table[hash] = data; /* over write */
        break;
      case ADD_VALUES:
        ta->table[hash] += data;
        break;
      case MAX_VALUES:
        ta->table[hash] = PetscMax(ta->table[hash],data);
        break;
      case MIN_VALUES:
        ta->table[hash] = PetscMin(ta->table[hash],data);
        break;
      case NOT_SET_VALUES:
      case INSERT_ALL_VALUES:
      case ADD_ALL_VALUES:
      case INSERT_BC_VALUES:
      case ADD_BC_VALUES:
        SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Unsupported InsertMode");
      }
      PetscFunctionReturn(0);
    } else if (!ta->keytable[hash]) {
      if (ta->count < 5*(ta->tablesize/6) - 1) {
        ta->count++; /* add */
        ta->keytable[hash] = key;
        ta->table[hash] = data;
      } else {
        PetscCall(PetscTableAddExpand(ta,key,data,imode));
      }
      PetscFunctionReturn(0);
    }
    hash = (hash + hashstep)%ta->tablesize;
  }
  SETERRQ(PETSC_COMM_SELF,PETSC_ERR_COR,"Full table");
  /* PetscFunctionReturn(0); */
}

static inline PetscErrorCode  PetscTableAddCount(PetscTable ta,PetscInt key)
{
  PetscInt i,hash   = (PetscInt)PetscHash(ta,(unsigned long)key);
  PetscInt hashstep = (PetscInt)PetscHashStep(ta,(unsigned long)key);

  PetscFunctionBegin;
  PetscCheck(key > 0,PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"key (value %" PetscInt_FMT ") <= 0",key);
  PetscCheck(key <= ta->maxkey,PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"key %" PetscInt_FMT " is greater than largest key allowed %" PetscInt_FMT,key,ta->maxkey);

  for (i=0; i<ta->tablesize; i++) {
    if (ta->keytable[hash] == key) {
      PetscFunctionReturn(0);
    } else if (!ta->keytable[hash]) {
      if (ta->count < 5*(ta->tablesize/6) - 1) {
        ta->count++; /* add */
        ta->keytable[hash] = key;
        ta->table[hash] = ta->count;
      } else {
        PetscCall(PetscTableAddCountExpand(ta,key));
      }
      PetscFunctionReturn(0);
    }
    hash = (hash + hashstep)%ta->tablesize;
  }
  SETERRQ(PETSC_COMM_SELF,PETSC_ERR_COR,"Full table");
  /* PetscFunctionReturn(0); */
}

/*
    PetscTableFind - finds data in table from a given key, if the key is valid but not in the table returns 0
*/
static inline PetscErrorCode  PetscTableFind(PetscTable ta,PetscInt key,PetscInt *data)
{
  PetscInt ii       = 0;
  PetscInt hash     = (PetscInt)PetscHash(ta,(unsigned long)key);
  PetscInt hashstep = (PetscInt)PetscHashStep(ta,(unsigned long)key);

  PetscFunctionBegin;
  *data = 0;
  PetscCheck(key > 0,PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"key (value %" PetscInt_FMT ") <= 0",key);
  PetscCheck(key <= ta->maxkey,PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"key %" PetscInt_FMT " is greater than largest key allowed %" PetscInt_FMT,key,ta->maxkey);

  while (ii++ < ta->tablesize) {
    if (!ta->keytable[hash]) break;
    else if (ta->keytable[hash] == key) {
      *data = ta->table[hash];
      break;
    }
    hash = (hash + hashstep)%ta->tablesize;
  }
  PetscFunctionReturn(0);
}

#endif
