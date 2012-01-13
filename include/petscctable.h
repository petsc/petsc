#ifndef __PETSCCTABLE_H
#define __PETSCCTABLE_H
#include "petscsys.h"
PETSC_EXTERN_CXX_BEGIN

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

#define HASH_FACT 79943
#define HASHT(ta,x) ((unsigned long)((HASH_FACT*(unsigned long)x)%ta->tablesize))

extern PetscErrorCode  PetscTableCreate(const PetscInt,PetscInt,PetscTable*);
extern PetscErrorCode  PetscTableCreateCopy(const PetscTable,PetscTable*);
extern PetscErrorCode  PetscTableDestroy(PetscTable*);
extern PetscErrorCode  PetscTableGetCount(const PetscTable,PetscInt*);
extern PetscErrorCode  PetscTableIsEmpty(const PetscTable,PetscInt*);
extern PetscErrorCode  PetscTableAddExpand(PetscTable,PetscInt,PetscInt,InsertMode);
extern PetscErrorCode  PetscTableAddCountExpand(PetscTable,PetscInt);
extern PetscErrorCode  PetscTableGetHeadPosition(PetscTable,PetscTablePosition*);
extern PetscErrorCode  PetscTableGetNext(PetscTable,PetscTablePosition*,PetscInt*,PetscInt*);
extern PetscErrorCode  PetscTableRemoveAll(PetscTable);

#undef __FUNCT__  
#define __FUNCT__ "PetscTableAdd"
PETSC_STATIC_INLINE PetscErrorCode PetscTableAdd(PetscTable ta,PetscInt key,PetscInt data,InsertMode imode)
{  
  PetscErrorCode ierr;
  PetscInt       ii = 0,hash = (PetscInt)HASHT(ta,key);
    
  PetscFunctionBegin;
  if (key <= 0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"key <= 0");
  if (key > ta->maxkey) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"key %D is greater than largest key allowed %D",key,ta->maxkey);
  if (!data) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Null data");
  
  if (ta->count < 5*(ta->tablesize/6) - 1) {
    while (ii++ < ta->tablesize){
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
        default: SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Unsupported InsertMode");
        }
	PetscFunctionReturn(0); 
      } else if (!ta->keytable[hash]) {
	ta->count++; /* add */
	ta->keytable[hash] = key;
        ta->table[hash] = data;
	PetscFunctionReturn(0);
      }
      hash = (hash == (ta->tablesize-1)) ? 0 : hash+1; 
    }  
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_COR,"Full table");
  } else {
    ierr = PetscTableAddExpand(ta,key,data,imode);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscTableAddCount"
PETSC_STATIC_INLINE PetscErrorCode  PetscTableAddCount(PetscTable ta,PetscInt key)
{  
  PetscErrorCode ierr;
  PetscInt       ii = 0,hash = (PetscInt)HASHT(ta,key);
  
  PetscFunctionBegin;
  if (key <= 0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"key <= 0");
  if (key > ta->maxkey) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"key %D is greater than largest key allowed %D",key,ta->maxkey);

  if (ta->count < 5*(ta->tablesize/6) - 1) {
    while (ii++ < ta->tablesize){
      if (ta->keytable[hash] == key) {
	PetscFunctionReturn(0); 
      } else if (!ta->keytable[hash]) {
	ta->count++; /* add */
	ta->keytable[hash] = key; ta->table[hash] = ta->count;
	PetscFunctionReturn(0);
      }
      hash = (hash == (ta->tablesize-1)) ? 0 : hash+1; 
    }  
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_COR,"Full table");
  } else {
    ierr = PetscTableAddCountExpand(ta,key);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}


#undef __FUNCT__  
#define __FUNCT__ "PetscTableFind"
/*
    PetscTableFind - checks if a key is in the table

    If data==0, then no table entry exists.
 
*/
PETSC_STATIC_INLINE PetscErrorCode  PetscTableFind(PetscTable ta,PetscInt key,PetscInt *data)
{  
  PetscInt hash,ii = 0;

  PetscFunctionBegin;
  *data = 0;
  if (key <= 0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Key <= 0");
  if (key > ta->maxkey) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"key %D is greater than largest key allowed %D",key,ta->maxkey);

  hash  = (PetscInt)HASHT(ta,key);
  while (ii++ < ta->tablesize) {
    if (!ta->keytable[hash]) break;
    else if (ta->keytable[hash] == key) { 
      *data = ta->table[hash]; 
      break; 
    }
    hash = (hash == (ta->tablesize-1)) ? 0 : hash+1; 
  }
  PetscFunctionReturn(0);
}

PETSC_EXTERN_CXX_END
#endif
