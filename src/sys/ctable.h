/* Contributed by - Mark Adams */

#if !defined(__CTABLE_H)
#define __CTABLE_H

typedef PetscInt* PetscTablePosition;  
typedef struct _p_PetscTable {
  PetscInt *keytable;
  PetscInt *table;
  PetscInt count;
  PetscInt tablesize;
  PetscInt head;
} *PetscTable;

EXTERN PetscErrorCode PetscTableCreate(const PetscInt,PetscTable*);
EXTERN PetscErrorCode PetscTableCreateCopy(const PetscTable,PetscTable*);
EXTERN PetscErrorCode PetscTableDelete(PetscTable);
EXTERN PetscErrorCode PetscTableGetCount(const PetscTable,PetscInt*);
EXTERN PetscErrorCode PetscTableIsEmpty(const PetscTable,PetscInt*);
EXTERN PetscErrorCode PetscTableAdd(PetscTable,const PetscInt,const PetscInt);
EXTERN PetscErrorCode PetscTableFind(PetscTable,const PetscInt,PetscInt*);
EXTERN PetscErrorCode PetscTableGetHeadPosition(PetscTable,PetscTablePosition*);
EXTERN PetscErrorCode PetscTableGetNext(PetscTable,PetscTablePosition*,PetscInt*,PetscInt*);
EXTERN PetscErrorCode PetscTableRemoveAll(PetscTable);

#endif
