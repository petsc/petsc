/* Contributed by - Mark Adams */

#if !defined(__CTABLE_H)
#define __CTABLE_H

typedef int* PetscTablePosition;  
typedef struct _p_PetscTable {
  int *keytable;
  int *table;
  int count;
  int tablesize;
  int head;
} *PetscTable;

EXTERN PetscErrorCode PetscTableCreate(const int,PetscTable*);
EXTERN PetscErrorCode PetscTableCreateCopy(const PetscTable,PetscTable*);
EXTERN PetscErrorCode PetscTableDelete(PetscTable);
EXTERN PetscErrorCode PetscTableGetCount(const PetscTable,int*);
EXTERN PetscErrorCode PetscTableIsEmpty(const PetscTable,int*);
EXTERN PetscErrorCode PetscTableAdd(PetscTable,const int,const int);
EXTERN PetscErrorCode PetscTableFind(PetscTable,const int,int*);
EXTERN PetscErrorCode PetscTableGetHeadPosition(PetscTable,PetscTablePosition*);
EXTERN PetscErrorCode PetscTableGetNext(PetscTable,PetscTablePosition*,int*,int*);
EXTERN PetscErrorCode PetscTableRemoveAll(PetscTable);

#endif
