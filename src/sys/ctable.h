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

EXTERN int PetscTableCreate(const int size,PetscTable *ta);
EXTERN int PetscTableCreateCopy(const PetscTable intable,PetscTable *rta);
EXTERN int PetscTableDelete(PetscTable ta);
EXTERN int PetscTableGetCount(const PetscTable ta,int *count);
EXTERN int PetscTableIsEmpty(const PetscTable ta,int* flag);
EXTERN int PetscTableAdd(PetscTable ta,const int key,const int data);
EXTERN int PetscTableFind(PetscTable ta,const int key,int *data);
EXTERN int PetscTableGetHeadPosition(PetscTable ta,PetscTablePosition *);
EXTERN int PetscTableGetNext(PetscTable ta,PetscTablePosition *rPosition,int *pkey,int *data);
EXTERN int PetscTableRemoveAll(PetscTable ta);

#endif
