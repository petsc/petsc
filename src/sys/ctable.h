/* $Id: ctable.h,v 1.5 1999/09/16 18:47:18 balay Exp bsmith $ */
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

extern int PetscTableCreate(const int size,PetscTable *ta);
extern int PetscTableCreateCopy(const PetscTable intable,PetscTable *rta);
extern int PetscTableDelete(PetscTable ta);
extern int PetscTableGetCount(const PetscTable ta,int *count);
extern int PetscTableIsEmpty(const PetscTable ta, int* flag);
extern int PetscTableAdd(PetscTable ta, const int key, const int data);
extern int PetscTableFind(PetscTable ta, const int key,int *data);
extern int PetscTableGetHeadPosition(PetscTable ta, PetscTablePosition *);
extern int PetscTableGetNext(PetscTable ta, PetscTablePosition *rPosition, int *pkey, int *data);
extern int PetscTableRemoveAll(PetscTable ta);

#endif
