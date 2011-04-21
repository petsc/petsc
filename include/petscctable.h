#ifndef __PETSCCTABLE_H
#define __PETSCCTABLE_H
#include "petscsys.h"
PETSC_EXTERN_CXX_BEGIN
/* Contributed by - Mark Adams */

typedef struct _n_PetscTable* PetscTable;
typedef PetscInt* PetscTablePosition;  

extern PetscErrorCode  PetscTableCreate(const PetscInt,PetscTable*);
extern PetscErrorCode  PetscTableCreateCopy(const PetscTable,PetscTable*);
extern PetscErrorCode  PetscTableDestroy(PetscTable*);
extern PetscErrorCode  PetscTableGetCount(const PetscTable,PetscInt*);
extern PetscErrorCode  PetscTableIsEmpty(const PetscTable,PetscInt*);
extern PetscErrorCode  PetscTableAdd(PetscTable,const PetscInt,const PetscInt);
extern PetscErrorCode  PetscTableAddCount(PetscTable,const PetscInt);
extern PetscErrorCode  PetscTableFind(PetscTable,const PetscInt,PetscInt*);
extern PetscErrorCode  PetscTableGetHeadPosition(PetscTable,PetscTablePosition*);
extern PetscErrorCode  PetscTableGetNext(PetscTable,PetscTablePosition*,PetscInt*,PetscInt*);
extern PetscErrorCode  PetscTableRemoveAll(PetscTable);
PETSC_EXTERN_CXX_END
#endif
