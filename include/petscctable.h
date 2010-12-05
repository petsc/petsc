#ifndef __PETSCCTABLE_H
#define __PETSCCTABLE_H
#include "petscsys.h"
PETSC_EXTERN_CXX_BEGIN
/* Contributed by - Mark Adams */

typedef struct _n_PetscTable* PetscTable;
typedef PetscInt* PetscTablePosition;  

extern PetscErrorCode PETSCSYS_DLLEXPORT PetscTableCreate(const PetscInt,PetscTable*);
extern PetscErrorCode PETSCSYS_DLLEXPORT PetscTableCreateCopy(const PetscTable,PetscTable*);
extern PetscErrorCode PETSCSYS_DLLEXPORT PetscTableDestroy(PetscTable);
extern PetscErrorCode PETSCSYS_DLLEXPORT PetscTableGetCount(const PetscTable,PetscInt*);
extern PetscErrorCode PETSCSYS_DLLEXPORT PetscTableIsEmpty(const PetscTable,PetscInt*);
extern PetscErrorCode PETSCSYS_DLLEXPORT PetscTableAdd(PetscTable,const PetscInt,const PetscInt);
extern PetscErrorCode PETSCSYS_DLLEXPORT PetscTableAddCount(PetscTable,const PetscInt);
extern PetscErrorCode PETSCSYS_DLLEXPORT PetscTableFind(PetscTable,const PetscInt,PetscInt*);
extern PetscErrorCode PETSCSYS_DLLEXPORT PetscTableGetHeadPosition(PetscTable,PetscTablePosition*);
extern PetscErrorCode PETSCSYS_DLLEXPORT PetscTableGetNext(PetscTable,PetscTablePosition*,PetscInt*,PetscInt*);
extern PetscErrorCode PETSCSYS_DLLEXPORT PetscTableRemoveAll(PetscTable);
PETSC_EXTERN_CXX_END
#endif
