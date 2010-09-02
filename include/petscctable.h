#ifndef __PETSCCTABLE_H
#define __PETSCCTABLE_H
#include "petscsys.h"
PETSC_EXTERN_CXX_BEGIN
/* Contributed by - Mark Adams */

typedef struct _n_PetscTable* PetscTable;
typedef PetscInt* PetscTablePosition;  

EXTERN PetscErrorCode PETSCSYS_DLLEXPORT PetscTableCreate(const PetscInt,PetscTable*);
EXTERN PetscErrorCode PETSCSYS_DLLEXPORT PetscTableCreateCopy(const PetscTable,PetscTable*);
EXTERN PetscErrorCode PETSCSYS_DLLEXPORT PetscTableDestroy(PetscTable);
EXTERN PetscErrorCode PETSCSYS_DLLEXPORT PetscTableGetCount(const PetscTable,PetscInt*);
EXTERN PetscErrorCode PETSCSYS_DLLEXPORT PetscTableIsEmpty(const PetscTable,PetscInt*);
EXTERN PetscErrorCode PETSCSYS_DLLEXPORT PetscTableAdd(PetscTable,const PetscInt,const PetscInt);
EXTERN PetscErrorCode PETSCSYS_DLLEXPORT PetscTableAddCount(PetscTable,const PetscInt);
EXTERN PetscErrorCode PETSCSYS_DLLEXPORT PetscTableFind(PetscTable,const PetscInt,PetscInt*);
EXTERN PetscErrorCode PETSCSYS_DLLEXPORT PetscTableGetHeadPosition(PetscTable,PetscTablePosition*);
EXTERN PetscErrorCode PETSCSYS_DLLEXPORT PetscTableGetNext(PetscTable,PetscTablePosition*,PetscInt*,PetscInt*);
EXTERN PetscErrorCode PETSCSYS_DLLEXPORT PetscTableRemoveAll(PetscTable);
PETSC_EXTERN_CXX_END
#endif
