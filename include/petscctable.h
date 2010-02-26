#ifndef __PETSCCTABLE_H
#define __PETSCCTABLE_H
#include "petscsys.h"
PETSC_EXTERN_CXX_BEGIN
/* Contributed by - Mark Adams */

typedef struct _n_PetscTable* PetscTable;
typedef PetscInt* PetscTablePosition;  

EXTERN PetscErrorCode PETSC_DLLEXPORT PetscTableCreate(const PetscInt,PetscTable*);
EXTERN PetscErrorCode PETSC_DLLEXPORT PetscTableCreateCopy(const PetscTable,PetscTable*);
EXTERN PetscErrorCode PETSC_DLLEXPORT PetscTableDestroy(PetscTable);
EXTERN PetscErrorCode PETSC_DLLEXPORT PetscTableGetCount(const PetscTable,PetscInt*);
EXTERN PetscErrorCode PETSC_DLLEXPORT PetscTableIsEmpty(const PetscTable,PetscInt*);
EXTERN PetscErrorCode PETSC_DLLEXPORT PetscTableAdd(PetscTable,const PetscInt,const PetscInt);
EXTERN PetscErrorCode PETSC_DLLEXPORT PetscTableAddCount(PetscTable,const PetscInt);
EXTERN PetscErrorCode PETSC_DLLEXPORT PetscTableFind(PetscTable,const PetscInt,PetscInt*);
EXTERN PetscErrorCode PETSC_DLLEXPORT PetscTableGetHeadPosition(PetscTable,PetscTablePosition*);
EXTERN PetscErrorCode PETSC_DLLEXPORT PetscTableGetNext(PetscTable,PetscTablePosition*,PetscInt*,PetscInt*);
EXTERN PetscErrorCode PETSC_DLLEXPORT PetscTableRemoveAll(PetscTable);
PETSC_EXTERN_CXX_END
#endif
