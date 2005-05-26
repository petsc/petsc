#ifndef __PETSCCTABLE_H
#define __PETSCCTABLE_H
/* Contributed by - Mark Adams */

typedef struct _n_PetscTable* PetscTable;
typedef PetscInt* PetscTablePosition;  

EXTERN PetscErrorCode PETSC_DLLEXPORT PetscTableCreate(const PetscInt,PetscTable*);
EXTERN PetscErrorCode PETSC_DLLEXPORT PetscTableCreateCopy(const PetscTable,PetscTable*);
EXTERN PetscErrorCode PETSC_DLLEXPORT PetscTableDelete(PetscTable);
EXTERN PetscErrorCode PETSC_DLLEXPORT PetscTableGetCount(const PetscTable,PetscInt*);
EXTERN PetscErrorCode PETSC_DLLEXPORT PetscTableIsEmpty(const PetscTable,PetscInt*);
EXTERN PetscErrorCode PETSC_DLLEXPORT PetscTableAdd(PetscTable,const PetscInt,const PetscInt);
EXTERN PetscErrorCode PETSC_DLLEXPORT PetscTableFind(PetscTable,const PetscInt,PetscInt*);
EXTERN PetscErrorCode PETSC_DLLEXPORT PetscTableGetHeadPosition(PetscTable,PetscTablePosition*);
EXTERN PetscErrorCode PETSC_DLLEXPORT PetscTableGetNext(PetscTable,PetscTablePosition*,PetscInt*,PetscInt*);
EXTERN PetscErrorCode PETSC_DLLEXPORT PetscTableRemoveAll(PetscTable);

#endif
