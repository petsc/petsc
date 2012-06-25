#ifndef _petscheap_h
#define _petscheap_h

#include <petscsys.h>

typedef struct _PetscHeap *PetscHeap;

PETSC_EXTERN PetscErrorCode PetscHeapCreate(PetscInt,PetscHeap*);
PETSC_EXTERN PetscErrorCode PetscHeapAdd(PetscHeap,PetscInt,PetscInt);
PETSC_EXTERN PetscErrorCode PetscHeapPop(PetscHeap,PetscInt*,PetscInt*);
PETSC_EXTERN PetscErrorCode PetscHeapPeek(PetscHeap,PetscInt*,PetscInt*);
PETSC_EXTERN PetscErrorCode PetscHeapStash(PetscHeap,PetscInt,PetscInt);
PETSC_EXTERN PetscErrorCode PetscHeapUnstash(PetscHeap);
PETSC_EXTERN PetscErrorCode PetscHeapDestroy(PetscHeap*);
PETSC_EXTERN PetscErrorCode PetscHeapView(PetscHeap,PetscViewer);

#endif
