#if !defined(_FreeSpace_h_)
#define _FreeSpace_h_

#include "petsc.h"

typedef struct _Space *FreeSpaceList;

typedef struct _Space {
  FreeSpaceList more_space;
  PetscInt      *array;
  PetscInt      *array_head;
  PetscInt      total_array_size;
  PetscInt      local_used;
  PetscInt      local_remaining;
} FreeSpace;  

PetscErrorCode GetMoreSpace(PetscInt,FreeSpaceList*);
PetscErrorCode MakeSpaceContiguous(FreeSpaceList*,PetscInt *);
PetscErrorCode DestroySpace(FreeSpaceList);

#endif
