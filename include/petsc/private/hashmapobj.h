#ifndef PETSC_HASHMAPOBJ_H
#define PETSC_HASHMAPOBJ_H

#include <petsc/private/hashmap.h>

/*
  Hash map from PetscInt64 --> PetscObject
*/
PETSC_HASH_MAP(HMapObj, PetscInt64, PetscObject, PetscHashInt, PetscHashEqual, NULL)

#endif /* PETSC_HASHMAPOBJ_H */
