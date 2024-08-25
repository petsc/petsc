#pragma once

#include <petsc/private/hashmap.h>

/*
  Hash map from PetscInt64 --> PetscObject
*/
PETSC_HASH_MAP(HMapObj, PetscInt64, PetscObject, PetscHashInt64, PetscHashEqual, NULL)
