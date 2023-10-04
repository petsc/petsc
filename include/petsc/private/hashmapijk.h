#pragma once

#include <petsc/private/hashmap.h>
#include <petsc/private/hashijkkey.h>

PETSC_HASH_MAP(HMapIJK, PetscHashIJKKey, PetscInt, PetscHashIJKKeyHash, PetscHashIJKKeyEqual, -1)
