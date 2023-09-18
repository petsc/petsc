#pragma once

#include <petsc/private/hashmap.h>
#include <petsc/private/hashijklkey.h>

PETSC_HASH_MAP(HMapIJKL, PetscHashIJKLKey, PetscInt, PetscHashIJKLKeyHash, PetscHashIJKLKeyEqual, -1)
