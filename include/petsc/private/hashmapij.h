#pragma once

#include <petsc/private/hashmap.h>
#include <petsc/private/hashijkey.h>

PETSC_HASH_MAP(HMapIJ, PetscHashIJKey, PetscInt, PetscHashIJKeyHash, PetscHashIJKeyEqual, -1)
