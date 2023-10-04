#pragma once

#include <petsc/private/hashmap.h>

#define PETSC_HMAPI_HAVE_EXTENDED_API 1

PETSC_HASH_MAP_EXTENDED(HMapI, PetscInt, PetscInt, PetscHashInt, PetscHashEqual, -1)
