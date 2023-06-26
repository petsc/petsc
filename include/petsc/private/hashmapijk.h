#ifndef PETSC_HASHMAPIJK_H
#define PETSC_HASHMAPIJK_H

#include <petsc/private/hashmap.h>
#include <petsc/private/hashijkkey.h>

PETSC_HASH_MAP(HMapIJK, PetscHashIJKKey, PetscInt, PetscHashIJKKeyHash, PetscHashIJKKeyEqual, -1)

#endif /* PETSC_HASHMAPIJK_H */
