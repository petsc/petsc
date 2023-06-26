#ifndef PETSC_HASHMAPIJKL_H
#define PETSC_HASHMAPIJKL_H

#include <petsc/private/hashmap.h>
#include <petsc/private/hashijklkey.h>

PETSC_HASH_MAP(HMapIJKL, PetscHashIJKLKey, PetscInt, PetscHashIJKLKeyHash, PetscHashIJKLKeyEqual, -1)

#endif /* PETSC_HASHMAPIJKL_H */
