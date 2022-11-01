#ifndef PETSC_HASHMAPIJ_H
#define PETSC_HASHMAPIJ_H

#include <petsc/private/hashmap.h>
#include <petsc/private/hashijkey.h>

PETSC_HASH_MAP(HMapIJ, PetscHashIJKey, PetscInt, PetscHashIJKeyHash, PetscHashIJKeyEqual, -1)

#endif /* PETSC_HASHMAPIJ_H */
