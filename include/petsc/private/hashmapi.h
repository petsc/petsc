#ifndef PETSC_HASHMAPI_H
#define PETSC_HASHMAPI_H

#include <petsc/private/hashmap.h>

PETSC_HASH_MAP(HMapI, PetscInt, PetscInt, PetscHashInt, PetscHashEqual, -1)

#endif /* PETSC_HASHMAPI_H */
