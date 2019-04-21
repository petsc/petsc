/*
 * hashmapiv.h
 *
 *  Created on: Apr 2, 2019
 *      Author: Fande Kong
 */

#if !defined(_PETSC_HASHMAPIV_H)
#define _PETSC_HASHMAPIV_H

#include <petsc/private/hashmap.h>

PETSC_HASH_MAP(HMapIV, PetscInt, PetscScalar, PetscHashInt, PetscHashEqual, -1)

#endif /* _PETSC_HASHMAPIV_H */

