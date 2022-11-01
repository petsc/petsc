#ifndef PETSC_HASHSETIJ_H
#define PETSC_HASHSETIJ_H

#include <petsc/private/hashset.h>
#include <petsc/private/hashijkey.h>

PETSC_HASH_SET(HSetIJ, PetscHashIJKey, PetscHashIJKeyHash, PetscHashIJKeyEqual)

#endif /* PETSC_HASHSETIJ_H */
