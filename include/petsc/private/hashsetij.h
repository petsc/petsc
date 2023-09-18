#pragma once

#include <petsc/private/hashset.h>
#include <petsc/private/hashijkey.h>

PETSC_HASH_SET(HSetIJ, PetscHashIJKey, PetscHashIJKeyHash, PetscHashIJKeyEqual)
