#if !defined(PETSC_HASHSETIJ_H)
#define PETSC_HASHSETIJ_H

#include <petsc/private/hashset.h>

#if !defined(PETSC_HASHIJKEY)
#define PETSC_HASHIJKEY
typedef struct _PetscHashIJKey { PetscInt i, j; } PetscHashIJKey;
#define PetscHashIJKeyHash(key) PetscHashCombine(PetscHashInt((key).i),PetscHashInt((key).j))
#define PetscHashIJKeyEqual(k1,k2) (((k1).i == (k2).i) ? ((k1).j == (k2).j) : 0)
#endif

PETSC_HASH_SET(HSetIJ, PetscHashIJKey, PetscHashIJKeyHash, PetscHashIJKeyEqual)

#endif /* PETSC_HASHSETIJ_H */
