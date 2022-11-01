#ifndef PETSC_HASHIJKEY_H
#define PETSC_HASHIJKEY_H

#define PETSC_HASHIJKEY

typedef struct _PetscHashIJKey {
  PetscInt i, j;
} PetscHashIJKey;

#define PetscHashIJKeyHash(key)     PetscHashCombine(PetscHashInt((key).i), PetscHashInt((key).j))
#define PetscHashIJKeyEqual(k1, k2) (((k1).i == (k2).i) ? ((k1).j == (k2).j) : 0)

#endif /* PETSC_HASHIJKEY_H */
