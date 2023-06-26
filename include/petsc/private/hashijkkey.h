#ifndef PETSC_HASHIJKKEY_H
#define PETSC_HASHIJKKEY_H

typedef struct _PetscHashIJKKey {
  PetscInt i, j, k;
} PetscHashIJKKey;

#define PetscHashIJKKeyHash(key)     PetscHashCombine(PetscHashInt((key).i), PetscHashCombine(PetscHashInt((key).j), PetscHashInt((key).k)))
#define PetscHashIJKKeyEqual(k1, k2) ((k1).i == (k2).i && (k1).j == (k2).j && (k1).k == (k2).k)

#endif /* PETSC_HASHIJKKEY_H */
