#pragma once

typedef struct _PetscHashIJKLKey {
  PetscInt i, j, k, l;
} PetscHashIJKLKey;

#define PetscHashIJKLKeyHash(key) PetscHashCombine(PetscHashCombine(PetscHashInt((key).i), PetscHashInt((key).j)), PetscHashCombine(PetscHashInt((key).k), PetscHashInt((key).l)))

#define PetscHashIJKLKeyEqual(k1, k2) ((k1).i == (k2).i && (k1).j == (k2).j && (k1).k == (k2).k && (k1).l == (k2).l)
