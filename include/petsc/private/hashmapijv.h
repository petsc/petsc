#ifndef _PETSC_HASHMAPIJV_H
#define _PETSC_HASHMAPIJV_H

#include <petsc/private/hashmap.h>
#include <petsc/private/hashijkey.h>

/* SUBMANSEC = Sys */
/*
   Hash map from (PetscInt,PetscInt) --> PetscScalar
*/
PETSC_HASH_MAP(HMapIJV, PetscHashIJKey, PetscScalar, PetscHashIJKeyHash, PetscHashIJKeyEqual, -1)

/*MC
  PetscHMapIJVQueryAdd - Add value to the value of a given key if the key exists,
  otherwise, insert a new (key,value) entry in the hash table

  Synopsis:
  #include <petsc/private/hashmapijv.h>
  PetscErrorCode PetscHMapIJVQueryAdd(PetscHMapT ht,PetscHashIJKey key,PetscScalar val,PetscBool *missing)

  Input Parameters:
+ ht  - The hash table
. key - The key
- val - The value

  Output Parameter:
. missing - `PETSC_TRUE` if the `PetscHMapIJV` did not already have the given key

  Level: developer

.seealso: `PetscHMapIJVSetWithMode()`, `PetscHMapIJV`, `PetscHMapIJVGet()`, `PetscHMapIJVIterSet()`, `PetscHMapIJVSet()`
M*/
static inline PetscErrorCode PetscHMapIJVQueryAdd(PetscHMapIJV ht, PetscHashIJKey key, PetscScalar val, PetscBool *missing)
{
  int      ret;
  khiter_t iter;
  PetscFunctionBeginHot;
  PetscValidPointer(ht, 1);
  iter = kh_put(HMapIJV, ht, key, &ret);
  PetscHashAssert(ret >= 0);
  if (ret) kh_val(ht, iter) = val;
  else kh_val(ht, iter) += val;
  *missing = ret ? PETSC_TRUE : PETSC_FALSE;
  PetscFunctionReturn(PETSC_SUCCESS);
}
#endif /* _PETSC_HASHMAPIJV_H */
