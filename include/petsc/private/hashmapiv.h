#if !defined(_PETSC_HASHMAPIV_H)
#define _PETSC_HASHMAPIV_H

#include <petsc/private/hashmap.h>

/*
 * Hash map from PetscInt --> PetscScalar
 * */
PETSC_HASH_MAP(HMapIV, PetscInt, PetscScalar, PetscHashInt, PetscHashEqual, -1)

/*MC
  PetscHMapIVAddValue - Add value to the value of a given key if the key exists,
  otherwise, insert a new (key,value) entry in the hash table

  Synopsis:
  #include <petsc/private/hashmapiv.h>
  PetscErrorCode PetscHMapIVAddValue(PetscHMapT ht,KeyType key,ValType val)

  Input Parameters:
+ ht  - The hash table
. key - The key
- val - The value

  Level: developer

.seealso: PetscHMapTGet(), PetscHMapTIterSet(), PetscHMapIVSet()
M*/
static inline
PetscErrorCode PetscHMapIVAddValue(PetscHMapIV ht,PetscInt key,PetscScalar val)
{
  int      ret;
  khiter_t iter;
  PetscFunctionBeginHot;
  PetscValidPointer(ht,1);
  iter = kh_put(HMapIV,ht,key,&ret);
  PetscHashAssert(ret>=0);
  if (ret) kh_val(ht,iter) = val;
  else  kh_val(ht,iter) += val;
  PetscFunctionReturn(0);
}

#endif /* _PETSC_HASHMAPIV_H */
