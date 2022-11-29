#if !defined(PETSC_HASHMAPP_H)
  #define PETSC_HASHMAPP_H

  #include <petsc/private/hashmap.h>

/*
  Hash map from PetscInt64 --> PetscObject*
*/
PETSC_HASH_MAP(HMapObj, PetscInt64, PetscObject, PetscHashInt, PetscHashEqual, NULL)

#endif /* PETSC_HASHMAPP_H */
