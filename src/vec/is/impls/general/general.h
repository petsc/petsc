
#if !defined(__GENERAL_H)
#define __GENERAL_H

/*
    Defines the data structure used for the general index set
*/
#include "src/vec/is/isimpl.h"
#include "petscsys.h"

typedef struct {
  PetscInt   N;         /* number of indices */ 
  PetscInt   n;         /* local number of indices */ 
  PetscTruth sorted;    /* indicates the indices are sorted */ 
  PetscInt   *idx;
} IS_General;

#endif
