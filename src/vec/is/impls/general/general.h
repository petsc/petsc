
#if !defined(__GENERAL_H)
#define __GENERAL_H

/*
    Defines the data structure used for the general index set
*/
#include "private/isimpl.h"

typedef struct {
  PetscInt   N;         /* number of indices */ 
  PetscInt   n;         /* local number of indices */ 
  PetscTruth sorted;    /* indicates the indices are sorted */ 
  PetscTruth allocated; /* did we allocate the index array ourselves? */
  PetscInt   *idx;
} IS_General;

#endif
