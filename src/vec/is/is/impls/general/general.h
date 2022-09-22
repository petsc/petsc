
#ifndef __GENERAL_H
#define __GENERAL_H

/*
    Defines the data structure used for the general index set
*/
#include <petsc/private/isimpl.h>

typedef struct {
  PetscBool sorted;    /* indicates the indices are sorted */
  PetscBool allocated; /* did we allocate the index array ourselves? */
  PetscInt *idx;
} IS_General;

#endif
